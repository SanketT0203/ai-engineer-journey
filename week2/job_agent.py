"""
Day 9 practical — harden Monday's job parser (fetch_and_extract_jobs.py).

Adds everything Tuesday's resilient_client.py covered, applied to a real
pipeline: tenacity retries, cross-PROVIDER model fallback (Claude -> OpenAI,
not just Claude -> Claude), a per-run budget limit (distinct from Tuesday's
lifetime cap — this one resets every run and protects THIS invocation from
runaway cost even if history is fine), and structured JSON logs.

    uv add anthropic openai instructor tenacity python-dotenv
    uv run python harden_job_parser.py

Uses your existing OPENAI_API_KEY (same one cost_tracker.track_openai_call
already reads) — no new key needed since you're already paying for it.

Why Claude -> OpenAI specifically, and not Claude -> Claude like Tuesday:
same-provider fallback only survives THAT provider's local hiccups (one
overloaded model). If Anthropic's whole API has an outage, every model in
an Anthropic-only chain fails together. Falling back to a different
provider on different infrastructure survives outages Tuesday's version
couldn't — and here it also happens to use capacity you've already paid for.

Model pricing note: I found conflicting info while researching current
OpenAI pricing (Aug 2026) — one source says gpt-4o-mini is still active at
$0.15/$0.60 per 1M tokens (matches cost_tracker.py, verified July 2026),
another suggested it may already be superseded by newer mini/nano models.
Went with gpt-4o-mini since it's the one I could verify against your own
already-working cost_tracker.py entry — double-check
platform.openai.com/docs/pricing before relying on this long-term, since
OpenAI's lineup has clearly been shifting.
"""

import json
import logging
import os
import re
import sys
import time

import anthropic
import instructor
import openai
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

# Reuse ONE shared cost-tracking file instead of duplicating it per folder —
# see resilient_client.py for the full explanation of why a plain
# `from cost_tracker import ...` isn't enough once files live in different
# week1/week2 folders, and why filesystem-path-style imports don't exist
# in Python at all.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (
    os.path.join(_THIS_DIR, "..", "week1"),  # week1 as a SIBLING of this script's folder (your actual layout)
    os.path.join(_THIS_DIR, ".."),           # shared file sitting at the project root instead
    _THIS_DIR,                                # shared file copied into this same folder
):
    sys.path.insert(0, os.path.abspath(_candidate))

try:
    from cost_calculator import log_call
except ImportError:
    from cost_tracker import log_call

from job_schema import JobPosting
from resilient_client import log_event, logger  # reuse Tuesday's structured-logging setup

API_URL = "https://www.arbeitnow.com/api/job-board-api"
KEYWORDS = [
    "ai engineer", "machine learning", "ml engineer", "data scientist",
    "llm", "artificial intelligence", "nlp", "deep learning",
    "genai", "generative ai", "agentic",
]

# Per-RUN budget: resets every time you run this script, independent of
# usage_log.csv history. Tuesday's CostCapExceeded protects your MONTH;
# this protects THIS invocation specifically — e.g. a bug that somehow
# fetches 500 postings instead of 10 shouldn't be allowed to spend freely
# just because you were nowhere near your monthly cap when it started.
RUN_BUDGET_EUR = 1.00


class RunBudgetExceeded(Exception):
    pass


# ── fetching + prompt building (same as Monday, copied in so this script
# doesn't depend on fetch_and_extract_jobs.py's module-level client setup
# running successfully just to import a helper function from it) ──────────

def strip_html(raw_html, max_chars=2500):
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def matches_keywords(job):
    haystack = (job.get("title", "") + " " + " ".join(job.get("tags", []))).lower()
    return any(kw in haystack for kw in KEYWORDS)


# Shared before_sleep callback for every @retry in this file (LLM calls
# further down AND the plain HTTP fetch loop above them) -- defined here,
# ahead of its first use, since a decorator argument is evaluated the
# moment the decorated function is defined, not when it's later called.
def _before_sleep(retry_state):
    exc = retry_state.outcome.exception()
    log_event(
        "retrying", level=logging.WARNING,
        attempt=retry_state.attempt_number,
        wait_seconds=round(retry_state.next_action.sleep, 2),
        error_type=type(exc).__name__,
        status_code=getattr(exc, "status_code", None),
    )


def _is_rate_limit_error(exc):
    import httpx
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


# Arbeitnow is a free public API with no published rate limit or docs on
# one -- your run hit a real 429 on page 11. This didn't show up before
# because harden_job_parser's own default was target_count=10 (usually
# satisfied in a handful of pages); jobradar.py now asks for
# target_count=40 for a richer digest, which means more pages, which
# tripped a limit that was always there but never exercised. Same
# tenacity retry pattern already used for the LLM calls below -- rate
# limits aren't only an LLM-API thing, this plain HTTP loop can hit them
# too.
# stop_after_attempt bumped 4 -> 5 and max wait 20 -> 30: running this
# script twice back-to-back (e.g. re-testing a fix minutes apart) hit the
# SAME rate-limit window twice, and the old budget (worst case ~14s of
# retrying) gave up before Arbeitnow's window had a chance to reset. Still
# not unlimited -- see the try/except in fetch_matching_postings below for
# what happens when even this isn't enough.
@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    before_sleep=_before_sleep,
    reraise=True,
)
def _fetch_page(http, page):
    resp = http.get(API_URL, params={"page": page})
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_matching_postings(target_count=10, max_pages=15, pause_seconds=0.4):
    """pause_seconds: brief pause between page requests -- being polite to
    a free public API and avoiding the 429 in the first place is better
    than only reacting to it after the fact. _fetch_page's retry above is
    the backstop for whatever 429s get through anyway.

    If Arbeitnow is STILL rate-limiting after _fetch_page exhausts its own
    retries (e.g. two runs close together, or a longer rate-limit window
    than the retry budget covers), that's treated as a normal stopping
    condition here -- same as `if not data: break` a few lines down -- not
    a fatal error. Whatever postings were already matched on earlier pages
    are returned as-is rather than thrown away by an unhandled exception
    that kills the whole script. A shorter digest beats a crashed one."""
    import httpx
    matched = []
    with httpx.Client(timeout=30) as http:
        for page in range(1, max_pages + 1):
            try:
                data = _fetch_page(http, page)
            except httpx.HTTPStatusError as e:
                log_event(
                    "arbeitnow_fetch_gave_up", level=logging.WARNING,
                    page=page, status_code=e.response.status_code,
                    matched_so_far=len(matched),
                )
                print(f"    Arbeitnow still rate-limiting after retries (page {page}) -- "
                      f"continuing with the {len(matched)} matching posting(s) already fetched.")
                break
            if not data:
                break
            for job in data:
                if matches_keywords(job):
                    matched.append(job)
                    if len(matched) >= target_count:
                        return matched
            if page < max_pages:
                time.sleep(pause_seconds)
    return matched


def build_prompt(job, description_text):
    return (
        f"Extract structured job data from this posting.\n\n"
        f"Title: {job.get('title')}\n"
        f"Company: {job.get('company_name')}\n"
        f"Location: {job.get('location')}\n"
        f"Tags: {', '.join(job.get('tags', []))}\n\n"
        f"Description:\n{description_text}"
    )


# ── which errors are worth retrying, across BOTH SDKs ──────────────────
# Anthropic and OpenAI model errors the same way: an APIStatusError
# subclass with .status_code for real HTTP error responses, and a separate
# APIConnectionError/APITimeoutError for connection-level failures that
# never got a response at all. Same predicate shape as Tuesday, just
# checking both hierarchies.
RETRYABLE_STATUS_CODES = {409, 429, 500, 503, 504, 529}


def is_retryable(exc):
    connection_level = (
        anthropic.APIConnectionError, anthropic.APITimeoutError,
        openai.APIConnectionError, openai.APITimeoutError,
    )
    if isinstance(exc, connection_level):
        return True
    return getattr(exc, "status_code", None) in RETRYABLE_STATUS_CODES


@retry(
    retry=retry_if_exception(is_retryable),
    wait=wait_random_exponential(multiplier=1, max=20),
    stop=stop_after_attempt(4),
    before_sleep=_before_sleep,
    reraise=True,
)
def _resilient_call(call_fn, *args):
    return call_fn(*args)


# ── two providers, two SDKs, two call shapes ────────────────────────────
# Instructor patches each underlying client's own method, so the CALL
# SHAPE differs by provider (.messages.create vs .chat.completions.create,
# input_tokens/output_tokens vs prompt_tokens/completion_tokens) even
# though Instructor gives both the same response_model= interface on top.
# instructor_max_retries here is a SEPARATE, smaller retry budget for "the
# model returned JSON that didn't validate against JobPosting" — a
# completely different failure mode from tenacity's job above, which is
# retrying transport failures (429/529/timeouts). They compose cleanly
# because they catch different exception types.

def call_anthropic(client, model, prompt, schema):
    result = client.messages.create(
        model=model, max_tokens=500, temperature=0, max_retries=1,
        response_model=schema,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = result._raw_response.usage
    return result, usage.input_tokens, usage.output_tokens


def call_openai(client, model, prompt, schema):
    result = client.chat.completions.create(
        model=model, temperature=0, max_retries=1,
        response_model=schema,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = result._raw_response.usage
    return result, usage.prompt_tokens, usage.completion_tokens  # OpenAI naming, not input_tokens/output_tokens


PROVIDER_CHAIN = [
    {
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "build_client": lambda: instructor.from_anthropic(anthropic.Anthropic()),
        "call": call_anthropic,
    },
    {
        "provider": "openai",
        "model": "gpt-4o-mini",
        # openai.OpenAI() with no args reads OPENAI_API_KEY from the
        # environment automatically, same convention as anthropic.Anthropic()
        # — identical to what cost_tracker.track_openai_call() already does.
        "build_client": lambda: instructor.from_openai(openai.OpenAI(), mode=instructor.Mode.TOOLS),
        "call": call_openai,
    },
]

_client_cache = {}


def get_client(provider_cfg):
    key = provider_cfg["provider"]
    if key not in _client_cache:
        _client_cache[key] = provider_cfg["build_client"]()
    return _client_cache[key]


def extract_with_fallback(job, description_text, run_spent):
    """Try each provider in PROVIDER_CHAIN in order. Returns (JobPosting, cost_eur).
    Raises RunBudgetExceeded before spending anything if the per-run cap is
    already hit; raises RuntimeError if every provider fails."""
    if run_spent >= RUN_BUDGET_EUR:
        log_event("run_budget_blocked", level=logging.WARNING,
                   run_spent_eur=round(run_spent, 4), run_budget_eur=RUN_BUDGET_EUR)
        raise RunBudgetExceeded(
            f"Per-run budget hit: already spent €{run_spent:.4f} of €{RUN_BUDGET_EUR:.2f} this run."
        )

    prompt = build_prompt(job, description_text)
    last_error = None

    for i, cfg in enumerate(PROVIDER_CHAIN):
        log_event("attempting_provider", provider=cfg["provider"], model=cfg["model"], is_fallback=(i > 0))
        try:
            client = get_client(cfg)
            result, in_tok, out_tok = _resilient_call(cfg["call"], client, cfg["model"], prompt, JobPosting)
        except Exception as e:
            last_error = e
            log_event(
                "provider_exhausted", level=logging.ERROR,
                provider=cfg["provider"], model=cfg["model"],
                error_type=type(e).__name__, status_code=getattr(e, "status_code", None),
            )
            continue

        cost_eur = log_call(cfg["provider"], cfg["model"], in_tok, out_tok)
        log_event(
            "extraction_succeeded", provider=cfg["provider"], model=cfg["model"],
            used_fallback=(i > 0), input_tokens=in_tok, output_tokens=out_tok, cost_eur=round(cost_eur, 6),
        )
        result.url = job.get("url")
        return result, cost_eur

    log_event("all_providers_failed", level=logging.ERROR,
              providers_tried=[c["provider"] for c in PROVIDER_CHAIN],
              final_error=type(last_error).__name__ if last_error else None)
    raise RuntimeError(f"All providers failed for '{job.get('title')}'. Last error: {last_error}") from last_error


if __name__ == "__main__":
    print(f"Fetching AI-engineer postings (per-run budget: €{RUN_BUDGET_EUR:.2f})...")
    postings = fetch_matching_postings()
    print(f"Found {len(postings)} matching postings.\n")

    results = []
    run_spent = 0.0

    for i, job in enumerate(postings, 1):
        print(f"[{i}/{len(postings)}] {job.get('title')} @ {job.get('company_name')}")
        try:
            desc = strip_html(job.get("description", ""))
            structured, cost_eur = extract_with_fallback(job, desc, run_spent)
            run_spent += cost_eur
            results.append(structured)
            print(f"    -> extracted OK (run spend so far: €{run_spent:.4f})")
        except RunBudgetExceeded as e:
            print(f"    -> STOPPING: {e}")
            break  # keep whatever was already extracted; don't discard partial progress
        except RuntimeError as e:
            print(f"    -> FAILED (both providers exhausted), skipping this posting: {e}")

    with open("jobs_extracted_hardened.json", "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)}/{len(postings)} postings to jobs_extracted_hardened.json")
    print(f"Total run spend: €{run_spent:.4f} (cap was €{RUN_BUDGET_EUR:.2f})")