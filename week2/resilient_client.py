"""
Day 9 — production reliability around Claude API calls: exponential backoff
(tenacity), timeouts, explicit 429/529 handling, a fallback model chain, a
hard cost cap, and structured (JSON) logging.

This wraps cost_tracker.log_call rather than replacing it, and reads the
SAME usage_log.csv every other script has been writing to — so the cost cap
below checks your REAL cumulative spend across the whole roadmap so far,
not just this one script.

    uv add anthropic tenacity python-dotenv
    uv run python resilient_client.py

Verified against Anthropic's current API error docs (July 2026) before
writing this — recheck platform.claude.com/docs/en/api/errors occasionally,
error-handling guidance is exactly the kind of thing that quietly changes.
"""

import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone

import anthropic
from tenacity import (
    retry, retry_if_exception, stop_after_attempt, wait_random_exponential,
)

# Reuse ONE shared cost-tracking file instead of duplicating it per folder.
# `from ./week1/cost_calculator import ...` is not valid Python — imports
# are dotted MODULE NAMES, never filesystem paths with slashes. To reach a
# module in a sibling folder, you add that folder to sys.path first (using
# this file's own location, not the current working directory — same fix
# as the learninglog/ path-resolution issue from a few days ago), THEN
# import it normally by name. Works whether you kept the file as
# cost_calculator.py (week1) or cost_tracker.py (as originally delivered).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (
    os.path.join(_THIS_DIR, "..", "week1"),  # week1 as a SIBLING of this script's folder (your actual layout)
    os.path.join(_THIS_DIR, ".."),           # shared file sitting at the project root instead
    _THIS_DIR,                                # shared file copied into this same folder
):
    sys.path.insert(0, os.path.abspath(_candidate))

try:
    from cost_calculator import log_call, CSV_PATH
except ImportError:
    from cost_tracker import log_call, CSV_PATH

# ── structured logging ───────────────────────────────────────────────────
# "Structured" just means: every log line is one JSON object with a
# consistent shape, not a free-text sentence. A human can still read it, but
# now it's also grep-able and aggregatable — "show me every retry where
# wait_seconds > 5", "count fallback_triggered events today" — without
# parsing prose. No extra dependency needed, just discipline about the
# shape of what you log.
logger = logging.getLogger("resilient_client")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))  # the message IS the JSON
logger.addHandler(_handler)
logger.propagate = False


def log_event(event, level=logging.INFO, **fields):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    logger.log(level, json.dumps(payload, default=str))


# ── cost cap ──────────────────────────────────────────────────────────────
# Retries and fallback models make your code more resilient to API hiccups,
# but they don't protect your WALLET — a bug that calls the API in a tight
# loop will happily retry itself into a real bill. This is a hard circuit
# breaker: before spending anything, check what you've ALREADY spent
# (summed straight from usage_log.csv, so it reflects every script you've
# run, not just this session) and refuse outright if you're over budget.
MAX_SPEND_EUR = 15.00  # tune to your actual monthly budget


class CostCapExceeded(Exception):
    pass


def current_spend_eur(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        return 0.0
    total = 0.0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += float(row["cost_eur"])
    return total


def check_cost_cap(cap_eur=MAX_SPEND_EUR):
    spent = current_spend_eur()
    if spent >= cap_eur:
        log_event("cost_cap_blocked", level=logging.WARNING,
                   spent_eur=round(spent, 4), cap_eur=cap_eur)
        raise CostCapExceeded(
            f"Refusing to call the API: usage_log.csv already shows €{spent:.4f} "
            f"spent against a €{cap_eur:.2f} cap. Raise MAX_SPEND_EUR if that's intentional."
        )
    return spent


# ── which errors are worth retrying ──────────────────────────────────────
# Straight from Anthropic's error docs: 409/429/500/504/529 are transient —
# someone else's server hiccup, or you briefly hit a rate limit — so
# retrying with backoff can succeed on attempt 2 or 3. 400/401/402/403/404/413
# are YOUR request being wrong (bad JSON, bad key, no permission, payload
# too big) — retrying just re-sends the same broken request and fails
# identically every time, burning attempts and time for nothing. Blindly
# doing `except anthropic.APIStatusError: retry` would retry ALL of these,
# including the ones that can never succeed — that's the mistake this
# predicate exists to avoid.
RETRYABLE_STATUS_CODES = {409, 429, 500, 504, 529}


def is_retryable(exc):
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return True  # connection-level issues have no status_code, but are transient
    return getattr(exc, "status_code", None) in RETRYABLE_STATUS_CODES


def _before_sleep(retry_state):
    exc = retry_state.outcome.exception()
    log_event(
        "retrying", level=logging.WARNING,
        attempt=retry_state.attempt_number,
        wait_seconds=round(retry_state.next_action.sleep, 2),
        error_type=type(exc).__name__,
        status_code=getattr(exc, "status_code", None),
    )


@retry(
    retry=retry_if_exception(is_retryable),
    # wait_random_exponential = exponential backoff WITH jitter (randomized
    # within the exponential curve) — without jitter, every client that got
    # rate-limited at the same moment retries at the same moment again,
    # which just recreates the pile-up you were trying to recover from.
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(4),
    before_sleep=_before_sleep,
    reraise=True,  # after exhausting attempts, raise the REAL exception, not tenacity's wrapper
)
def _call_model(client, model, messages, max_tokens, timeout):
    # .create() itself has no timeout= parameter — a per-request override
    # requires .with_options(timeout=...). The client-level Anthropic(...)
    # constructor default is 10 MINUTES, way too long to sit waiting during
    # an outage, so an explicit override here matters.
    return client.with_options(timeout=timeout).messages.create(
        model=model, max_tokens=max_tokens, messages=messages,
    )


DEFAULT_MODEL_CHAIN = ["claude-haiku-4-5", "claude-sonnet-5"]


def call_resilient(messages, model_chain=None, max_tokens=500, timeout=30.0, cap_eur=MAX_SPEND_EUR):
    """Try each model in model_chain in order. A model that's still failing
    after its full retry budget triggers fallback to the NEXT model, rather
    than failing the whole request outright — e.g. if claude-haiku-4-5 is
    overloaded (529) for an extended stretch, fall through to
    claude-sonnet-5 instead of making the user wait or see an error."""
    check_cost_cap(cap_eur)  # refuse BEFORE spending anything, if already over budget

    model_chain = model_chain or DEFAULT_MODEL_CHAIN
    client = anthropic.Anthropic()
    last_error = None

    for i, model in enumerate(model_chain):
        log_event("attempting_model", model=model, is_fallback=(i > 0))
        try:
            response = _call_model(client, model, messages, max_tokens, timeout)
        except Exception as e:
            last_error = e
            log_event(
                "model_exhausted", level=logging.ERROR, model=model,
                error_type=type(e).__name__, status_code=getattr(e, "status_code", None),
            )
            continue  # fall through to the next model in the chain

        cost_eur = log_call("anthropic", model, response.usage.input_tokens, response.usage.output_tokens)
        log_event(
            "call_succeeded", model=model, used_fallback=(i > 0),
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
            cost_eur=round(cost_eur, 6),
        )
        return response

    log_event("all_models_failed", level=logging.ERROR, models_tried=model_chain,
              final_error=type(last_error).__name__ if last_error else None)
    raise RuntimeError(f"All models in {model_chain} failed. Last error: {last_error}") from last_error


if __name__ == "__main__":
    try:
        response = call_resilient(
            [{"role": "user", "content": "Say hi in exactly five words."}],
            max_tokens=50,
        )
        print("\nFinal answer:", response.content[0].text)
    except CostCapExceeded as e:
        print(f"\nBlocked by cost cap: {e}")
    except RuntimeError as e:
        print(f"\nAll fallback models failed even after retries: {e}")