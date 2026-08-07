"""
JobRadar — the capstone. Daily pipeline:

    fetch new postings (Arbeitnow + RSS feeds)
    -> skip ones already seen
    -> structured extraction (retries, Claude->OpenAI fallback, per-run
       budget, structured logs -- reused directly from harden_job_parser.py,
       not rebuilt)
    -> score match against YOUR profile (job_profile.py): skills overlap +
       embedding similarity, combined
    -> ranked HTML + JSON digest

    uv add anthropic openai instructor tenacity sentence-transformers feedparser beautifulsoup4 httpx python-dotenv
    uv run python jobradar.py

Run this daily (Task Scheduler / cron / a scheduled task) -- jobradar_seen.json
tracks which postings you've already been shown, so reruns only spend API
budget on genuinely NEW postings, not the same ones every day.

Email delivery (top 5 matches + why each matched, sent after the digest
files are written): add to your .env --

    EMAIL_FROM=you@gmail.com
    EMAIL_APP_PASSWORD=xxxxxxxxxxxxxxxx   # NOT your normal Gmail password
    EMAIL_TO=you@gmail.com                # optional, defaults to EMAIL_FROM

An app password needs 2FA turned on for the Google account first --
Google Account -> Security -> 2-Step Verification -> App passwords. This
is still the right free option for a personal script sending ~1 email/day
to yourself in 2026 (checked current Gmail SMTP limits before wiring this
in: personal accounts cap at 500 sends/day, nowhere close to what a daily
digest needs). If EMAIL_FROM/EMAIL_APP_PASSWORD aren't set, sending is
skipped with a printed warning -- the JSON/HTML digest files still get
written either way, so a missing or wrong password never loses a day's
results, it just means you open the file yourself instead of getting the
email.

On the LinkedIn part of the brief, said plainly: LinkedIn does not offer an
official RSS feed for job search results. Third-party "RSS generator"
services scrape LinkedIn's search pages on your behalf to fake one --
LinkedIn's User Agreement explicitly prohibits automated scraping, and
they've pursued this legally before (hiQ Labs v. LinkedIn). That's a real
risk to your account, not a technicality, so I didn't wire one in by
default. What I DID build: fetch_rss_postings() works against ANY RSS/Atom
feed URL, and RSS_FEED_URLS ships with We Work Remotely's official,
ToS-compliant programming-jobs feed as a working default. If you still
want a LinkedIn feed via a third-party generator, that's your call to make
with the risk understood -- just add the URL to RSS_FEED_URLS below.
"""

import json
import os
import re
import smtplib
import sys
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Same cross-folder bootstrap as resilient_client.py / harden_job_parser.py
# -- lets this file live in week2/ (or wherever) and still find week1's
# cost file, resolved from THIS file's own location, not cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (
    os.path.join(_THIS_DIR, "..", "week1"),
    os.path.join(_THIS_DIR, ".."),
    _THIS_DIR,
):
    sys.path.insert(0, os.path.abspath(_candidate))

import job_agent as hjp  # fetch_matching_postings, strip_html, extract_with_fallback, log_event, ...
from job_profile import PROFILE_SKILLS, PROFILE_TEXT

# JobRadar's own per-run budget -- separate from harden_job_parser's default,
# since a daily run across TWO sources can process more postings than a
# single manual run did. extract_with_fallback() reads RUN_BUDGET_EUR as a
# MODULE-level global inside harden_job_parser (not a function parameter),
# so overriding it means setting the attribute on the imported module
# itself, not a local variable here.
JOBRADAR_RUN_BUDGET_EUR = 2.00
hjp.RUN_BUDGET_EUR = JOBRADAR_RUN_BUDGET_EUR

# Hard cap on how many postings get PAID extraction per run. The €2 budget
# above is a euro-based safety net (protects against a bug that somehow
# spends wildly); this is a count-based one, because at your actual
# per-posting cost (€0.1583 / 65 postings last run ≈ €0.0024 each) the euro
# cap alone would happily let ~800 postings through before it ever kicked
# in -- nowhere near what you asked for. Set to 10 if you want tighter.
MAX_POSTINGS_PER_RUN = 20

# The roadmap's own target for a full daily run, separate from both caps
# above -- those protect against runaway spend, this is what "on budget"
# actually means day to day. Checked against it explicitly after every run
# (see the __main__ block) rather than just trusting the €2 cap, since €2
# is 40x looser than this and wouldn't itself tell you if you'd drifted
# over the real target.
DAILY_COST_TARGET_EUR = 0.05

SEEN_PATH = "jobradar_seen.json"
DIGEST_JSON_PATH = "jobradar_digest.json"
DIGEST_HTML_PATH = "jobradar_digest.html"

TOP_N_FOR_EMAIL = 5

# Read from .env via the same convention every other credential in this
# project uses (uv auto-loads .env, so there's no explicit load_dotenv()
# call needed -- see the module docstring for what to set). Read once at
# import time, not inside send_digest_email(), so a missing config is
# obvious immediately rather than only surfacing after the whole run.
#
# "".join(v.split()) rather than v.strip(): Google's app-password dialog
# displays the password as 4 groups of 4 ("wgmw zgjx rdpq wmrn"), and
# copying it out of the browser very commonly copies U+00A0 (non-breaking
# space) as the group separator instead of a normal space -- invisible in
# an editor, but smtplib's PLAIN-auth mechanism does a hard
# `.encode('ascii')` on the credential string during login(), which blows
# up on that character. .strip() only trims the ends, so it wouldn't have
# caught a separator sitting in the MIDDLE of the password -- .split()
# with no argument splits on ANY whitespace (space, tab, NBSP, ...) and
# drops the empties, so joining it back with "" removes internal
# separators too, not just leading/trailing ones.
def _clean_env(name, default=None):
    val = os.environ.get(name, default)
    return "".join(val.split()) if val else val


SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
EMAIL_FROM = _clean_env("EMAIL_FROM")
EMAIL_APP_PASSWORD = _clean_env("EMAIL_APP_PASSWORD")
EMAIL_TO = _clean_env("EMAIL_TO") or EMAIL_FROM

RSS_FEED_URLS = [
    "https://weworkremotely.com/categories/remote-programming-jobs.rss",
    # LinkedIn: no official feed -- see module docstring above. Add a URL
    # here (from a third-party generator, with the ToS risk understood) if
    # you decide you want to try it anyway.
]

SCORE_WEIGHT_SKILLS = 0.5  # vs (1 - this) for embedding similarity -- see match_score()

# ── employment-type LABEL: working student / internship vs full-time ────
# Both employment types are fetched, extracted, scored, and shown -- this
# used to be a hard pre-extraction filter that dropped full-time postings
# entirely, but you asked to see both (e.g. full-time roles are still
# worth tracking for after graduation, working-student/internship roles
# are what you can take right now). is_student_eligible() now just tags
# each result so the digest can badge them distinctly rather than silently
# excluding one category.
#
# English + German terms, since German tech postings routinely use the
# German term even when the rest of the listing is in English -- "Werkstudent"
# is a distinct, legally-defined German employment category (part-time,
# enrolled-student-only, capped hours) from "Praktikum" (internship), and
# both are genuinely different from "junior full-time". I checked
# Arbeitnow's API response shape before relying on this: it DOES have a
# `job_types` field, but it's empty on most postings and inconsistently
# capitalized ("Full Time" vs "Full time") on the rest -- not reliable
# enough to tag on alone, so this checks title/description/tags too, which
# is where postings actually self-label in practice (German labor law
# requires employers to state this status clearly for tax/visa reasons, so
# it's almost always right there in the title).
STUDENT_ELIGIBLE_KEYWORDS = [
    "working student", "werkstudent", "werkstudent:in", "werkstudentin",
    "intern", "interns", "internship", "praktikum", "praktikant", "praktikantin",
    "working student (m/w/d)", "werkstudent (m/w/d)",
]

# Plain `kw in haystack` substring matching (what this used to do) is the
# same bug class as string.find() vs a real tokenizer: "intern" as a bare
# substring also matches inside "INTERNational team", "INTERNal tools",
# "INTERNet access" -- all extremely common in job-posting boilerplate and
# NOTHING to do with an internship. That's what mislabeled two full-time
# Auxmoney/Vidalytics postings as Werkstudent/Praktikum -- not the LLM
# hallucinating, this check runs on the raw posting text before the LLM
# ever sees it. \b word-boundary regex fixes it: \bintern\b matches the
# standalone word "intern" but NOT "intern" as a prefix glued to more
# letters, since \b only fires at a transition between a word character
# and a non-word character (or the string edge) -- there's no such
# transition between "intern" and "ational".
_STUDENT_ELIGIBLE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in STUDENT_ELIGIBLE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def is_student_eligible(job):
    haystack = " ".join([
        job.get("title", ""),
        job.get("description", ""),
        " ".join(job.get("tags", []) or []),
        " ".join(job.get("job_types", []) or []),
    ])
    return bool(_STUDENT_ELIGIBLE_PATTERN.search(haystack))


def filter_student_eligible(postings):
    """Not called by default anymore (see note above) -- kept here in case
    you want to switch back to a hard filter later, e.g. via a command-line
    flag. `uv run python jobradar.py` today shows both employment types."""
    eligible = [p for p in postings if is_student_eligible(p)]
    dropped = len(postings) - len(eligible)
    if dropped:
        hjp.log_event("ineligible_postings_filtered", dropped=dropped, kept=len(eligible))
    return eligible


# ── source 2: any RSS/Atom feed, including (optionally) LinkedIn ────────

def _split_wwr_title(title):
    """We Work Remotely-style feeds format titles as 'Company: Role' --
    best-effort split; falls back to 'Unknown' rather than guessing wrong
    on feeds that don't follow this convention."""
    if ":" in title:
        company, _, role = title.partition(":")
        return company.strip(), role.strip()
    return "Unknown", title.strip()


def fetch_rss_postings(feed_urls=None):
    import feedparser
    feed_urls = feed_urls if feed_urls is not None else RSS_FEED_URLS
    postings = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        if getattr(feed, "bozo", False):
            hjp.log_event("rss_feed_parse_error", level=hjp.logging.WARNING,
                           url=url, error=str(getattr(feed, "bozo_exception", "")))
            continue
        for entry in feed.entries:
            company, role_title = _split_wwr_title(entry.get("title", ""))
            postings.append({
                "title": role_title,
                "company_name": company,
                "location": entry.get("location", "") or "Remote",
                "tags": [],
                "description": entry.get("summary", "") or entry.get("description", ""),
                "url": entry.get("link", ""),
                "source": "rss",
            })
        hjp.log_event("rss_feed_fetched", url=url, count=len(feed.entries))
    return postings


# ── dedup against what's already been digested ──────────────────────────

def load_seen(path=SEEN_PATH):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))


def save_seen(seen, path=SEEN_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f, ensure_ascii=False, indent=2)


# ── cheap pre-filter: rank by a free heuristic before spending API budget ──
# Fetching from Arbeitnow/RSS is free (public APIs); extraction is what
# costs real money (an LLM call per posting). So the efficient order when
# MAX_POSTINGS_PER_RUN forces a cut is: fetch broadly, rank for free, only
# pay to extract+embed the postings most likely to actually matter -- not
# just whichever 20 happened to come first out of Arbeitnow+RSS.
#
# Uses the SAME word-boundary fix as is_student_eligible() above, for the
# same reason: a naive `"Git" in text` substring check would also match
# "GitHub", "digital", "legitimate" -- false hits that would skew which
# postings get through the cap.
_PROFILE_SKILLS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in PROFILE_SKILLS) + r")\b",
    re.IGNORECASE,
)


def rough_skills_score(job):
    """Free proxy for match quality, run BEFORE extraction: how many of
    YOUR skills appear, by name, in the posting's raw title/description/
    tags. Cruder than the real skills_overlap_score() below (which runs on
    the LLM's cleanly extracted skills list, not noisy raw text) -- good
    enough to triage a big pool down to the ones worth actually paying
    for, not precise enough to be the final score."""
    haystack = " ".join([
        job.get("title", ""),
        job.get("description", ""),
        " ".join(job.get("tags", []) or []),
    ])
    return len(_PROFILE_SKILLS_PATTERN.findall(haystack))


# ── scoring: skills overlap + embedding similarity, combined ────────────
# Two different, imperfect signals, deliberately. Same reasoning as the
# cosine-similarity-vs-clustering conversation from a few days ago: skills
# overlap is precise and explainable ("you match 4/6 requested skills") but
# brittle to synonyms -- a posting asking for "GenAI" scores ZERO overlap
# against your "LLM" skill even though they mean almost the same thing.
# Embedding similarity catches that kind of thematic/semantic match, but is
# fuzzier and can be fooled by superficially similar wording with no real
# skill match behind it. Combining both is more robust than trusting either
# alone -- same "two signals, different failure modes" idea, here a
# weighted blend instead of a multi-stage pipeline.

def skills_overlap_score(profile_skills, posting_skills):
    """Fraction of the posting's REQUESTED skills that you actually have --
    "you match 4/6 requested skills" is the most legible number to put in a
    digest. (A symmetric Jaccard score would also penalize you for having
    skills irrelevant to THIS posting, which isn't actually a bad sign.)"""
    if not posting_skills:
        return 0.0
    profile_set = {s.lower() for s in profile_skills}
    posting_set = {s.lower() for s in posting_skills}
    overlap = profile_set & posting_set
    return len(overlap) / len(posting_set)


def matched_skill_names(profile_skills, posting_skills):
    """The actual skill NAMES behind skills_overlap_score()'s fraction --
    the email digest wants to say "matches on Python, LLM, RAG", not just
    "60%". Returns names in your own profile's capitalization, since
    that's the spelling you'll recognize."""
    if not posting_skills:
        return []
    profile_by_lower = {s.lower(): s for s in profile_skills}
    posting_set = {s.lower() for s in posting_skills}
    return [profile_by_lower[s] for s in posting_set if s in profile_by_lower]


def posting_to_text(posting_dict):
    skills = ", ".join(posting_dict.get("skills", []) or [])
    return f"{posting_dict.get('title', '')}. Skills: {skills}."


def match_score(overlap, embed_sim, weight_skills=SCORE_WEIGHT_SKILLS):
    embed_sim = max(0.0, embed_sim)  # cosine can go negative; clip for a clean 0-100% reading
    return weight_skills * overlap + (1 - weight_skills) * embed_sim


# ── digest rendering ──────────────────────────────────────────────────────
# Meter bars follow the dataviz skill's conventions: sequential magnitude
# encoding = one hue, light-to-dark (blue), fill is the value, unfilled
# track is a LIGHTER step of the same ramp (not a different color), 4px
# rounded ends. Both light and dark surfaces are selected, not an
# automatic invert -- palette values straight from references/palette.md.

def _score_badge_pct(score):
    return f"{round(score * 100)}%"


def render_html_digest(ranked, generated_at, run_spent):
    rows = []
    for r in ranked:
        pct = round(r["score"] * 100)
        salary = ""
        if r.get("salary_min") or r.get("salary_max"):
            lo = f"€{r['salary_min']:,.0f}" if r.get("salary_min") else "?"
            hi = f"€{r['salary_max']:,.0f}" if r.get("salary_max") else "?"
            salary = f"<span class='pill'>{lo}–{hi}/yr</span>"
        visa = "<span class='pill'>visa sponsorship</span>" if r.get("visa_sponsorship") else ""
        german = "<span class='pill'>German required</span>" if r.get("german_required") else ""
        skills_html = "".join(f"<span class='pill pill-skill'>{s}</span>" for s in (r.get("skills") or [])[:8])

        # Employment-type badge: distinguishes working-student/internship
        # postings (what you're eligible for right now) from full-time ones
        # (worth tracking for after graduation) now that both flow through
        # the digest instead of one being silently filtered out. Uses the
        # dataviz skill's fixed status "good" green for eligible-now roles
        # -- not because full-time is "bad", just to make the one category
        # that's actionable today scannable at a glance -- and a neutral
        # chip for full-time so it doesn't read as a warning.
        if r.get("student_eligible"):
            badge = "<span class='badge badge-student'>Werkstudent / Praktikum</span>"
        else:
            badge = "<span class='badge badge-fulltime'>Full-time</span>"

        rows.append(f"""
        <article class="card">
          <div class="card-top">
            <div>
              {badge}
              <h2><a href="{r.get('url', '#')}" target="_blank" rel="noopener">{r.get('title', 'Untitled')}</a></h2>
              <p class="meta">{r.get('company', 'Unknown')} · {r.get('location') or '—'} · <span class="source">{r.get('source', '')}</span></p>
            </div>
            <div class="score-block">
              <div class="score-value">{pct}%</div>
              <div class="meter"><div class="meter-fill" style="width:{pct}%"></div></div>
              <div class="score-breakdown">skills {round(r['skills_overlap']*100)}% · semantic {round(r['embedding_similarity']*100)}%</div>
            </div>
          </div>
          <div class="pills">{salary}{visa}{german}{skills_html}</div>
        </article>""")

    cost_ok = run_spent <= DAILY_COST_TARGET_EUR
    cost_note = (
        f"<span class='cost-ok'>✓ under the €{DAILY_COST_TARGET_EUR:.2f}/day target</span>"
        if cost_ok else
        f"<span class='cost-over'>⚠ over the €{DAILY_COST_TARGET_EUR:.2f}/day target</span>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JobRadar Digest — {generated_at[:10]}</title>
<style>
  /* Vars live on :root/body, not .viz-root -- body's own `background:
     var(--page)` rule needs to READ these, and CSS custom properties only
     inherit DOWNWARD to descendants, never upward from a child to its
     parent. Scoping them to .viz-root (a child of body) left body's
     background unable to see the dark-mode override -- confirmed visually
     via a Playwright screenshot before shipping, page background stayed
     white while cards correctly went dark. */
  :root {{
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
    --gridline: #e1e0d9; --border: rgba(11,11,11,0.10);
    --seq-fill: #2a78d6; --seq-track: #cde2fb;
    --status-good: #0ca30c; --status-good-bg: #e5f5e5;
    --status-warn: #c98500; --status-warn-bg: #fdf1d9;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --surface-1: #1a1a19; --page: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
      --gridline: #2c2c2a; --border: rgba(255,255,255,0.10);
      --seq-fill: #3987e5; --seq-track: #184f95;
      --status-good: #0ca30c; --status-good-bg: #14311a;
      --status-warn: #eda100; --status-warn-bg: #3a2c0c;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--page); color: var(--text-primary); }}
  .wrap {{ max-width: 760px; margin: 0 auto; padding: 32px 20px 64px; }}
  header {{ margin-bottom: 28px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: var(--text-secondary); font-size: 14px; margin: 0; }}
  .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; margin-bottom: 14px; }}
  .card-top {{ display:flex; justify-content: space-between; gap: 16px; align-items: flex-start; }}
  h2 {{ font-size: 16px; margin: 0 0 4px; }}
  h2 a {{ color: var(--text-primary); text-decoration: none; }}
  h2 a:hover {{ text-decoration: underline; }}
  .meta {{ color: var(--text-secondary); font-size: 13px; margin: 0; }}
  .source {{ color: var(--text-muted); text-transform: uppercase; font-size: 11px; letter-spacing: 0.04em; }}
  .badge {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; margin-bottom: 6px; letter-spacing: 0.01em; }}
  .badge-student {{ color: var(--status-good); background: var(--status-good-bg); }}
  .badge-fulltime {{ color: var(--text-muted); background: var(--gridline); }}
  .score-block {{ flex-shrink:0; width: 140px; text-align: right; }}
  .score-value {{ font-size: 20px; font-weight: 600; font-variant-numeric: proportional-nums; }}
  .meter {{ height: 6px; border-radius: 3px; background: var(--seq-track); margin: 6px 0 4px; overflow: hidden; }}
  .meter-fill {{ height: 100%; background: var(--seq-fill); border-radius: 3px; }}
  .score-breakdown {{ font-size: 11px; color: var(--text-muted); }}
  .pills {{ margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }}
  .pill {{ font-size: 12px; padding: 3px 9px; border-radius: 999px; background: var(--gridline); color: var(--text-secondary); }}
  .pill-skill {{ background: transparent; border: 1px solid var(--gridline); }}
  .cost-ok {{ color: var(--status-good); font-weight: 600; }}
  .cost-over {{ color: var(--status-warn); font-weight: 600; }}
  footer {{ color: var(--text-muted); font-size: 12px; margin-top: 24px; }}
</style>
</head>
<body>
<div class="viz-root wrap">
  <header>
    <h1>JobRadar — Daily Digest</h1>
    <p class="sub">{len(ranked)} new posting{'s' if len(ranked) != 1 else ''} · generated {generated_at} · run cost €{run_spent:.4f} · {cost_note}</p>
  </header>
  {"".join(rows) if rows else "<p class='sub'>No new postings today.</p>"}
  <footer>Scored against job_profile.py — skills overlap × {SCORE_WEIGHT_SKILLS:.0%} + embedding similarity × {1 - SCORE_WEIGHT_SKILLS:.0%}.</footer>
</div>
</body>
</html>"""


# ── email delivery: top N + a plain-language "why" per posting ──────────

def build_match_reason(r):
    """Short, honest explanation of why a posting scored the way it did --
    built directly from numbers already computed during scoring, NOT a
    separate LLM call asked to "explain itself." Asking a model to justify
    a score after the fact risks the same failure mode as the
    hallucination-check exercise: a fluent explanation that doesn't
    actually trace back to the real signal, since the model wasn't shown
    the scoring math, just asked to sound convincing. This can't drift
    from the score because it's built out of the exact same variables."""
    matched = r.get("matched_skills") or []
    skills_bit = (
        f"matches {len(matched)} of your skills ({', '.join(matched)})"
        if matched else "no direct skill-keyword overlap"
    )
    sem_pct = round(r["embedding_similarity"] * 100)
    tag = "Werkstudent/Praktikum" if r.get("student_eligible") else "full-time"
    return f"{skills_bit}; {sem_pct}% semantic similarity to your profile ({tag})."


def render_email_text(top, generated_at):
    """Plain-text part of the email -- always renders correctly regardless
    of the recipient's mail client, and is what most spam filters give
    the most weight to having present at all alongside an HTML part."""
    lines = [f"JobRadar — Top {len(top)} match{'es' if len(top) != 1 else ''} (generated {generated_at})", ""]
    for i, r in enumerate(top, 1):
        pct = round(r["score"] * 100)
        lines.append(f"{i}. {r.get('title', 'Untitled')} @ {r.get('company', 'Unknown')} — {pct}% match")
        lines.append(f"   {r.get('url', '')}")
        lines.append(f"   Why: {build_match_reason(r)}")
        lines.append("")
    return "\n".join(lines)


def render_email_html(top, generated_at):
    """Deliberately NOT reusing render_html_digest()'s markup -- that one
    relies on CSS custom properties (--seq-fill etc) defined on :root,
    which plenty of real-world email clients (Outlook desktop especially)
    either strip <style> blocks from or don't support CSS variables in at
    all. Email HTML needs inline styles on every element to be safe, so
    this is a separate, simpler renderer for just the top N."""
    rows = []
    for r in top:
        pct = round(r["score"] * 100)
        reason = build_match_reason(r)
        if r.get("student_eligible"):
            badge_color, badge_label = "#0ca30c", "WERKSTUDENT / PRAKTIKUM"
        else:
            badge_color, badge_label = "#666666", "FULL-TIME"
        rows.append(f"""
        <tr>
          <td style="padding:14px 0;border-bottom:1px solid #e1e0d9;">
            <div style="font-size:11px;font-weight:700;color:{badge_color};letter-spacing:0.04em;margin-bottom:4px;">{badge_label}</div>
            <a href="{r.get('url', '#')}" style="font-size:16px;font-weight:600;color:#0b0b0b;text-decoration:none;">{r.get('title', 'Untitled')}</a>
            <div style="font-size:13px;color:#52514e;margin-top:2px;">{r.get('company', 'Unknown')} · {r.get('location') or '—'} · {pct}% match</div>
            <div style="font-size:13px;color:#333333;margin-top:8px;">{reason}</div>
          </td>
        </tr>""")
    return f"""<html><body style="font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#f9f9f7;color:#0b0b0b;margin:0;padding:24px;">
<div style="max-width:600px;margin:0 auto;">
  <h1 style="font-size:20px;margin:0 0 4px;">JobRadar — Top {len(top)} Match{'es' if len(top) != 1 else ''}</h1>
  <p style="font-size:13px;color:#52514e;margin:0 0 16px;">generated {generated_at}</p>
  <table style="width:100%;border-collapse:collapse;">{"".join(rows)}</table>
  <p style="font-size:12px;color:#898781;margin-top:24px;">Full ranked digest: {DIGEST_HTML_PATH} on this machine.</p>
</div>
</body></html>"""


def send_digest_email(top, generated_at):
    """Best-effort delivery -- by the time this runs, the JSON/HTML digest
    is ALREADY written to disk (see __main__ below), so a bad password or
    an SMTP hiccup here means "you open the file yourself today," never
    "the run failed and nothing was produced." Delivery-channel failures
    shouldn't be able to take down a pipeline that already succeeded at
    its actual job."""
    if not EMAIL_FROM or not EMAIL_APP_PASSWORD:
        print("  Email not sent: EMAIL_FROM / EMAIL_APP_PASSWORD not set in .env (see module docstring).")
        return False
    if not top:
        print("  Email not sent: no scored postings to report.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"JobRadar — {len(top)} top match{'es' if len(top) != 1 else ''} — {generated_at[:10]}"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    # Attach plain text FIRST, then HTML -- email clients render the LAST
    # part in a multipart/alternative that they understand, so this order
    # (plain, then html) is what makes HTML-capable clients prefer the
    # HTML version while plain-text-only clients still get something readable.
    msg.attach(MIMEText(render_email_text(top, generated_at), "plain"))
    msg.attach(MIMEText(render_email_html(top, generated_at), "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
        print(f"  Emailed top {len(top)} match{'es' if len(top) != 1 else ''} to {EMAIL_TO}")
        hjp.log_event("email_sent", to=EMAIL_TO, count=len(top))
        return True
    except Exception as e:
        print(f"  Email send FAILED ({type(e).__name__}: {e}) -- digest files were still written, just not emailed.")
        hjp.log_event("email_send_failed", level=hjp.logging.ERROR, error_type=type(e).__name__)
        return False


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer, util

    print("Fetching new postings...")
    # target_count=40, not the default 10: fetching more candidates up
    # front just means a richer, more varied digest across BOTH employment
    # types below -- this only costs extra HTTP requests to a free public
    # API, not API tokens.
    arbeitnow_postings = hjp.fetch_matching_postings(target_count=40)
    for j in arbeitnow_postings:
        j["source"] = "arbeitnow"
    rss_postings = fetch_rss_postings()
    all_postings = arbeitnow_postings + rss_postings
    print(f"  Arbeitnow: {len(arbeitnow_postings)}, RSS: {len(rss_postings)}, total: {len(all_postings)}")

    # Both employment types go through -- is_student_eligible() is now a
    # LABEL attached to every result (student_eligible: true/false), not a
    # gate. Nothing gets dropped before extraction anymore.
    seen = load_seen()
    new_postings = [p for p in all_postings if p.get("url") and p["url"] not in seen]
    print(f"  {len(new_postings)} are new (not already in {SEEN_PATH})")

    if not new_postings:
        print("Nothing new today -- no digest written.")
        raise SystemExit(0)

    # MAX_POSTINGS_PER_RUN is what actually limits API spend per day now,
    # not the euro budget (see the comment by MAX_POSTINGS_PER_RUN above).
    # Rank by the free rough_skills_score() first so a cut favors postings
    # that look like actual matches, not just whichever ones happened to
    # come first out of Arbeitnow+RSS.
    if len(new_postings) > MAX_POSTINGS_PER_RUN:
        new_postings.sort(key=rough_skills_score, reverse=True)
        dropped = len(new_postings) - MAX_POSTINGS_PER_RUN
        new_postings = new_postings[:MAX_POSTINGS_PER_RUN]
        print(f"  Capping to top {MAX_POSTINGS_PER_RUN} by rough skills match "
              f"({dropped} lower-priority postings skipped this run to save API budget)")
        hjp.log_event("postings_capped", cap=MAX_POSTINGS_PER_RUN, dropped=dropped)

    print("\nLoading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    profile_embedding = embed_model.encode(PROFILE_TEXT, normalize_embeddings=True)

    results = []
    run_spent = 0.0
    for i, job in enumerate(new_postings, 1):
        print(f"[{i}/{len(new_postings)}] {job.get('title')} @ {job.get('company_name')} ({job.get('source')})")
        try:
            desc = hjp.strip_html(job.get("description", ""))
            structured, cost_eur = hjp.extract_with_fallback(job, desc, run_spent)
            run_spent += cost_eur
        except hjp.RunBudgetExceeded as e:
            print(f"    -> STOPPING: {e}")
            break
        except RuntimeError as e:
            print(f"    -> FAILED (both providers exhausted), skipping: {e}")
            continue

        posting_dict = structured.model_dump()
        overlap = skills_overlap_score(PROFILE_SKILLS, posting_dict.get("skills", []))
        matched_skills = matched_skill_names(PROFILE_SKILLS, posting_dict.get("skills", []))
        posting_embedding = embed_model.encode(posting_to_text(posting_dict), normalize_embeddings=True)
        embed_sim = float(util.cos_sim(profile_embedding, posting_embedding)[0][0])
        score = match_score(overlap, embed_sim)

        results.append({
            **posting_dict, "score": score, "skills_overlap": overlap,
            "matched_skills": matched_skills,
            "embedding_similarity": embed_sim, "source": job.get("source"),
            "student_eligible": is_student_eligible(job),
        })
        seen.add(job["url"])
        tag = "student-eligible" if results[-1]["student_eligible"] else "full-time"
        print(f"    -> match {score:.0%} (skills {overlap:.0%}, semantic {embed_sim:.0%}, {tag}) — run spend €{run_spent:.4f}")

    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(DIGEST_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=2)

    with open(DIGEST_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(render_html_digest(ranked, generated_at, run_spent))

    save_seen(seen)

    print(f"\n{len(ranked)} new postings scored and ranked.")
    print(f"Digest written: {DIGEST_HTML_PATH} / {DIGEST_JSON_PATH}")
    print(f"Total run spend: €{run_spent:.4f} (cap was €{JOBRADAR_RUN_BUDGET_EUR:.2f})")
    if run_spent <= DAILY_COST_TARGET_EUR:
        print(f"On budget: €{run_spent:.4f} is under the €{DAILY_COST_TARGET_EUR:.2f}/day target.")
    else:
        over_by = run_spent - DAILY_COST_TARGET_EUR
        print(f"Over the €{DAILY_COST_TARGET_EUR:.2f}/day target by €{over_by:.4f} -- "
              f"consider lowering MAX_POSTINGS_PER_RUN (currently {MAX_POSTINGS_PER_RUN}).")

    # Digest files are already safely on disk at this point -- email
    # delivery happens last and can't undo any of the above even if it fails.
    send_digest_email(ranked[:TOP_N_FOR_EMAIL], generated_at)