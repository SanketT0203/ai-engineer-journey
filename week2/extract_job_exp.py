"""
Day 6 practical — fetch real AI-engineer postings from the Arbeitnow API,
extract structured data with Instructor + Pydantic.


This is literally the first building block of JobRadar (Friday's project) —
today you're proving the extraction step works before you automate it daily.
"""

import json
import re

import anthropic
import httpx
import instructor
from bs4 import BeautifulSoup

from cost_calculator import log_call
from job_schema import JobPosting

API_URL = "https://www.arbeitnow.com/api/job-board-api"
MODEL = "claude-haiku-4-5"
TARGET_COUNT = 10
MAX_PAGES = 15  # safety cap so a bad filter can't loop forevers

# Keywords used to filter the firehose of ALL postings down to  .
# The public API has no server-side search, so this filtering happens client-side —
# exactly what a real JobRadar needs to do on every run.
KEYWORDS = [
    "ai engineer", "machine learning", "ml engineer", "data scientist",
    "llm", "artificial intelligence", "nlp", "deep learning",
    "genai", "generative ai", "agentic",
]

client = instructor.from_anthropic(anthropic.Anthropic())


def strip_html(raw_html, max_chars=2500):
    """Descriptions come back as HTML. Strip tags, collapse whitespace, and
    TRUNCATE — a full job description can run 5000+ tokens, and we don't need
    all of it to extract five structured fields. This truncation IS a token
    cost decision, not just tidiness."""
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def matches_keywords(job):
    haystack = (job.get("title", "") + " " + " ".join(job.get("tags", []))).lower()
    return any(kw in haystack for kw in KEYWORDS)


def fetch_matching_postings(target_count=TARGET_COUNT, max_pages=MAX_PAGES):
    matched = []
    with httpx.Client(timeout=30) as http:
        for page in range(1, max_pages + 1):
            resp = http.get(API_URL, params={"page": page})
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if not data:  # ran out of pages
                break

            for job in data:
                if matches_keywords(job):
                    matched.append(job)
                    if len(matched) >= target_count:
                        return matched

            print(f"  ...page {page}: {len(matched)}/{target_count} matches so far")

    return matched


def extract_structured(job):
    description_text = strip_html(job.get("description", ""))
    prompt = (
        f"Extract structured job data from this posting.\n\n"
        f"Title: {job.get('title')}\n"
        f"Company: {job.get('company_name')}\n"
        f"Location: {job.get('location')}\n"
        f"Tags: {', '.join(job.get('tags', []))}\n\n"
        f"Description:\n{description_text}"
    )

    result = client.messages.create(
        model=MODEL,
        max_tokens=500,
        temperature=0,
        max_retries=2,
        response_model=JobPosting,
        messages=[{"role": "user", "content": prompt}],
    )

    usage = getattr(getattr(result, "_raw_response", None), "usage", None)
    if usage:
        log_call("anthropic", MODEL, usage.input_tokens, usage.output_tokens)

    result.url = job.get("url")  # not LLM-extracted, just carried through
    return result


if __name__ == "__main__":
    print(f"Fetching AI-engineer postings from Arbeitnow (target: {TARGET_COUNT})...")
    postings = fetch_matching_postings()
    print(f"\nFound {len(postings)} matching postings. Extracting structured data...\n")

    results = []
    for i, job in enumerate(postings, 1):
        print(f"[{i}/{len(postings)}] {job.get('title')} @ {job.get('company_name')}")
        try:
            structured = extract_structured(job)
            results.append(structured)
            print(f"    -> seniority={structured.seniority}, skills={structured.skills[:5]}, "
                  f"german_required={structured.german_required}, visa_sponsorship={structured.visa_sponsorship}")
        except Exception as e:
            print(f"    -> FAILED to extract: {e}")

    with open("jobs_extracted.json", "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} structured postings to jobs_extracted.json")
    print("This file is now real seed data for JobRadar.")