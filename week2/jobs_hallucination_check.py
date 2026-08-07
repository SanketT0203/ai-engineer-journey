"""
Day 6 reflection, made measurable — run the SAME postings through a naive
schema (no null guidance) and your guarded schema (has "null if not stated"),
then mechanically cross-check every non-null value against the source text.

A value that ISN'T null but also isn't backed by anything in the actual
description is the real signature of hallucination — not "field is null",
which (as you correctly suspected) can just mean the info was genuinely absent.

    uv run python hallucination_check.py
"""

import json
import re

from extract_job_exp import fetch_matching_postings, strip_html, client, MODEL
from cost_calculator import log_call
from job_schema import JobPosting
from job_schema_naive import NaiveJobPosting

SALARY_EVIDENCE = re.compile(r"€|\$|EUR|gehalt|salary|k\s*p\.a\.?|per year|/year|/annum|p\.a\.", re.I)
GERMAN_EVIDENCE = ["german", "deutsch", "language requirement", "language skills"]
VISA_EVIDENCE = ["visa", "sponsor", "relocation", "work permit"]


def build_prompt(job, description_text):
    return (
        f"Extract structured job data from this posting.\n\n"
        f"Title: {job.get('title')}\n"
        f"Company: {job.get('company_name')}\n"
        f"Location: {job.get('location')}\n"
        f"Tags: {', '.join(job.get('tags', []))}\n\n"
        f"Description:\n{description_text}"
    )


def extract_with(schema_cls, job, description_text):
    result = client.messages.create(
        model=MODEL, max_tokens=500, temperature=0, max_retries=2,
        response_model=schema_cls,
        messages=[{"role": "user", "content": build_prompt(job, description_text)}],
    )
    usage = getattr(getattr(result, "_raw_response", None), "usage", None)
    if usage:
        log_call("anthropic", MODEL, usage.input_tokens, usage.output_tokens)
    return result


def check_salary(desc_text, salary_min, salary_max):
    if (salary_min is not None or salary_max is not None) and not SALARY_EVIDENCE.search(desc_text):
        return "LIKELY HALLUCINATED — no salary/currency text found in description at all"
    return None


def check_boolean(desc_text, value, keywords, field_name):
    if value is not None and not any(kw in desc_text.lower() for kw in keywords):
        return f"LIKELY GUESSED — {field_name}={value} but no related keyword found in description"
    return None


def check_skills(desc_text, skills):
    """Skills the model listed that don't appear verbatim anywhere in the source text —
    could be legitimate inference (title implies Python) or could be fabrication.
    Flagged for YOUR manual read, not auto-condemned."""
    return [s for s in skills if s.lower() not in desc_text.lower()]


if __name__ == "__main__":
    print("Fetching postings (reusing the same filter as fetch_and_extract_jobs.py)...")
    postings = fetch_matching_postings(target_count=5)  # 5 is enough to see the pattern, keeps cost tiny
    print(f"Got {len(postings)} postings. Running BOTH schemas on each...\n")

    report = []
    for job in postings:
        desc = strip_html(job.get("description", ""))
        print(f"--- {job.get('title')} @ {job.get('company_name')} ---")

        naive = extract_with(NaiveJobPosting, job, desc)
        guarded = extract_with(JobPosting, job, desc)

        flags = []
        for label, result in [("NAIVE", naive), ("GUARDED", guarded)]:
            salary_flag = check_salary(desc, result.salary_min, result.salary_max)
            german_flag = check_boolean(desc, result.german_required, GERMAN_EVIDENCE, "german_required")
            visa_flag = check_boolean(desc, result.visa_sponsorship, VISA_EVIDENCE, "visa_sponsorship")
            unverified_skills = check_skills(desc, result.skills)

            print(f"  [{label}] salary={result.salary_min}-{result.salary_max}  "
                  f"german_required={result.german_required}  visa_sponsorship={result.visa_sponsorship}")
            print(f"           skills={result.skills}")
            if salary_flag: print(f"           !! {label}: {salary_flag}")
            if german_flag: print(f"           !! {label}: {german_flag}")
            if visa_flag: print(f"           !! {label}: {visa_flag}")
            if unverified_skills: print(f"           ?  {label}: skills not found verbatim in text (check manually): {unverified_skills}")

            flags.append({
                "schema": label, "salary_flag": salary_flag, "german_flag": german_flag,
                "visa_flag": visa_flag, "unverified_skills": unverified_skills,
            })

        report.append({
            "title": job.get("title"), "description_excerpt": desc[:300],
            "naive": naive.model_dump(), "guarded": guarded.model_dump(), "flags": flags,
        })
        print()

    with open("hallucination_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Full report with source-text excerpts saved to hallucination_report.json")
    print(
        "\nRead each '!!' flag yourself — the heuristic can misfire (e.g. a salary "
        "mentioned as 'competitive' with no number won't trigger SALARY_EVIDENCE, "
        "so a null there is correct even without a flag). The heuristic narrows down "
        "WHERE to look; it doesn't replace actually reading the source text."
    )