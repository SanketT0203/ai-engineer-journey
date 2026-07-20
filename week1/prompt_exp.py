"""
Day 4 practical — run all 5 prompt versions against all 10 listings, score
each field, and watch accuracy climb from v1 to v5.

    uv run python run_prompt_experiment.py

Uses temperature=0: for an EVAL run you want reproducibility, not creativity —
same prompt should score the same way if you rerun it tomorrow.
"""

import csv
import json
import re

import anthropic

from cost_calculator import log_call
from rental_data import FIELDS, LISTINGS
from prompt_versions import PROMPT_VERSIONS

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5"


def extract_json(raw_text):
    """Pull a JSON object out of a model response, tolerating extra prose
    (v1/v2 will have none at all; v5 may have reasoning before the JSON)."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.rfind("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def to_float(value):
    """Normalize a value (number, German-formatted string, or null-ish string) to float or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace("€", "").replace("EUR", "").replace(" ", "")
        if s.lower() in ("null", "none", ""):
            return None
        s = s.replace(".", "").replace(",", ".") if ("," in s and "." in s) else s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def score_listing(predicted, ground_truth):
    """Return (correct_count, results_dict) comparing predicted vs ground_truth per field."""
    results = {}
    correct = 0
    for field in FIELDS:
        gt_val = to_float(ground_truth[field])
        pred_val = to_float(predicted.get(field)) if isinstance(predicted, dict) else None
        is_correct = (gt_val is None and pred_val is None) or (
            gt_val is not None and pred_val is not None and abs(gt_val - pred_val) < 1.0
        )
        results[field] = is_correct
        if is_correct:
            correct += 1
    return correct, results


def run_experiment():
    rows = []  # for the CSV export

    print(f"Running {len(PROMPT_VERSIONS)} prompt versions x {len(LISTINGS)} listings "
          f"= {len(PROMPT_VERSIONS) * len(LISTINGS)} calls...\n")

    version_scores = {}

    for version_name, template in PROMPT_VERSIONS:
        total_correct = 0
        total_fields = 0
        parse_failures = 0

        for listing in LISTINGS:
            prompt = template.format(text=listing["text"])
            msg = client.messages.create(
                model=MODEL,
                max_tokens=400,
                temperature=0,  # deterministic — this is an eval, not creative writing
                messages=[{"role": "user", "content": prompt}],
            )
            log_call("anthropic", MODEL, msg.usage.input_tokens, msg.usage.output_tokens)

            reply_text = msg.content[0].text
            predicted = extract_json(reply_text)

            if predicted is None:
                parse_failures += 1
                correct, field_results = 0, {f: False for f in FIELDS}
            else:
                correct, field_results = score_listing(predicted, listing["ground_truth"])

            total_correct += correct
            total_fields += len(FIELDS)

            rows.append({
                "version": version_name, "listing_id": listing["id"],
                "parsed_ok": predicted is not None, "correct_fields": correct,
                "total_fields": len(FIELDS),
                **{f"correct_{f}": field_results[f] for f in FIELDS},
            })

        accuracy = total_correct / total_fields
        version_scores[version_name] = accuracy
        print(f"{version_name:20s}  accuracy={accuracy:5.1%}   "
              f"parse failures={parse_failures}/{len(LISTINGS)}")

    with open("prompt_experiment_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("SUMMARY — accuracy by version")
    print("=" * 60)
    baseline = version_scores["v1_naive"]
    for name, acc in version_scores.items():
        delta = acc - baseline
        print(f"{name:20s}  {acc:5.1%}   ({delta:+.1%} vs v1)")

    print("\nFull per-field breakdown saved to prompt_experiment_results.csv")


if __name__ == "__main__":
    run_experiment()