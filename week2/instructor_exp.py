"""
Same 10 listings, same scoring, same ground truth as Friday's
run_prompt_experiment.py — but reliability now comes from Instructor +
Pydantic instead of prompt wording. Direct apples-to-apples comparison.

    uv add instructor pydantic
    uv run python run_instructor_experiment.py
"""

import anthropic
import instructor
from pydantic import ValidationError

from cost_calculator import log_call
from rental_data import FIELDS, LISTINGS
from rental_schema import RentalExtraction

MODEL = "claude-haiku-4-5"

# instructor.from_anthropic() patches the client so `response_model=` becomes
# a valid argument on .messages.create(). Under the hood it forces a tool call
# matching RentalExtraction's schema, and if the result fails Pydantic
# validation, it automatically retries (up to max_retries) with the validation
# error fed back to the model as extra context.
client = instructor.from_anthropic(anthropic.Anthropic())


def to_float(value):
    if value is None:
        return None
    return float(value)


def score_listing(predicted: RentalExtraction, ground_truth: dict):
    """Same scoring logic as Friday, adapted for a Pydantic object instead of a dict."""
    results = {}
    correct = 0
    for field in FIELDS:
        gt_val = to_float(ground_truth[field])
        pred_val = to_float(getattr(predicted, field, None))
        is_correct = (gt_val is None and pred_val is None) or (
            gt_val is not None and pred_val is not None and abs(gt_val - pred_val) < 1.0
        )
        results[field] = is_correct
        if is_correct:
            correct += 1
    return correct, results


def run_experiment():
    total_correct = 0
    total_fields = 0
    hard_failures = 0  # exhausted all retries, never got valid data at all

    print(f"Running Instructor + Pydantic extraction on {len(LISTINGS)} listings...\n")

    for listing in LISTINGS:
        try:
            result = client.messages.create(
                model=MODEL,
                max_tokens=400,
                temperature=0,
                max_retries=2,  # Instructor will retry up to twice on validation failure
                response_model=RentalExtraction,
                messages=[{
                    "role": "user",
                    "content": f"Extract the rental data from this German listing:\n\n{listing['text']}",
                }],
            )
        except ValidationError as e:
            print(f"Listing {listing['id']}: FAILED even after retries -> {e}")
            hard_failures += 1
            total_fields += len(FIELDS)
            continue

        # Instructor attaches the original API response (with usage info) as
        # ._raw_response on the returned Pydantic instance. Access defensively —
        # this is an implementation detail that could shift between versions.
        usage = getattr(getattr(result, "_raw_response", None), "usage", None)
        if usage:
            log_call("anthropic", MODEL, usage.input_tokens, usage.output_tokens)
        else:
            print("  (note: couldn't find usage info on this response, skipping cost log)")

        correct, field_results = score_listing(result, listing["ground_truth"])
        total_correct += correct
        total_fields += len(FIELDS)

        wrong = [f for f, ok in field_results.items() if not ok]
        status = "all correct" if not wrong else f"wrong: {wrong}"
        print(f"Listing {listing['id']}: {correct}/{len(FIELDS)} correct ({status})")
        print(f"  -> {result.model_dump()}")

    accuracy = total_correct / total_fields
    print("\n" + "=" * 60)
    print(f"Instructor + Pydantic accuracy: {accuracy:.1%}")
    print(f"Hard failures (exhausted retries): {hard_failures}/{len(LISTINGS)}")
    print("=" * 60)
    print(
        "\nCompare this number against your v3/v4/v5 results in "
        "prompt_experiment_results.csv from Friday. Same task, same data — "
        "the question is whether tooling beat prompting, and by how much."
    )


if __name__ == "__main__":
    run_experiment()