"""
Reusable cost tracker — wraps any Claude/GPT call, logs tokens + EUR cost to a CSV.

You'll import this in almost every script for the rest of the roadmap, so keep
it dependency-light (stdlib csv only) and provider-agnostic.

Usage:
    from cost_tracker import track_anthropic_call, track_openai_call

    text, cost_eur = track_anthropic_call("Explain RAG in one sentence.")
    print(text, f"(cost: €{cost_eur:.6f})")
"""

import csv
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY / OPENAI_API_KEY from .env into the environment

CSV_PATH = "usage_log.csv"

# $ per 1M tokens (input, output). Verified July 2026 — prices change, recheck
# every few weeks at anthropic.com/pricing and openai.com/api/pricing.
MODEL_PRICING = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),   # introductory rate through 2026-08-31
    "gpt-4o-mini": (0.15, 0.60),
}

# USD -> EUR. Approximate, fluctuates daily — good enough for tracking your
# own learning-project spend, not for anything you'd bill a client for.
USD_TO_EUR = 0.86


def log_call(provider, model, input_tokens, output_tokens, csv_path=CSV_PATH):
    """Compute cost and append one row to the CSV. Returns cost in EUR."""
    # Defensive coercion: this function is a boundary — callers might hand it
    # a string by accident (e.g. an f-string result). Fail loudly and clearly
    # here with a ValueError, instead of a confusing TypeError deep in the math.
    try:
        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"log_call expected input_tokens/output_tokens to be numbers, "
            f"got input_tokens={input_tokens!r} ({type(input_tokens).__name__}), "
            f"output_tokens={output_tokens!r} ({type(output_tokens).__name__}). "
            f"Check what you're passing in at the call site."
        ) from e

    price_in, price_out = MODEL_PRICING[model]
    cost_usd = (input_tokens / 1e6) * price_in + (output_tokens / 1e6) * price_out
    cost_eur = cost_usd * USD_TO_EUR

    is_new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp", "provider", "model", "input_tokens",
                              "output_tokens", "cost_usd", "cost_eur"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            provider, model, input_tokens, output_tokens,
            round(cost_usd, 8), round(cost_eur, 8),
        ])

    print(f"[{provider}/{model}] {input_tokens} in / {output_tokens} out "
          f"-> ${cost_usd:.6f} (€{cost_eur:.6f})")
    return cost_eur


def track_anthropic_call(prompt, model="claude-haiku-4-5", max_tokens=200):
    """Call Claude, log the cost, return (reply_text, cost_eur)."""
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    cost_eur = log_call("anthropic", model,
                         msg.usage.input_tokens, msg.usage.output_tokens)
    return msg.content[0].text, cost_eur


def track_openai_call(prompt, model="gpt-4o-mini"):
    """Call GPT, log the cost, return (reply_text, cost_eur)."""
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
    )
    cost_eur = log_call("openai", model,
                         resp.usage.prompt_tokens, resp.usage.completion_tokens)
    return resp.choices[0].message.content, cost_eur


if __name__ == "__main__":
    text, cost = track_anthropic_call("Say hi in exactly five words.")
    print(f"\nReply: {text}")
    print(f"This call cost approximately €{cost:.6f}")