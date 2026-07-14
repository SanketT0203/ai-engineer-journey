import csv
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "usage_log.csv"
#[rices are in usd per million tokens, so we divide by 1_000_000 to get the price per token]
MODEL_PRICING = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),   # introductory rate through 2026-08-31
    "gpt-4o-mini": (0.15, 0.60),
}

USD_TO_EUR = 0.86

def log_call(provider, model, input_tokens, output_tokens, csv_path=CSV_PATH):
    """Compute cost and append one row to the CSV. Returns cost in EUR."""
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
def track_anthropic_call(prompt,model="claude-haiku-4-5", max_tokens=200):
    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    cost_eur = log_call("anthropic", model, message.usage.input_tokens, message.usage.output_tokens)
    return message.content[0].text, cost_eur

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

if __name__=="__main__":
    text, cost = track_anthropic_call("Say hi in exactly five words.")
    print(f"\nReply: {text}")
    print(f"This call cost approximately €{cost:.6f}")  