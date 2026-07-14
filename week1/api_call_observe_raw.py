import json
import os
 
import httpx
from dotenv import load_dotenv
 
load_dotenv()
 
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

PROMPT = "In one short sentence, explain what a token is in the context of LLMs."

def call_anthropic_raw():
    print("\n" + "=" * 60)
    print("ANTHROPIC — raw httpx POST")
    print("=" * 60)
 
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        # Anthropic auth: a custom header, NOT "Authorization: Bearer ..."
        "x-api-key": ANTHROPIC_API_KEY,
        # Anthropic versions its API via a header, not the URL
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-haiku-4-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": PROMPT}],
    }
 
    resp = httpx.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()  # raises an exception on 4xx/5xx — read the message if it fires
    data = resp.json()
 
    print("\nFull raw response JSON:")
    print(json.dumps(data, indent=2))
 
    # This is the bit that matters for cost tracking:
    usage = data["usage"]
    print("\n--- extracted ---")
    print(f"input_tokens:  {usage['input_tokens']}")
    print(f"output_tokens: {usage['output_tokens']}")
    print(f"reply text:    {data['content'][0]['text']}")
    return usage

if __name__ == "__main__":
    call_anthropic_raw()