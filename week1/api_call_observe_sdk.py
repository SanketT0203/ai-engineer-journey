import os
 
from dotenv import load_dotenv
 
load_dotenv()
 
PROMPT = "In one short sentence, explain what a token is in the context of LLMs."


def call_anthropic_sdk():
    import anthropic
 
    print("\n" + "=" * 60)
    print("ANTHROPIC — official SDK")
    print("=" * 60)
 
    
    client = anthropic.Anthropic()              #imp
 
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": PROMPT}],
    )
 
    # `message` is a typed Python object, not a raw dict — this is the other
    # thing the SDK buys you: autocomplete + validation instead of ["key"]["errors"]
    print(f"reply text:    {message.content[0].text}")
    print(f"input_tokens:  {message.usage.input_tokens}")
    print(f"output_tokens: {message.usage.output_tokens}")
 
    # Proof it's the same data as the raw call, just wrapped in an object:
    print(f"\nAs a dict (message.model_dump()):")
    print(message.model_dump_json(indent=2))
 
    return message.usage

if __name__ == "__main__":
    call_anthropic_sdk()