import os
import anthropic
from dotenv import load_dotenv
from cost_calculator import log_call
client= anthropic.Anthropic()
MODEL = "claude-haiku-4-5"


system_prompt = "You are a concise, helpful assistant. Keep replies to 2-3 sentences unless asked for more."

def stream_reply(history):
    full_text   = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=500,
        system=system_prompt,
        messages=history,
    )as stream:
        for chunk in stream.text_stream:
            print(chunk, end="", flush=True)
            full_text += chunk
        final_message = stream.get_final_message()
    print()
    return full_text, final_message.usage.input_tokens, final_message.usage.output_tokens


def main():
    # `history` is the single source of truth for the conversation. Every
    # turn appends to it; every turn sends the WHOLE thing back to the API.
    history = []
    turn = 0
 
    print(f"Chatting with {MODEL}. Type 'exit' or 'quit' to stop.\n")
 
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
 
        turn += 1
 
        # 1. Append the user's turn to history BEFORE calling the API —
        #    the API needs to see it as part of `messages`.
        history.append({"role": "user", "content": user_input})
 
        print("Claude: ", end="", flush=True)
        reply_text, input_tokens, output_tokens = stream_reply(history)
 
        # 2. Append the assistant's own reply back into history too.
        #    Skip this and the model "forgets" everything it just said —
        #    the next turn would look like the start of a brand new chat.
        history.append({"role": "assistant", "content": reply_text})
 
        cost_eur = log_call("anthropic", MODEL, input_tokens, output_tokens)
        print(f"    [turn {turn}] input_tokens={input_tokens}  output_tokens={output_tokens}  cost=€{cost_eur:.6f}\n")
 
    print(f"\nConversation ended after {turn} turns. Full history is in usage_log.csv.")
    print("Open cost_summary.py or plot input_tokens per turn — you should see it grow roughly linearly.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted — bye!")