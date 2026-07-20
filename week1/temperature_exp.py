import anthropic
from cost_calculator import log_call

client = anthropic.Anthropic()
 
PROMPT = "Write one creative sentence about coffee."
MODEL = "claude-haiku-4-5"

def ask(temperature):
    msg = client.messages.create(
        model=MODEL,
        max_tokens=60,
        temperature=temperature,
        messages=[{"role": "user", "content": PROMPT}],
    )
    log_call("anthropic", MODEL, msg.usage.input_tokens, msg.usage.output_tokens)
    return msg.content[0].text.strip()

if __name__ == "__main__":
    print("=" * 60)
    print("TEMPERATURE = 0  (deterministic — expect identical or near-identical output)")
    print("=" * 60)
    t0_results = []
    for i in range(3):  # 3 is enough to prove the point at t=0
        text = ask(temperature=0)
        t0_results.append(text)
        print(f"{i+1}. {text}")
 
    print("\n" + "=" * 60)
    print("TEMPERATURE = 1  (sampling — expect real variation)")
    print("=" * 60)
    t1_results = []
    for i in range(10):
        text = ask(temperature=1)
        t1_results.append(text)
        print(f"{i+1}. {text}")
 
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"t=0 -> {len(set(t0_results))} unique output(s) out of {len(t0_results)} runs")
    print(f"t=1 -> {len(set(t1_results))} unique output(s) out of {len(t1_results)} runs")
    print(
        "\nIf t=0 gave you >1 unique output: that's real, and worth noting in your "
        "log — most providers' 'temperature=0' is 'almost always the top token' "
        "rather than a strict guarantee, due to floating-point non-determinism on "
        "their serving hardware. This is a genuinely useful fact for interviews: "
        "'temperature=0 is not a 100% determinism guarantee' surprises people."
    )