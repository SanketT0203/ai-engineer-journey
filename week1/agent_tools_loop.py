"""
Day 7 build — the tool-use loop, by hand, no framework.

Three tools: calculator, get_current_time, search_jobs(query).
The loop: call API -> if tool_use, run the real function -> feed result back -> repeat.

    uv run python agent_tool_loop.py
"""

import ast
import json
import operator
import os
from datetime import datetime

import anthropic

from cost_calculator import log_call

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5"


# ── Tool implementations ─────────────────────────────────────────────

# Never eval() model-provided text — it's untrusted input (same category of
# risk as prompt injection from Week 1). This walks the parse TREE and only
# allows numbers and basic math operators; anything else (function calls,
# imports, attribute access) is rejected before it can run.
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: could not safely evaluate '{expression}' ({e})"


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def search_jobs(query: str) -> str:
    """Search your real jobs_extracted.json if it exists; falls back to a
    tiny mock dataset so this tool works even on a fresh checkout."""
    path = "jobs_extracted.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            jobs = json.load(f)
    else:
        jobs = [
            {"title": "AI Engineer", "company": "MockCorp", "skills": ["Python", "LangChain"], "location": "Berlin"},
            {"title": "Senior ML Engineer", "company": "MockAI", "skills": ["PyTorch", "AWS"], "location": "Munich"},
        ]

    q = query.lower()
    matches = [
        j for j in jobs
        if q in j.get("title", "").lower() or any(q in s.lower() for s in j.get("skills", []))
    ]
    if not matches:
        return f"No jobs found matching '{query}'."
    return json.dumps(matches[:5], ensure_ascii=False)


TOOL_FUNCTIONS = {
    "calculator": lambda **kwargs: calculator(kwargs["expression"]),
    "get_current_time": lambda **kwargs: get_current_time(),
    "search_jobs": lambda **kwargs: search_jobs(kwargs["query"]),
}

# ── Tool schemas the model sees ──────────────────────────────────────

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression, e.g. '2 + 2 * 3'. Supports +, -, *, /, **, parentheses, and unary minus.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "The math expression to evaluate."}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current local date and time.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_jobs",
        "description": "Search previously fetched AI-engineer job postings by keyword — matches against job title or listed skills. Returns up to 5 matches.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keyword, e.g. 'LangChain' or 'senior'."}},
            "required": ["query"],
        },
    },
]


# ── The loop itself ───────────────────────────────────────────────────

def run_agent(user_message, max_iterations=8, verbose=True):
    messages = [{"role": "user", "content": user_message}]

    for iteration in range(1, max_iterations + 1):
        response = client.messages.create(
            model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages,
        )
        log_call("anthropic", MODEL, response.usage.input_tokens, response.usage.output_tokens)

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            if verbose:
                print(f"[iteration {iteration}] final answer, stop_reason={response.stop_reason}")
            return final_text

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = TOOL_FUNCTIONS.get(block.name)
                if verbose:
                    print(f"[iteration {iteration}] model wants to call {block.name}({block.input})")
                try:
                    result = fn(**block.input) if fn else f"Unknown tool: {block.name}"
                except Exception as e:
                    result = f"Error executing {block.name}: {e}"
                if verbose:
                    print(f"                -> result: {result}")
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(result)})

        messages.append({"role": "user", "content": tool_results})

    return "Reached max_iterations without a final answer — the model may be stuck in a tool-call loop."


if __name__ == "__main__":
    questions = [
        "What's 47 * 12 + 8?",
        "What time is it right now, and separately, what's 100 / 4?",
        "Search my job listings for anything mentioning LangChain.",
    ]
    for q in questions:
        print("=" * 60)
        print("Q:", q)
        answer = run_agent(q)
        print("\nFINAL ANSWER:", answer, "\n")