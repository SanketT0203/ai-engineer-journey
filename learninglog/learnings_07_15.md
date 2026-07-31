worked on prompting techniques, zero shot, fewshot,role prompting,xml structures.
created an info extraction task from data via prompting.
checked accuracies with different prompting techniques 
10 German rental listings
with hand-written ground truth, ran the same extraction task through 5 prompt versions,
scored each one automatically against the ground truth instead of judging "does this
look right" by eye.

Checked accuracy across all 5 — giving more info, instructions, and structure
consistently helps. Biggest single jump: forcing JSON output (v3) — v1/v2 often
didn't even produce something parseable, so a huge chunk of the "accuracy" gain
wasn't really about better understanding, it was about the model FAILING LESS at
following a format. Worth remembering: unstructured output isn't just harder to
read, it's often literally unusable downstream without extra parsing hacks

Lesson: "prompt engineering" for a real extraction task isn't really about phrasing
things politely — it's about explicitly encoding the business/domain rules a human
would apply automatically. The model doesn't know your edge cases unless you tell it.

All of these accuracies were tested at temperature=0 because I didn't want to test
the variety of outputs, I wanted the deterministic ones.


Built the manual agent loop: give the model a `tools` schema (calculator,
get_current_time, search_jobs), send `messages`, check `response.stop_reason`. If
it's `"tool_use"`, the response contains one or more `tool_use` blocks — I execute
the REAL Python function myself, wrap the result as a `tool_result` block matched by
`tool_use_id`, append it as a new user-role message, and loop again. Only stop when
`stop_reason` is something other than `tool_use`.
