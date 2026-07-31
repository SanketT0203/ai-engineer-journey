how attention works for llms, 
difference in http raw calls and sdk calls. prefer sdk calls it handles crucial error handling by itself instead of writiing it manually.

The SDK isn't hiding anything mysterious — it's building the same JSON body and
headers, just with retries, backoff on rate limits, and typed response objects for
free. Also played with temperature and top_p: temperature controls how random the
next-token choice is, top_p restricts the choice to the smallest set of tokens whose
combined probability crosses a threshold.

how do I count tokens, and where do the numbers actually come from?
For plain text, the API hands you real counts after every call —
`response.usage.input_tokens` and `response.usage.output_tokens` (Anthropic naming;
OpenAI calls the same thing `prompt_tokens`/`completion_tokens`). Cost math is
`(input_tokens / 1e6) * price_in + (output_tokens / 1e6) * price_out`, since pricing
is quoted per million tokens.

For images, tokens aren't just "however much text is in it" — there's an actual
formula: `ceil(width/28) * ceil(height/28)`. A PDF's token cost is the image cost
of each rendered page PLUS the text cost of anything extracted from it, combined.
Spreadsheets have no native multimodal token path — they get converted to text
first, so their cost is just normal text-token counting on whatever got extracted.
Context window = the max total tokens (input + output) the model can hold in one
call before it starts truncating or refusing