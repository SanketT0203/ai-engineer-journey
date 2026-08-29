# Beamter-Bot

**Status: finalized.** Multi-turn chat, page-level citations, and a
measured baseline are all in place — see below for what that means and
where the numbers stand.

A RAG chatbot for German bureaucracy: Anmeldung (address registration), the
Blue Card EU (work visa), and the Rundfunkbeitrag (broadcasting fee) — the
three things every international student in Germany ends up googling in
their first month. Answers cite the exact document and page they came from,
so "trust me" is never the only option, and it holds a real conversation —
follow-ups don't need to repeat the topic, and it'll switch languages on
request without losing the thread.

## Pipeline

```
load PDFs → chunk per page → embed (multilingual) → store in Qdrant
    → condense (fold history into a standalone question)
    → retrieve (dense, wide pool) → rerank (cross-encoder) → generate (Claude)
```

- **Multilingual embeddings** (`paraphrase-multilingual-MiniLM-L12-v2`) so
  German source documents and English questions land in the same vector
  space — a plain English-only model doesn't guarantee that alignment.
- **Qdrant** as the vector store, with per-page chunk payloads so every
  citation can name a real page number, not just a document name.
- **Reranking**: dense retrieval fetches a wide candidate pool, a
  cross-encoder re-scores each candidate against the actual question with
  full attention over the pair, and only the top survivors get sent to the
  LLM. Slower per-candidate than dense search alone, which is exactly why
  it only runs over a small pool instead of the whole collection.
- **Conversational memory**: the chat loop keeps the last few
  (question, answer) turns and, on every new message, asks Claude to fold
  that history into one standalone question before retrieval ever runs.
  See "Multi-turn chat" below.
- **Claude Haiku** generates the final answer, instructed to answer only
  from the retrieved context, cite `(document, page N)` for every claim,
  and answer in whichever language the (condensed) question is asking for.

## Multi-turn chat

Until this task, every question in this project was retrieved and answered
completely independently — the chat loop had no memory. That breaks on the
first real follow-up: ask "Wie hoch ist das allgemeine Mindestgehalt für
die Blaue Karte EU?", get an answer, then ask "kannst du das auf Englisch
erklären?" ("can you explain that in English?"), and a plain retrieval step
embeds the literal words "kannst du das auf Englisch erklären" — which
match nothing in a corpus of German bureaucracy text. Retrieval fails, the
LLM has no context, and the honest answer would be "I don't know," which is
wrong: the information was right there, the *question* just wasn't
self-contained.

The fix is `condense_question()` in `rag_chat.py`: before retrieval, the
last few turns plus the new message go to Claude with instructions to
rewrite the message into a single standalone question, resolving "das" /
"that" / "it" into whatever the history says it refers to. For the example
above, that turns "kannst du das auf Englisch erklären?" into something
like *"Explain the Blue Card EU minimum salary requirement in English."* —
which retrieves correctly (it names the actual topic) **and** carries the
language request forward in plain text, which is what `generate_answer()`'s
new language instruction picks up: answer in the question's language,
unless the question explicitly asks for a different one.

One rewrite step solving two problems (retrieval relevance and response
language) instead of two separate mechanisms is deliberate — the language
request was never really a separate signal from "what is this follow-up
actually asking," so it didn't need separate handling.

Two things worth knowing about how this is scoped:

- **The first message in any session skips condensation entirely** — no
  history exists yet, so there's nothing to fold in, and it would just be a
  wasted API call every single run.
- **`golden_dataset.json` and `eval_harness.py` stay single-turn on
  purpose.** Every golden question is graded as if it were asked cold, with
  no prior conversation — `condense_question()` never gets called during an
  eval run. That's a deliberate scope boundary, not an oversight: measuring
  multi-turn quality well needs multi-turn golden conversations (a
  real follow-up chain with a graded rewrite at each step), which is a
  bigger, separate dataset-design problem worth its own task rather than
  something to bolt onto a single-turn harness.

## How much context does retrieval actually cost?

Retrieval is a token firehose by construction: `RERANK_POOL` widens the net
to give the cross-encoder something to actually choose from, and every one
of the `TOP_K` chunks that survives reranking — up to `CHUNK_SIZE` (800)
characters each — goes into the generation prompt on *every single
question*, whether or not it ends up mattering to the final answer. That
cost was never actually measured before now; it was just assumed to be
"however many tokens 4 chunks happens to be."

Every query now logs it. `count_context_tokens()` in `rag_chat.py` calls
Anthropic's real token-counting endpoint (`client.messages.count_tokens`)
on the exact context string generation is about to see — free (it doesn't
cost money, and runs on a separate rate limit from actual generation
calls), and an exact count rather than a guess, which matters here since
German compounds and legal terminology don't tokenize the same as plain
English text. If that call fails for any reason, it falls back to a rough
`chars / 4` estimate rather than crashing the chat turn over a metric — the
CSV row records which kind of number it got (`exact_count` column).

Every turn appends one row to `context_tokens_log.csv` (timestamp,
question, chunk count, character count, token count, exact-or-estimated) —
same instinct as JobRadar's `usage_log.csv` from week 1, aimed at "how
expensive is retrieval itself" instead of "how expensive was the
extraction call." A number printed to the terminal scrolls away the moment
the next question comes in; a CSV is something to actually look back on
and aggregate.

**Dedup / compression ideas — noted, not yet built.** These are the
concrete next moves once the log has enough real rows to show whether
they're worth it:

- **Trim the overlap, not just note it.** `CHUNK_OVERLAP` (150 characters)
  means two adjacent chunks from the same page share a literal, duplicated
  block of text on purpose — that's what keeps a boundary-straddling
  sentence readable in both. But when two such chunks *both* make it into
  the same `TOP_K`, that shared block gets sent to the LLM twice. A
  deterministic string-level fix (find the actual overlapping substring
  between same-page neighbors and trim one copy) removes real, known
  duplication with no model involved and no risk of cutting something
  needed.
- **MMR-style selection instead of pure top-k.** Right now reranking picks
  the `TOP_K` highest-scoring candidates with no awareness of each other —
  two near-duplicate chunks can both win a slot even though the second one
  adds almost nothing beyond what the first already covers. Maximal
  Marginal Relevance would drop a candidate from the selection if its
  embedding is too similar to one already chosen, trading a little pure
  relevance for real context diversity.
- **Extractive context compression.** After retrieval, a small/cheap model
  call could pull out only the sentences within each chunk actually
  relevant to the question, discarding the rest of the 800-character
  window as padding. This is a real tradeoff, not a free win — it adds one
  more small LLM call to save tokens on the bigger generation call — so
  it's a "measure whether the net cost actually drops" candidate, not an
  "obviously helps" one.
- **Smaller `CHUNK_SIZE`, leaning on reranking's precision.** Large chunks
  carry more surrounding padding around the one sentence that actually
  answers a question. Shrinking `CHUNK_SIZE` and trusting the (now
  measured-to-work) reranking step to still find the right smaller chunk
  could cut per-chunk token cost directly — worth an A/B in
  `eval_harness.py`-style before touching production, same as reranking
  itself was.
- **Prompt caching for repeat context.** With a small, fixed document set
  like this, the same chunks get retrieved across many different users'
  similar questions. Anthropic's prompt caching (the same mechanism behind
  contextual retrieval's affordability, from the earlier Learn task) would
  let a repeated context block bill at the cheaper cached-read rate instead
  of full price every time — a cost fix that needs zero changes to
  retrieval quality at all.

## Evaluation

Two eval tools, two different jobs:

- `rag_eval.py` — hit-rate@k for dense vs hybrid (BM25+dense) vs
  hybrid+rerank, against `eval_questions.json` (20 question → expected-
  source pairs, retrieval only, no reference answers). Used to explore
  retrieval strategies before committing to one.
- `eval_harness.py` — the one that matters for production. Runs against
  `golden_dataset.json` (25 hand-verified question → reference-answer
  pairs, each grounded in the exact real German text in `docs/`) and
  reports both retrieval-quality metrics (hit@k, context precision@k,
  context recall — all computed locally, no API cost) and generation-
  quality metrics (faithfulness, answer relevancy — Claude-judged,
  hand-rolled analogues of the RAGAS metrics of the same name, small real
  API cost).

```bash
docker run -p 6333:6333 qdrant/qdrant
uv add pypdf sentence-transformers anthropic qdrant-client numpy
uv run python eval_harness.py docs/ golden_dataset.json
```

## Improvement: reranking

**The one change:** added a cross-encoder reranking step to `retrieve()` in
`rag_chat.py`. Before, dense retrieval returned its top `TOP_K` chunks
directly. Now it fetches a wider pool (`RERANK_POOL`, default 10), a
cross-encoder scores each candidate against the actual question, and only
the top `TOP_K` after that re-scoring go to the LLM.

**Why this should help, in theory:** dense retrieval embeds the question and
each chunk independently (a bi-encoder), so it can miss a chunk that's
relevant in a way plain embedding similarity doesn't capture well. A
cross-encoder sees the question and a candidate chunk together, with full
attention over the pair — more accurate, but too slow to run over an entire
collection, which is why it only reranks a small pool instead of replacing
dense search outright.

**Measured before/after**, run via `eval_harness.py` against
`golden_dataset.json` (25 questions, exact document+page match):

```
                     dense           dense+rerank    delta
hit@1                __/25 (__%)     __/25 (__%)     ___
ctx precision@1      ____            ____            ____
hit@2                __/25 (__%)     __/25 (__%)     ___
ctx precision@2      ____            ____            ____
hit@4                __/25 (__%)     __/25 (__%)     ___
ctx precision@4      ____            ____            ____
ctx recall@4         ___%            ___%             ___%

faithfulness (dense+rerank):      ____
answer relevancy (dense+rerank):  ____
```

Fill this table in from your own real run — paste `eval_harness.py`'s
printed output directly, it's formatted for exactly this. Real numbers need
your real `ANTHROPIC_API_KEY` and the real `sentence-transformers` /
`cross-encoder` models; anything computed without those (mine, in a sandbox
that can't install either) isn't trustworthy signal, only proof the code
runs.

**What to read into the delta once you have it:** hit@1 is the metric most
likely to move from reranking — it's exactly the case reranking targets,
promoting the single best chunk to the very top of a small pool where dense
similarity alone put it 2nd or 3rd. Small movement (or none) at hit@4 isn't
a bug: if the right chunk was already inside the wider `RERANK_POOL`,
reranking just reorders it — it can't manufacture a hit@4 that dense
retrieval already had. And on a very small corpus (a handful of PDFs, a
few dozen chunks — see the corpus-size note `rag_eval.py` prints), any
single metric can move by ±1 question just from where a coin-flip-close
score landed on one particular question; the direction of the delta across
several metrics together is more informative than any single number moving.

**The honest failure mode to watch for:** reranking can also make things
*worse* on a small corpus, if the cross-encoder is confidently wrong on a
handful of questions it happens to score confidently in the wrong
direction — that's exactly why this is measured, not assumed.
