"""
rag_chat.py -- chat with a folder of PDFs, Qdrant as the vector DB.

    load -> chunk -> embed -> store (Qdrant) -> condense -> retrieve -> rerank -> generate

Migrated from the numpy-array version: same stages, but storage and
retrieval are now a real Qdrant collection -- persistent, filterable by
payload, running as its own service -- instead of one numpy array living
inside this script's own memory. Every answer cites which document AND
which page it came from, not just which document. Retrieval reranks a wide
dense pool with a cross-encoder before the top TOP_K reach the LLM -- see
golden_dataset.json + eval_harness.py for the measured before/after effect
of that change, and BEAMTER_BOT.md for the writeup.

FINALIZED (this task): the chat loop now remembers recent turns and
condenses each new message against them before retrieval runs, so a
follow-up like "kannst du das auf Englisch erklären?" ("can you explain
that in English?") resolves "das" against the previous answer's topic
instead of being embedded and searched on its own -- which would retrieve
nothing useful, since the literal words "auf Englisch erklären" don't
match any bureaucracy chunk. The condensed question also carries the
language request forward explicitly (see condense_question() and the
language instruction in generate_answer()), so the SAME rewrite step
fixes both retrieval relevance and which language the answer comes back
in, without needing two separate mechanisms.

TOKEN LOGGING (this task): every query now logs how many tokens its
retrieved context actually cost, to context_tokens_log.csv -- see
count_context_tokens() / log_context_tokens() below, and BEAMTER_BOT.md
for what the numbers mean and for dedup/compression ideas that would
shrink them (not yet built -- noted for a future task).

    docker run -p 6333:6333 qdrant/qdrant
    uv add pypdf sentence-transformers numpy anthropic qdrant-client
    uv run python rag_chat.py docs/

Needs ANTHROPIC_API_KEY set, and a Qdrant instance reachable at QDRANT_URL
(defaults to http://localhost:6333 -- the docker command above starts it).
"""

import csv
import glob
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
)

# Multilingual on purpose -- these docs are German, questions are often
# English, and this is the model that was swapped in specifically so those
# two don't have to match. See the earlier chat about all-MiniLM-L6-v2
# being English-only for why a plain monolingual model wasn't enough.
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# Multilingual cross-encoder, same family already vetted for German docs +
# English questions in rag_eval.py's hybrid+rerank experiment.
CROSS_ENCODER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CHUNK_SIZE = 800     # characters, not tokens -- simple and good enough here
CHUNK_OVERLAP = 150  # keeps a sentence that straddles a boundary readable in both chunks
TOP_K = 4            # how many chunks to hand the LLM per question, AFTER reranking
RERANK_POOL = 10     # how many chunks dense retrieval fetches BEFORE reranking narrows to TOP_K
HISTORY_TURNS = 3    # how many prior (question, answer) turns get shown to the condenser -- bounded so a long chat doesn't slowly bloat every request
COLLECTION_NAME = "bureaucracy_docs"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
CONTEXT_LOG_PATH = os.environ.get("CONTEXT_LOG_PATH", "context_tokens_log.csv")


# ── load: pull text out of every PDF, PAGE BY PAGE ───────────────────────
# The numpy version joined all of a document's pages into one string before
# chunking, which meant it had no idea which page any given chunk came
# from. Keeping pages separate here is what makes a real page number in
# every citation possible instead of just a document name.
def load_pdf_pages(folder):
    pages = []  # [{"source": filename, "page": 1-indexed page number, "text": ...}, ...]
    for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        for i, page in enumerate(PdfReader(path).pages, start=1):
            pages.append({"source": os.path.basename(path), "page": i, "text": page.extract_text() or ""})
    return pages


# ── chunk: split EACH PAGE into overlapping windows ──────────────────────
# Chunking per page instead of per whole-document trades a little context
# at page boundaries for something worth more here: every chunk belongs to
# exactly one page, so its citation can name a real page number, not a guess.
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ── store: (re)create the Qdrant collection, embed every chunk, upsert ───
def build_index(folder, embed_model, client):
    dim = embed_model.get_sentence_embedding_dimension()
    try:
        client.delete_collection(COLLECTION_NAME)  # rebuilt fresh every run, same simplicity as the numpy version
    except Exception:
        pass  # fine if it didn't exist yet (first run ever)
    client.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    records = []  # one per chunk: becomes that chunk's Qdrant PAYLOAD below
    for page in load_pdf_pages(folder):
        for chunk in chunk_text(page["text"]):
            records.append({"text": chunk, "source": page["source"], "page": page["page"]})
    if not records:
        return 0

    vectors = embed_model.encode([r["text"] for r in records], normalize_embeddings=True)
    client.upsert(COLLECTION_NAME, points=[
        PointStruct(id=i, vector=vec.tolist(), payload=rec)
        for i, (vec, rec) in enumerate(zip(vectors, records))
    ])
    return len(records)


# ── condense: fold conversation history into ONE standalone question ─────
# Every question so far in this project has been embedded and retrieved on
# exactly as typed. That breaks the moment there's a real conversation --
# "kannst du das auf Englisch erklären?" contains no words a vector search
# over German bureaucracy text would ever match. This rewrites a follow-up
# into a standalone question BEFORE it reaches retrieve(), using the recent
# history to resolve "das" / "that" / "it" into the actual topic. It
# deliberately does NOT strip out an explicit language request like "auf
# Englisch" -- keeping it in the rewritten question is what lets
# generate_answer()'s language instruction (below) pick it up later,
# without a second mechanism just for language.
#
# Skips the LLM call entirely on the first turn (no history yet) -- there's
# nothing to condense against, and it'd be a wasted API call every single
# session.
def condense_question(history, question):
    if not history:
        return question
    from anthropic import Anthropic
    history_text = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in history[-HISTORY_TURNS:])
    prompt = (
        "Given this conversation history and a NEW follow-up message, rewrite the follow-up "
        "into a single, fully standalone question that makes sense with no prior context. "
        "Resolve pronouns and references (\"that\", \"it\", \"das\", \"davon\") into the actual "
        "topic from the history. If the follow-up asks for the answer in a different language "
        "(for example \"auf Englisch\", \"in English\", \"auf Deutsch\"), KEEP that language "
        "request explicit in the rewritten question -- do not translate it away. "
        "Reply with ONLY the rewritten question, nothing else.\n\n"
        f"HISTORY:\n{history_text}\n\nNEW MESSAGE: {question}"
    )
    client = Anthropic()
    reply = client.messages.create(
        model="claude-haiku-4-5", max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return reply.content[0].text.strip()


# ── retrieve: Qdrant ranks a wide pool, then a cross-encoder reranks it ──
# `source` is optional payload filtering in action (last lesson's topic) --
# pass a filename to restrict the search to just that document.
#
# Two-stage on purpose: dense retrieval (bi-encoder) is fast and searches
# the whole collection, but it embeds the question and each chunk
# INDEPENDENTLY, so it can miss chunks that are relevant in a way that
# doesn't show up as raw embedding similarity. The cross-encoder sees the
# question and each candidate chunk TOGETHER (full attention over the pair),
# which is slower -- too slow to run over an entire collection -- but far
# more accurate. So: cast a wide net with the cheap method (RERANK_POOL
# candidates), then let the expensive, accurate method pick the real top
# TOP_K out of that pool.
#
# collection_name is a parameter (not just the COLLECTION_NAME constant) so
# eval_harness.py can point this SAME function at its own eval collection --
# an eval that calls a different retrieval implementation than production
# isn't measuring production.
def retrieve(question, embed_model, cross_encoder, client, k=TOP_K, source=None, collection_name=COLLECTION_NAME):
    q_vec = embed_model.encode([question], normalize_embeddings=True)[0]
    query_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]) if source else None
    pool = client.query_points(
        collection_name, query=q_vec.tolist(), query_filter=query_filter, limit=RERANK_POOL,
    ).points
    if not pool:
        return []
    pairs = [(question, hit.payload["text"]) for hit in pool]
    ce_scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(pool, ce_scores), key=lambda pair: -pair[1])[:k]
    return [(hit.payload, float(score)) for hit, score in reranked]


# ── shared: build the context block retrieve() feeds to the LLM ──────────
# Pulled out so generate_answer() and the token-logging step below build
# EXACTLY the same string -- logging a token count for text that isn't
# quite what generation actually saw would defeat the point of measuring.
def format_context(hits):
    return "\n\n".join(f"[{r['source']}, page {r['page']}]\n{r['text']}" for r, _ in hits)


# ── measure: how many tokens did retrieval just hand the LLM? ────────────
# Retrieval is a token firehose by construction -- RERANK_POOL widens the
# net to give the cross-encoder something to choose from, TOP_K chunks (at
# CHUNK_SIZE=800 chars each, worst case) go to generation, and every one of
# those tokens is billed on every single question, whether or not it ends
# up mattering to the answer. This never got measured before now; it was
# just assumed to be "however many tokens 4 chunks happens to be."
#
# Uses Anthropic's real token-counting endpoint (client.messages.count_tokens)
# -- it's free (doesn't cost money, separate rate limit from actual
# generation calls) and gives an EXACT count instead of a guess, which
# matters here since German compounds and legal terminology don't tokenize
# at the same rate as plain English. Falls back to a rough chars/4 estimate
# if that call fails for any reason (offline, rate-limited) -- a metric
# breaking shouldn't break the chat turn it's measuring.
def count_context_tokens(context):
    if not context:
        return 0, True
    from anthropic import Anthropic
    try:
        client = Anthropic()
        resp = client.messages.count_tokens(
            model="claude-haiku-4-5", messages=[{"role": "user", "content": context}],
        )
        return resp.input_tokens, True  # True = exact count from the API
    except Exception:
        return len(context) // 4, False  # rough fallback -- False = estimated, not exact


# ── log: append one row per query to a running CSV ───────────────────────
# A number printed to the terminal scrolls away the moment the next question
# comes in. Appending to CSV means the cost of retrieval is something you
# can actually look back on and aggregate later (sum a column, plot it,
# notice a slow creep) -- same instinct as JobRadar's usage_log.csv from
# Week 1, applied to the "how expensive is retrieval itself" question
# rather than "how expensive was the extraction call."
def log_context_tokens(question, hits, context, n_tokens, exact, path=CONTEXT_LOG_PATH):
    is_new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp_utc", "question", "n_chunks", "context_chars", "context_tokens", "exact_count"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            question[:120],
            len(hits),
            len(context),
            n_tokens,
            exact,
        ])


# ── generate: hand the retrieved, page-tagged chunks to the LLM ──────────
def generate_answer(question, hits):
    from anthropic import Anthropic
    context = format_context(hits)
    prompt = (
        "Answer the question using ONLY the context below, which comes from "
        "real German bureaucracy documents. If the answer isn't in the "
        "context, say you don't know instead of guessing -- these are "
        "visa/registration/fee rules, being wrong is worse than saying "
        "nothing. Cite the document name and page number for every claim, "
        "in the form (source, page N), right where you state it -- keep "
        "citing sources no matter which language you answer in.\n\n"
        "Answer in the same language the question is written in, UNLESS "
        "the question explicitly asks for a different language (e.g. "
        "\"auf Englisch\", \"in English\", \"auf Deutsch\") -- in that "
        "case, answer in the requested language instead, still grounded "
        "in the same German-language context above.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    reply = client.messages.create(
        model="claude-haiku-4-5", max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return reply.content[0].text


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (same as the rest of the project).")

    client = QdrantClient(url=QDRANT_URL)
    try:
        client.get_collections()  # cheap call purely to confirm the server is actually reachable
    except Exception as e:
        sys.exit(f"Can't reach Qdrant at {QDRANT_URL} ({e}).\nStart it first: docker run -p 6333:6333 qdrant/qdrant")

    from sentence_transformers import SentenceTransformer, CrossEncoder
    print(f"Loading + embedding PDFs from {folder}/ into Qdrant ({QDRANT_URL})...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
    n_chunks = build_index(folder, embed_model, client)
    n_sources = len(glob.glob(os.path.join(folder, "*.pdf")))
    print(f"Indexed {n_chunks} chunks from {n_sources} PDF(s). Ask a question (Ctrl+C to quit).\n")

    history = []  # [{"question": raw text as typed, "answer": ...}, ...] -- oldest first, bounded below
    while True:
        raw_question = input("> ").strip()
        if not raw_question:
            continue

        question = condense_question(history, raw_question)
        if question != raw_question:
            print(f"  (understood as: {question})")  # visible so a bad rewrite is obvious, not silent

        hits = retrieve(question, embed_model, cross_encoder, client)

        context = format_context(hits)
        n_tokens, exact = count_context_tokens(context)
        if hits:
            log_context_tokens(question, hits, context, n_tokens, exact)

        answer = generate_answer(question, hits)
        real_sources = ", ".join(sorted({f"{r['source']} p.{r['page']}" for r, _ in hits}))
        tok_label = f"~{n_tokens} tokens" if exact else f"~{n_tokens} tokens (estimated)"
        print(f"\n{answer}\n\n  (actually retrieved: {real_sources})\n  (context: {len(hits)} chunks, {tok_label})\n")

        # store the RAW question, not the condensed one -- condensation re-reads this
        # history fresh from the original wording every time, so it isn't compounding
        # its own earlier rewrites turn over turn.
        history.append({"question": raw_question, "answer": answer})
        history = history[-HISTORY_TURNS:]


if __name__ == "__main__":
    main()