"""
eval_harness.py -- baseline scores for Beamter-Bot against golden_dataset.json.

Runs the CURRENT production pipeline (rag_chat.py's retrieve() + generate_answer()
-- dense-only retrieval, since hybrid/rerank only exist as a separate measurement
tool in rag_eval.py and were never wired into rag_chat.py itself) against 25
hand-verified question -> reference-answer pairs, and reports two families of
metric:

  Retrieval-side (no LLM, computed from real Qdrant hits + golden_dataset.json):
    - hit@k                        did the exact expected (source, page) show up in top k
    - context precision@k          RAGAS-style: are the RELEVANT chunks ranked near the top,
                                    not just present somewhere in top k (hand-rolled analogue
                                    of RAGAS's NonLLMContextPrecisionWithReference)
    - context recall (token overlap)  what fraction of the reference answer's words are
                                    actually backed by SOME retrieved chunk -- a cheap,
                                    no-LLM approximation of RAGAS's context recall

  Generation-side (needs a real Claude call to judge -- these are the two metrics
  that cannot be computed without ANTHROPIC_API_KEY and real money, even a
  little):
    - faithfulness                 does the generated answer only say things the
                                    retrieved context actually supports? (claim
                                    decomposition + verification, RAGAS-style,
                                    collapsed into one judge call per question here
                                    rather than two, to keep this cheap)
    - answer relevancy              does the answer actually address the question
                                    asked? (reverse-question + embedding similarity,
                                    RAGAS-style)

These are hand-rolled, not the `ragas` package -- consistent with the rest of
this project's from-scratch approach, and it means every number here can be
traced back to code you can read in this file, not a metric implementation
you have to trust blindly.

    docker run -p 6333:6333 qdrant/qdrant
    uv add pypdf sentence-transformers anthropic qdrant-client numpy
    uv run python eval_harness.py docs/ golden_dataset.json

Needs ANTHROPIC_API_KEY set. This makes real API calls: 1 generation call +
1 faithfulness-judge call + 1 relevancy-judge call per question, all on
claude-haiku-4-5 (cheap) -- for 25 questions that's ~75 calls, well under a
cent at current Haiku pricing, but it is real spend, not free like the
retrieval metrics above.
"""

import json
import re
import sys

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from rag_chat import load_pdf_pages, chunk_text, EMBED_MODEL_NAME, QDRANT_URL, generate_answer

GOLDEN_COLLECTION = "golden_eval_collection"  # separate from rag_chat.py's own + rag_eval.py's own, on purpose
TOP_K = 4  # matches rag_chat.py's own default -- an eval that retrieves more than production isn't measuring production
K_VALUES = [1, 2, 4]
JUDGE_MODEL = "claude-haiku-4-5"


def build_corpus(folder):
    records = []  # {"text", "source", "page"} -- index in this list is its id everywhere below
    for page in load_pdf_pages(folder):
        for chunk in chunk_text(page["text"]):
            records.append({"text": chunk, "source": page["source"], "page": page["page"]})
    return records


def build_index(records, embed_model, client):
    dim = embed_model.get_sentence_embedding_dimension()
    try:
        client.delete_collection(GOLDEN_COLLECTION)
    except Exception:
        pass
    client.create_collection(GOLDEN_COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    vectors = embed_model.encode([r["text"] for r in records], normalize_embeddings=True)
    client.upsert(GOLDEN_COLLECTION, points=[
        PointStruct(id=i, vector=v.tolist(), payload=r) for i, (v, r) in enumerate(zip(vectors, records))
    ])


def retrieve_ranked(question, embed_model, client, k):
    q_vec = embed_model.encode([question], normalize_embeddings=True)[0]
    hits = client.query_points(GOLDEN_COLLECTION, query=q_vec.tolist(), limit=k).points
    return [h.payload for h in hits]  # ranked list of {"text","source","page"}


# ── retrieval-side metrics: no LLM, pure code ────────────────────────────

def _is_relevant(chunk, expected_source, expected_page):
    return chunk["source"] == expected_source and chunk["page"] == expected_page


def hit_at_k(ranked, expected_source, expected_page, k):
    return any(_is_relevant(c, expected_source, expected_page) for c in ranked[:k])


def context_precision_at_k(ranked, expected_source, expected_page, k):
    """RAGAS-style: average precision over the ranked list, not just 'is the
    right chunk present somewhere in top k'. A relevant chunk at rank 1 scores
    higher than the same relevant chunk buried at rank 4 -- ranking quality,
    not just recall."""
    relevant_flags = [_is_relevant(c, expected_source, expected_page) for c in ranked[:k]]
    if not any(relevant_flags):
        return 0.0
    precisions = []
    hits_so_far = 0
    for i, is_rel in enumerate(relevant_flags, start=1):
        if is_rel:
            hits_so_far += 1
            precisions.append(hits_so_far / i)
    return sum(precisions) / sum(relevant_flags)


def context_recall_token_overlap(ranked, ground_truth, k):
    """Cheap, no-LLM approximation of RAGAS context recall: what fraction of
    the reference answer's distinct words are found SOMEWHERE in the
    retrieved context. Real RAGAS context recall checks whether each CLAIM in
    the reference is traceable to context (needs an LLM); this is a rougher
    proxy that at least needs no API call, so it's free to run on all 25."""
    gt_words = set(re.findall(r"\w+", ground_truth.lower()))
    gt_words = {w for w in gt_words if len(w) > 2}  # drop tiny stopword-ish tokens
    if not gt_words:
        return 0.0
    context_words = set()
    for c in ranked[:k]:
        context_words |= set(re.findall(r"\w+", c["text"].lower()))
    covered = gt_words & context_words
    return len(covered) / len(gt_words)


# ── generation-side metrics: real Claude calls, real cost ────────────────

def faithfulness(question, answer, context_text, client):
    """Decompose the answer into atomic claims and check each against the
    retrieved context, in one judge call. Score = supported claims / total
    claims. RAGAS normally splits decomposition and verification into two
    calls; collapsing them here halves the cost for a baseline run -- worth
    knowing if you compare these numbers to a real `ragas` run later."""
    prompt = (
        "You are auditing an AI assistant's answer against the context it was "
        "given. Break the ANSWER into its individual factual claims, then for "
        "each claim say whether the CONTEXT actually supports it.\n\n"
        f"CONTEXT:\n{context_text}\n\nANSWER:\n{answer}\n\n"
        "Respond with ONLY a JSON array, no other text, like: "
        '[{"claim": "...", "supported": true}, {"claim": "...", "supported": false}]'
    )
    reply = client.messages.create(
        model=JUDGE_MODEL, max_tokens=500, messages=[{"role": "user", "content": prompt}],
    )
    text = reply.content[0].text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    claims = json.loads(match.group(0)) if match else []
    if not claims:
        return None  # couldn't parse -- don't silently report 0, surface it as missing
    supported = sum(1 for c in claims if c.get("supported"))
    return supported / len(claims)


def answer_relevancy(question, answer, embed_model, client):
    """Ask the judge to generate a few questions the ANSWER would be a good
    reply to, embed those alongside the original question, and average their
    cosine similarity. An answer that drifts off-topic generates reverse-
    questions that don't resemble the original question -- that's the
    signal."""
    prompt = (
        "Given only this ANSWER (not the original question), write 3 different "
        "questions that this answer would be a good, direct reply to.\n\n"
        f"ANSWER:\n{answer}\n\n"
        'Respond with ONLY a JSON array of 3 strings, like: ["question 1", "question 2", "question 3"]'
    )
    reply = client.messages.create(
        model=JUDGE_MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}],
    )
    text = reply.content[0].text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    reverse_questions = json.loads(match.group(0)) if match else []
    if not reverse_questions:
        return None
    vecs = embed_model.encode([question] + reverse_questions, normalize_embeddings=True)
    q_vec, reverse_vecs = vecs[0], vecs[1:]
    sims = [float(q_vec @ rv) for rv in reverse_vecs]  # both normalized -> dot product is cosine similarity
    return sum(sims) / len(sims)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"
    golden_path = sys.argv[2] if len(sys.argv) > 2 else "golden_dataset.json"

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (same as rag_chat.py) -- generation and judging both need it.")

    from sentence_transformers import SentenceTransformer
    from anthropic import Anthropic
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL)
    try:
        client.get_collections()
    except Exception as e:
        sys.exit(f"Can't reach Qdrant at {QDRANT_URL} ({e}).\nStart it first: docker run -p 6333:6333 qdrant/qdrant")
    judge = Anthropic()

    records = build_corpus(folder)
    build_index(records, embed_model, client)
    print(f"Indexed {len(records)} chunks from {len({r['source'] for r in records})} PDF(s). "
          f"Running baseline over {len(golden)} golden questions...\n"
          f"(makes ~{len(golden) * 3} Claude Haiku calls -- generation + faithfulness judge + "
          f"relevancy judge, per question -- real spend, though small at Haiku pricing)\n")

    totals = {k: {"hit": 0, "precision": 0.0} for k in K_VALUES}
    recall_sum = 0.0
    faithfulness_scores, relevancy_scores = [], []

    for i, item in enumerate(golden, start=1):
        q, gt = item["question"], item["ground_truth"]
        exp_source, exp_page = item["expected_source"], item["expected_page"]

        ranked = retrieve_ranked(q, embed_model, client, k=max(K_VALUES))
        for k in K_VALUES:
            totals[k]["hit"] += hit_at_k(ranked, exp_source, exp_page, k)
            totals[k]["precision"] += context_precision_at_k(ranked, exp_source, exp_page, k)
        recall_sum += context_recall_token_overlap(ranked, gt, TOP_K)

        # generation uses TOP_K chunks, same as production rag_chat.py -- not the wider `ranked` list above
        hits_for_gen = [(r, None) for r in ranked[:TOP_K]]  # generate_answer expects (payload, score) pairs
        answer = generate_answer(q, hits_for_gen)
        context_text = "\n\n".join(f"[{r['source']}, page {r['page']}]\n{r['text']}" for r in ranked[:TOP_K])

        f_score = faithfulness(q, answer, context_text, judge)
        r_score = answer_relevancy(q, answer, embed_model, judge)
        if f_score is not None:
            faithfulness_scores.append(f_score)
        if r_score is not None:
            relevancy_scores.append(r_score)

        print(f"  [{i}/{len(golden)}] {q[:60]}")

    n = len(golden)
    print("\n=== BASELINE SCORES (dense-only retrieval, current production rag_chat.py) ===\n")
    print("Retrieval (no LLM, exact-page-match against golden_dataset.json):")
    for k in K_VALUES:
        hit_pct = totals[k]["hit"] / n
        prec_avg = totals[k]["precision"] / n
        print(f"  hit@{k}: {totals[k]['hit']}/{n} ({hit_pct:.0%})   context precision@{k}: {prec_avg:.2f}")
    print(f"  context recall (token overlap, @{TOP_K}): {recall_sum / n:.0%}")

    print("\nGeneration (Claude-judged, real API calls):")
    if faithfulness_scores:
        print(f"  faithfulness:     {sum(faithfulness_scores) / len(faithfulness_scores):.2f}  "
              f"(n={len(faithfulness_scores)}/{n} -- some judge replies may fail to parse)")
    else:
        print("  faithfulness:     no valid judge responses parsed")
    if relevancy_scores:
        print(f"  answer relevancy: {sum(relevancy_scores) / len(relevancy_scores):.2f}  "
              f"(n={len(relevancy_scores)}/{n})")
    else:
        print("  answer relevancy: no valid judge responses parsed")

    print("\nThis is ONE run against dense-only retrieval -- the reference point future changes "
          "(hybrid search, reranking, chunk-size tuning) get compared against, not a verdict on "
          "its own. Re-run after any pipeline change and diff against these numbers.")


if __name__ == "__main__":
    main()