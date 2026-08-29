"""
rag_eval.py -- hit-rate@k for dense-only vs hybrid (dense+BM25 via RRF) vs
hybrid+rerank (cross-encoder), measured against eval_questions.json.
Numbers, not vibes.

    docker run -p 6333:6333 qdrant/qdrant
    uv add pypdf sentence-transformers rank_bm25 qdrant-client numpy
    uv run python rag_eval.py docs/ eval_questions.json

Reuses load_pdf_pages/chunk_text from rag_chat.py so the eval corpus is
built exactly the same way the real chat tool builds its own index --
an eval that indexes differently from production isn't measuring
production.
"""

import json
import re
import sys
from collections import Counter
from dotenv import load_dotenv

load_dotenv()
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

from rag_chat import load_pdf_pages, chunk_text, EMBED_MODEL_NAME, QDRANT_URL

EVAL_COLLECTION = "eval_collection"     # separate from rag_chat.py's own collection, on purpose
HYBRID_POOL = 10   # candidates each retriever contributes to RRF fusion before narrowing to K
RRF_K = 60         # standard constant from the original RRF paper (Cormack et al., 2009)
CROSS_ENCODER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilingual -- matches German docs + English questions
K_VALUES = [1, 2, 5]  # @5 is what was asked for; @1/@2 are reported too -- see note in main()


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def build_corpus(folder):
    records = []  # one dict per chunk: {"text", "source", "page"} -- index in this list IS its id everywhere below
    for page in load_pdf_pages(folder):
        for chunk in chunk_text(page["text"]):
            records.append({"text": chunk, "source": page["source"], "page": page["page"]})
    return records


def build_dense_index(records, embed_model, client):
    dim = embed_model.get_sentence_embedding_dimension()
    try:
        client.delete_collection(EVAL_COLLECTION)
    except Exception:
        pass
    client.create_collection(EVAL_COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    vectors = embed_model.encode([r["text"] for r in records], normalize_embeddings=True)
    client.upsert(EVAL_COLLECTION, points=[
        PointStruct(id=i, vector=v.tolist(), payload=r) for i, (v, r) in enumerate(zip(vectors, records))
    ])


def dense_ranked_ids(question, embed_model, client, limit):
    q_vec = embed_model.encode([question], normalize_embeddings=True)[0]
    hits = client.query_points(EVAL_COLLECTION, query=q_vec.tolist(), limit=limit).points
    return [h.id for h in hits]


def build_sparse_index(records):
    return BM25Okapi([_tokenize(r["text"]) for r in records])


def sparse_ranked_ids(question, bm25, limit):
    scores = bm25.get_scores(_tokenize(question))
    return list(scores.argsort()[::-1][:limit])


def rrf_fuse(ranked_id_lists, k=RRF_K, top_n=HYBRID_POOL):
    """Reciprocal Rank Fusion: combine ranked lists by POSITION, not raw
    score -- a BM25 score and a cosine similarity live on incomparable
    scales, so summing them directly would be meaningless."""
    scores = Counter()
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in scores.most_common(top_n)]


def hit_at_k(ids, records, expected_source, k):
    sources = {records[i]["source"] for i in ids[:k]}
    return expected_source in sources


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"
    questions_path = sys.argv[2] if len(sys.argv) > 2 else "eval_questions.json"

    with open(questions_path, encoding="utf-8") as f:
        eval_set = json.load(f)

    from sentence_transformers import SentenceTransformer, CrossEncoder
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
    client = QdrantClient(url=QDRANT_URL)

    records = build_corpus(folder)
    build_dense_index(records, embed_model, client)
    bm25 = build_sparse_index(records)
    print(f"Indexed {len(records)} chunks from {len({r['source'] for r in records})} PDF(s). "
          f"Evaluating {len(eval_set)} questions...\n")

    # hits[method][k] = count of questions where the correct source landed in the top k
    hits = {m: {k: 0 for k in K_VALUES} for m in ("dense", "hybrid", "hybrid+rerank")}
    rows = []

    for item in eval_set:
        q, expected = item["question"], item["expected_source"]

        dense_ids = dense_ranked_ids(q, embed_model, client, limit=max(K_VALUES))
        sparse_ids = sparse_ranked_ids(q, bm25, limit=HYBRID_POOL)
        dense_wide = dense_ranked_ids(q, embed_model, client, limit=HYBRID_POOL)
        fused_ids = rrf_fuse([dense_wide, sparse_ids], top_n=HYBRID_POOL)

        pairs = [(q, records[i]["text"]) for i in fused_ids]
        ce_scores = cross_encoder.predict(pairs)
        reranked_ids = [doc_id for doc_id, _ in sorted(zip(fused_ids, ce_scores), key=lambda x: -x[1])]

        row = {"question": q, "expected": expected}
        for k in K_VALUES:
            hd = hit_at_k(dense_ids, records, expected, k)
            hh = hit_at_k(fused_ids, records, expected, k)
            hr = hit_at_k(reranked_ids, records, expected, k)
            hits["dense"][k] += hd
            hits["hybrid"][k] += hh
            hits["hybrid+rerank"][k] += hr
            row[k] = (hd, hh, hr)
        rows.append(row)

    n = len(eval_set)
    print(f"{'method':<16}" + "".join(f"hit@{k:<9}" for k in K_VALUES))
    for method in ("dense", "hybrid", "hybrid+rerank"):
        line = f"{method:<16}"
        for k in K_VALUES:
            h = hits[method][k]
            line += f"{h}/{n} ({h/n:.0%})".ljust(13)
        print(line)

    # At a small corpus (a handful of documents, single-digit chunk counts
    # per doc), hit@5 retrieves a large fraction of the ENTIRE corpus on
    # every query -- it can saturate near 100% for every method regardless
    # of quality, which is real but uninformative. hit@1 and hit@2 are
    # reported alongside it for exactly that reason: they're the metrics
    # actually capable of showing a difference between methods until the
    # corpus is big enough for @5 to mean something.
    if len(records) < 20:
        print(f"\nNote: only {len(records)} chunks in this corpus -- hit@5 retrieves most of it on every "
              "query and may be saturated for all three methods. hit@1/@2 above are more likely to "
              "actually discriminate between methods at this size; re-run against the full multi-page "
              "official PDFs for a hit@5 number with real signal in it.")

    print("\nper-question (dense | hybrid | hybrid+rerank) @5:")
    for row in rows:
        hd, hh, hr = row[5]
        marks = "".join("v" if x else "x" for x in (hd, hh, hr))
        print(f"  [{marks}] {row['question']}  (expected: {row['expected']})")


if __name__ == "__main__":
    main()