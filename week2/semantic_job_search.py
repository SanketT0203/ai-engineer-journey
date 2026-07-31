"""
Day 8 practical — embeddings + cosine similarity, hands-on.

Turns your job postings into a tiny local semantic search engine. No API
calls, no tokens billed, no internet needed once the model is downloaded —
this whole thing runs on your CPU.

    uv add sentence-transformers
    uv run python semantic_job_search.py

First run downloads the model (~80MB, one-time, cached afterwards).
"""

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer, util


# ── Part 1: cosine similarity from scratch ──────────────────────────────
# Before letting a library hide this, do it by hand once so you actually
# know what "similarity score" means when a black box hands you 0.83.
#
# An embedding is just a list of floats — a point in (here) 384-dimensional
# space. Two texts with similar MEANING land near each other in that space,
# even if they don't share a single word. Cosine similarity measures the
# ANGLE between two vectors, not the distance between their tips — so two
# vectors pointing the same direction score 1.0 even if one is "longer"
# (e.g. a longer paragraph vs a short phrase about the same topic).

def cosine_similarity(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sanity_check_cosine_similarity():
    print("=== cosine similarity, hand-verified on toy vectors ===")
    print("identical vectors        ->", cosine_similarity([1, 2, 3], [1, 2, 3]))        # 1.0
    print("opposite vectors         ->", cosine_similarity([1, 2, 3], [-1, -2, -3]))      # -1.0
    print("orthogonal (unrelated)   ->", cosine_similarity([1, 0], [0, 1]))               # 0.0
    print("same direction, 2x mag.  ->", cosine_similarity([1, 2], [2, 4]))               # 1.0 (magnitude ignored!)
    print()


# ── Part 2: real sentence embeddings ─────────────────────────────────────
# all-MiniLM-L6-v2: 22M params, 384-dim output, ~80MB, runs fast on CPU.
# Not the most accurate model that exists, but the standard "good enough,
# free, instant" default — exactly what a €10-20/month budget wants you
# reaching for before you ever pay per-token for an API embedding call.
MODEL_NAME = "all-MiniLM-L6-v2"


def demo_semantic_vs_keyword(model):
    print("=== why this beats keyword matching ===")
    pairs = [
        ("AI engineer", "machine learning engineer"),   # near-synonyms, ZERO shared words
        ("AI engineer", "barista"),                      # unrelated
        ("remote friendly, visa sponsorship available", "no relocation needed, work from anywhere"),
    ]
    for a, b in pairs:
        emb_a, emb_b = model.encode([a, b], normalize_embeddings=True)
        score = cosine_similarity(emb_a, emb_b)
        print(f"  '{a}'  <->  '{b}'\n    cosine similarity = {score:.3f}\n")


# ── Part 3: semantic search over your real job postings ─────────────────

def load_postings():
    path = "jobs_extracted.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    # fallback so this still runs on a fresh checkout
    return [
        {"title": "AI Engineer", "company": "MockCorp", "skills": ["Python", "LangChain", "RAG"], "location": "Berlin"},
        {"title": "Senior ML Engineer", "company": "MockAI", "skills": ["PyTorch", "AWS", "MLOps"], "location": "Munich"},
        {"title": "Frontend Developer", "company": "MockWeb", "skills": ["React", "TypeScript"], "location": "Remote"},
        {"title": "Data Analyst", "company": "MockData", "skills": ["SQL", "Excel", "Tableau"], "location": "Hamburg"},
        {"title": "Agentic AI Engineer", "company": "MockGen", "skills": ["LLMs", "tool use", "Python"], "location": "Remote"},
    ]


def posting_to_text(job):  
    """One string per posting to embed. What you include here IS the search
    surface — leave out a field and the model can never match on it."""
    skills = ", ".join(job.get("skills", []) or [])
    return f"{job.get('title', '')}. Skills: {skills}. Location: {job.get('location', '')}."


def semantic_search(query, postings, corpus_embeddings, model, top_k=3):
    query_embedding = model.encode(query, normalize_embeddings=True)
    # util.cos_sim does exactly the cosine_similarity() math above, batched
    # and fast, across every posting at once instead of a python for-loop.
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    ranked = sorted(zip(postings, scores.tolist()), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    sanity_check_cosine_similarity()

    print("Loading embedding model (first run downloads it, then it's cached)...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Output dimension: {model.get_sentence_embedding_dimension()}\n")

    demo_semantic_vs_keyword(model)

    postings = load_postings()
    texts = [posting_to_text(j) for j in postings]
    # normalize_embeddings=True makes the vectors unit length, so cosine
    # similarity and plain dot product become the same computation — this
    # is what util.cos_sim relies on for speed.
    corpus_embeddings = model.encode(texts, normalize_embeddings=True)

    print(f"=== semantic search over {len(postings)} postings ===")
    queries = [
        "remote role building LLM agents",
        "no coding, just spreadsheets and dashboards",
    ]
    for q in queries:
        print(f"\nQuery: '{q}'")
        for job, score in semantic_search(q, postings, corpus_embeddings, model):
            print(f"  {score:.3f}  {job['title']} @ {job['company']} ({job.get('location')})")