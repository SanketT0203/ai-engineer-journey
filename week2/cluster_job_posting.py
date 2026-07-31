"""
Day 8 extension — embed job postings, compute pairwise similarity, cluster
with scikit-learn, and auto-name each cluster. No LLM call anywhere in this
pipeline: embedding is local (sentence-transformers), clustering is local
(scikit-learn), and naming is a deterministic word-count heuristic, not an
API call. That "don't spend tokens on something math can already do" instinct
is exactly the mindset your token-management capstone is about.

    uv add sentence-transformers scikit-learn
    uv run python cluster_job_postings.py
"""

import json
import os
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"


def load_postings():
    path = "jobs_extracted.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data:
            return data
    # Fallback mock set — deliberately spans 4 obvious themes (agentic/LLM,
    # traditional ML/MLOps, data/BI, unrelated frontend role) so clustering
    # has something real to find even with jobs_extracted.json missing.
    return [
        {"title": "AI Engineer", "company": "MockGen", "skills": ["Python", "LangChain", "RAG"], "location": "Berlin"},
        {"title": "Agentic AI Engineer", "company": "MockAgent", "skills": ["LLMs", "tool use", "Python"], "location": "Remote"},
        {"title": "GenAI Developer", "company": "MockLLM", "skills": ["OpenAI API", "prompt engineering", "Python"], "location": "Munich"},
        {"title": "Senior ML Engineer", "company": "MockAI", "skills": ["PyTorch", "AWS", "MLOps"], "location": "Munich"},
        {"title": "ML Platform Engineer", "company": "MockPlatform", "skills": ["Kubernetes", "MLOps", "PyTorch"], "location": "Berlin"},
        {"title": "Machine Learning Engineer", "company": "MockScale", "skills": ["TensorFlow", "AWS", "MLOps"], "location": "Hamburg"},
        {"title": "Data Analyst", "company": "MockData", "skills": ["SQL", "Excel", "Tableau"], "location": "Hamburg"},
        {"title": "Business Intelligence Analyst", "company": "MockBI", "skills": ["SQL", "Power BI", "Tableau"], "location": "Frankfurt"},
        {"title": "Frontend Developer", "company": "MockWeb", "skills": ["React", "TypeScript"], "location": "Remote"},
    ]


def posting_to_text(job):
    skills = ", ".join(job.get("skills", []) or [])
    return f"{job.get('title', '')}. Skills: {skills}. Location: {job.get('location', '')}."


def pick_k(embeddings, k_min=2, k_max=None):
    """Data-driven cluster count via silhouette score — same idea as
    elbow/silhouette analysis from any intro ML course, applied to
    embeddings instead of raw tabular features. Treat this as a starting
    point, not gospel: silhouette rewards well-separated clusters, so on a
    small/ambiguous dataset it can merge two related-but-distinct themes
    into one if they aren't far enough apart in embedding space. Always
    eyeball the actual clusters it hands back."""
    n = len(embeddings)
    k_max = k_max or min(6, n - 1)
    if n <= k_min:
        return 1
    best_k, best_score = k_min, -1
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"  k={k}: silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def name_cluster(postings_in_cluster, top_n=3):
    """No LLM call — deterministic and free. Name = the most common skills
    shared across postings in this cluster. Falls back to common title
    words if a cluster's postings have no skills listed at all."""
    skill_counts = Counter()
    for job in postings_in_cluster:
        for s in job.get("skills", []) or []:
            skill_counts[s] += 1

    if skill_counts:
        top = [s for s, _ in skill_counts.most_common(top_n)]
        return " / ".join(top)

    title_words = Counter()
    for job in postings_in_cluster:
        for w in job.get("title", "").split():
            if len(w) > 3:
                title_words[w] += 1
    top = [w for w, _ in title_words.most_common(top_n)]
    return " / ".join(top) if top else "Unlabeled cluster"


if __name__ == "__main__":
    postings = load_postings()
    texts = [posting_to_text(j) for j in postings]
    print(f"Loaded {len(postings)} postings.\n")

    print("Loading embedding model (first run downloads it, then it's cached)...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True)
    print(f"Embedded {len(embeddings)} postings into {embeddings.shape[1]}-dim vectors.\n")

    # ── pairwise similarity matrix (sklearn's version of the same cosine
    # math from semantic_job_search.py — same formula, different library) ──
    sim_matrix = cosine_similarity(embeddings)
    n_preview = min(5, len(postings))
    print(f"=== pairwise cosine similarity (first {n_preview} x {n_preview}) ===")
    print("        " + "".join(f"{i:>7}" for i in range(n_preview)))
    for i in range(n_preview):
        row = "".join(f"{sim_matrix[i][j]:7.3f}" for j in range(n_preview))
        print(f"  [{i}]  {row}")
    print()

    # ── clustering ─────────────────────────────────────────────────────
    print("=== choosing k via silhouette score ===")
    k = pick_k(embeddings)
    print(f"-> chosen k = {k}\n")

    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embeddings)

    clusters = {i: [] for i in range(k)}
    for job, label in zip(postings, labels):
        clusters[int(label)].append(job)

    print("=== clusters ===")
    for cluster_id, jobs in clusters.items():
        name = name_cluster(jobs)
        print(f"\nCluster {cluster_id}: \"{name}\"  ({len(jobs)} postings)")
        for job in jobs:
            print(f"    - {job['title']} @ {job['company']} ({job.get('location')})")