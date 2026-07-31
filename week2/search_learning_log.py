"""
Day 8 practical — semantic search over your own learning log.

Points at your real learninglog/ folder (learnings_07_08.md, learnings_07_14.md,
...) — reads each one, embeds it, and lets you search by MEANING instead of
exact keywords. Runs fully local (sentence-transformers), no API cost.

    uv add sentence-transformers
    uv run python search_learning_log.py

Works no matter which folder you run/place this script in relative to
learninglog/ — it searches its own folder, one level up, and two levels up
(covers AI-ENGINEER-JOURNEY/week1/this_script.py with
AI-ENGINEER-JOURNEY/learninglog/ as a sibling of week1/, which is your
actual layout) before giving up.
"""

import glob
import os
import re

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"
LOG_DIR = "learninglog"

# A relative path like "learninglog" is resolved against whatever directory
# you HAPPEN to run the command from (the "current working directory") — not
# against where this .py file lives. Run it from a different folder (or via
# an IDE "Run" button that uses a different cwd) and a plain relative path
# silently misses, even though the folder is right there. SCRIPT_DIR anchors
# the search to this file's own location instead, so it works regardless.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DATE_IN_NAME = re.compile(r"(\d{1,2})_(\d{1,2})")


def resolve_log_dir(name=LOG_DIR):
    """Search, in order: next to this script, one level up, two levels up,
    then the same three relative to the current working directory, then the
    name exactly as given (covers an absolute path you set manually).

    The "one/two levels up" cases matter for your actual layout — this
    script lives in week1/ or week2/, while learninglog/ is a SIBLING of
    those folders under AI-ENGINEER-JOURNEY/, i.e. one level above wherever
    this script sits, not next to it.

    Returns the first candidate that exists as a real folder; if none do,
    returns the script-relative guess so the error message has a concrete
    path to show you instead of just a folder name."""
    bases = [SCRIPT_DIR, os.getcwd()]
    candidates = []
    for base in bases:
        candidates.append(os.path.join(base, name))
        candidates.append(os.path.join(base, "..", name))
        candidates.append(os.path.join(base, "..", "..", name))
    candidates.append(name)

    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    return os.path.abspath(candidates[0])


def label_from_filename(filename):
    """learnings_07_08.txt -> 'Jul 08'. Falls back to the raw filename if it
    doesn't match the MM_DD pattern, so a differently-named file still works,
    it just won't get a pretty date label."""
    match = DATE_IN_NAME.search(filename)
    if match:
        month, day = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return f"{MONTHS[month]} {day:02d}"
    return os.path.splitext(filename)[0]


def load_notes(log_dir=None):
    """One note per file in learninglog/. Picks up .md, .txt, and — just in
    case some of yours have no extension at all — any file whose name
    starts with 'learnings_'. Whatever the file contains is embedded as-is;
    no title line or markdown structure is assumed or stripped out."""
    log_dir = log_dir or resolve_log_dir()
    paths = set(glob.glob(os.path.join(log_dir, "*.md")))
    paths |= set(glob.glob(os.path.join(log_dir, "*.txt")))
    paths |= {p for p in glob.glob(os.path.join(log_dir, "learnings_*")) if os.path.isfile(p)}

    notes = []
    for path in sorted(paths):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue  # skip empty files rather than embedding nothing
        filename = os.path.basename(path)
        notes.append({"path": path, "label": label_from_filename(filename), "text": text})
    return notes


def semantic_search(query, notes, embeddings, model, top_k=3):
    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    ranked = sorted(zip(notes, scores.tolist()), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def print_result(note, score, snippet_chars=160):
    snippet = note["text"].replace("\n", " ")
    snippet = snippet[:snippet_chars] + ("…" if len(snippet) > snippet_chars else "")
    print(f"  {score:.3f}  [{note['label']}] ({os.path.basename(note['path'])})")
    print(f"          \"{snippet}\"")


if __name__ == "__main__":
    resolved_dir = resolve_log_dir()
    notes = load_notes(resolved_dir)
    if not notes:
        if os.path.isdir(resolved_dir):
            existing = os.listdir(resolved_dir)
            detail = (f"Folder found at {resolved_dir}, but nothing matched *.md/*.txt/learnings_*. "
                      f"Files actually in there: {existing}")
        else:
            detail = f"No folder found at {resolved_dir} (or any of the other locations this script checked)."
        raise SystemExit(f"No notes found. {detail}")

    print(f"Loaded {len(notes)} notes from {resolved_dir}: "
          f"{', '.join(os.path.basename(n['path']) for n in notes)}\n")

    print("Loading embedding model (first run downloads it, then it's cached)...")
    model = SentenceTransformer(MODEL_NAME)
    texts = [n["text"] for n in notes]
    embeddings = model.encode(texts, normalize_embeddings=True)
    print(f"Embedded {len(embeddings)} notes into {embeddings.shape[1]}-dim vectors.\n")

    # Try your own queries here — this is just a starting example.
    queries = ["how do I count tokens"]
    for q in queries:
        print(f"Query: \"{q}\"")
        for note, score in semantic_search(q, notes, embeddings, model, top_k=3):
            print_result(note, score)
        print()

    # Interactive: keep querying your real log until you hit Enter on empty input.
    while True:
        q = input("Search your log (Enter to quit): ").strip()
        if not q:
            break
        for note, score in semantic_search(q, notes, embeddings, model, top_k=3):
            print_result(note, score)
        print()