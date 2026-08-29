"""
streamlit_app.py -- Streamlit UI for Beamter-Bot, deployed on Streamlit
Community Cloud.

Same relationship to rag_chat.py as the Gradio version this replaces: every
real RAG function -- condense_question, retrieve, generate_answer,
format_context, count_context_tokens, log_context_tokens, build_index -- is
IMPORTED unchanged. The only new code is Streamlit's own chat primitives
and the response logic that adapts between them. Same pipeline as
everywhere else in this project, just a different UI framework -- Hugging
Face stopped offering free CPU hosting for Gradio apps to free accounts
partway through this task, so this replaces space_deploy/ as the actual
deploy path. See BEAMTER_BOT.md for the full story.

Why @st.cache_resource matters here: Streamlit reruns this ENTIRE script,
top to bottom, on every single interaction -- every new chat message.
Without caching, that would mean reloading both ML models and rebuilding
the whole Qdrant index on every question. @st.cache_resource makes
Streamlit run load_pipeline() exactly once (shared across every visitor
hitting this deployed app, not per-session) and hand back the same objects
on every rerun after that -- the same thing module-level loading did in
app.py for Gradio, just expressed the way Streamlit expects it.

    uv run streamlit run streamlit_app.py

Needs ANTHROPIC_API_KEY -- locally, in the environment as usual; on
Community Cloud, as a ROOT-LEVEL key in the app's secrets (see DEPLOY.md
for exactly why root-level matters).
"""

import os

import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_chat import (
    CROSS_ENCODER_NAME,
    EMBED_MODEL_NAME,
    build_index,
    condense_question,
    count_context_tokens,
    format_context,
    generate_answer,
    log_context_tokens,
    retrieve,
)

DOCS_FOLDER = "docs"
QDRANT_PATH = os.environ.get("QDRANT_PATH", "qdrant_data")  # embedded, on-disk for this run only -- no Docker either way
FOOTER_SEP = "\n\n---\n"

# Streamlit's secrets.toml auto-exposes ROOT-LEVEL keys as os.environ too --
# see DEPLOY.md. This line is a defensive bridge in case a key ends up
# nested under a [section] instead (which does NOT auto-bridge), or when
# running locally before st.secrets has anything in it. Same category of
# gotcha as the earlier ".env doesn't auto-load into os.environ" lesson --
# a secrets store existing isn't the same as a specific library actually
# being able to read it.
#
# Wrapped in try/except on purpose: st.secrets raises
# StreamlitSecretNotFoundError on ANY access -- even a plain `in` check --
# when no secrets have been configured for this app at all yet, not just
# an empty dict the way a normal Python dict would behave. Without this
# guard, that exception crashes the app before main() ever runs, instead
# of falling through to the friendlier "Missing API key" message main()
# shows further down.
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass  # no secrets configured yet -- main()'s own check below handles this cleanly


@st.cache_resource(show_spinner="Loading models and indexing documents (first load only)...")
def load_pipeline():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
    client = QdrantClient(path=QDRANT_PATH)
    n_chunks = build_index(DOCS_FOLDER, embed_model, client)
    return embed_model, cross_encoder, client, n_chunks


def strip_footer(text):
    """Undo the citation/token footer before feeding a past answer back
    into condense_question() -- it's UI decoration, not something that
    should shape how a follow-up gets condensed."""
    return text.split(FOOTER_SEP)[0]


# ── the actual response logic, kept as a PURE function ────────────────────
# No st.* calls in here on purpose: this is directly unit-testable without
# a running Streamlit server, the same way app.py's respond() was for the
# Gradio version. `history_messages` is st.session_state.messages as it
# stood BEFORE this turn's user message was appended -- completed
# {"role", "content"} pairs, same shape the Gradio version consumed.
def respond(raw_question, history_messages, embed_model, cross_encoder, qdrant_client):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ("**Missing API key.** This app needs `ANTHROPIC_API_KEY` set as a "
                "*root-level* key in this app's secrets. See DEPLOY.md."), raw_question

    turns = [
        {"question": history_messages[i]["content"], "answer": strip_footer(history_messages[i + 1]["content"])}
        for i in range(0, len(history_messages) - 1, 2)
    ]
    question = condense_question(turns, raw_question)
    hits = retrieve(question, embed_model, cross_encoder, qdrant_client)

    if not hits:
        return ("I couldn't find anything relevant to that in the Anmeldung, Blue Card, "
                 "or Rundfunkbeitrag documents."), question

    context = format_context(hits)
    n_tokens, exact = count_context_tokens(context)
    log_context_tokens(question, hits, context, n_tokens, exact)  # non-fatal on failure, see log_context_tokens

    answer = generate_answer(question, hits)
    sources = ", ".join(sorted({f"{r['source']} p.{r['page']}" for r, _ in hits}))
    tok_note = f"~{n_tokens} tokens" if exact else f"~{n_tokens} tokens (estimated)"
    return f"{answer}{FOOTER_SEP}*Sources: {sources} &middot; {tok_note} retrieved*", question


# ── Streamlit UI wiring -- thin, calls respond() for everything real ─────
def main():
    st.set_page_config(page_title="Beamter-Bot", page_icon="🏛️")
    st.title("Beamter-Bot 🏛️")
    st.caption(
        "Ask about Anmeldung, the Blue Card EU, or the Rundfunkbeitrag -- grounded in real "
        "official German documents, with a citation on every claim. Follow-ups work "
        "naturally (\"what about for graduates?\", \"kannst du das auf Englisch erklären?\") "
        "-- it remembers the conversation."
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error(
            "**Missing API key.** Set `ANTHROPIC_API_KEY` as a root-level key in this app's "
            "secrets (Settings -> Secrets on Community Cloud, or a local `.streamlit/"
            "secrets.toml` / environment variable when running locally). See DEPLOY.md."
        )
        st.stop()

    embed_model, cross_encoder, qdrant_client, n_chunks = load_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown(
            f"*Indexed {n_chunks} chunks. Try asking, for example:*\n"
            "- Wie viele Tage Zeit habe ich für die Anmeldung nach dem Einzug?\n"
            "- What documents do I need to bring to register my address?\n"
            "- Wie hoch ist das allgemeine Mindestgehalt für die Blaue Karte EU?\n"
            "- Wie viel kostet der Rundfunkbeitrag pro Monat?"
        )

    # Redraw the full conversation on every rerun -- Streamlit has no
    # persistent DOM, the whole script re-executes from scratch each time.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    raw_question = st.chat_input("Ask about Anmeldung, the Blue Card, or the Rundfunkbeitrag...")
    if raw_question:
        history_before = list(st.session_state.messages)  # snapshot -- BEFORE this turn's message is appended
        st.session_state.messages.append({"role": "user", "content": raw_question})
        with st.chat_message("user"):
            st.markdown(raw_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer_display, condensed_question = respond(
                    raw_question, history_before, embed_model, cross_encoder, qdrant_client,
                )
            if condensed_question != raw_question:
                st.caption(f"Understood as: {condensed_question}")
            st.markdown(answer_display)

        st.session_state.messages.append({"role": "assistant", "content": answer_display})


if __name__ == "__main__":
    main()