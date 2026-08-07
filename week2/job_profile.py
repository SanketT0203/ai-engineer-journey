"""
YOUR profile — edit this to actually describe you. JobRadar scores every
posting against exactly what's in this file, so a vague or stale profile
means a useless digest no matter how good the rest of the pipeline is.

PROFILE_SKILLS drives the exact-match scoring (skills_overlap_score in
jobradar.py) — keep it to real, concrete skills/tools, matching how job
postings actually phrase them (e.g. "LangChain" not "agent frameworks").

PROFILE_TEXT drives the embedding-similarity scoring — a few honest
sentences about what you want and what you bring, written the way you'd
describe yourself, not a keyword dump (that's what PROFILE_SKILLS is for).
"""

PROFILE_SKILLS = [
    "Python", "LLM", "LangChain", "RAG", "Pydantic", "prompt engineering",
    "tool use", "agentic AI", "API integration", "REST API", "Git",
    "data science", "embeddings", "vector search",
]

PROFILE_TEXT = (
    "Master's student in Germany with a web development background and "
    "strong Python skills, transitioning into AI engineering and agentic "
    "AI. Comfortable building LLM-powered pipelines: structured extraction, "
    "RAG, tool-use agents, and production concerns like retries, cost "
    "tracking, and prompt injection defenses. Currently enrolled full-time, "
    "so eligible right now for working student (Werkstudent) roles and "
    "internships (Praktikum), and also open to full-time AI Engineer or "
    "Agentic AI Engineer roles that would start after graduation. Looking "
    "for opportunities in Germany, ideally with visa sponsorship support "
    "and some flexibility on German language requirements while still "
    "learning the language."
)

# jobradar.py no longer enforces the working-student/internship constraint
# as a hard filter before scoring -- both full-time and working-student/
# internship postings are fetched, extracted, and shown, tagged via
# is_student_eligible() in jobradar.py (STUDENT_ELIGIBLE_KEYWORDS) so the
# digest can badge them distinctly rather than picking one category for
# you. This text drives the embedding-similarity half of the score for
# EITHER category now, which is why it mentions both being wanted --
# skewing it toward student-only language would have quietly suppressed
# the semantic-similarity score for full-time postings even though nothing
# was filtering them out anymore.