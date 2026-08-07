"""
Pydantic schema for AI-engineer job postings extracted from free-text descriptions.

Notice german_required and visa_sponsorship are Optional[bool], not bool.
Most listings never explicitly say either way — forcing True/False would mean
guessing. This is the same "null vs. explicit value" distinction from the
rental-listing exercise: absence of information is a real, distinct state,
not the same as "False".
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

Seniority = Literal["working student", "intern", "junior", "mid", "senior", "lead", "not specified"]


class JobPosting(BaseModel):
    title: str
    company: str
    location: Optional[str] = None

    seniority: Seniority = Field(
        description="Infer from title/description wording (e.g. 'Senior', 'Junior', years of experience required). Use 'not specified' if genuinely unclear."
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Technical skills, tools, and frameworks explicitly mentioned (e.g. Python, LangChain, AWS, PyTorch). Empty list if none stated.",
    )

    salary_min: Optional[float] = Field(None, description="Minimum stated salary in EUR/year, null if not stated.")
    salary_max: Optional[float] = Field(None, description="Maximum stated salary in EUR/year, null if not stated.")

    german_required: Optional[bool] = Field(
        None,
        description="true if German language is explicitly required, false if explicitly stated as NOT required or the role is explicitly English-only, null if language requirements are never mentioned at all.",
    )
    visa_sponsorship: Optional[bool] = Field(
        None,
        description="true if visa sponsorship / relocation support is explicitly offered, false if explicitly stated as unavailable, null if never mentioned.",
    )

    url: Optional[str] = None  # carried through from the raw listing, not LLM-extracted