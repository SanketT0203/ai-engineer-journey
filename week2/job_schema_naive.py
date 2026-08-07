from typing import Literal, Optional
from pydantic import BaseModel, Field
 
Seniority = Literal["intern", "junior", "mid", "senior", "lead", "not specified"]
 
 
class NaiveJobPosting(BaseModel):
    title: str
    company: str
    location: Optional[str] = None
    seniority: Seniority
    skills: list[str] = Field(default_factory=list)
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    german_required: Optional[bool] = None
    visa_sponsorship: Optional[bool] = None
    url: Optional[str] = None
 