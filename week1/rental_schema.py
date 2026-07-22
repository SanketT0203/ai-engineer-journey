"""
The Pydantic schema for rental-listing extraction.

Compare this against v4/v5's prompt text from Friday — notice the domain rules
("keine Kaution means 0", "gesetzliche Kündigungsfrist means 3") haven't gone
away. They've just moved INTO the schema as Field descriptions, which Instructor
includes in the tool definition it sends to the model. Same knowledge, different
home: prose paragraph vs. structured metadata attached to each field.
"""

from typing import Optional
from pydantic import BaseModel, Field


class RentalExtraction(BaseModel):
    kaltmiete: Optional[float] = Field(
        None, description="Cold rent (Kaltmiete/KM) in EUR, excluding utilities. null if not explicitly stated."
    )
    nebenkosten: Optional[float] = Field(
        None, description="Utility costs (Nebenkosten/NK) in EUR. null if not stated."
    )
    warmmiete: Optional[float] = Field(
        None,
        description=(
            "Total rent (Warmmiete/WM) in EUR. Use the stated value if given directly; "
            "otherwise compute kaltmiete + nebenkosten if both are known; otherwise null."
        ),
    )
    kaution: float = Field(
        ...,
        ge=0,
        description=(
            'Deposit in EUR. If given as a multiple of rent (e.g. "3 Nettokaltmieten", '
            '"2 Monatsmieten"), compute it by multiplying that number by the relevant rent. '
            'If the listing says "keine Kaution", use 0.'
        ),
    )
    kuendigungsfrist_monate: Optional[int] = Field(
        None,
        description=(
            'Notice period in months. "gesetzliche Kündigungsfrist" means the statutory '
            "default of 3 months (BGB §573c). null only if truly unmentioned."
        ),
    )