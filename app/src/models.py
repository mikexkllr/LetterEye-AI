# Define a Pydantic model for structured data extraction
from pydantic import BaseModel, Field


class LetterDetails(BaseModel):
    sender: str = Field(..., description="The sender of the letter")
    receiver: str = Field(..., description="The receiver of the letter. NAME ONLY (no other details)")
    organisation: str = Field(..., description="The organisation/ company/name of the sender - empty if not present or private person")
    date_of_writing: str = Field(..., description="The date the letter was written")
    type_of_letter: str = Field(..., description="The type of the letter. Maximal 5 words which describes a summary what the letter is about")
    responsible_person: str = Field(..., description="The person responsible for the letter. Name only (no other details)")
