"""Extracting the facts needed for the file name (sender, date, subject) with LangChain structured output.

This is the only place where a model *generates* text. Routing decisions are made by the decision engine,
never here. The structured-output runnable comes from ai/factory.py, for whichever provider is configured.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..settings import LANGUAGE_NAMES_EN


class LetterFacts(BaseModel):
    sender_organization: str = Field("", description="Company, authority or institution that sent the letter. Empty if a private person sent it.")
    sender_person: str = Field("", description="Name of the person who signed or sent the letter. Empty if not stated.")
    recipient_name: str = Field("", description="Name of the addressee (person or company the letter is written to). Name only, no address.")
    letter_date: str = Field("", description="Date the letter was written, exactly as printed. Empty if there is none.")
    subject: str = Field("", description="Very short subject of the letter, at most 6 words.")

    @property
    def sender(self) -> str:
        return self.sender_organization or self.sender_person


class FactExtractor:
    def __init__(self, chain: Any, *, language: str = "de"):
        """`chain`: a runnable that turns messages into LetterFacts (a chat model with structured output)."""
        self.language = language
        self._chain = chain

    def extract(self, text: str) -> LetterFacts:
        language = LANGUAGE_NAMES_EN.get(self.language, "the letter's language")
        system = (
            "You extract facts from a scanned letter. Use only information that is present in the text; "
            "leave a field empty when it is not there. The text may contain OCR errors; fix obvious ones in names. "
            f"Write the subject in {language}."
        )
        result = self._chain.invoke([SystemMessage(system), HumanMessage(text[:12000])])
        if result is None:  # the model did not use the structured output
            return LetterFacts()
        if isinstance(result, dict):  # some LangChain versions and providers return dicts
            result = LetterFacts.model_validate({k: v or "" for k, v in result.items()})
        return result
