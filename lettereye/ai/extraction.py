"""Extracting the facts needed for the file name (sender, date, subject) with LangChain structured output.

This is the only place where the local model *generates* text. Routing decisions are made by the
decision engine, never here.
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
    def __init__(self, base_url: str, model: str, *, language: str = "de", num_ctx: int = 8192,
                 keep_alive: str | int = "15m", llm: Any | None = None):
        self.language = language
        if llm is None:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(model=model, base_url=base_url, temperature=0, reasoning=False, num_ctx=num_ctx,
                             keep_alive=keep_alive, num_predict=400)
        self._chain = llm.with_structured_output(LetterFacts, method="json_schema")

    def extract(self, text: str) -> LetterFacts:
        language = LANGUAGE_NAMES_EN.get(self.language, "the letter's language")
        system = (
            "You extract facts from a scanned letter. Use only information that is present in the text; "
            "leave a field empty when it is not there. The text may contain OCR errors; fix obvious ones in names. "
            f"Write the subject in {language}."
        )
        result = self._chain.invoke([SystemMessage(system), HumanMessage(text[:12000])])
        if isinstance(result, dict):  # some LangChain versions return dicts
            result = LetterFacts.model_validate(result)
        return result
