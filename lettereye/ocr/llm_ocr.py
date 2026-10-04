"""OCR with a vision LLM on Ollama (GLM-OCR by default). Slower but much more robust than classic OCR
on bad scans, handwriting-like fonts, stamps and tables. Only used when the decision model is unsure.
"""

from __future__ import annotations

import base64
import io
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage
from PIL import Image

from . import OcrResult

_PROMPTS = {
    "glm-ocr": "Text Recognition:",
    "deepseek-ocr": "Free OCR.",
}
_GENERIC_PROMPT = "Transcribe all text in this document image exactly, line by line. Output only the text."


def prompt_for(model: str) -> str:
    base = model.split(":")[0].split("/")[-1].lower()
    return _PROMPTS.get(base, _GENERIC_PROMPT)


def _content_lines(text: str) -> list[str]:
    return [line.strip() for line in re.sub(r"```[a-zA-Z]*", "", text).splitlines() if line.strip()]


def repetition_start(lines: list[str], probe: int = 2) -> int | None:
    """Index where the transcription starts over (its first lines come back), or None."""
    if len(lines) < probe * 2 + 1:
        return None
    head = lines[:probe]
    for start in range(probe, len(lines) - probe + 1):
        if lines[start:start + probe] == head:
            return start
    return None


def clean_ocr_output(text: str) -> str:
    """Remove markdown fences and the repeated copies some OCR models append to their answer."""
    lines = _content_lines(text)
    cut = repetition_start(lines)
    return "\n".join(lines[:cut] if cut else lines).strip()


class LlmOCR:
    def __init__(self, base_url: str, model: str = "glm-ocr", *, keep_alive: str | int = "15m",
                 max_side: int = 1600, llm: Any | None = None):
        self.model = model
        self.max_side = max_side
        if llm is None:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(model=model, base_url=base_url, temperature=0, keep_alive=keep_alive,
                             num_ctx=16384, num_predict=3072, repeat_penalty=1.05)
        self._llm = llm

    def recognize(self, images: list[Image.Image]) -> OcrResult:
        start = time.perf_counter()
        pages = []
        for image in images:
            page = image.convert("RGB")
            page.thumbnail((self.max_side, self.max_side))
            buffer = io.BytesIO()
            page.save(buffer, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            message = HumanMessage(content=[
                {"type": "image_url", "image_url": data_url},
                {"type": "text", "text": prompt_for(self.model)},
            ])
            pages.append(self._read_page(message))
        return OcrResult(
            text="\n\n".join(p for p in pages if p).strip(),
            source="ocr_llm",
            seconds=time.perf_counter() - start,
            device="Ollama",
            pages=len(images),
            details={"model": self.model},
        )

    def _read_page(self, message: HumanMessage) -> str:
        """Stream the transcription and stop as soon as the model starts repeating the page.

        Small OCR models sometimes finish the page and then begin again (often inside a markdown
        block). Closing the stream makes Ollama stop generating, so no time is wasted on the copy.
        """
        text = ""
        checked_lines = 0
        for chunk in self._llm.stream([message]):
            text += str(chunk.content)
            if "\n" in str(chunk.content):
                lines = _content_lines(text)
                if len(lines) != checked_lines:
                    checked_lines = len(lines)
                    if repetition_start(lines) is not None:
                        break
        return clean_ocr_output(text)
