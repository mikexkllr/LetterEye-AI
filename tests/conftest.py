from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from lettereye.ai.decision import DecisionEngine
from lettereye.ai.extraction import LetterFacts
from lettereye.db import Database
from lettereye.ocr import OcrResult

LEGACY_CSV = Path(__file__).resolve().parent.parent / "examples" / "legacy_csv"


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LETTEREYE_DATA_DIR", str(tmp_path / "appdata"))
    return tmp_path / "appdata"


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


def logprob_response(probabilities: dict[str, float], prefix_tokens: tuple[str, ...] = ()) -> AIMessage:
    """An AIMessage shaped like ChatOllama's reply with logprobs=True."""
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    best = ranked[0][0]
    top = [{"token": label, "logprob": math.log(max(p, 1e-12))} for label, p in ranked]
    positions = [{"token": t, "logprob": -0.01, "top_logprobs": [{"token": t, "logprob": -0.01}]} for t in prefix_tokens]
    positions.append({"token": best, "logprob": math.log(max(ranked[0][1], 1e-12)), "top_logprobs": top})
    return AIMessage(content=best, response_metadata={"logprobs": positions})


class ScriptedLLM:
    """Fake chat model. `scorer(state, question, option) -> weight` decides the answer distribution."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.calls: list[tuple[str, list[str]]] = []

    def invoke(self, messages):
        prompt = messages[-1].content
        state = prompt.split("<<<", 1)[1].split(">>>", 1)[0]
        question = re.search(r"QUESTION: (.*)", prompt).group(1)
        options = re.findall(r"^([A-T])\) (.*)$", prompt.split("OPTIONS:", 1)[1], flags=re.MULTILINE)
        self.calls.append((question, [o for _, o in options]))
        weights = {label: max(self.scorer(state, question, option), 1e-6) for label, option in options}
        total = sum(weights.values())
        return logprob_response({label: w / total for label, w in weights.items()})


def letter_scorer(state: str, question: str, option: str) -> float:
    """Behaves like a sensible decision model on clean text, and is lost on garbage."""
    readable = "Sehr geehrte" in state or "Dear" in state
    if question.startswith("Is this letter addressed to"):
        names = question[len("Is this letter addressed to "):].rstrip("?").split(" / ")
        found = readable and any(n in state for n in names)
        return (0.97 if found else 0.03) if option == "Yes" else (0.03 if found else 0.97)
    if question.startswith("Who is the addressee"):
        if not readable:
            return 1.0  # uniform: no idea
        name = option.split(" (also")[0]
        aliases = re.findall(r"also written as: (.*)\)", option)
        candidates = [name] + (aliases[0].split(", ") if aliases else [])
        if any(c in state for c in candidates):
            return 50.0
        if option.startswith("Someone else"):
            return 8.0 if not any(n in state for n in ("Bob Smith", "Eve Adams")) else 0.1
        return 0.2
    if question.startswith("Which colleague"):
        if "Finanzamt" in state and "Finanzamt" in option:
            return 30.0
        return 2.0 if option.startswith("None") else 0.5
    if question.startswith("Which category"):
        if "Rechnung" in option and "Abrechnung" in state:
            return 20.0
        return 1.0
    return 1.0


@pytest.fixture
def decision_engine():
    llm = ScriptedLLM(letter_scorer)
    engine = DecisionEngine(llm=llm, debias=True)
    engine.llm = llm
    return engine


class FakeOCR:
    def __init__(self, text: str, confidence: float = 0.98, source: str = "fast_ocr", model: str = "fake-ocr"):
        self.text, self.confidence, self.source, self.model = text, confidence, source, model
        self.calls = 0

    def recognize(self, images) -> OcrResult:
        self.calls += 1
        return OcrResult(text=self.text, source=self.source, confidence=self.confidence, device="cpu", pages=len(images))


class FakeExtractor:
    def __init__(self, facts: LetterFacts | None = None, fail: bool = False):
        self.facts = facts or LetterFacts()
        self.fail = fail
        self.texts: list[str] = []

    def extract(self, text: str) -> LetterFacts:
        self.texts.append(text)
        if self.fail:
            raise RuntimeError("model unavailable")
        return self.facts


CLEAN_LETTER = """Stadtwerke München GmbH
Herrn
Bob Smith
Hauptstraße 12, 10115 Berlin
München, 14.03.2026
Betreff: Jahresabrechnung Strom 2025
Sehr geehrter Herr Smith,
anbei erhalten Sie Ihre Jahresabrechnung. Mit freundlichen Grüßen"""

GARBLED = "St@dtw3rke Mn ch3n G bH | H rrn | B b Sm th | H ptstr 12 | ..ahr sabr chn ng | xx yy zz qq ww"


@pytest.fixture
def scan_image(tmp_path) -> Path:
    path = tmp_path / "scan.png"
    Image.new("RGB", (400, 560), "white").save(path)
    return path
