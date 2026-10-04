"""The document pipeline.

    read text ──► decide ──► (unsure?) ──► OCR LLM ──► decide again ──► extract facts ──► filed | review

1. Text: the PDF's own text layer if it has one, otherwise fast OCR (PP-OCRv6 on the GPU).
2. Decisions (local decision model, typed answers with probabilities):
     • Who is the letter addressed to?  (your recipients + "someone else")  → worker via your data
     • Is it really addressed to <winner>?  (verification)
     • If the recipient is unknown: which worker is responsible, by their description?
     • Which document type is it?
3. Only if that is not confident enough is the slow OCR LLM (GLM-OCR) run, and the decisions repeated.
4. A generative pass extracts sender, date and subject for the file name.
Anything still below the confidence threshold goes to the review queue in the UI.

The processor does not move files or touch the database; the engine does that.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ..ai.decision import ChoiceResult, DecisionEngine, DecisionError
from ..ai.extraction import FactExtractor, LetterFacts
from ..db import Recipient, Worker
from ..ocr import OcrResult
from ..ocr.documents import LoadedDocument
from ..settings import Settings
from .dates import find_letter_date, normalize_date
from .filing import FilingValues, build_relative_path
from .names import clean_person_name

log = logging.getLogger(__name__)

NOT_IN_LIST = "Someone else (not in this list)"
NOBODY = "None of them / unclear"
MIN_USEFUL_CHARS = 40

Q_RECIPIENT = "Who is the addressee (recipient) of this letter?"
I_RECIPIENT = ("The addressee is the person or company the letter is written TO (usually in the address block "
               "or the salutation), not the sender, not a contact person at the sender, and not someone only "
               "mentioned in the text. Names may contain small OCR errors.")
Q_WORKER = "Which colleague is responsible for this letter, based on their area of responsibility?"
Q_TYPE = "Which category best describes this document?"


@dataclass
class Directory:
    """Snapshot of the workers and recipients the decisions are made against."""
    workers: list[Worker]
    recipients: list[Recipient]

    def worker(self, worker_id: int | None) -> Worker | None:
        return next((w for w in self.workers if w.id == worker_id), None)


@dataclass
class DecisionSet:
    recipient: Recipient | None = None
    worker: Worker | None = None
    recipient_confidence: float = 0.0
    worker_confidence: float = 0.0
    doc_type: str = ""
    doc_type_confidence: float = 0.0
    decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return self.recipient_confidence if self.recipient else self.worker_confidence


@dataclass
class Analysis:
    ocr: OcrResult
    decision: DecisionSet
    facts: LetterFacts
    letter_date: str
    status: str  # filed | review
    reason: str = ""
    escalated: bool = False
    new_recipient_name: str = ""
    stages: list[dict[str, Any]] = field(default_factory=list)
    preview: Image.Image | None = None

    @property
    def confidence(self) -> float:
        return self.decision.confidence

    def trace(self) -> dict[str, Any]:
        return {
            "stages": self.stages,
            "decisions": self.decision.decisions,
            "escalated": self.escalated,
            "facts": self.facts.model_dump(),
            "reason": self.reason,
        }


def recipient_label(recipient: Recipient) -> str:
    if recipient.aliases:
        return f"{recipient.name} (also written as: {', '.join(recipient.aliases[:4])})"
    return recipient.name


class Processor:
    def __init__(
        self,
        settings: Settings,
        decision: DecisionEngine,
        extractor: FactExtractor,
        fast_ocr: Any,
        llm_ocr: Any | None,
        directory: Callable[[], Directory],
    ):
        self.settings = settings
        self.decision = decision
        self.extractor = extractor
        self.fast_ocr = fast_ocr
        self.llm_ocr = llm_ocr
        self.directory = directory

    # ------------------------------------------------------------------ main entry
    def analyze(self, path: str | Path, on_stage: Callable[[str], None] | None = None) -> Analysis:
        notify = on_stage or (lambda _msg: None)
        s = self.settings
        stages: list[dict[str, Any]] = []
        directory = self.directory()
        document = LoadedDocument(path)
        images: list[Image.Image] | None = None

        def page_images() -> list[Image.Image]:
            nonlocal images
            if images is None:
                started = time.perf_counter()
                images = document.render(s.max_pages, s.render_dpi)
                stages.append({"name": "render", "seconds": round(time.perf_counter() - started, 3), "pages": len(images)})
            return images

        # 1) first text: text layer, else fast OCR (or the OCR LLM if configured to always use it)
        notify("Reading text layer")
        started = time.perf_counter()
        text_layer = document.text_layer(s.max_pages)
        stages.append({"name": "text_layer", "seconds": round(time.perf_counter() - started, 3), "chars": len(text_layer)})
        if len(text_layer) >= s.min_text_layer_chars:
            ocr = OcrResult(text=text_layer, source="text_layer", confidence=1.0, pages=min(s.max_pages, document.page_count()))
        elif s.ocr_mode == "llm_always" and self.llm_ocr is not None:
            ocr = self._run_llm_ocr(page_images(), stages, notify)
        else:
            notify("Fast OCR (PP-OCRv6)")
            ocr = self.fast_ocr.recognize(page_images())
            stages.append({"name": "fast_ocr", "seconds": round(ocr.seconds, 3), "device": ocr.device,
                           "confidence": round(ocr.confidence or 0, 4), "chars": len(ocr.text)})

        can_escalate = s.ocr_mode == "auto" and self.llm_ocr is not None and ocr.source != "ocr_llm"

        # 2) decide; 3) escalate to the OCR LLM only if the decisions are not good enough
        decision = None
        if len(ocr.text) >= MIN_USEFUL_CHARS or not can_escalate:
            decision = self._decide(ocr.text, directory, stages, notify)
        escalated = False
        if can_escalate and self._needs_escalation(decision, ocr):
            escalated = True
            llm = self._run_llm_ocr(page_images(), stages, notify)
            if len(llm.text) >= MIN_USEFUL_CHARS or decision is None:
                second = self._decide(llm.text, directory, stages, notify, round_name="decisions_after_ocr_llm")
                earlier = decision.decisions if decision else []
                if decision is None or self._is_confident(second) or second.confidence >= decision.confidence:
                    decision, ocr = second, llm
                    decision.decisions = earlier + decision.decisions  # keep both rounds for the UI
                else:
                    decision.decisions = earlier + second.decisions
        assert decision is not None

        # 4) facts for the file name
        notify("Extracting sender, date and subject")
        started = time.perf_counter()
        try:
            facts = self.extractor.extract(ocr.text) if ocr.text.strip() else LetterFacts()
        except Exception as exc:  # naming details are nice to have; never fail a letter because of them
            log.warning("Fact extraction failed: %s", exc)
            facts = LetterFacts()
        facts.recipient_name = clean_person_name(facts.recipient_name)
        stages.append({"name": "extract", "seconds": round(time.perf_counter() - started, 3)})
        letter_date = normalize_date(facts.letter_date, s.language) or find_letter_date(ocr.text, s.language)

        status, reason, new_name = self._outcome(decision, facts, directory)
        preview = (images or document.render(1, 110))[0] if (images or document.page_count()) else None
        return Analysis(ocr=ocr, decision=decision, facts=facts, letter_date=letter_date, status=status,
                        reason=reason, escalated=escalated, new_recipient_name=new_name, stages=stages, preview=preview)

    # ------------------------------------------------------------------ steps
    def _run_llm_ocr(self, images: list[Image.Image], stages: list[dict[str, Any]], notify) -> OcrResult:
        notify(f"Reading with OCR LLM ({self.llm_ocr.model})")
        result = self.llm_ocr.recognize(images)
        stages.append({"name": "ocr_llm", "seconds": round(result.seconds, 3), "model": self.llm_ocr.model,
                       "chars": len(result.text)})
        return result

    def _needs_escalation(self, decision: DecisionSet | None, ocr: OcrResult) -> bool:
        """Run the OCR LLM only when the decision model is not enough on the cheap text."""
        if decision is None:
            return True
        if self._is_confident(decision):
            return False
        # Sure that the recipient is someone new, and the text is clean: better OCR would not change that.
        first = next((d for d in decision.decisions if d.get("key") == "recipient"), None)
        if (first and first["choice"] == NOT_IN_LIST and first["confidence"] >= self.settings.confidence_threshold
                and (ocr.confidence or 0) >= 0.9):
            return False
        return True

    def _is_confident(self, decision: DecisionSet) -> bool:
        threshold = self.settings.confidence_threshold
        if decision.recipient is not None:
            return decision.recipient_confidence >= threshold
        return decision.worker is not None and decision.worker_confidence >= threshold and self.settings.auto_create_recipients

    def _decide(self, text: str, directory: Directory, stages: list[dict[str, Any]], notify,
                round_name: str = "decisions") -> DecisionSet:
        s = self.settings
        started = time.perf_counter()
        result = DecisionSet()

        def record(key: str, choice: ChoiceResult) -> None:
            result.decisions.append({"key": key, "round": round_name, **choice.as_dict()})

        try:
            recipients = directory.recipients
            if recipients:
                notify("Decision model: who is the letter for?")
                options = [recipient_label(r) for r in recipients] + [NOT_IN_LIST]
                choice = self.decision.choice(text, Q_RECIPIENT, options, instructions=I_RECIPIENT, none_option=NOT_IN_LIST)
                record("recipient", choice)
                if choice.choice != NOT_IN_LIST:
                    recipient = recipients[choice.index]
                    confidence = choice.confidence
                    if s.verify_recipient:
                        names = " / ".join([recipient.name, *recipient.aliases[:3]])
                        check = self.decision.choice(
                            text, f"Is this letter addressed to {names}?", ["Yes", "No"], instructions=I_RECIPIENT)
                        record("verify_recipient", check)
                        confidence = min(confidence, check.probability_of("Yes"))
                    result.recipient = recipient
                    result.worker = directory.worker(recipient.worker_id)
                    result.recipient_confidence = confidence

            if result.recipient is None and s.route_unknown_by_description:
                described = [w for w in directory.workers if w.description.strip()]
                if described:
                    notify("Decision model: which worker is responsible?")
                    options = [f"{w.name}: {w.description.strip()}" for w in described] + [NOBODY]
                    choice = self.decision.choice(text, Q_WORKER, options, none_option=NOBODY)
                    record("worker", choice)
                    if choice.choice != NOBODY:
                        result.worker = described[choice.index]
                        result.worker_confidence = choice.confidence

            types = s.document_types
            if types:
                notify("Decision model: what kind of document?")
                options = [f"{t.name}: {t.description}" if t.description else t.name for t in types]
                choice = self.decision.choice(text, Q_TYPE, options)
                record("doc_type", choice)
                result.doc_type = types[choice.index].name
                result.doc_type_confidence = choice.confidence
        except DecisionError as exc:
            log.warning("Decision model gave no usable answer: %s", exc)
            result.decisions.append({"key": "error", "round": round_name, "error": str(exc)})

        stages.append({"name": round_name, "seconds": round(time.perf_counter() - started, 3),
                       "questions": len([d for d in result.decisions if d.get("round") == round_name])})
        return result

    def _outcome(self, decision: DecisionSet, facts: LetterFacts, directory: Directory) -> tuple[str, str, str]:
        threshold = self.settings.confidence_threshold
        if decision.recipient is not None:
            if decision.recipient_confidence >= threshold:
                return "filed", "", ""
            return ("review", f"Not sure it is for {decision.recipient.name} "
                    f"({decision.recipient_confidence:.0%}, needs {threshold:.0%}).", "")
        if not directory.recipients and not any(w.description for w in directory.workers):
            return "review", "No workers and recipients are set up yet.", ""
        name = facts.recipient_name.strip()
        unknown = f"Recipient '{name}' is not in your list." if name else "The recipient could not be identified."
        if decision.worker is not None and decision.worker_confidence >= threshold:
            if self.settings.auto_create_recipients and name and not any(
                    r.name.lower() == name.lower() for r in directory.recipients):
                return "filed", "", name
            return "review", f"{unknown} Suggested worker: {decision.worker.name} ({decision.worker_confidence:.0%}).", ""
        return "review", unknown, ""


def relative_target(settings: Settings, worker: Worker, recipient_name: str, recipient_folder: str,
                    analysis_facts: LetterFacts, letter_date: str, doc_type: str, original: Path) -> Path:
    values = FilingValues(
        worker=worker.folder,
        recipient=recipient_folder or recipient_name,
        date=letter_date,
        sender=analysis_facts.sender,
        doc_type=doc_type,
        subject=analysis_facts.subject,
        original=original.stem,
    )
    return build_relative_path(settings.folder_template, settings.filename_template, values, original.suffix)
