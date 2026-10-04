"""Export what LetterEye has learned from its users as training data.

Every letter a human accepted or corrected becomes a labelled example:

* decisions.jsonl  – one typed decision per line (state, question, options, correct option, what the AI said):
                     the format decision models like Jev / Von / Kev are trained and evaluated on
* chat_sft.jsonl   – the same examples as chat messages, exactly as LetterEye prompts the model, with the
                     correct option letter as the answer – ready for LoRA fine-tuning (Unsloth, axolotl, …)
* extraction.jsonl – OCR text with the human-corrected sender, date, subject and recipient
* documents.csv / feedback.csv – everything for your own analysis
* images/          – page previews (optional)
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

from .ai.decision import LABELS, SYSTEM_PROMPT, DecisionEngine
from .db import Database
from .insights import latest_outcomes
from .paths import previews_dir
from .pipeline.processor import I_RECIPIENT, NOBODY, NOT_IN_LIST

README = """# LetterEye training data

Exported {created} · {examples} decision examples from {letters} human-verified letters.

All data comes from letters a person accepted or corrected in LetterEye's approval queue, so every label
is human-verified. Auto-filed letters nobody checked are not included.

## decisions.jsonl
One typed decision per line:

    {{"task": "recipient", "state": "<OCR text>", "question": "...", "options": ["...", ...],
     "label": "<correct option>", "label_index": 2, "ai_choice": "...", "ai_probabilities": [...],
     "ai_correct": false, "source": "corrected", "decision_model": "qwen3.5:4b", "doc_id": 17}}

`task` is one of recipient, verify_recipient, worker, doc_type. Use it to evaluate or train a decision
model (a classifier head over an encoder, or a LoRA on a small LLM). `ai_correct` lets you measure the
current model; the examples where it is false are the interesting ones.

## chat_sft.jsonl
The same examples as chat messages in exactly the format LetterEye sends to the model, with the correct
option letter as the assistant answer:

    {{"messages": [{{"role": "system", ...}}, {{"role": "user", ...}}, {{"role": "assistant", "content": "C"}}]}}

Fine-tune the decision model on it (e.g. Unsloth LoRA on qwen3.5), export a GGUF, create an Ollama model
from it and select it in LetterEye under Settings → AI models. Hold back ~20 % for evaluation.

## extraction.jsonl
OCR text with what the AI extracted and what the human kept or corrected (sender, letter date,
subject, recipient) – for improving the extraction step.

## documents.csv, feedback.csv
Every letter and every human decision, for spreadsheets and your own analysis.
"""


def _label_option(options: list[str], wanted: str, fallback: str | None) -> str | None:
    wanted_cf = wanted.casefold().strip()
    for option in options:
        head = option.split(" (also written as")[0].split(":")[0].strip().casefold()
        if head == wanted_cf or option.casefold() == wanted_cf:
            return option
    return fallback if fallback in options else None


def _decision_examples(doc, outcome) -> list[dict]:
    final = outcome.feedback.final
    examples = []
    for decision in doc.trace.get("decisions", []):
        options = [o["option"] for o in decision.get("options", [])]
        if not options:
            continue
        task = decision["key"]
        if task == "recipient":
            label = _label_option(options, final.get("recipient", ""), NOT_IN_LIST)
            instructions = I_RECIPIENT
        elif task == "worker":
            label = _label_option(options, final.get("worker", ""), NOBODY)
            instructions = ""
        elif task == "doc_type":
            label = _label_option(options, final.get("doc_type", ""), None)
            instructions = ""
        elif task == "verify_recipient":
            asked = decision["question"].removeprefix("Is this letter addressed to ").rstrip("?").split(" / ")
            label = "Yes" if final.get("recipient", "").casefold() in (a.casefold() for a in asked) else "No"
            instructions = I_RECIPIENT
        else:
            continue
        if label is None:
            continue
        probabilities = [o["p"] for o in decision["options"]]
        examples.append({
            "task": task, "state": doc.ocr_text, "question": decision["question"], "instructions": instructions,
            "options": options, "label": label, "label_index": options.index(label),
            "ai_choice": decision.get("choice"), "ai_probabilities": probabilities,
            "ai_correct": decision.get("choice") == label, "round": decision.get("round", "decisions"),
            "source": outcome.feedback.action, "decision_model": outcome.feedback.decision_model, "doc_id": doc.id,
        })
    return examples


def export(db: Database, folder: str | Path, include_images: bool = False) -> tuple[Path, int, int]:
    """Write a zip into `folder`. Returns (path, number of decision examples, number of letters)."""
    outcomes = {k: v for k, v in latest_outcomes(db.list_feedback()).items() if v.verified}
    prompt_engine = DecisionEngine(llm=object())  # only used to render prompts exactly like the app does
    decisions, chats, extraction = [], [], []
    docs_by_id = {d.id: d for d in db.list_documents(limit=1_000_000)}
    for doc_id, outcome in outcomes.items():
        doc = docs_by_id.get(doc_id)
        if doc is None or not doc.ocr_text:
            continue
        for example in _decision_examples(doc, outcome):
            decisions.append(example)
            user = prompt_engine.render_prompt(example["state"], example["question"], example["options"],
                                               example["instructions"])
            chats.append({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": LABELS[example["label_index"]]},
            ], "task": example["task"], "doc_id": doc_id})
        ai, final = outcome.feedback.ai, outcome.feedback.final
        extraction.append({
            "doc_id": doc_id, "text": doc.ocr_text,
            "ai": {k: ai.get(k, "") for k in ("sender", "letter_date", "subject", "recipient")},
            "label": {k: final.get(k, "") for k in ("sender", "letter_date", "subject", "recipient")},
            "changed": outcome.feedback.changed,
        })

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"lettereye-training-data-{datetime.now():%Y%m%d-%H%M}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("decisions.jsonl", "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in decisions))
        archive.writestr("chat_sft.jsonl", "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in chats))
        archive.writestr("extraction.jsonl", "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in extraction))
        archive.writestr("documents.csv", _documents_csv(docs_by_id.values(), latest_outcomes(db.list_feedback())))
        archive.writestr("feedback.csv", _feedback_csv(db))
        archive.writestr("README.md", README.format(created=f"{datetime.now():%Y-%m-%d %H:%M}",
                                                    examples=len(decisions), letters=len(extraction)))
        if include_images:
            for doc_id in outcomes:
                image = previews_dir() / f"{doc_id}.png"
                if image.exists():
                    archive.write(image, f"images/{doc_id}.png")
    return target, len(decisions), len(extraction)


def _documents_csv(docs, outcomes) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "file", "status", "outcome", "changed", "ai_worker", "ai_recipient", "ai_doc_type",
                     "ai_confidence", "worker", "recipient", "doc_type", "letter_date", "sender", "subject",
                     "ocr_source", "escalated", "decision_model", "processing_s", "created_at"])
    for d in sorted(docs, key=lambda d: d.id):
        outcome = outcomes.get(d.id)
        p = d.proposal or {}
        writer.writerow([
            d.id, d.original_name, d.status, outcome.kind if outcome else "", ";".join(outcome.feedback.changed)
            if outcome else "", p.get("worker", ""), p.get("recipient", ""), p.get("doc_type", ""), d.confidence,
            d.worker_name, d.recipient_label, d.doc_type, d.letter_date, d.sender, d.subject, d.ocr_source,
            bool(p.get("escalated")), p.get("decision_model", ""),
            round(d.duration_ms / 1000, 2) if d.duration_ms else "", d.created_at,
        ])
    return buffer.getvalue()


def _feedback_csv(db: Database) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "doc_id", "created_at", "action", "mode", "changed", "ai", "final", "ai_confidence",
                     "decision_model", "ocr_source", "escalated", "seconds_to_decide", "processing_ms", "undone"])
    for f in db.list_feedback(include_undone=True):
        writer.writerow([f.id, f.doc_id, f.created_at, f.action, f.mode, ";".join(f.changed),
                         json.dumps(f.ai, ensure_ascii=False), json.dumps(f.final, ensure_ascii=False),
                         f.ai_confidence, f.decision_model, f.ocr_source, f.escalated, f.seconds_to_decide,
                         f.processing_ms, f.undone])
    return buffer.getvalue()
