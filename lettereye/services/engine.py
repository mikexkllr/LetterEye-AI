"""The processing engine: watches the inbox, runs the pipeline on one letter at a time (the GPU is
shared), files the result and records everything for the UI."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import paths
from ..ai.decision import DecisionEngine
from ..ai.extraction import FactExtractor, LetterFacts
from ..ai.ollama_service import OllamaService
from ..db import Database, Document, now_iso
from ..events import ActivityFeed
from ..ocr.documents import save_preview
from ..ocr.fast_ocr import FastOCR
from ..ocr.llm_ocr import LlmOCR
from ..pipeline.filing import transfer
from ..pipeline.names import clean_person_name
from ..pipeline.processor import Analysis, Directory, Processor, relative_target
from ..settings import Settings, SettingsStore
from .watcher import InboxWatcher, wait_until_ready

log = logging.getLogger(__name__)


def fingerprint(path: Path) -> str:
    try:
        st = path.stat()
        return f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return ""


def preview_path(doc_id: int) -> Path:
    return paths.previews_dir() / f"{doc_id}.png"


@dataclass
class Current:
    doc_id: int
    name: str
    stage: str
    started: float


class Engine:
    def __init__(self, db: Database, store: SettingsStore, feed: ActivityFeed):
        self.db = db
        self.store = store
        self.feed = feed
        self.state = "stopped"  # stopped | starting | running | stopping
        self.current: Current | None = None
        self.last_error = ""
        self.ocr_device = ""
        self._queue: queue.Queue[int] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._watcher: InboxWatcher | None = None
        self._processor: Processor | None = None
        self._settings: Settings = store.get()
        self._fast_ocr: FastOCR | None = None
        self._lifecycle = threading.Lock()
        store.on_change(self._on_settings_changed)

    # ----------------------------------------------------------------- building blocks
    def directory(self) -> Directory:
        return Directory(self.db.list_workers(active_only=True), self.db.list_recipients(active_workers_only=True))

    def fast_ocr(self, settings: Settings) -> FastOCR:
        key = (settings.fast_ocr_size, settings.fast_ocr_device, settings.language)
        if self._fast_ocr is None or (self._fast_ocr.size, self._fast_ocr.preference, self._fast_ocr.language) != key:
            self._fast_ocr = FastOCR(settings.fast_ocr_size, settings.fast_ocr_device, settings.language)
        return self._fast_ocr

    def build_processor(self, settings: Settings) -> Processor:
        decision = DecisionEngine(settings.ollama_url, settings.decision_model, num_ctx=settings.num_ctx,
                                  keep_alive=settings.keep_alive, debias=settings.debias_order)
        extractor = FactExtractor(settings.ollama_url, settings.effective_extraction_model, language=settings.language,
                                  num_ctx=settings.num_ctx, keep_alive=settings.keep_alive)
        llm_ocr = None
        if settings.ocr_mode != "fast_only" and settings.ocr_llm_model:
            llm_ocr = LlmOCR(settings.ollama_url, settings.ocr_llm_model, keep_alive=settings.keep_alive)
        return Processor(settings, decision, extractor, self.fast_ocr(settings), llm_ocr, self.directory)

    def preflight(self, settings: Settings | None = None) -> list[str]:
        """Everything that must be true before letters can be processed, as user-facing messages."""
        settings = settings or self.store.get()
        problems = settings.problems()
        ollama = OllamaService(settings.ollama_url)
        if ollama.version() is None:
            problems.append(f"Ollama is not reachable at {settings.ollama_url}. Is it installed and running?")
            return problems
        needed = [settings.decision_model, settings.effective_extraction_model]
        if settings.ocr_mode != "fast_only":
            needed.append(settings.ocr_llm_model)
        for model in dict.fromkeys(m for m in needed if m):
            if not ollama.has_model(model):
                problems.append(f"Model '{model}' is not downloaded yet (Settings → AI models).")
        return problems

    # ----------------------------------------------------------------- lifecycle
    def start(self) -> list[str]:
        with self._lifecycle:
            if self.state in ("running", "starting"):
                return []
            settings = self.store.get()
            problems = self.preflight(settings)
            if problems:
                self.last_error = problems[0]
                return problems
            self.state = "starting"
            self._settings = settings
            self._processor = self.build_processor(settings)
            self._stop = threading.Event()  # a fresh event per run: an old worker still busy keeps its own
            self._recover_interrupted(settings)
            self._thread = threading.Thread(target=self._loop, args=(self._stop,), name="lettereye-worker", daemon=True)
            self._thread.start()
            self._watcher = InboxWatcher(settings.inbox_folder, self._on_file, recursive=settings.recursive_watch,
                                         ignore=self._ignore_path)
            self._watcher.start(scan_existing=settings.process_existing_on_start)
            self.state = "running"
            self.last_error = ""
            self.feed.add(f"Watching {settings.inbox_folder}", "success")
            threading.Thread(target=self._warm_up, args=(settings,), name="warm-up", daemon=True).start()
            return []

    def stop(self) -> None:
        with self._lifecycle:
            if self.state == "stopped":
                return
            self.state = "stopping"
            self._stop.set()
            if self._watcher:
                self._watcher.stop()
                self._watcher = None
            if self._thread:
                self._thread.join(timeout=10)
                self._thread = None
            while not self._queue.empty():  # queued letters stay where they are and are picked up next time
                try:
                    doc_id = self._queue.get_nowait()
                except queue.Empty:
                    break
                doc = self.db.get_document(doc_id)
                if doc is not None:
                    self._release(doc, self._settings)
            self.state = "stopped"
            self.current = None
            self.feed.add("Stopped watching", "info")

    def restart(self) -> list[str]:
        self.stop()
        return self.start()

    def _on_settings_changed(self, settings: Settings) -> None:
        changed = {k for k, v in settings.model_dump().items() if getattr(self._settings, k, None) != v}
        if self.state == "running" and changed - LIVE_SETTINGS:
            threading.Thread(target=self.restart, name="restart", daemon=True).start()
        elif self.state == "running":
            self._settings = settings  # takes effect for the next letter without restarting

    def _warm_up(self, settings: Settings) -> None:
        try:
            self.ocr_device = self.fast_ocr(settings).warm_up()
            OllamaService(settings.ollama_url).warm_up(settings.decision_model, settings.keep_alive)
            self.feed.touch()
        except Exception as exc:
            log.warning("Warm-up failed: %s", exc)

    def _recover_interrupted(self, settings: Settings) -> None:
        for doc in self.db.interrupted_documents():
            self._release(doc, settings)

    def _in_inbox(self, path: Path, settings: Settings) -> bool:
        try:
            return path.resolve().is_relative_to(Path(settings.inbox_folder).resolve())
        except (OSError, ValueError):
            return False

    def _release(self, doc: Document, settings: Settings) -> None:
        """Undo 'queued/processing' for a letter that was not finished (stop or crash)."""
        path = Path(doc.current_path)
        if path.exists() and self._in_inbox(path, settings):
            self.db.delete_document(doc.id)  # still in the inbox: it is detected again next time
        elif path.exists():
            self.db.update_document(doc.id, status="review", stage="", review_reason="Interrupted – run the AI again.")
        else:
            self.db.update_document(doc.id, status="failed", stage="", error="Interrupted because the app was closed.")

    def _ignore_path(self, path: Path) -> bool:
        output = self._settings.output_folder
        if output and path.resolve().is_relative_to(Path(output).resolve()):
            return True
        return self.db.is_tracked(path, fingerprint(path))

    def _on_file(self, path: Path) -> None:
        if self._stop.is_set() or self.db.is_tracked(path, fingerprint(path)):
            return
        doc = self.db.create_document(path, fingerprint=fingerprint(path))
        self._queue.put(doc.id)
        self.feed.add(f"New letter: {path.name}", "info", doc.id)

    # ----------------------------------------------------------------- processing
    def _loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                doc_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(doc_id, stop=stop)
            except Exception as exc:  # never let one letter kill the worker thread
                log.exception("Unexpected error while processing document %s", doc_id)
                self.db.update_document(doc_id, status="failed", error=str(exc))
            finally:
                self.current = None
                self.feed.touch()

    def _stage(self, doc_id: int, stage: str) -> None:
        if self.current and self.current.doc_id == doc_id:
            self.current.stage = stage
        self.db.update_document(doc_id, stage=stage)
        self.feed.touch()

    def _process(self, doc_id: int, processor: Processor | None = None, stop: threading.Event | None = None) -> None:
        doc = self.db.get_document(doc_id)
        if doc is None:
            return
        settings = self._settings
        processor = processor or self._processor
        assert processor is not None
        path = Path(doc.current_path)
        self.current = Current(doc_id, doc.original_name, "Waiting until the file is complete", time.monotonic())
        self.db.update_document(doc_id, status="processing", stage=self.current.stage, error="")
        self.feed.touch()
        stop = stop or self._stop
        if not wait_until_ready(path, stop):
            if not path.exists():
                self.db.delete_document(doc_id)
                self.feed.add(f"{doc.original_name} disappeared before it could be read", "warning")
            elif not stop.is_set():
                self.db.update_document(doc_id, status="failed", error="The file stayed locked or incomplete.")
            return

        started = time.perf_counter()
        try:
            analysis = processor.analyze(path, on_stage=lambda stage: self._stage(doc_id, stage))
        except Exception as exc:
            log.exception("Processing %s failed", path)
            target = self._park(path, settings.failed_dir, settings)
            self.db.update_document(doc_id, status="failed", stage="", error=str(exc), current_path=str(target),
                                    duration_ms=int((time.perf_counter() - started) * 1000))
            self.feed.add(f"Failed: {doc.original_name} – {exc}", "error", doc_id)
            return

        if analysis.preview is not None:
            try:
                save_preview(analysis.preview, preview_path(doc_id))
            except OSError:
                pass
        self._apply(doc, analysis, path, settings, int((time.perf_counter() - started) * 1000))

    def _proposal(self, analysis: Analysis, settings: Settings) -> dict[str, Any]:
        """Snapshot of what the AI proposed, kept unchanged so human decisions can be compared with it."""
        d = analysis.decision
        return {
            "verdict": analysis.status,  # what the AI would do on its own: filed | review
            "worker": d.worker.name if d.worker else "",
            "worker_id": d.worker.id if d.worker else None,
            "recipient": d.recipient.name if d.recipient else analysis.new_recipient_name,
            "recipient_id": d.recipient.id if d.recipient else None,
            "new_recipient": bool(analysis.new_recipient_name and not d.recipient),
            "recipient_read": analysis.facts.recipient_name,
            "doc_type": d.doc_type,
            "letter_date": analysis.letter_date,
            "sender": analysis.facts.sender,
            "subject": analysis.facts.subject,
            "confidence": round(analysis.confidence, 4),
            "doc_type_confidence": round(d.doc_type_confidence, 4),
            "decision_model": settings.decision_model,
            "ocr_source": analysis.ocr.source,
            "escalated": analysis.escalated,
            "reason": analysis.reason,
            "ready_at": now_iso(),
        }

    def _apply(self, doc: Document, analysis: Analysis, path: Path, settings: Settings, duration_ms: int) -> None:
        decision = analysis.decision
        proposal = self._proposal(analysis, settings)
        common: dict[str, Any] = dict(
            stage="", sender=analysis.facts.sender, letter_date=analysis.letter_date, doc_type=decision.doc_type,
            subject=analysis.facts.subject, recipient_name=analysis.facts.recipient_name,
            confidence=round(analysis.confidence, 4), ocr_source=analysis.ocr.source, ocr_text=analysis.ocr.text,
            trace=analysis.trace(), duration_ms=duration_ms, proposal=proposal,
            worker_id=decision.worker.id if decision.worker else None,
            recipient_id=decision.recipient.id if decision.recipient else None,
        )
        escalated = " (after OCR LLM)" if analysis.escalated else ""
        confident = analysis.status == "filed" and decision.worker is not None
        if confident and self.store.get().workflow_mode == "automatic":
            recipient = decision.recipient
            if recipient is None and analysis.new_recipient_name:
                try:
                    recipient = self.db.create_recipient(decision.worker.id, analysis.new_recipient_name)
                    self.feed.add(f"New recipient '{recipient.name}' added to {decision.worker.name}", "ai", doc.id)
                except ValueError:
                    recipient = self.db.find_recipient(analysis.new_recipient_name)
            if recipient is not None:
                rel = relative_target(settings, decision.worker, recipient.name, recipient.folder, analysis.facts,
                                      analysis.letter_date, decision.doc_type, Path(doc.original_name))
                target = transfer(path, Path(settings.output_folder) / rel, move=self._should_move(path, settings))
                self.db.update_document(doc.id, status="filed", current_path=str(target), review_reason="",
                                        **{**common, "recipient_id": recipient.id})
                self.db.add_feedback(doc.id, "auto_filed", mode="automatic", ai=_fields(proposal),
                                     final=_fields(proposal), ai_confidence=proposal["confidence"],
                                     decision_model=settings.decision_model, ocr_source=analysis.ocr.source,
                                     escalated=analysis.escalated, processing_ms=duration_ms)
                self.feed.add(f"Filed {doc.original_name} → {decision.worker.name} / {recipient.name} "
                              f"({analysis.confidence:.0%}){escalated}", "success", doc.id)
                return
        target = self._park(path, settings.review_dir, settings)
        if confident:
            self.db.update_document(doc.id, status="pending", current_path=str(target), review_reason="", **common)
            self.feed.add(f"Ready for approval: {doc.original_name} → {proposal['worker']} / {proposal['recipient']} "
                          f"({analysis.confidence:.0%}){escalated}", "ai", doc.id)
        else:
            self.db.update_document(doc.id, status="review", current_path=str(target), review_reason=analysis.reason,
                                    **common)
            self.feed.add(f"Needs review: {doc.original_name} – {analysis.reason}", "warning", doc.id)

    def _park(self, path: Path, folder: Path, settings: Settings) -> Path:
        """Move a letter into the review/failed folder (unless it is already there)."""
        if not path.exists():
            return path
        if path.parent.resolve() == folder.resolve():
            return path
        return transfer(path, folder / path.name, move=self._should_move(path, settings))

    def _should_move(self, path: Path, settings: Settings) -> bool:
        """Originals in the inbox are copied in 'copy' mode; our own copies (review/failed) are always moved."""
        return settings.move_files or not self._in_inbox(path, settings)

    # ----------------------------------------------------------------- human in the loop
    def approve(self, doc_id: int, *, recipient_id: int | None = None, worker_id: int | None = None,
                new_recipient_name: str = "", doc_type: str | None = None, letter_date: str | None = None,
                sender: str | None = None, subject: str | None = None, learn_alias: bool = True) -> Path:
        """Accept the AI's proposal (no arguments) or file the letter with corrections.

        Works for letters waiting for approval or review, and for letters already filed (a correction after
        the fact moves the file). Every call is recorded as feedback for the metrics and training data.
        """
        doc = self.db.get_document(doc_id)
        if doc is None:
            raise ValueError("Document not found.")
        if doc.status not in ("pending", "review", "filed"):
            raise ValueError("This letter is not waiting for a decision.")
        settings = self.store.get()
        proposal = doc.proposal or {}
        recipient = self._pick_recipient(doc, recipient_id, worker_id, new_recipient_name)
        worker = self.db.get_worker(recipient.worker_id)
        assert worker is not None
        final = {
            "worker": worker.name,
            "recipient": recipient.name,
            "doc_type": doc.doc_type if doc_type is None else doc_type,
            "letter_date": doc.letter_date if letter_date is None else letter_date,
            "sender": doc.sender if sender is None else sender,
            "subject": doc.subject if subject is None else subject,
        }
        source = Path(doc.current_path)
        if not source.exists():
            raise FileNotFoundError(f"The file is no longer at {source}")
        facts = LetterFacts(sender_organization=final["sender"], subject=final["subject"])
        rel = relative_target(settings, worker, recipient.name, recipient.folder, facts, final["letter_date"],
                              final["doc_type"], Path(doc.original_name))
        destination = Path(settings.output_folder) / rel
        target = source if source == destination else transfer(source, destination, move=True)

        if doc.status == "filed":  # a correction after the fact replaces the earlier verdict
            previous = self.db.last_feedback(doc_id)
            if previous and previous.action in ("accepted", "corrected", "confirmed"):
                self.db.undo_feedback(previous.id)
        changed = [k for k in FIELDS if _norm(final[k]) != _norm(proposal.get(k, ""))]
        action = "corrected" if changed else ("confirmed" if doc.status == "filed" else "accepted")
        self.db.update_document(doc_id, status="filed", current_path=str(target), worker_id=worker.id,
                                recipient_id=recipient.id, doc_type=final["doc_type"],
                                letter_date=final["letter_date"], sender=final["sender"], subject=final["subject"],
                                review_reason="")
        self.db.add_feedback(doc_id, action, mode=settings.workflow_mode, ai=_fields(proposal), final=final,
                             changed=changed, ai_confidence=doc.confidence,
                             decision_model=proposal.get("decision_model", ""), ocr_source=doc.ocr_source,
                             escalated=bool(proposal.get("escalated")),
                             seconds_to_decide=_seconds_since(proposal.get("ready_at") or doc.updated_at),
                             processing_ms=doc.duration_ms)
        verb = {"accepted": "Accepted", "confirmed": "Confirmed", "corrected": "Corrected"}[action]
        detail = f" ({', '.join(changed)})" if changed else ""
        self.feed.add(f"{verb}: {doc.original_name} → {worker.name} / {recipient.name}{detail}", "success", doc_id)
        spelling = clean_person_name(doc.recipient_name)
        if learn_alias and "recipient" in changed and len(spelling) >= 3 and self.db.add_alias(recipient.id, spelling):
            self.feed.add(f"Learned: '{spelling}' is {recipient.name}", "ai", doc_id)
        return target

    assign = approve  # name used by the first version

    def confirm(self, doc_id: int) -> None:
        """'This was right' for a letter the AI filed on its own – turns it into verified training data."""
        doc = self.db.get_document(doc_id)
        if doc is None or doc.status != "filed":
            raise ValueError("Only filed letters can be confirmed.")
        self.approve(doc_id, learn_alias=False)

    def approve_many(self, doc_ids: list[int]) -> tuple[int, list[str]]:
        done, problems = 0, []
        for doc_id in doc_ids:
            try:
                self.approve(doc_id)
                done += 1
            except Exception as exc:
                problems.append(str(exc))
        return done, problems

    def reject(self, doc_id: int, reason: str = "Not a letter") -> Path:
        """Set a letter aside: it is moved to the rejected folder and not filed."""
        doc = self.db.get_document(doc_id)
        if doc is None:
            raise ValueError("Document not found.")
        settings = self.store.get()
        source = Path(doc.current_path)
        target = transfer(source, settings.rejected_dir / source.name, move=True) if source.exists() else source
        self.db.update_document(doc_id, status="ignored", current_path=str(target), review_reason=reason)
        proposal = doc.proposal or {}
        self.db.add_feedback(doc_id, "rejected", mode=settings.workflow_mode, ai=_fields(proposal),
                             final={"reason": reason}, ai_confidence=doc.confidence,
                             decision_model=proposal.get("decision_model", ""), ocr_source=doc.ocr_source,
                             escalated=bool(proposal.get("escalated")),
                             seconds_to_decide=_seconds_since(proposal.get("ready_at") or doc.updated_at))
        self.feed.add(f"Set aside: {doc.original_name} – {reason}", "info", doc_id)
        return target

    dismiss = reject

    def undo(self, doc_id: int) -> None:
        """Take back the last human decision: the letter returns to the approval queue."""
        doc = self.db.get_document(doc_id)
        last = self.db.last_feedback(doc_id) if doc else None
        if doc is None or last is None or last.action not in ("accepted", "corrected", "confirmed", "rejected"):
            raise ValueError("Nothing to undo for this letter.")
        settings = self.store.get()
        proposal = doc.proposal or {}
        source = Path(doc.current_path)
        target = transfer(source, settings.review_dir / doc.original_name, move=True) if source.exists() else source
        status = "pending" if proposal.get("verdict") == "filed" else "review"
        self.db.update_document(
            doc_id, status=status, current_path=str(target), worker_id=proposal.get("worker_id"),
            recipient_id=proposal.get("recipient_id"), doc_type=proposal.get("doc_type", doc.doc_type),
            letter_date=proposal.get("letter_date", doc.letter_date), sender=proposal.get("sender", doc.sender),
            subject=proposal.get("subject", doc.subject), review_reason=proposal.get("reason", ""))
        self.db.undo_feedback(last.id)
        self.feed.add(f"Undone: {doc.original_name} is back in the approval queue", "info", doc_id)

    def _pick_recipient(self, doc: Document, recipient_id: int | None, worker_id: int | None, new_name: str):
        if recipient_id is not None:
            recipient = self.db.get_recipient(recipient_id)
        elif worker_id is not None and new_name.strip():
            recipient = self.db.find_recipient(new_name) or self.db.create_recipient(worker_id, new_name)
        elif doc.recipient_id is not None:
            recipient = self.db.get_recipient(doc.recipient_id)
        elif doc.proposal.get("new_recipient") and doc.proposal.get("worker_id"):
            name = doc.proposal["recipient"]
            recipient = self.db.find_recipient(name) or self.db.create_recipient(doc.proposal["worker_id"], name)
        else:
            raise ValueError("Choose a recipient first.")
        if recipient is None:
            raise ValueError("Recipient not found.")
        return recipient

    def reprocess(self, doc_id: int) -> None:
        doc = self.db.get_document(doc_id)
        if doc is None or not Path(doc.current_path).exists():
            raise FileNotFoundError("The file for this document is gone.")
        if self.state != "running":
            raise RuntimeError("Start watching first, then re-run the AI.")
        self.db.update_document(doc_id, status="queued", stage="Queued for another try", error="")
        self._queue.put(doc_id)
        self.feed.add(f"Re-running AI on {doc.original_name}", "info", doc_id)

    def analyze_file(self, path: str | Path, on_stage=None) -> Analysis:
        """Dry run for the 'Test a letter' feature: full pipeline, nothing is moved or saved."""
        settings = self.store.get()
        problems = [p for p in self.preflight(settings) if "folder" not in p.lower()]
        if problems:
            raise RuntimeError(problems[0])
        return self.build_processor(settings).analyze(path, on_stage=on_stage)

    # ----------------------------------------------------------------- UI helpers
    @property
    def queue_size(self) -> int:
        return self._queue.qsize()


FIELDS = ("worker", "recipient", "doc_type", "letter_date", "sender", "subject")
# Settings that apply immediately; changing anything else restarts the watcher.
LIVE_SETTINGS = {"workflow_mode", "theme", "update_channel", "auto_update", "update_check_minutes", "autostart",
                 "setup_completed"}


def _fields(values: dict[str, Any]) -> dict[str, Any]:
    return {key: values.get(key, "") or "" for key in FIELDS}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _seconds_since(iso: str | None) -> float | None:
    from datetime import UTC, datetime

    try:
        return round((datetime.now(UTC) - datetime.fromisoformat(iso)).total_seconds(), 1) if iso else None
    except ValueError:
        return None
