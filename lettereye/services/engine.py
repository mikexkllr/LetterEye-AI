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
from ..db import Database, Document
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
        if self.state == "running":
            threading.Thread(target=self.restart, name="restart", daemon=True).start()

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

    def _apply(self, doc: Document, analysis: Analysis, path: Path, settings: Settings, duration_ms: int) -> None:
        decision = analysis.decision
        common: dict[str, Any] = dict(
            stage="", sender=analysis.facts.sender, letter_date=analysis.letter_date, doc_type=decision.doc_type,
            subject=analysis.facts.subject, recipient_name=analysis.facts.recipient_name,
            confidence=round(analysis.confidence, 4), ocr_source=analysis.ocr.source, ocr_text=analysis.ocr.text,
            trace=analysis.trace(), duration_ms=duration_ms,
            worker_id=decision.worker.id if decision.worker else None,
            recipient_id=decision.recipient.id if decision.recipient else None,
        )
        escalated = " (after OCR LLM)" if analysis.escalated else ""
        if analysis.status == "filed" and decision.worker is not None:
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
                self.feed.add(f"Filed {doc.original_name} → {decision.worker.name} / {recipient.name} "
                              f"({analysis.confidence:.0%}){escalated}", "success", doc.id)
                return
        target = self._park(path, settings.review_dir, settings)
        self.db.update_document(doc.id, status="review", current_path=str(target), review_reason=analysis.reason, **common)
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

    # ----------------------------------------------------------------- user actions
    def assign(self, doc_id: int, *, recipient_id: int | None = None, worker_id: int | None = None,
               new_recipient_name: str = "", doc_type: str | None = None, learn_alias: bool = True) -> Path:
        """File a document from the review queue by hand. Optionally remember the spelling for next time."""
        doc = self.db.get_document(doc_id)
        if doc is None:
            raise ValueError("Document not found.")
        settings = self.store.get()
        if recipient_id is not None:
            recipient = self.db.get_recipient(recipient_id)
        elif worker_id is not None and new_recipient_name.strip():
            recipient = self.db.find_recipient(new_recipient_name) or self.db.create_recipient(worker_id, new_recipient_name)
        else:
            raise ValueError("Choose a recipient or enter a new one.")
        if recipient is None:
            raise ValueError("Recipient not found.")
        worker = self.db.get_worker(recipient.worker_id)
        assert worker is not None
        source = Path(doc.current_path)
        if not source.exists():
            raise FileNotFoundError(f"The file is no longer at {source}")
        facts = LetterFacts.model_validate(doc.trace.get("facts", {}))
        final_type = doc_type or doc.doc_type or (settings.document_types[-1].name if settings.document_types else "")
        rel = relative_target(settings, worker, recipient.name, recipient.folder, facts, doc.letter_date, final_type,
                              Path(doc.original_name))
        target = transfer(source, Path(settings.output_folder) / rel, move=True)
        trace = {**doc.trace, "manual": {"recipient": recipient.name, "worker": worker.name, "type": final_type}}
        self.db.update_document(doc_id, status="filed", current_path=str(target), worker_id=worker.id,
                                recipient_id=recipient.id, doc_type=final_type, review_reason="", trace=trace)
        self.feed.add(f"Filed by hand: {doc.original_name} → {worker.name} / {recipient.name}", "success", doc_id)
        spelling = clean_person_name(doc.recipient_name)
        if learn_alias and len(spelling) >= 3 and self.db.add_alias(recipient.id, spelling):
            self.feed.add(f"Learned: '{spelling}' is {recipient.name}", "ai", doc_id)
        return target

    def reprocess(self, doc_id: int) -> None:
        doc = self.db.get_document(doc_id)
        if doc is None or not Path(doc.current_path).exists():
            raise FileNotFoundError("The file for this document is gone.")
        if self.state != "running":
            raise RuntimeError("Start watching first, then re-run the AI.")
        self.db.update_document(doc_id, status="queued", stage="Queued for another try", error="")
        self._queue.put(doc_id)
        self.feed.add(f"Re-running AI on {doc.original_name}", "info", doc_id)

    def dismiss(self, doc_id: int) -> None:
        self.db.update_document(doc_id, status="ignored", review_reason="Dismissed")
        self.feed.touch()

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
