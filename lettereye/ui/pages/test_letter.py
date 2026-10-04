"""'Test a letter': run the full pipeline on a file and show what would happen. Nothing is moved."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from nicegui import run, ui

from ...ocr.documents import SUPPORTED_EXTENSIONS
from ...pipeline.processor import Analysis, relative_target
from .. import theme
from ..context import ctx
from ..document_detail import DECISION_TITLES, STAGE_TITLES
from ..widgets import confidence_bar, probability_bars

ACCEPT = ",".join(sorted(SUPPORTED_EXTENSIONS))


def open_test_dialog() -> None:
    c = ctx()
    progress = {"stage": ""}

    with ui.dialog().props("maximized=false") as dialog, ui.card().classes("le-card").style("width: 1100px; max-width: 96vw"):
        with ui.row().classes("w-full items-start justify-between no-wrap"):
            with ui.column().classes("gap-1"):
                ui.label("Test a letter").classes("text-xl font-bold")
                ui.label("Runs OCR and the decision model on a file and shows the result. Nothing is moved or saved.") \
                    .classes("le-muted text-sm")
            ui.button(icon="close", on_click=dialog.close).props("flat round")
        upload = ui.upload(label="Drop a PDF or image here, or click to choose", auto_upload=True, max_files=1) \
            .props(f'accept="{ACCEPT}" flat bordered color=primary').classes("w-full")
        busy = ui.row().classes("items-center gap-3 py-2")
        with busy:
            ui.spinner("dots", size="lg", color="primary")
            stage_label = ui.label("").classes("text-sm font-medium")
        busy.set_visibility(False)
        result = ui.column().classes("w-full gap-4")
        timer = ui.timer(0.3, lambda: stage_label.set_text(progress["stage"] or "Starting…"), active=False)
    dialog.on("hide", dialog.delete)

    async def handle_upload(e) -> None:
        workdir = Path(tempfile.mkdtemp(prefix="lettereye-test-"))
        target = workdir / Path(e.file.name).name
        await e.file.save(target)
        result.clear()
        busy.set_visibility(True)
        progress["stage"] = "Loading models (first run can take a moment)"
        timer.activate()
        try:
            analysis = await run.io_bound(c.engine.analyze_file, target,
                                          lambda stage: progress.__setitem__("stage", stage))
        except Exception as exc:
            with result:
                with ui.row().classes("items-center gap-2 p-4 rounded-xl bg-red-50 text-red-900 w-full"):
                    ui.icon("error")
                    ui.label(str(exc)).classes("text-sm")
            return
        finally:
            timer.deactivate()
            busy.set_visibility(False)
            upload.reset()
        try:
            with result:
                render_analysis(analysis, target.name)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    upload.on_upload(handle_upload)
    dialog.open()


def render_analysis(analysis: Analysis, filename: str) -> None:
    c = ctx()
    s = c.store.get()
    d = analysis.decision
    if analysis.status == "filed" and d.worker is not None:
        recipient_name = d.recipient.name if d.recipient else analysis.new_recipient_name
        folder = d.recipient.folder if d.recipient else analysis.new_recipient_name
        rel = relative_target(s, d.worker, recipient_name, folder, analysis.facts, analysis.letter_date, d.doc_type,
                              Path(filename))
        with ui.row().classes("w-full items-center gap-4 p-4 rounded-2xl no-wrap").style(
                "background: rgba(22,163,74,.10); border: 1px solid rgba(22,163,74,.25)"):
            ui.icon("task_alt", size="32px").classes("text-green-600")
            with ui.column().classes("gap-0 flex-grow"):
                ui.label(f"Would be filed for {d.worker.name} → {recipient_name}").classes("text-base font-semibold")
                ui.label(str(Path(s.output_folder or "<output>") / rel)).classes("le-kbd break-all mt-1")
            confidence_bar(analysis.confidence, "w-32")
    else:
        with ui.row().classes("w-full items-center gap-4 p-4 rounded-2xl no-wrap").style(
                "background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.3)"):
            ui.icon("rate_review", size="32px").classes("text-amber-600")
            with ui.column().classes("gap-0 flex-grow"):
                ui.label("Would go to the review queue").classes("text-base font-semibold")
                ui.label(analysis.reason).classes("text-sm")
            confidence_bar(analysis.confidence, "w-32")

    with ui.row().classes("w-full gap-6 no-wrap items-start"):
        with ui.column().classes("gap-2").style("width: 34%; min-width: 240px"):
            if analysis.preview is not None:
                ui.image(analysis.preview).classes("le-preview w-full")
            label, icon = theme.OCR_SOURCES.get(analysis.ocr.source, (analysis.ocr.source, "text_fields"))
            with ui.row().classes("gap-2 items-center"):
                ui.chip(label, icon=icon).props("dense outline")
                if analysis.ocr.device:
                    ui.chip(analysis.ocr.device.replace("ExecutionProvider", ""), icon="memory").props("dense outline")
                if analysis.escalated:
                    ui.chip("escalated to OCR LLM", icon="auto_awesome", color="accent", text_color="white").props("dense")
        with ui.column().classes("flex-grow gap-3").style("min-width: 0"):
            facts = analysis.facts
            with ui.grid(columns="130px 1fr").classes("w-full gap-y-1 gap-x-4"):
                for key, value in (("Sender", facts.sender), ("Recipient (read)", facts.recipient_name),
                                   ("Letter date", analysis.letter_date), ("Type", d.doc_type), ("Subject", facts.subject)):
                    ui.label(key).classes("le-muted text-sm")
                    ui.label(value or "–").classes("text-sm font-medium")
            for decision in d.decisions:
                if not decision.get("options"):
                    continue
                with ui.card().classes("le-card w-full p-4").props("flat"):
                    title = DECISION_TITLES.get(decision["key"], decision["key"])
                    if decision.get("round") == "decisions_after_ocr_llm":
                        title += " · after OCR LLM"
                    ui.label(title).classes("text-xs uppercase tracking-wider le-muted font-semibold")
                    probability_bars(decision)
            with ui.expansion("Recognized text & timing", icon="notes").classes("w-full"):
                ui.label(" · ".join(f"{STAGE_TITLES.get(st['name'], st['name'])}: {st.get('seconds', 0):.2f}s"
                                    for st in analysis.stages)).classes("le-muted text-xs mb-2")
                ui.label(analysis.ocr.text or "No text").classes("le-ocr w-full")
