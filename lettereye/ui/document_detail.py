"""Dialog showing everything about one processed letter."""

from __future__ import annotations

from pathlib import Path

from nicegui import run, ui

from ..services.engine import preview_path
from . import theme
from .context import ctx
from .widgets import confidence_bar, open_path, probability_bars, relative_time, reveal_path, status_chip

DECISION_TITLES = {
    "recipient": "Recipient",
    "verify_recipient": "Verification",
    "worker": "Responsible worker",
    "doc_type": "Document type",
}
STAGE_TITLES = {
    "text_layer": "Read PDF text layer",
    "render": "Render pages",
    "fast_ocr": "Fast OCR (PP-OCRv6)",
    "decisions": "Decision model",
    "ocr_llm": "OCR LLM (escalation)",
    "decisions_after_ocr_llm": "Decision model (2nd round)",
    "extract": "Extract sender/date/subject",
}


def preview_image(doc_id: int, updated_at: str, classes: str = "") -> None:
    path = preview_path(doc_id)
    if path.exists():
        ui.image(f"/previews/{doc_id}.png?v={updated_at}").classes(f"le-preview {classes}").props("fit=contain")
    else:
        with ui.column().classes(f"le-preview items-center justify-center {classes}").style("min-height: 240px"):
            ui.icon("description", size="48px").classes("le-muted")
            ui.label("No preview").classes("le-muted text-sm")


def show_document(doc_id: int) -> None:
    doc = ctx().db.get_document(doc_id)
    if doc is None:
        ui.notify("Document not found", type="warning")
        return
    with ui.dialog() as dialog, ui.card().classes("le-card").style("width: 1080px; max-width: 96vw"):
        with ui.row().classes("w-full items-start justify-between no-wrap"):
            with ui.column().classes("gap-1"):
                ui.label(doc.original_name).classes("text-lg font-semibold break-all")
                with ui.row().classes("items-center gap-2"):
                    status_chip(doc.status)
                    ui.label(relative_time(doc.created_at)).classes("le-muted text-sm")
                    if doc.ocr_source:
                        label, icon = theme.OCR_SOURCES.get(doc.ocr_source, (doc.ocr_source, "text_fields"))
                        ui.chip(label, icon=icon).props("dense outline").classes("text-xs")
                    if doc.trace.get("escalated"):
                        ui.chip("escalated to OCR LLM", icon="auto_awesome", color="accent", text_color="white") \
                            .props("dense").classes("text-xs")
            ui.button(icon="close", on_click=dialog.close).props("flat round")

        with ui.row().classes("w-full gap-6 no-wrap items-start"):
            with ui.column().classes("gap-2").style("width: 38%; min-width: 260px"):
                preview_image(doc.id, doc.updated_at, "w-full")
                with ui.row().classes("gap-2"):
                    ui.button("Open", icon="open_in_new", on_click=lambda: open_path(doc.current_path)).props("flat")
                    ui.button("Show in folder", icon="folder", on_click=lambda: reveal_path(doc.current_path)).props("flat")
                _verdict_box(doc, dialog)

            with ui.column().classes("flex-grow gap-2").style("min-width: 0"):
                with ui.tabs().classes("w-full").props("align=left dense") as tabs:
                    t_details = ui.tab("Details", icon="info")
                    t_ai = ui.tab("AI decisions", icon="psychology")
                    t_text = ui.tab("Text", icon="notes")
                    t_time = ui.tab("Timeline", icon="timeline")
                with ui.tab_panels(tabs, value=t_details).classes("w-full bg-transparent"):
                    with ui.tab_panel(t_details).classes("p-0 pt-2"):
                        _details(doc)
                    with ui.tab_panel(t_ai).classes("p-0 pt-2"):
                        decisions = [d for d in doc.trace.get("decisions", []) if d.get("options")]
                        if not decisions:
                            ui.label("No decisions recorded.").classes("le-muted")
                        rounds = sorted({d.get("round", "decisions") for d in decisions}, key=lambda r: r != "decisions")
                        for round_name in rounds:
                            if len(rounds) > 1:
                                ui.label(STAGE_TITLES.get(round_name, round_name)).classes("le-section-title mt-2")
                            for decision in (d for d in decisions if d.get("round", "decisions") == round_name):
                                with ui.card().classes("le-card w-full p-4").props("flat"):
                                    ui.label(DECISION_TITLES.get(decision["key"], decision["key"])).classes(
                                        "text-xs uppercase tracking-wider le-muted font-semibold")
                                    probability_bars(decision)
                    with ui.tab_panel(t_text).classes("p-0 pt-2"):
                        ui.label(doc.ocr_text or "No text").classes("le-ocr w-full")
                    with ui.tab_panel(t_time).classes("p-0 pt-2"):
                        with ui.timeline(side="right").props("dense color=primary"):
                            for stage in doc.trace.get("stages", []):
                                extra = ", ".join(f"{k}: {v}" for k, v in stage.items() if k not in ("name", "seconds"))
                                ui.timeline_entry(extra, title=STAGE_TITLES.get(stage["name"], stage["name"]),
                                                  subtitle=f"{stage.get('seconds', 0):.2f} s")
                        if doc.duration_ms:
                            ui.label(f"Total: {doc.duration_ms / 1000:.1f} s").classes("le-muted text-sm")
    dialog.on("hide", dialog.delete)
    dialog.open()


def _verdict_box(doc, dialog) -> None:
    """Human feedback on the AI's decision: confirm or fix filed letters, jump to approvals for waiting ones."""
    c = ctx()
    if doc.status in ("pending", "review"):
        def go() -> None:
            dialog.close()
            ui.navigate.to("/approvals")

        ui.button("Decide in Approvals", icon="fact_check", on_click=go).props("unelevated color=primary no-caps")
        return
    if doc.status != "filed":
        return
    last = c.db.last_feedback(doc.id)
    with ui.column().classes("w-full gap-2 p-3 rounded-2xl").style("background: var(--le-soft)"):
        if last is not None and last.verified:
            with ui.row().classes("items-center gap-2"):
                ui.icon("verified", color="positive")
                label = {"accepted": "Accepted by you", "confirmed": "Confirmed by you",
                         "corrected": f"Corrected by you ({', '.join(last.changed)})"}[last.action]
                ui.label(label).classes("text-sm font-medium")
        else:
            ui.label("Filed automatically – was this right?").classes("text-sm font-medium")
        with ui.row().classes("gap-2"):
            if last is None or not last.verified:
                async def confirm() -> None:
                    await run.io_bound(c.engine.confirm, doc.id)
                    ui.notify("Thanks – recorded as correct", type="positive")
                    dialog.close()

                ui.button("Yes, correct", icon="thumb_up", on_click=confirm).props("unelevated color=positive no-caps dense")
            ui.button("Fix it", icon="edit", on_click=lambda: _fix_dialog(doc.id, dialog)) \
                .props("outline color=primary no-caps dense")


def _fix_dialog(doc_id: int, parent) -> None:
    from .proposal_editor import ProposalEditor

    doc = ctx().db.get_document(doc_id)
    with ui.dialog() as dialog, ui.card().classes("le-card").style("width: 520px; max-width: 95vw"):
        ui.label("Fix this letter").classes("text-lg font-semibold")
        ui.label("The file is moved to the corrected place and the correction is recorded for training.") \
            .classes("le-muted text-sm")

        def done(message: str, _doc_id: int) -> None:
            ui.notify(message, type="positive")
            dialog.close()
            parent.close()

        ProposalEditor(doc, on_done=done, compact=True)
    dialog.on("hide", dialog.delete)
    dialog.open()


def _details(doc) -> None:
    facts = doc.trace.get("facts", {})
    rows = [
        ("Worker", doc.worker_name or "–"),
        ("Recipient", doc.recipient_label or (f"{doc.recipient_name} (not in list)" if doc.recipient_name else "–")),
        ("Sender", doc.sender or "–"),
        ("Letter date", doc.letter_date or "–"),
        ("Type", doc.doc_type or "–"),
        ("Subject", doc.subject or "–"),
        ("Signed by", facts.get("sender_person") or "–"),
    ]
    with ui.grid(columns="140px 1fr").classes("w-full gap-y-2 gap-x-4 items-center"):
        for key, value in rows:
            ui.label(key).classes("le-muted text-sm")
            ui.label(value).classes("text-sm font-medium break-words")
        ui.label("Confidence").classes("le-muted text-sm")
        confidence_bar(doc.confidence, "w-40")
        ui.label("Location").classes("le-muted text-sm")
        ui.label(str(Path(doc.current_path))).classes("text-sm break-all le-kbd")
    if doc.review_reason and doc.status == "review":
        with ui.row().classes("items-center gap-2 mt-3 p-3 rounded-xl bg-amber-50 text-amber-900 w-full"):
            ui.icon("info")
            ui.label(doc.review_reason).classes("text-sm")
    if doc.error:
        with ui.row().classes("items-center gap-2 mt-3 p-3 rounded-xl bg-red-50 text-red-900 w-full"):
            ui.icon("error")
            ui.label(doc.error).classes("text-sm")
