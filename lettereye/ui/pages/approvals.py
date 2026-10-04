"""Approvals – the human-in-the-loop queue.

Every letter the AI has processed but not filed waits here: confident proposals (approval mode) and letters
the AI was unsure about. One look at the page, one key press to accept; corrections are recorded as
training data and spellings are learned.
"""

from __future__ import annotations

from nicegui import run, ui

from ...db import Document
from .. import theme
from ..context import ctx
from ..document_detail import preview_image
from ..proposal_editor import ProposalEditor
from ..widgets import card, empty_state, on_data_change, open_path, relative_time

FILTERS = {"all": "All", "pending": "Proposals", "review": "Unsure"}
AUTO_ACCEPT_LEVEL = 0.95


def mode_toggle() -> ui.toggle:
    c = ctx()

    def change(e) -> None:
        c.store.update(workflow_mode=e.value)
        ui.notify("Every letter now waits for your approval" if e.value == "approve"
                  else "Confident letters are filed automatically; only unsure ones wait for you", type="info")

    return ui.toggle({"approve": "Approve each letter", "automatic": "Automatic"}, value=c.store.get().workflow_mode,
                     on_change=change).props("unelevated no-caps toggle-color=primary rounded").classes("le-mode-toggle")


def page() -> None:
    c = ctx()
    state: dict = {"selected": None, "filter": "all", "ids": (), "editor": None, "last_done": None, "snack_timer": None}

    with ui.column().classes("le-page le-page-wide"):
        with ui.row().classes("w-full items-end justify-between gap-4 no-wrap"):
            with ui.column().classes("gap-1"):
                ui.label("Approvals").classes("le-title")
                ui.label("Check what the AI proposes, then accept or correct it. Nothing is filed without you in "
                         "approval mode.").classes("le-subtitle")
            mode_toggle()
        with ui.row().classes("w-full items-center gap-3"):
            filters = ui.toggle(FILTERS, value="all", on_change=lambda e: set_filter(e.value)) \
                .props("dense unelevated no-caps toggle-color=primary")
            counts = ui.row().classes("items-center gap-2")
            ui.space()
            bulk = ui.button("", icon="done_all", on_click=lambda: accept_all()).props("outline color=positive no-caps")
        body = ui.row().classes("w-full gap-4 no-wrap items-stretch le-approvals")
        with ui.row().classes("w-full justify-center gap-4 le-muted text-xs"):
            for key, text in (("⏎", "accept"), ("↑ ↓", "previous / next"), ("E", "change recipient"), ("U", "undo")):
                with ui.row().classes("items-center gap-1"):
                    ui.label(key).classes("le-kbd")
                    ui.label(text)

    with ui.element("div").classes("le-snackbar") as snackbar:
        snack_icon = ui.icon("task_alt", size="20px")
        snack_text = ui.label("").classes("text-sm font-medium")
        snack_undo = ui.button("Undo", on_click=lambda: undo_last()).props("flat dense no-caps color=white")
    snackbar.set_visibility(False)

    # ------------------------------------------------------------------ data
    def waiting() -> list[Document]:
        status = ["pending", "review"] if state["filter"] == "all" else state["filter"]
        return c.db.list_documents(status=status, limit=500, oldest_first=True)

    def set_filter(value: str) -> None:
        state["filter"] = value
        state["selected"] = None
        render()

    # ------------------------------------------------------------------ rendering
    def render() -> None:
        docs = waiting()
        state["ids"] = tuple(d.id for d in docs)
        if state["selected"] not in state["ids"]:
            state["selected"] = docs[0].id if docs else None
        all_waiting = c.db.list_documents(status=["pending", "review"], limit=1000)
        n_pending = sum(1 for d in all_waiting if d.status == "pending")
        n_review = len(all_waiting) - n_pending
        counts.clear()
        with counts:
            ui.chip(f"{n_pending} proposal{'s' if n_pending != 1 else ''}", icon="pending_actions").props("dense outline color=primary")
            ui.chip(f"{n_review} unsure", icon="help_outline").props("dense outline color=amber-8")
        sure = [d for d in all_waiting if d.status == "pending" and (d.confidence or 0) >= AUTO_ACCEPT_LEVEL]
        bulk.text = f"Accept all ≥ {AUTO_ACCEPT_LEVEL:.0%} ({len(sure)})"
        bulk.set_visibility(bool(sure))

        body.clear()
        with body:
            if not docs:
                with card("items-center justify-center").style("height: 100%"):
                    empty_state("auto_awesome", "All caught up!",
                                "New letters appear here as soon as the AI has read them. "
                                + ("Every letter waits for your approval." if c.store.get().workflow_mode == "approve"
                                   else "Confident letters are filed automatically; unsure ones wait here."))
                state["editor"] = None
                return
            with ui.column().classes("le-card le-queue p-2 gap-1"):
                for doc in docs:
                    queue_item(doc)
            selected = next(d for d in docs if d.id == state["selected"])
            with ui.column().classes("le-card p-3 gap-2 flex-grow le-preview-pane"):
                preview_pane(selected)
            with ui.scroll_area().classes("le-card le-decision-pane"):
                with ui.column().classes("w-full p-2"):
                    state["editor"] = ProposalEditor(selected, on_done=done)

    def queue_item(doc: Document) -> None:
        active = doc.id == state["selected"]
        threshold = c.store.get().confidence_threshold
        with ui.row().classes(f"le-queue-item w-full items-center gap-3 no-wrap {'active' if active else ''}") \
                .on("click", lambda d=doc.id: select(d)):
            preview_image(doc.id, doc.updated_at, "le-thumb")
            with ui.column().classes("gap-0 flex-grow").style("min-width: 0"):
                target = " / ".join(x for x in (doc.worker_name, doc.recipient_label) if x)
                ui.label(target or doc.recipient_name or "Unknown recipient").classes("text-sm font-semibold truncate")
                ui.label(f"{doc.sender or doc.original_name}").classes("le-muted text-xs truncate")
                with ui.row().classes("items-center gap-1 mt-1"):
                    conf = doc.confidence or 0
                    tone = "le-pill-good" if conf >= threshold else "le-pill-warn"
                    ui.label(f"{conf:.0%}").classes(f"le-pill {tone}")
                    ui.label("proposal" if doc.status == "pending" else "unsure").classes(
                        f"le-pill {'le-pill-info' if doc.status == 'pending' else 'le-pill-warn'}")
                    ui.label(relative_time(doc.created_at)).classes("le-muted text-[11px]")

    def preview_pane(doc: Document) -> None:
        zoom = {"on": False}
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            with ui.column().classes("gap-0").style("min-width: 0"):
                ui.label(doc.original_name).classes("text-sm font-semibold truncate")
                label, icon = theme.OCR_SOURCES.get(doc.ocr_source, (doc.ocr_source or "–", "text_fields"))
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"received {relative_time(doc.created_at)}").classes("le-muted text-xs")
                    ui.chip(label, icon=icon).props("dense outline").classes("text-[11px]")
                    if doc.proposal.get("escalated"):
                        ui.chip("re-read by OCR LLM", icon="auto_awesome").props("dense outline color=accent") \
                            .classes("text-[11px]")
            with ui.row().classes("gap-1"):
                zoom_button = ui.button(icon="zoom_in", on_click=lambda: toggle_zoom()).props("flat round dense") \
                    .tooltip("Zoom")
                ui.button(icon="open_in_new", on_click=lambda: open_path(doc.current_path)).props("flat round dense") \
                    .tooltip("Open the file")
        with ui.scroll_area().classes("w-full flex-grow le-preview-scroll"):
            holder = ui.column().classes("w-full items-center")
            with holder:
                preview_image(doc.id, doc.updated_at, "le-page-preview")

        def toggle_zoom() -> None:
            zoom["on"] = not zoom["on"]
            holder.classes(add="le-zoomed") if zoom["on"] else holder.classes(remove="le-zoomed")
            zoom_button.props(f"icon={'zoom_out' if zoom['on'] else 'zoom_in'}")

    def select(doc_id: int) -> None:
        state["selected"] = doc_id
        render()

    def move(step: int) -> None:
        ids = state["ids"]
        if not ids:
            return
        index = ids.index(state["selected"]) if state["selected"] in ids else 0
        state["selected"] = ids[max(0, min(len(ids) - 1, index + step))]
        render()

    # ------------------------------------------------------------------ actions
    def done(message: str, doc_id: int) -> None:
        ids = list(state["ids"])
        if doc_id in ids:
            index = ids.index(doc_id)
            remaining = ids[:index] + ids[index + 1:]
            state["selected"] = remaining[min(index, len(remaining) - 1)] if remaining else None
        show_snack(message, doc_id)
        render()

    def show_snack(message: str, doc_id: int | None, icon: str = "task_alt") -> None:
        state["last_done"] = doc_id
        snack_text.text = message
        snack_icon.props(f"name={icon}")
        snack_undo.set_visibility(doc_id is not None)
        snackbar.set_visibility(True)
        if state["snack_timer"] is not None:
            state["snack_timer"].cancel()
        state["snack_timer"] = ui.timer(8, lambda: snackbar.set_visibility(False), once=True)

    async def undo_last() -> None:
        doc_id = state["last_done"]
        if doc_id is None:
            return
        try:
            await run.io_bound(c.engine.undo, doc_id)
        except Exception as exc:
            ui.notify(str(exc), type="warning")
            return
        snackbar.set_visibility(False)
        state["last_done"] = None
        state["filter"] = "all"
        filters.value = "all"
        state["selected"] = doc_id
        render()

    async def accept_all() -> None:
        sure = [d for d in c.db.list_documents(status="pending", limit=1000)
                if (d.confidence or 0) >= AUTO_ACCEPT_LEVEL]
        if not sure:
            return
        with ui.dialog() as dialog, ui.card().classes("le-card").style("width: 460px"):
            ui.label(f"Accept {len(sure)} proposals?").classes("text-lg font-semibold")
            ui.label(f"All proposals the AI is at least {AUTO_ACCEPT_LEVEL:.0%} sure about are filed as proposed.") \
                .classes("le-muted text-sm")
            with ui.column().classes("w-full gap-0 max-h-60 overflow-auto"):
                for d in sure[:50]:
                    ui.label(f"• {d.worker_name} / {d.recipient_label} – {d.sender or d.original_name}").classes("text-sm")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
                ui.button("Accept all", icon="done_all", on_click=lambda: dialog.submit(True)) \
                    .props("unelevated color=positive")
        if not await dialog:
            return
        count, problems = await run.io_bound(c.engine.approve_many, [d.id for d in sure])
        for problem in problems[:3]:
            ui.notify(problem, type="warning")
        show_snack(f"Filed {count} letters", None, "done_all")
        render()

    def handle_key(e) -> None:
        if not e.action.keydown or e.modifiers.ctrl or e.modifiers.meta or e.modifiers.alt:
            return
        key = e.key.name if hasattr(e.key, "name") else str(e.key)
        editor: ProposalEditor | None = state["editor"]
        if key in ("Enter", "a", "A") and editor is not None:
            ui.timer(0, editor.accept, once=True)
        elif key in ("ArrowDown", "j", "J"):
            move(1)
        elif key in ("ArrowUp", "k", "K"):
            move(-1)
        elif key in ("e", "E") and editor is not None:
            editor.focus_recipient()
        elif key in ("u", "U"):
            ui.timer(0, undo_last, once=True)

    ui.keyboard(on_key=handle_key)

    def maybe_refresh() -> None:
        ids = tuple(d.id for d in waiting())
        if ids != state["ids"]:
            render()

    render()
    on_data_change(maybe_refresh)
