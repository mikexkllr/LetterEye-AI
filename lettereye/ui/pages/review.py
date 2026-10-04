"""Review queue: letters the decision model was not sure about. One click files them; the spelling is
remembered so the same letter goes through automatically next time."""

from __future__ import annotations

from nicegui import run, ui

from ...db import Document
from ...pipeline.names import clean_person_name
from ..context import ctx
from ..document_detail import preview_image, show_document
from ..widgets import card, confidence_bar, empty_state, on_data_change, open_path, page_header, relative_time


def page() -> None:
    c = ctx()
    shown_ids: dict[str, tuple[int, ...]] = {"ids": ()}

    with ui.column().classes("le-page"):
        page_header("Review", "Letters the AI was not sure about. Choose the right recipient and file them with one "
                              "click; new spellings are learned automatically.")
        container = ui.column().classes("w-full gap-4")

    def render() -> None:
        docs = c.db.list_documents(status="review", limit=100)
        shown_ids["ids"] = tuple(d.id for d in docs)
        container.clear()
        with container:
            if not docs:
                with card():
                    empty_state("done_all", "All caught up!", "Nothing needs your attention right now. Letters the AI "
                                "is unsure about will show up here.")
                return
            recipients = c.db.list_recipients()
            workers = c.db.list_workers()
            for doc in docs:
                _review_card(doc, recipients, workers)

    def maybe_refresh() -> None:
        ids = tuple(d.id for d in c.db.list_documents(status="review", limit=100))
        if ids != shown_ids["ids"]:
            render()

    render()
    on_data_change(maybe_refresh)


def _review_card(doc: Document, recipients, workers) -> None:
    c = ctx()
    s = c.store.get()
    trace = doc.trace
    suggestion = next((d for d in reversed(trace.get("decisions", [])) if d.get("key") == "recipient"), None)
    worker_suggestion = next((d for d in reversed(trace.get("decisions", [])) if d.get("key") == "worker"), None)
    recipient_options = {r.id: f"{r.name}  ·  {r.worker_name}" for r in recipients}
    worker_options = {w.id: w.name for w in workers}
    type_options = [t.name for t in s.document_types]

    with card("p-0 overflow-hidden"):
        with ui.row().classes("w-full no-wrap items-stretch gap-0"):
            with ui.column().classes("p-4 gap-2 items-center").style("width: 260px; background: var(--le-soft)"):
                preview_image(doc.id, doc.updated_at, "w-full cursor-pointer")
                ui.button("Details", icon="info", on_click=lambda: show_document(doc.id)).props("flat dense")
            with ui.column().classes("p-5 gap-3 flex-grow").style("min-width: 0"):
                with ui.row().classes("w-full items-start justify-between no-wrap"):
                    with ui.column().classes("gap-0").style("min-width: 0"):
                        ui.label(doc.original_name).classes("text-base font-semibold truncate")
                        ui.label(relative_time(doc.created_at)).classes("le-muted text-xs")
                    with ui.column().classes("gap-0 items-end"):
                        confidence_bar(doc.confidence)
                        ui.label("AI confidence").classes("le-muted text-[11px]")
                with ui.row().classes("items-center gap-2 px-3 py-2 rounded-xl w-full no-wrap").style(
                        "background: rgba(245,158,11,.12)"):
                    ui.icon("lightbulb", size="18px").classes("text-amber-600")
                    ui.label(doc.review_reason or "Please check this letter.").classes("text-sm")

                facts = [("From", doc.sender), ("To (as read)", doc.recipient_name), ("Date", doc.letter_date),
                         ("Subject", doc.subject)]
                with ui.row().classes("gap-x-8 gap-y-1"):
                    for key, value in facts:
                        with ui.column().classes("gap-0"):
                            ui.label(key).classes("le-muted text-xs")
                            ui.label(value or "–").classes("text-sm font-medium")

                if suggestion and suggestion.get("options"):
                    top = sorted(suggestion["options"], key=lambda o: o["p"], reverse=True)[:3]
                    with ui.row().classes("items-center gap-2"):
                        ui.label("AI candidates:").classes("le-muted text-xs")
                        for option in top:
                            ui.chip(f"{option['option'].split(' (also')[0]} · {option['p']:.0%}").props("dense outline") \
                                .classes("text-xs")

                # ------------------------------------------------------------ form
                suggested_recipient = doc.recipient_id
                if suggested_recipient is None and suggestion and suggestion.get("choice"):
                    match = next((r for r in recipients if suggestion["choice"].split(" (also")[0] == r.name), None)
                    suggested_recipient = match.id if match else None
                suggested_worker = doc.worker_id
                if suggested_worker is None and worker_suggestion and worker_suggestion.get("choice"):
                    match_w = next((w for w in workers if worker_suggestion["choice"].startswith(f"{w.name}:")), None)
                    suggested_worker = match_w.id if match_w else None

                mode = ui.toggle({"existing": "Existing recipient", "new": "New recipient"},
                                 value="existing" if (suggested_recipient or not doc.recipient_name) else "new") \
                    .props("dense no-caps unelevated toggle-color=primary").classes("mt-1")
                with ui.row().classes("w-full items-start gap-3") as existing_row:
                    recipient_select = ui.select(recipient_options, value=suggested_recipient, with_input=True,
                                                 label="Recipient").props("outlined dense options-dense") \
                        .classes("flex-grow").style("min-width: 260px")
                with ui.row().classes("w-full items-start gap-3") as new_row:
                    new_name = ui.input("Recipient name", value=doc.recipient_name).props("outlined dense") \
                        .classes("flex-grow")
                    worker_select = ui.select(worker_options, value=suggested_worker, label="Worker") \
                        .props("outlined dense").classes("w-56")
                existing_row.bind_visibility_from(mode, "value", value="existing")
                new_row.bind_visibility_from(mode, "value", value="new")

                with ui.row().classes("w-full items-center gap-3"):
                    type_select = ui.select(type_options, value=doc.doc_type if doc.doc_type in type_options else None,
                                            label="Document type").props("outlined dense").classes("w-56")
                    learn = ui.checkbox(f"Remember “{clean_person_name(doc.recipient_name)}” as a spelling", value=True) \
                        .classes("text-sm")
                    learn.set_visibility(bool(doc.recipient_name))

                async def file_it() -> None:
                    try:
                        if mode.value == "existing":
                            if not recipient_select.value:
                                ui.notify("Choose a recipient first", type="warning")
                                return
                            kwargs = {"recipient_id": recipient_select.value}
                        else:
                            if not (new_name.value or "").strip() or not worker_select.value:
                                ui.notify("Enter a name and choose a worker", type="warning")
                                return
                            kwargs = {"worker_id": worker_select.value, "new_recipient_name": new_name.value}
                        target = await run.io_bound(lambda: c.engine.assign(
                            doc.id, doc_type=type_select.value, learn_alias=learn.value, **kwargs))
                        ui.notify(f"Filed to {target.parent}", type="positive")
                    except Exception as exc:
                        ui.notify(str(exc), type="negative")

                def rerun() -> None:
                    try:
                        c.engine.reprocess(doc.id)
                        ui.notify("Queued – the AI will look at it again", type="info")
                    except Exception as exc:
                        ui.notify(str(exc), type="warning")

                with ui.row().classes("w-full items-center gap-2 pt-1"):
                    ui.button("File letter", icon="task_alt", on_click=file_it).props("unelevated color=primary")
                    ui.button("Run AI again", icon="refresh", on_click=rerun).props("flat")
                    ui.button("Open", icon="open_in_new", on_click=lambda: open_path(doc.current_path)).props("flat")
                    ui.space()
                    ui.button("Dismiss", icon="visibility_off", on_click=lambda: c.engine.dismiss(doc.id)) \
                        .props("flat color=grey")
