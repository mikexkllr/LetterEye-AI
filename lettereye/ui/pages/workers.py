"""Workers and the recipients (clients) they are responsible for – replaces the old hand-made CSV files."""

from __future__ import annotations

import tempfile
from pathlib import Path

from nicegui import run, ui

from ...db import WORKER_COLORS, Recipient, Worker
from ..context import ctx
from ..widgets import card, empty_state, page_header, pick_folder, worker_avatar


def page() -> None:
    c = ctx()
    state: dict = {"selected": None, "filter": "", "recipient_filter": ""}

    with ui.column().classes("le-page"):
        with page_header("Workers", "Who handles which recipients. The decision model routes every letter to the "
                                    "worker responsible for its recipient."):
            with ui.row().classes("gap-2 no-wrap shrink-0"):
                with ui.button("Import / export", icon="import_export").props("outline color=primary"):
                    with ui.menu():
                        ui.menu_item("Import CSV file (worker, recipient, aliases)", on_click=lambda: import_csv_dialog())
                        ui.menu_item("Import old LetterEye CSV folder", on_click=lambda: import_legacy())
                        ui.separator()
                        ui.menu_item("Export to CSV…", on_click=lambda: export_csv())
                ui.button("Add worker", icon="person_add", on_click=lambda: add_worker_dialog()).props(
                    "unelevated color=primary")

        with ui.row().classes("w-full gap-5 no-wrap items-start"):
            with card("p-3").style("width: 300px; flex-shrink: 0"):
                search = ui.input(placeholder="Find worker…").props("outlined dense clearable").classes("w-full mb-2")
                with search.add_slot("prepend"):
                    ui.icon("search")
                worker_list = ui.column().classes("w-full gap-1")
            detail = ui.column().classes("flex-grow gap-5").style("min-width: 0")

    # ---------------------------------------------------------------- left: worker list
    def render_list() -> None:
        workers = c.db.list_workers()
        if state["selected"] is None and workers:
            state["selected"] = workers[0].id
        if state["selected"] is not None and not any(w.id == state["selected"] for w in workers):
            state["selected"] = workers[0].id if workers else None
        term = (search.value or "").lower()
        worker_list.clear()
        with worker_list:
            if not workers:
                ui.label("No workers yet").classes("le-muted text-sm p-3")
            for worker in workers:
                if term and term not in worker.name.lower():
                    continue
                active = worker.id == state["selected"]
                with ui.row().classes(f"le-worker-item w-full items-center gap-3 no-wrap {'active' if active else ''}") \
                        .on("click", lambda w=worker: select(w.id)):
                    worker_avatar(worker, "40px")
                    with ui.column().classes("gap-0 flex-grow").style("min-width: 0"):
                        ui.label(worker.name).classes("text-sm font-semibold truncate")
                        n = worker.recipient_count
                        ui.label(f"{n} recipient{'s' if n != 1 else ''}" + ("" if worker.active else " · paused")) \
                            .classes("le-muted text-xs")

    def select(worker_id: int) -> None:
        state["selected"] = worker_id
        state["recipient_filter"] = ""
        render_list()
        render_detail()

    search.on("update:model-value", lambda: render_list(), throttle=0.2)

    # ---------------------------------------------------------------- right: details
    def render_detail() -> None:
        detail.clear()
        worker = c.db.get_worker(state["selected"]) if state["selected"] else None
        with detail:
            if worker is None:
                with card():
                    empty_state("groups", "Add your first worker",
                                "Workers are the people letters are sorted for. Give each one the recipients "
                                "(clients, residents, customers…) they are responsible for.")
                    with ui.row().classes("w-full justify-center pb-6 gap-2"):
                        ui.button("Add worker", icon="person_add", on_click=lambda: add_worker_dialog()) \
                            .props("unelevated color=primary")
                        ui.button("Import old CSV folder", icon="upload_file", on_click=lambda: import_legacy()) \
                            .props("outline color=primary")
                return
            _worker_card(worker)
            _recipients_card(worker)

    def _worker_card(worker: Worker) -> None:
        with card("p-6"):
            with ui.row().classes("w-full items-center gap-4 no-wrap"):
                worker_avatar(worker, "64px")
                with ui.column().classes("gap-1 flex-grow"):
                    name = ui.input(value=worker.name).props("dense borderless input-class='text-2xl font-bold'") \
                        .classes("w-full")
                    name.on("blur", lambda: save(name=name.value))
                    name.on("keydown.enter", lambda: save(name=name.value))
                    with ui.row().classes("items-center gap-1"):
                        for color in WORKER_COLORS:
                            ui.element("div").classes("w-5 h-5 rounded-full cursor-pointer").style(
                                f"background:{color}; outline: {'2px solid ' + color if color == worker.color else 'none'};"
                                "outline-offset: 2px").on("click", lambda col=color: save(color=col))
                ui.switch("Active", value=worker.active, on_change=lambda e: save(active=e.value)) \
                    .tooltip("Paused workers get no letters")
                with ui.button(icon="more_vert").props("flat round"):
                    with ui.menu():
                        ui.menu_item("Delete worker…", on_click=lambda: confirm_delete(worker))
            ui.separator().classes("my-4")
            with ui.grid(columns=2).classes("w-full gap-4"):
                description = ui.textarea(
                    "Responsibilities (optional)", value=worker.description,
                    placeholder="e.g. All letters from the tax office and health insurers") \
                    .props("outlined autogrow").classes("col-span-2")
                description.on("blur", lambda: save(description=description.value))
                ui.label("The decision model uses this description to route letters whose recipient is not in "
                         "anybody's list.").classes("le-muted text-xs col-span-2 -mt-3")
                folder = ui.input("Folder name", value=worker.folder_name, placeholder=worker.name) \
                    .props("outlined dense").classes("w-full")
                folder.on("blur", lambda: save(folder_name=folder.value))
                email = ui.input("E-mail (optional)", value=worker.email).props("outlined dense").classes("w-full")
                email.on("blur", lambda: save(email=email.value))

    def save(**fields) -> None:
        if state["selected"] is None:
            return
        current = c.db.get_worker(state["selected"])
        if current is None or all(getattr(current, k) == v for k, v in fields.items()):
            return
        try:
            c.db.update_worker(state["selected"], **fields)
            c.feed.touch()
            ui.notify("Saved", type="positive", timeout=900)
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
        render_list()
        if "color" in fields or "active" in fields or "name" in fields:
            render_detail()

    def _recipients_card(worker: Worker) -> None:
        recipients = c.db.list_recipients(worker_id=worker.id)
        with card("p-6"):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.column().classes("gap-0"):
                    ui.label(f"Recipients · {len(recipients)}").classes("le-section-title")
                    ui.label("Letters addressed to these people or companies go to this worker.") \
                        .classes("le-muted text-xs")
                rsearch = ui.input(placeholder="Filter…", value=state["recipient_filter"]) \
                    .props("outlined dense clearable").classes("w-56")
            with ui.row().classes("w-full items-start gap-2 no-wrap mt-3"):
                new = ui.textarea(placeholder="Add recipient – one name per line, paste a whole list if you like") \
                    .props("outlined dense autogrow rows=1").classes("flex-grow")

                def add() -> None:
                    names = [n.strip() for n in (new.value or "").replace(";", "\n").splitlines() if n.strip()]
                    added, problems = 0, []
                    for n in names:
                        try:
                            c.db.create_recipient(worker.id, n)
                            added += 1
                        except ValueError as exc:
                            problems.append(str(exc))
                    if added:
                        ui.notify(f"Added {added} recipient{'s' if added != 1 else ''}", type="positive")
                        c.feed.touch()
                    for problem in problems[:3]:
                        ui.notify(problem, type="warning")
                    render_list()
                    render_detail()

                new.on("keydown.enter.prevent", add)
                ui.button("Add", icon="add", on_click=add).props("unelevated color=primary").style("height: 40px")

            rows = ui.column().classes("w-full gap-0 mt-3")

            def render_rows() -> None:
                term = (rsearch.value or "").lower()
                state["recipient_filter"] = rsearch.value or ""
                rows.clear()
                with rows:
                    shown = [r for r in recipients if not term or term in r.name.lower()
                             or any(term in a.lower() for a in r.aliases)]
                    if not recipients:
                        ui.label("No recipients yet. Add the people or companies this worker is responsible for.") \
                            .classes("le-muted text-sm py-6")
                    elif not shown:
                        ui.label("No match").classes("le-muted text-sm py-4")
                    for recipient in shown[:400]:
                        _recipient_row(recipient)
                    if len(shown) > 400:
                        ui.label(f"… and {len(shown) - 400} more – use the filter").classes("le-muted text-xs py-2")

            rsearch.on("update:model-value", lambda: render_rows(), throttle=0.2)
            render_rows()

    def _recipient_row(recipient: Recipient) -> None:
        with ui.row().classes("le-feed-item w-full items-center gap-3 no-wrap"):
            ui.icon("person", size="20px").classes("le-muted")
            with ui.column().classes("gap-0 flex-grow").style("min-width: 0"):
                ui.label(recipient.name).classes("text-sm font-medium truncate")
                if recipient.aliases:
                    ui.label("also: " + ", ".join(recipient.aliases)).classes("le-muted text-xs truncate")
            ui.button(icon="edit", on_click=lambda: edit_recipient_dialog(recipient)).props("flat round dense") \
                .tooltip("Edit name, spellings, move to another worker")
            ui.button(icon="delete_outline", on_click=lambda: delete_recipient(recipient)).props(
                "flat round dense color=grey").tooltip("Remove")

    # ---------------------------------------------------------------- dialogs & actions
    def add_worker_dialog() -> None:
        with ui.dialog() as dialog, ui.card().classes("le-card").style("width: 520px"):
            ui.label("Add worker").classes("text-lg font-semibold")
            name = ui.input("Name", placeholder="e.g. John Doe").props("outlined autofocus").classes("w-full")
            description = ui.textarea("Responsibilities (optional)",
                                      placeholder="e.g. Tax office, pension fund and health insurance letters") \
                .props("outlined autogrow").classes("w-full")
            recipients = ui.textarea("Recipients (optional)", placeholder="One name per line") \
                .props("outlined autogrow").classes("w-full")

            def create() -> None:
                try:
                    worker = c.db.create_worker(name.value or "", description.value or "")
                except ValueError as exc:
                    ui.notify(str(exc), type="negative")
                    return
                problems = []
                for n in (recipients.value or "").splitlines():
                    if n.strip():
                        try:
                            c.db.create_recipient(worker.id, n)
                        except ValueError as exc:
                            problems.append(str(exc))
                for problem in problems[:3]:
                    ui.notify(problem, type="warning")
                c.feed.touch()
                dialog.close()
                select(worker.id)
                ui.notify(f"Added {worker.name}", type="positive")

            name.on("keydown.enter", create)
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Add worker", icon="person_add", on_click=create).props("unelevated color=primary")
        dialog.on("hide", dialog.delete)
        dialog.open()

    def edit_recipient_dialog(recipient: Recipient) -> None:
        workers = {w.id: w.name for w in c.db.list_workers()}
        with ui.dialog() as dialog, ui.card().classes("le-card").style("width: 560px"):
            ui.label("Edit recipient").classes("text-lg font-semibold")
            name = ui.input("Name", value=recipient.name).props("outlined").classes("w-full")
            aliases = ui.input_chips("Other spellings / names", value=list(recipient.aliases), new_value_mode="add-unique") \
                .props("outlined").classes("w-full")
            ui.label("Type and press Enter. LetterEye adds spellings automatically when you correct it in the "
                     "review queue.").classes("le-muted text-xs -mt-2")
            worker = ui.select(workers, value=recipient.worker_id, label="Worker").props("outlined").classes("w-full")
            folder = ui.input("Folder name", value=recipient.folder_name, placeholder=recipient.name) \
                .props("outlined").classes("w-full")

            def store() -> None:
                try:
                    c.db.update_recipient(recipient.id, name=name.value, aliases=list(aliases.value or []),
                                          worker_id=worker.value, folder_name=folder.value or "")
                except ValueError as exc:
                    ui.notify(str(exc), type="negative")
                    return
                c.feed.touch()
                dialog.close()
                render_list()
                render_detail()

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Save", icon="check", on_click=store).props("unelevated color=primary")
        dialog.on("hide", dialog.delete)
        dialog.open()

    def delete_recipient(recipient: Recipient) -> None:
        c.db.delete_recipient(recipient.id)
        c.feed.touch()
        render_list()
        render_detail()
        ui.notify(f"Removed {recipient.name}")

    def confirm_delete(worker: Worker) -> None:
        with ui.dialog() as dialog, ui.card().classes("le-card"):
            ui.label(f"Delete {worker.name}?").classes("text-lg font-semibold")
            ui.label(f"This also removes their {worker.recipient_count} recipients. Already filed letters stay "
                     "where they are.").classes("le-muted text-sm")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def really() -> None:
                    c.db.delete_worker(worker.id)
                    state["selected"] = None
                    c.feed.touch()
                    dialog.close()
                    render_list()
                    render_detail()

                ui.button("Delete", icon="delete", on_click=really).props("unelevated color=negative")
        dialog.on("hide", dialog.delete)
        dialog.open()

    async def import_legacy() -> None:
        folder = await pick_folder("", "Folder with the old Firstname_Lastname.csv files")
        if not folder:
            return
        try:
            workers_added, recipients_added = await run.io_bound(c.db.import_legacy_csv_folder, folder)
        except Exception as exc:
            ui.notify(f"Import failed: {exc}", type="negative")
            return
        c.feed.touch()
        ui.notify(f"Imported {workers_added} workers and {recipients_added} recipients", type="positive")
        state["selected"] = None
        render_list()
        render_detail()

    def import_csv_dialog() -> None:
        with ui.dialog() as dialog, ui.card().classes("le-card").style("width: 560px"):
            ui.label("Import CSV").classes("text-lg font-semibold")
            ui.label("Columns: worker, recipient, aliases (optional, separated by ';'). A header row is optional; "
                     "comma, semicolon and tab separated files work.").classes("le-muted text-sm")

            async def handle(e) -> None:
                tmp = Path(tempfile.mkdtemp()) / "import.csv"
                await e.file.save(tmp)
                try:
                    workers_added, recipients_added = await run.io_bound(c.db.import_csv, tmp)
                except Exception as exc:
                    ui.notify(f"Import failed: {exc}", type="negative")
                    return
                c.feed.touch()
                dialog.close()
                ui.notify(f"Imported {workers_added} workers and {recipients_added} recipients", type="positive")
                render_list()
                render_detail()

            ui.upload(on_upload=handle, auto_upload=True, max_files=1, label="Choose CSV file") \
                .props('accept=".csv,.txt" flat bordered').classes("w-full")
        dialog.on("hide", dialog.delete)
        dialog.open()

    async def export_csv() -> None:
        folder = await pick_folder(c.store.get().output_folder, "Save the CSV into…")
        if not folder:
            return
        target = Path(folder) / "lettereye-workers.csv"
        count = await run.io_bound(c.db.export_csv, target)
        ui.notify(f"Exported {count} recipients to {target}", type="positive")

    render_list()
    render_detail()
