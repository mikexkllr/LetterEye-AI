"""First-run setup assistant: language → AI engine & models → folders → workers → start."""

from __future__ import annotations

from pathlib import Path

from nicegui import run, ui

from ... import paths
from ...ai.ollama_service import OllamaService, normalize_model_name
from ...settings import LANGUAGES, default_document_types
from ..context import ctx
from ..models_panel import ModelsPanel
from ..widgets import folder_field, pick_folder, worker_avatar


def page(first_run: bool = False) -> None:
    c = ctx()
    s = c.store.get()
    draft = {
        "language": s.language,
        "decision_model": s.decision_model,
        "ocr_llm_model": s.ocr_llm_model,
        "inbox_folder": s.inbox_folder or str(paths.default_inbox()),
        "output_folder": s.output_folder or str(paths.default_output()),
    }

    def put(key: str, value) -> None:
        draft[key] = value

    def persist(**extra) -> None:
        changes = dict(draft, **extra)
        if changes["language"] != c.store.get().language:
            changes["document_types"] = default_document_types(changes["language"])
        c.store.update(**changes)

    outer = "w-full min-h-screen items-center py-10 px-4" if first_run else "le-page items-center"
    with ui.column().classes(outer).style(
            "background: radial-gradient(circle at 10% 0%, rgba(34,211,238,.10), transparent 40%),"
            "radial-gradient(circle at 90% 10%, rgba(217,70,239,.10), transparent 40%)" if first_run else ""):
        with ui.column().classes("w-full items-center gap-6").style("max-width: 980px"):
            with ui.row().classes("items-center gap-4"):
                ui.image("/assets/icon.png").classes("w-14 h-14")
                with ui.column().classes("gap-0"):
                    ui.label("Welcome to LetterEye AI").classes("text-3xl font-bold tracking-tight")
                    ui.label("Let's get your letters sorting themselves – this takes about five minutes.") \
                        .classes("le-muted")

            with ui.stepper().props("vertical=false animated header-nav=false alternative-labels") \
                    .classes("w-full") as stepper:

                # ----------------------------------------------------------- 1 welcome
                with ui.step("Welcome", icon="waving_hand"):
                    ui.label("How LetterEye works").classes("text-xl font-semibold")
                    with ui.grid(columns=3).classes("w-full gap-4 my-2"):
                        for icon, title, text in (
                            ("document_scanner", "1 · Read",
                             "New scans in your inbox are read with fast OCR (PP-OCRv6) on your GPU."),
                            ("psychology", "2 · Decide",
                             "A local decision model picks the recipient, worker and document type – with a "
                             "probability for every choice."),
                            ("folder_special", "3 · File",
                             "Sure? Filed into Worker / Recipient. Unsure? A better OCR model re-reads it, otherwise "
                             "you decide in the review queue."),
                        ):
                            with ui.column().classes("p-5 rounded-2xl gap-2").style("background: var(--le-soft)"):
                                ui.icon(icon, size="30px").classes("le-gradient-text")
                                ui.label(title).classes("font-semibold")
                                ui.label(text).classes("text-sm le-muted")
                    with ui.row().classes("items-center gap-3 mt-2"):
                        ui.icon("lock", color="positive")
                        ui.label("Everything runs on this computer. No letter ever leaves it.").classes("text-sm font-medium")
                    ui.select(LANGUAGES, value=draft["language"], label="Language of your letters",
                              on_change=lambda e: put("language", e.value)).props("outlined").classes("w-72 mt-4")
                    with ui.stepper_navigation():
                        ui.button("Let's start", icon="arrow_forward", on_click=lambda: (persist(), stepper.next())) \
                            .props("unelevated color=primary")

                # ----------------------------------------------------------- 2 AI engine
                with ui.step("AI engine", icon="memory"):
                    ui.label("Local AI models").classes("text-xl font-semibold")
                    ui.label("LetterEye uses Ollama to run models on your GPU. Pick the models and download them; "
                             "downloads continue in the background.").classes("le-muted text-sm")
                    ModelsPanel(draft["decision_model"], draft["ocr_llm_model"], "auto", put)
                    with ui.row().classes("items-center gap-3 mt-2 p-4 rounded-2xl w-full").style("background: var(--le-soft)"):
                        ui.icon("bolt", size="26px").classes("le-gradient-text")
                        with ui.column().classes("gap-0 flex-grow"):
                            ui.label("Fast OCR · PP-OCRv6 (about 30 MB)").classes("font-semibold")
                            ocr_status = ui.label("Downloaded automatically on first use – or now:").classes("text-sm le-muted")

                        async def get_ocr() -> None:
                            ocr_status.text = "Downloading and testing…"
                            try:
                                provider = await run.io_bound(lambda: c.engine.fast_ocr(c.store.get()).warm_up())
                                ocr_status.text = f"Ready · running on {provider.replace('ExecutionProvider', '')}"
                            except Exception as exc:
                                ocr_status.text = f"Failed: {exc}"

                        ui.button("Download now", icon="download", on_click=get_ocr).props("outline color=primary")

                    async def next_from_models() -> None:
                        persist()
                        service = OllamaService(c.store.get().ollama_url)
                        installed = await run.io_bound(service.list_models)
                        missing = [m for m in (draft["decision_model"], draft["ocr_llm_model"])
                                   if normalize_model_name(m) not in {normalize_model_name(i) for i in installed}]
                        if missing:
                            ui.notify(f"Still downloading or missing: {', '.join(missing)}. You can continue – "
                                      "LetterEye starts once they are ready.", type="warning", timeout=6000)
                        stepper.next()

                    with ui.stepper_navigation():
                        ui.button("Back", on_click=stepper.previous).props("flat")
                        ui.button("Next", icon="arrow_forward", on_click=next_from_models).props("unelevated color=primary")

                # ----------------------------------------------------------- 3 folders
                with ui.step("Folders", icon="folder"):
                    ui.label("Where are your scans, and where should letters go?").classes("text-xl font-semibold")
                    ui.label("Point your scanner (or scan software) at the inbox folder. Both folders are created if "
                             "they do not exist.").classes("le-muted text-sm mb-2")
                    folder_field("Inbox – your scanner saves here", draft["inbox_folder"], lambda v: put("inbox_folder", v))
                    folder_field("Sorted letters – output", draft["output_folder"], lambda v: put("output_folder", v))

                    def next_from_folders() -> None:
                        inbox, output = Path(draft["inbox_folder"]).expanduser(), Path(draft["output_folder"]).expanduser()
                        if inbox.resolve() == output.resolve():
                            ui.notify("Inbox and output must be different folders", type="warning")
                            return
                        try:
                            inbox.mkdir(parents=True, exist_ok=True)
                            output.mkdir(parents=True, exist_ok=True)
                        except OSError as exc:
                            ui.notify(f"Cannot create folder: {exc}", type="negative")
                            return
                        draft["inbox_folder"], draft["output_folder"] = str(inbox), str(output)
                        persist()
                        stepper.next()
                        render_workers()

                    with ui.stepper_navigation():
                        ui.button("Back", on_click=stepper.previous).props("flat")
                        ui.button("Next", icon="arrow_forward", on_click=next_from_folders).props("unelevated color=primary")

                # ----------------------------------------------------------- 4 workers
                with ui.step("Workers", icon="groups"):
                    ui.label("Who gets which letters?").classes("text-xl font-semibold")
                    ui.label("Add the people letters are sorted for and the recipients each one is responsible for. "
                             "You can change all of this later on the Workers page.").classes("le-muted text-sm")
                    with ui.row().classes("w-full gap-5 no-wrap items-start mt-2"):
                        with ui.column().classes("gap-3 p-5 rounded-2xl").style("flex: 1; background: var(--le-soft)"):
                            w_name = ui.input("Worker name", placeholder="e.g. John Doe").props("outlined dense") \
                                .classes("w-full")
                            w_recipients = ui.textarea("Recipients (one per line)",
                                                       placeholder="Alice Johnson\nBob Smith\nACME GmbH") \
                                .props("outlined autogrow").classes("w-full")

                            def add_worker() -> None:
                                try:
                                    worker = c.db.create_worker(w_name.value or "")
                                except ValueError as exc:
                                    ui.notify(str(exc), type="warning")
                                    return
                                for line in (w_recipients.value or "").splitlines():
                                    if line.strip():
                                        try:
                                            c.db.create_recipient(worker.id, line)
                                        except ValueError as exc:
                                            ui.notify(str(exc), type="warning")
                                w_name.value, w_recipients.value = "", ""
                                c.feed.touch()
                                render_workers()

                            with ui.row().classes("gap-2"):
                                ui.button("Add worker", icon="person_add", on_click=add_worker) \
                                    .props("unelevated color=primary")

                                async def import_legacy() -> None:
                                    folder = await pick_folder("", "Folder with the old Firstname_Lastname.csv files")
                                    if folder:
                                        added = await run.io_bound(c.db.import_legacy_csv_folder, folder)
                                        ui.notify(f"Imported {added[0]} workers and {added[1]} recipients", type="positive")
                                        render_workers()

                                ui.button("Import old CSV folder", icon="upload_file", on_click=import_legacy) \
                                    .props("flat color=primary")
                        workers_box = ui.column().classes("gap-2").style("flex: 1")

                    def render_workers() -> None:
                        workers_box.clear()
                        with workers_box:
                            workers = c.db.list_workers()
                            if not workers:
                                ui.label("No workers yet.").classes("le-muted text-sm p-2")
                            for worker in workers:
                                names = [r.name for r in c.db.list_recipients(worker_id=worker.id)]
                                with ui.row().classes("w-full items-center gap-3 p-3 rounded-2xl no-wrap") \
                                        .style("border: 1px solid var(--le-border)"):
                                    worker_avatar(worker, "38px")
                                    with ui.column().classes("gap-0").style("min-width: 0"):
                                        ui.label(worker.name).classes("font-semibold text-sm")
                                        ui.label(", ".join(names[:6]) + (" …" if len(names) > 6 else "") or "no recipients yet") \
                                            .classes("le-muted text-xs truncate")

                    render_workers()

                    async def to_summary() -> None:
                        stepper.next()
                        await render_summary()

                    with ui.stepper_navigation():
                        ui.button("Back", on_click=stepper.previous).props("flat")
                        ui.button("Next", icon="arrow_forward", on_click=to_summary).props("unelevated color=primary")

                # ----------------------------------------------------------- 5 done
                with ui.step("Ready", icon="rocket_launch"):
                    ui.label("All set!").classes("text-xl font-semibold")
                    summary = ui.column().classes("w-full gap-2 my-2")

                    async def render_summary() -> None:
                        settings = c.store.get()
                        problems = await run.io_bound(c.engine.preflight, settings)
                        summary.clear()
                        with summary:
                            checks = [
                                ("Inbox", settings.inbox_folder, True),
                                ("Sorted letters", settings.output_folder, True),
                                ("Decision model", settings.decision_model,
                                 not any(settings.decision_model in p for p in problems)),
                                ("OCR LLM", settings.ocr_llm_model, not any(settings.ocr_llm_model in p for p in problems)),
                                ("Workers", f"{len(c.db.list_workers())} workers · {len(c.db.list_recipients())} recipients",
                                 bool(c.db.list_workers())),
                            ]
                            for title, value, ok in checks:
                                with ui.row().classes("items-center gap-3"):
                                    ui.icon("check_circle" if ok else "pending", color="positive" if ok else "warning")
                                    ui.label(title).classes("font-medium w-36")
                                    ui.label(value).classes("le-muted text-sm break-all")
                            for problem in problems:
                                ui.label(f"• {problem}").classes("text-sm text-amber-700")
                    start_now = ui.switch("Start watching the inbox now", value=True)
                    ui.label("Tip: use “Test a letter” on the dashboard to see the AI in action on any PDF.") \
                        .classes("le-muted text-sm")

                    async def finish() -> None:
                        persist(setup_completed=True, autostart=True)
                        if start_now.value:
                            problems = await run.io_bound(c.engine.start)
                            if problems:
                                ui.notify(problems[0], type="warning", timeout=8000)
                        ui.run_javascript("window.location.href = '/'")

                    with ui.stepper_navigation():
                        ui.button("Back", on_click=stepper.previous).props("flat")
                        ui.button("Open LetterEye", icon="done", on_click=finish).props("unelevated color=primary size=lg")
