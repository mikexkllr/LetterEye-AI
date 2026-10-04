"""Settings, grouped in tabs, with plain-language explanations. Saved explicitly (the watcher restarts)."""

from __future__ import annotations

from nicegui import run, ui

from ... import __version__, paths
from ...ai.ollama_service import OllamaService
from ...gpu import ONNX_DEVICE_LABELS, best_onnx_device, detect_gpus, onnx_providers
from ...pipeline.filing import FILENAME_TOKENS, FOLDER_TOKENS, FilingValues, build_relative_path
from ...settings import LANGUAGES, DocumentType, Settings, default_document_types
from ..context import ctx
from ..models_panel import ModelsPanel
from ..widgets import card, folder_field, open_path, page_header


def page() -> None:
    c = ctx()
    draft = c.store.get().model_dump()
    dirty = {"value": False}

    def set_(key: str, value) -> None:
        if draft.get(key) != value:
            draft[key] = value
            dirty["value"] = True
            footer.set_visibility(True)

    with ui.column().classes("le-page"):
        page_header("Settings", "Everything LetterEye needs, in one place. Changes apply after you save.")
        with ui.tabs().props("align=left inline-label no-caps").classes("w-full") as tabs:
            t_folders = ui.tab("Folders", icon="folder")
            t_models = ui.tab("AI models", icon="psychology")
            t_ocr = ui.tab("OCR & GPU", icon="memory")
            t_decisions = ui.tab("Decisions", icon="rule")
            t_filing = ui.tab("File names", icon="drive_file_rename_outline")
            t_types = ui.tab("Document types", icon="category")
            t_general = ui.tab("General", icon="settings")

        with ui.tab_panels(tabs, value=t_folders).classes("w-full bg-transparent").props("animated=false"):
            # ------------------------------------------------------------------ folders
            with ui.tab_panel(t_folders).classes("p-0"), card("p-6 gap-4"):
                ui.label("Where letters come from and where they go").classes("le-section-title")
                folder_field("Inbox – the folder your scanner saves to", draft["inbox_folder"],
                             lambda v: set_("inbox_folder", v), "New PDFs and images here are processed automatically.")
                folder_field("Sorted letters – the output folder", draft["output_folder"],
                             lambda v: set_("output_folder", v), "Letters are filed here as Worker / Recipient / file.")
                with ui.row().classes("w-full gap-4"):
                    ui.input("Review folder name", value=draft["review_folder_name"],
                             on_change=lambda e: set_("review_folder_name", e.value)).props("outlined dense").classes("w-60")
                    ui.input("Failed folder name", value=draft["failed_folder_name"],
                             on_change=lambda e: set_("failed_folder_name", e.value)).props("outlined dense").classes("w-60")
                ui.label("Uncertain and failed letters are kept in these subfolders of the output folder.") \
                    .classes("le-muted text-xs -mt-3")
                ui.separator()
                ui.radio({True: "Move letters out of the inbox (recommended)",
                          False: "Copy letters and leave the originals in the inbox"},
                         value=draft["move_files"], on_change=lambda e: set_("move_files", e.value))
                ui.switch("Also process files that are already in the inbox when watching starts",
                          value=draft["process_existing_on_start"],
                          on_change=lambda e: set_("process_existing_on_start", e.value))
                ui.switch("Watch subfolders of the inbox too", value=draft["recursive_watch"],
                          on_change=lambda e: set_("recursive_watch", e.value))

            # ------------------------------------------------------------------ models
            with ui.tab_panel(t_models).classes("p-0"), card("p-6 gap-4"):
                with ui.row().classes("w-full items-end gap-3"):
                    url = ui.input("Ollama address", value=draft["ollama_url"],
                                   on_change=lambda e: set_("ollama_url", e.value)).props("outlined dense").classes("w-80")

                    async def test_connection() -> None:
                        version = await run.io_bound(OllamaService(url.value).version)
                        ui.notify(f"Connected to Ollama {version}" if version else "Ollama is not reachable at that address",
                                  type="positive" if version else "negative")

                    ui.button("Test", icon="lan", on_click=test_connection).props("outline color=primary")
                ModelsPanel(draft["decision_model"], draft["ocr_llm_model"], draft["ocr_mode"], set_)
                with ui.expansion("Advanced", icon="tune").classes("w-full"):
                    with ui.row().classes("gap-4 items-start"):
                        ui.input("Extraction model (empty = decision model)", value=draft["extraction_model"],
                                 on_change=lambda e: set_("extraction_model", e.value)).props("outlined dense").classes("w-80")
                        ui.input("Keep models loaded for", value=draft["keep_alive"],
                                 on_change=lambda e: set_("keep_alive", e.value)).props(
                            'outlined dense hint="e.g. 15m, 1h, -1 = forever"').classes("w-56")
                        ui.select([4096, 8192, 16384, 32768], value=draft["num_ctx"], label="Context window (tokens)",
                                  on_change=lambda e: set_("num_ctx", e.value)).props("outlined dense").classes("w-56")

            # ------------------------------------------------------------------ OCR & GPU
            with ui.tab_panel(t_ocr).classes("p-0"), card("p-6 gap-4"):
                ui.label("How letters are read").classes("le-section-title")
                ui.radio({
                    "auto": "Smart (recommended): fast GPU OCR first, the OCR LLM only when the decision model is unsure",
                    "fast_only": "Fast only: never use the OCR LLM (lowest GPU memory)",
                    "llm_always": "Best quality: always read scans with the OCR LLM (slower)",
                }, value=draft["ocr_mode"], on_change=lambda e: set_("ocr_mode", e.value))
                ui.separator()
                ui.label("Fast OCR · PP-OCRv6").classes("le-section-title")
                with ui.row().classes("gap-6 items-start"):
                    with ui.column().classes("gap-1"):
                        ui.label("Model size").classes("text-sm le-muted")
                        ui.toggle({"tiny": "Tiny", "small": "Small", "medium": "Medium"}, value=draft["fast_ocr_size"],
                                  on_change=lambda e: set_("fast_ocr_size", e.value)).props("no-caps unelevated toggle-color=primary")
                        ui.label("Small is fast and accurate; Medium is a little better on poor scans and wants a GPU.") \
                            .classes("le-muted text-xs w-80")
                    with ui.column().classes("gap-1"):
                        ui.label("Run on").classes("text-sm le-muted")
                        ui.toggle({"auto": "Auto", "gpu": "GPU", "cpu": "CPU"}, value=draft["fast_ocr_device"],
                                  on_change=lambda e: set_("fast_ocr_device", e.value)).props("no-caps unelevated toggle-color=primary")
                        device = best_onnx_device(draft["fast_ocr_device"])
                        ui.label(f"Will use: {ONNX_DEVICE_LABELS.get(device, device)}").classes("text-xs font-medium")
                        ui.label("Available: " + (", ".join(p.replace("ExecutionProvider", "") for p in onnx_providers())
                                                  or "none")).classes("le-muted text-xs")
                with ui.row().classes("gap-4 items-center"):
                    ui.number("Pages to read per letter", value=draft["max_pages"], min=1, max=10, step=1,
                              on_change=lambda e: set_("max_pages", int(e.value or 1))).props("outlined dense").classes("w-56")
                    ui.select({150: "150 dpi (fast)", 200: "200 dpi (recommended)", 300: "300 dpi (small print)"},
                              value=draft["render_dpi"], label="Scan resolution for OCR",
                              on_change=lambda e: set_("render_dpi", e.value)).props("outlined dense").classes("w-56")

                    async def download_ocr() -> None:
                        n = ui.notification("Downloading and loading PP-OCRv6…", spinner=True, timeout=None)
                        try:
                            provider = await run.io_bound(lambda: c.engine.fast_ocr(Settings.model_validate(draft)).warm_up())
                            n.message, n.spinner, n.type = f"PP-OCRv6 ready on {provider}", False, "positive"
                        except Exception as exc:
                            n.message, n.spinner, n.type = f"Failed: {exc}", False, "negative"
                        n.timeout = 4

                    ui.button("Download / test fast OCR", icon="download", on_click=download_ocr).props("outline color=primary")
                ui.separator()
                gpus = detect_gpus()
                ui.label("Detected GPU").classes("le-section-title")
                if gpus:
                    for g in gpus:
                        ui.label(f"{g.name}" + (f" · {g.memory_total_gb} GB" if g.memory_total_gb else "")).classes("text-sm")
                else:
                    ui.label("No GPU detected. Everything will work, but slowly on the CPU.").classes("text-sm le-muted")
                ui.label("Ollama uses the GPU automatically. The dashboard shows whether each model fits completely "
                         "into GPU memory.").classes("le-muted text-xs")

            # ------------------------------------------------------------------ decisions
            with ui.tab_panel(t_decisions).classes("p-0"), card("p-6 gap-4"):
                ui.label("How sure must the AI be?").classes("le-section-title")
                ui.markdown(
                    "The decision model works like a **System One model**: it never writes free text, it picks one of "
                    "the options it is given (your recipients, your workers, your document types) and reports a "
                    "probability. Letters below the threshold go to the **Review** queue."
                ).classes("text-sm le-muted")
                with ui.row().classes("w-full items-center gap-4 no-wrap"):
                    threshold_label = ui.label(f"{draft['confidence_threshold']:.0%}").classes("text-2xl font-bold w-20")
                    ui.slider(min=0.5, max=0.99, step=0.01, value=draft["confidence_threshold"],
                              on_change=lambda e: (set_("confidence_threshold", round(e.value, 2)),
                                                   threshold_label.set_text(f"{e.value:.0%}"))) \
                        .props("label-always=false color=primary").classes("flex-grow")
                ui.label("Lower = more automatic, higher = more letters to check by hand. 80% is a good start.") \
                    .classes("le-muted text-xs -mt-2")
                ui.separator()
                ui.switch("Double-check the recipient with a yes/no question", value=draft["verify_recipient"],
                          on_change=lambda e: set_("verify_recipient", e.value))
                ui.switch("Ask every question twice with shuffled options (removes position bias)",
                          value=draft["debias_order"], on_change=lambda e: set_("debias_order", e.value))
                ui.switch("Route letters for unknown recipients by the workers' responsibility descriptions",
                          value=draft["route_unknown_by_description"],
                          on_change=lambda e: set_("route_unknown_by_description", e.value))
                ui.switch("Then add the unknown recipient to that worker automatically",
                          value=draft["auto_create_recipients"],
                          on_change=lambda e: set_("auto_create_recipients", e.value))

            # ------------------------------------------------------------------ filing
            with ui.tab_panel(t_filing).classes("p-0"), card("p-6 gap-4"):
                ui.label("Folder and file names").classes("le-section-title")
                folder_input = ui.input("Folder structure", value=draft["folder_template"],
                                        on_change=lambda e: (set_("folder_template", e.value), update_preview())) \
                    .props("outlined").classes("w-full")
                _token_row(FOLDER_TOKENS, folder_input)
                file_input = ui.input("File name", value=draft["filename_template"],
                                      on_change=lambda e: (set_("filename_template", e.value), update_preview())) \
                    .props("outlined").classes("w-full")
                _token_row(FILENAME_TOKENS, file_input)
                with ui.row().classes("items-center gap-2 p-3 rounded-xl w-full").style("background: var(--le-soft)"):
                    ui.icon("visibility", size="18px").classes("le-muted")
                    ui.label("Example:").classes("text-sm le-muted")
                    preview = ui.label("").classes("le-kbd break-all")

                def update_preview() -> None:
                    values = FilingValues(worker="John Doe", recipient="Bob Smith", date="2026-03-14",
                                          sender="Stadtwerke München", doc_type="Rechnung",
                                          subject="Jahresabrechnung Strom 2025", original="scan_0042")
                    rel = build_relative_path(draft["folder_template"], draft["filename_template"], values, ".pdf")
                    preview.text = str(rel)

                update_preview()

            # ------------------------------------------------------------------ document types
            with ui.tab_panel(t_types).classes("p-0"), card("p-6 gap-3"):
                ui.label("Document types").classes("le-section-title")
                ui.label("The decision model picks one of these for every letter. The description helps it choose.") \
                    .classes("le-muted text-sm")
                types_box = ui.column().classes("w-full gap-2")

                def render_types() -> None:
                    types_box.clear()
                    with types_box:
                        for index, t in enumerate(draft["document_types"]):
                            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                                ui.input(value=t["name"], on_change=lambda e, i=index: _edit_type(i, "name", e.value)) \
                                    .props("outlined dense").classes("w-48")
                                ui.input(value=t["description"], placeholder="Description",
                                         on_change=lambda e, i=index: _edit_type(i, "description", e.value)) \
                                    .props("outlined dense").classes("flex-grow")
                                ui.button(icon="delete_outline", on_click=lambda i=index: _remove_type(i)) \
                                    .props("flat round dense color=grey")

                def _edit_type(index: int, key: str, value: str) -> None:
                    types = [dict(t) for t in draft["document_types"]]
                    types[index][key] = value
                    set_("document_types", types)

                def _remove_type(index: int) -> None:
                    set_("document_types", [t for i, t in enumerate(draft["document_types"]) if i != index])
                    render_types()

                def _add_type() -> None:
                    set_("document_types", [*draft["document_types"], {"name": "New type", "description": ""}])
                    render_types()

                def _reset_types() -> None:
                    set_("document_types", [t.model_dump() for t in default_document_types(draft["language"])])
                    render_types()

                render_types()
                with ui.row().classes("gap-2"):
                    ui.button("Add type", icon="add", on_click=_add_type).props("outline color=primary")
                    ui.button("Reset to defaults", icon="restart_alt", on_click=_reset_types).props("flat")

            # ------------------------------------------------------------------ general
            with ui.tab_panel(t_general).classes("p-0"), card("p-6 gap-4"):
                with ui.row().classes("gap-4 items-start"):
                    ui.select(LANGUAGES, value=draft["language"], label="Language of your letters",
                              on_change=lambda e: set_("language", e.value)).props("outlined dense").classes("w-64")
                    ui.select({"auto": "Follow system", "light": "Light", "dark": "Dark"}, value=draft["theme"],
                              label="Appearance", on_change=lambda e: set_("theme", e.value)).props("outlined dense").classes("w-64")
                ui.switch("Start watching automatically when LetterEye opens", value=draft["autostart"],
                          on_change=lambda e: set_("autostart", e.value))
                ui.separator()
                with ui.row().classes("gap-2"):
                    ui.button("Run the setup assistant", icon="auto_fix_high", on_click=lambda: ui.navigate.to("/setup")) \
                        .props("outline color=primary")
                    ui.button("Open data folder", icon="folder_open", on_click=lambda: open_path(paths.data_dir())).props("flat")
                    ui.button("Open logs", icon="receipt_long", on_click=lambda: open_path(paths.log_dir())).props("flat")
                ui.label(f"LetterEye AI {__version__} · data: {paths.data_dir()}").classes("le-muted text-xs")

    # ---------------------------------------------------------------------- save bar
    def discard() -> None:
        ui.navigate.reload()

    async def save() -> None:
        try:
            settings = Settings.model_validate({**draft, "document_types": [DocumentType(**t) if isinstance(t, dict) else t
                                                                            for t in draft["document_types"]]})
        except Exception as exc:
            ui.notify(f"Cannot save: {exc}", type="negative")
            return
        old_theme = c.store.get().theme
        await run.io_bound(c.store.save, settings)
        dirty["value"] = False
        footer.set_visibility(False)
        problems = settings.problems()
        ui.notify("Saved" + (" – the watcher restarts with the new settings" if c.engine.state == "running" else ""),
                  type="positive")
        for problem in problems[:2]:
            ui.notify(problem, type="warning")
        if settings.theme != old_theme:
            ui.navigate.reload()

    with ui.page_sticky(position="bottom", y_offset=20):
        with ui.row().classes("le-card items-center gap-4 px-5 py-3 no-wrap").style(
                "box-shadow: 0 12px 40px -10px rgba(15,23,42,.35)") as footer:
            ui.icon("edit_note", color="primary")
            ui.label("You have unsaved changes").classes("text-sm font-medium")
            ui.button("Discard", on_click=discard).props("flat")
            ui.button("Save changes", icon="save", on_click=save).props("unelevated color=primary")
    footer.set_visibility(False)


def _token_row(tokens: dict[str, str], target: ui.input) -> None:
    with ui.row().classes("gap-1 -mt-2"):
        for token, description in tokens.items():
            ui.chip(f"{{{token}}}", on_click=lambda t=token: target.set_value(f"{target.value}{{{t}}}")) \
                .props("dense outline clickable").classes("text-xs le-token").tooltip(description)
