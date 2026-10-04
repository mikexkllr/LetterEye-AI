"""Dashboard: start/stop, today's numbers, live activity, system health (GPU, models)."""

from __future__ import annotations

from datetime import UTC, datetime, time

from nicegui import run, ui

from ...ai.ollama_service import OLLAMA_DOWNLOAD_URL, OllamaService, normalize_model_name
from ...gpu import ONNX_DEVICE_LABELS, best_onnx_device, detect_gpus
from ..context import ctx
from ..document_detail import show_document
from ..widgets import card, on_data_change, open_path, page_header, relative_time, status_chip
from .test_letter import open_test_dialog

LEVEL_ICONS = {
    "info": ("radio_button_checked", "text-slate-400"),
    "success": ("task_alt", "text-green-500"),
    "warning": ("rate_review", "text-amber-500"),
    "error": ("error", "text-red-500"),
    "ai": ("auto_awesome", "text-fuchsia-500"),
}


def _today_utc_iso() -> str:
    local_midnight = datetime.combine(datetime.now().date(), time.min).astimezone()
    return local_midnight.astimezone(UTC).isoformat(timespec="seconds")


async def toggle_engine() -> None:
    c = ctx()
    if c.engine.state == "running":
        await run.io_bound(c.engine.stop)
        ui.notify("Stopped watching the inbox")
        return
    problems = await run.io_bound(c.engine.start)
    if problems:
        for problem in problems[:3]:
            ui.notify(problem, type="warning", position="top", timeout=8000)
    else:
        ui.notify("Watching the inbox – new letters are sorted automatically", type="positive")


def page() -> None:
    c = ctx()
    with ui.column().classes("le-page"):
        with page_header("Dashboard", "Letters are read, decided and filed by AI running on this computer."):
            ui.button("Test a letter", icon="science", on_click=open_test_dialog).props("outline color=primary")

        @ui.refreshable
        def hero() -> None:
            s = c.store.get()
            state = c.engine.state
            running = state == "running"
            with ui.element("div").classes("le-hero w-full p-7"):
                with ui.row().classes("w-full items-center justify-between no-wrap gap-6 relative z-10"):
                    with ui.column().classes("gap-2"):
                        with ui.row().classes("items-center gap-3"):
                            ui.element("span").classes(f"le-status-dot {'le-dot-on' if running else 'le-dot-off'}") \
                                .style("background: white" if not running else "")
                            ui.label("Watching for new letters" if running else
                                     "Starting…" if state == "starting" else "Not watching").classes("text-2xl font-bold")
                        if running and c.engine.current:
                            cur = c.engine.current
                            with ui.row().classes("items-center gap-2 text-white/90"):
                                ui.spinner("dots", size="md", color="white")
                                ui.label(f"{cur.name} — {cur.stage}").classes("text-[15px]")
                        elif running:
                            ui.label(f"Drop scans into {s.inbox_folder}").classes("text-white/85 text-[15px] break-all")
                        else:
                            ui.label(c.engine.last_error or "Press start to sort letters from your inbox folder.") \
                                .classes("text-white/85 text-[15px]")
                        with ui.row().classes("gap-2 pt-1"):
                            if c.engine.queue_size:
                                ui.chip(f"{c.engine.queue_size} waiting", icon="hourglass_top").props("dense") \
                                    .classes("bg-white/20 text-white")
                            if s.inbox_folder:
                                ui.button("Inbox", icon="move_to_inbox", on_click=lambda: open_path(s.inbox_folder)) \
                                    .props("flat dense color=white no-caps")
                            if s.output_folder:
                                ui.button("Sorted letters", icon="folder_special",
                                          on_click=lambda: open_path(s.output_folder)).props("flat dense color=white no-caps")
                    ui.button("Stop" if running else "Start", icon="stop" if running else "play_arrow",
                              on_click=toggle_engine) \
                        .props(f"size=lg unelevated {'outline color=white' if running else 'color=white text-color=primary'}") \
                        .classes("px-6 shrink-0").style("border-radius: 14px")

        hero()

        @ui.refreshable
        def stats() -> None:
            today = c.db.count_by_status(since=_today_utc_iso())
            totals = c.db.count_by_status()
            items = [
                ("Processed today", today["filed"] + today["review"] + today["failed"] + today["ignored"], "mark_email_read", "#6366f1"),
                ("Filed automatically", today["filed"], "task_alt", "#16a34a"),
                ("Waiting for review", totals["review"], "rate_review", "#f59e0b"),
                ("Failed today", today["failed"], "error_outline", "#dc2626"),
            ]
            with ui.grid(columns=4).classes("w-full gap-4"):
                for label, value, icon, color in items:
                    with card("p-5"):
                        with ui.row().classes("items-center justify-between w-full no-wrap"):
                            with ui.column().classes("gap-1"):
                                ui.label(label).classes("le-muted text-sm font-medium")
                                ui.label(str(value)).classes("le-stat-value")
                            with ui.element("div").classes("rounded-2xl p-3").style(f"background: {color}1a"):
                                ui.icon(icon, size="26px").style(f"color: {color}")

        stats()

        with ui.row().classes("w-full gap-5 no-wrap items-start"):
            with ui.column().classes("gap-5").style("flex: 3; min-width: 0"):
                with card("p-5"):
                    ui.label("Live activity").classes("le-section-title mb-2")

                    @ui.refreshable
                    def feed() -> None:
                        entries = c.feed.recent(40)
                        if not entries:
                            ui.label("Nothing happened yet. Start watching and drop a scanned letter into the inbox.") \
                                .classes("le-muted text-sm py-6")
                        with ui.column().classes("w-full gap-0"):
                            for entry in entries:
                                icon, color = LEVEL_ICONS.get(entry.level, LEVEL_ICONS["info"])
                                with ui.row().classes("le-feed-item w-full items-start gap-3 no-wrap"):
                                    ui.icon(icon, size="18px").classes(f"{color} mt-[2px]")
                                    label = ui.label(entry.message).classes("text-sm flex-grow break-words")
                                    if entry.doc_id:
                                        label.classes("cursor-pointer hover:underline")
                                        label.on("click", lambda d=entry.doc_id: show_document(d))
                                    ui.label(entry.time.strftime("%H:%M:%S")).classes("le-muted text-xs shrink-0")

                    with ui.scroll_area().classes("w-full").style("height: 380px"):
                        feed()

            with ui.column().classes("gap-5").style("flex: 2; min-width: 320px"):
                with card("p-5"):
                    async def load_health() -> None:
                        info = await run.io_bound(_collect_health)
                        health_box.clear()
                        with health_box:
                            _render_health(info)

                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("System").classes("le-section-title")
                        ui.button(icon="refresh", on_click=load_health).props("flat round dense")
                    health_box = ui.column().classes("w-full gap-3 pt-1")
                    with health_box:
                        ui.skeleton().classes("w-full h-24")
                    ui.timer(20, load_health, immediate=True)

                with card("p-5"):
                    ui.label("Recent letters").classes("le-section-title mb-1")

                    @ui.refreshable
                    def recent() -> None:
                        docs = c.db.list_documents(limit=6)
                        if not docs:
                            ui.label("No letters processed yet.").classes("le-muted text-sm py-4")
                        for doc in docs:
                            with ui.row().classes("le-feed-item w-full items-center gap-3 no-wrap cursor-pointer") \
                                    .on("click", lambda d=doc.id: show_document(d)):
                                with ui.column().classes("gap-0 flex-grow").style("min-width: 0"):
                                    ui.label(doc.original_name).classes("text-sm font-medium truncate")
                                    target = " / ".join(x for x in (doc.worker_name, doc.recipient_label) if x)
                                    ui.label(target or doc.review_reason or doc.stage or relative_time(doc.created_at)) \
                                        .classes("le-muted text-xs truncate")
                                status_chip(doc.status)

                    recent()

    def refresh_all() -> None:
        hero.refresh()
        stats.refresh()
        feed.refresh()
        recent.refresh()

    on_data_change(refresh_all)
    last_state = {"state": c.engine.state}

    def watch_state() -> None:  # engine start/stop happen outside the activity feed sometimes
        if c.engine.state != last_state["state"]:
            last_state["state"] = c.engine.state
            hero.refresh()

    ui.timer(0.7, watch_state)


def _collect_health() -> dict:
    c = ctx()
    s = c.store.get()
    ollama = OllamaService(s.ollama_url)
    version = ollama.version()
    installed = ollama.list_models() if version else []
    loaded = {m.name: m for m in ollama.loaded_models()} if version else {}
    gpus = detect_gpus()
    device = c.engine.ocr_device or best_onnx_device(s.fast_ocr_device)
    return {"settings": s, "version": version, "installed": installed, "loaded": loaded, "gpus": gpus,
            "ocr_device": device}


def _row(icon: str, title: str, value: str, ok: bool | None, hint: str = "") -> None:
    color = "text-green-500" if ok else "text-amber-500" if ok is None else "text-red-500"
    with ui.row().classes("w-full items-start gap-3 no-wrap"):
        ui.icon(icon, size="20px").classes(f"{color} mt-[1px]")
        with ui.column().classes("gap-0 flex-grow").style("min-width: 0"):
            ui.label(title).classes("text-sm font-medium")
            ui.label(value).classes("le-muted text-xs break-words")
            if hint:
                ui.label(hint).classes("text-xs text-amber-600")


def _render_health(info: dict) -> None:
    s = info["settings"]
    if not info["version"]:
        _row("cloud_off", "Ollama", f"Not reachable at {s.ollama_url}", False)
        ui.link("Download Ollama", OLLAMA_DOWNLOAD_URL, new_tab=True).classes("text-sm text-primary ml-8")
    else:
        _row("hub", "Ollama", f"Running · v{info['version']}", True)

    def model_row(icon: str, title: str, name: str) -> None:
        wanted = normalize_model_name(name)
        present = any(normalize_model_name(m) == wanted for m in info["installed"])
        key = next((k for k in info["loaded"] if normalize_model_name(k) == wanted), None)
        if not present:
            _row(icon, title, f"{name} · not downloaded", False, "Settings → AI models")
            return
        has_gpu = bool(info["gpus"])
        placement = (info["loaded"][key].placement if has_gpu else "CPU") if key else "not loaded yet"
        warn = has_gpu and key is not None and info["loaded"][key].gpu_share < 0.99
        _row(icon, title, f"{name} · {placement}", None if warn else True,
             "Partly on CPU – a smaller model would be faster." if warn else "")

    model_row("psychology", "Decision model", s.decision_model)
    if s.ocr_mode != "fast_only":
        model_row("document_scanner", "OCR LLM (fallback)", s.ocr_llm_model)
    device = info["ocr_device"]
    label = ONNX_DEVICE_LABELS.get(device, device.replace("ExecutionProvider", ""))
    _row("bolt", "Fast OCR · PP-OCRv6", label, "cpu" not in device.lower() or None)
    gpus = info["gpus"]
    if gpus:
        gpu = gpus[0]
        mem = f" · {gpu.memory_total_gb} GB" if gpu.memory_total_gb else ""
        _row("memory", "GPU", f"{gpu.name}{mem}", True)
    else:
        _row("memory", "GPU", "No GPU detected – everything runs on the CPU (slow).", None)
