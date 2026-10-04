"""Choosing and downloading the local models (decision model, OCR LLM, fast OCR) with progress."""

from __future__ import annotations

import threading
from collections.abc import Callable

from nicegui import run, ui

from ..ai.ollama_service import OLLAMA_DOWNLOAD_URL, OllamaService, normalize_model_name
from ..gpu import DECISION_MODELS, OCR_LLM_MODELS, ModelChoice, recommended_models, total_vram_gb
from .context import ctx

# Downloads run in background threads and outlive page changes.
_downloads: dict[str, dict] = {}
_lock = threading.Lock()


def start_download(base_url: str, model: str) -> None:
    key = normalize_model_name(model)
    with _lock:
        if key in _downloads and not _downloads[key].get("done"):
            return
        _downloads[key] = {"status": "Starting…", "fraction": None, "done": False, "error": ""}

    def work() -> None:
        def progress(status: str, fraction: float | None) -> None:
            _downloads[key].update(status=status, fraction=fraction)

        try:
            OllamaService(base_url).pull(model, progress)
            _downloads[key].update(status="Downloaded", fraction=1.0, done=True)
            ctx().feed.add(f"Model {model} downloaded", "success")
        except Exception as exc:
            _downloads[key].update(status="Failed", error=str(exc), done=True)
            ctx().feed.add(f"Download of {model} failed: {exc}", "error")

    threading.Thread(target=work, name=f"pull-{model}", daemon=True).start()


def download_state(model: str) -> dict | None:
    return _downloads.get(normalize_model_name(model))


def ollama_status_banner(on_retry: Callable[[], None] | None = None) -> None:
    """Shown when Ollama is not reachable: what to do, per OS."""
    with ui.column().classes("w-full gap-2 p-4 rounded-2xl").style(
            "background: rgba(220,38,38,.07); border: 1px solid rgba(220,38,38,.25)"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("cloud_off", size="22px").classes("text-red-600")
            ui.label("Ollama is not running").classes("font-semibold")
        ui.markdown(
            "LetterEye runs all AI models locally through **Ollama** (free). It uses your GPU automatically "
            "(NVIDIA CUDA, AMD ROCm, Apple Metal).\n\n"
            "1. Download and install Ollama for your system.\n"
            "2. Start it (on Windows it runs in the system tray after installation).\n"
            "3. Click *Check again*."
        ).classes("text-sm")
        with ui.row().classes("gap-2"):
            ui.button("Download Ollama", icon="download",
                      on_click=lambda: ui.navigate.to(OLLAMA_DOWNLOAD_URL, new_tab=True)).props("unelevated color=primary")
            if on_retry:
                ui.button("Check again", icon="refresh", on_click=on_retry).props("outline color=primary")


def model_cards(choices: list[ModelChoice], selected: str, installed: list[str], recommended: str,
                on_select: Callable[[str], None], base_url: str, gpu_known: bool = True) -> None:
    installed_norm = {normalize_model_name(m) for m in installed}
    with ui.grid(columns=2).classes("w-full gap-3"):
        for choice in choices:
            is_selected = normalize_model_name(choice.name) == normalize_model_name(selected)
            is_installed = normalize_model_name(choice.name) in installed_norm
            border = "2px solid #6366f1" if is_selected else "1px solid var(--le-border)"
            with ui.column().classes("p-4 rounded-2xl gap-1 cursor-pointer").style(
                    f"border: {border}; background: var(--le-card)").on("click", lambda n=choice.name: on_select(n)):
                with ui.row().classes("w-full items-center justify-between no-wrap"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("radio_button_checked" if is_selected else "radio_button_unchecked",
                                color="primary" if is_selected else "grey-5")
                        ui.label(choice.title).classes("font-semibold")
                    if choice.name == recommended:
                        ui.chip("Recommended for your GPU" if gpu_known else "Recommended", icon="star").props("dense color=primary text-color=white") \
                            .classes("text-xs")
                ui.label(choice.note).classes("le-muted text-xs")
                with ui.row().classes("items-center justify-between w-full mt-1"):
                    ui.label(f"{choice.name} · {choice.size_gb:.1f} GB · ~{choice.min_vram_gb:.0f} GB VRAM") \
                        .classes("le-kbd")
                    download_control(choice.name, is_installed, base_url)


def download_control(model: str, is_installed: bool, base_url: str) -> None:
    """'Installed' badge, a Download button, or a live progress bar."""
    box = ui.row().classes("items-center gap-2")

    def render() -> None:
        box.clear()
        state = download_state(model)
        with box:
            if is_installed or (state and state.get("done") and not state.get("error")):
                ui.chip("Installed", icon="check_circle", color="positive", text_color="white").props("dense") \
                    .classes("text-xs")
            elif state and not state.get("done"):
                fraction = state.get("fraction")
                ui.linear_progress(value=fraction or 0, show_value=False).props(
                    f"rounded size=8px color=primary {'indeterminate' if fraction is None else ''}").classes("w-28")
                ui.label(f"{fraction:.0%}" if fraction else state.get("status", "")[:18]).classes("text-xs le-muted")
            else:
                if state and state.get("error"):
                    ui.icon("error", color="negative").tooltip(state["error"])
                ui.button("Download", icon="download").props("dense unelevated color=primary size=sm") \
                    .on("click.stop", lambda: (start_download(base_url, model), render()))

    last: dict = {"sig": None}

    def tick() -> None:
        state = download_state(model) or {}
        sig = (state.get("status"), state.get("fraction"), state.get("done"))
        if sig != last["sig"]:
            last["sig"] = sig
            render()

    tick()
    ui.timer(0.8, tick)


class ModelsPanel:
    """Decision model + OCR LLM selection with downloads. `on_change(field, value)` receives the choices."""

    def __init__(self, decision_model: str, ocr_llm_model: str, ocr_mode: str,
                 on_change: Callable[[str, str], None]):
        self.decision_model = decision_model
        self.ocr_llm_model = ocr_llm_model
        self.ocr_mode = ocr_mode
        self.on_change = on_change
        self.container = ui.column().classes("w-full gap-5")
        ui.timer(0.05, self.refresh, once=True)

    async def refresh(self) -> None:
        base_url = ctx().store.get().ollama_url
        service = OllamaService(base_url)
        version = await run.io_bound(service.version)
        installed = await run.io_bound(service.list_models) if version else []
        vram = await run.io_bound(total_vram_gb)
        rec_decision, rec_ocr = recommended_models(vram)
        self.container.clear()
        with self.container:
            if not version:
                ollama_status_banner(on_retry=self.refresh)
                return
            with ui.row().classes("items-center gap-2"):
                ui.icon("check_circle", color="positive")
                ui.label(f"Ollama {version} is running").classes("text-sm font-medium")
                if vram:
                    ui.label(f"· GPU memory: {vram} GB").classes("text-sm le-muted")
            ui.label("Decision model").classes("le-section-title")
            ui.label("Makes the typed decisions (who is it for, which document type) and reads sender/date/subject. "
                     "Runs on the GPU through Ollama.").classes("le-muted text-sm -mt-2")
            model_cards(DECISION_MODELS, self.decision_model, installed, rec_decision, self._pick_decision, base_url, vram is not None)
            self._other_installed(installed, "decision_model", self.decision_model,
                                  [c.name for c in DECISION_MODELS])
            if self.ocr_mode != "fast_only":
                ui.label("OCR LLM (only when needed)").classes("le-section-title mt-2")
                ui.label("A vision model that re-reads the page when the fast OCR text is not good enough for a "
                         "confident decision.").classes("le-muted text-sm -mt-2")
                model_cards(OCR_LLM_MODELS, self.ocr_llm_model, installed, rec_ocr, self._pick_ocr, base_url, vram is not None)
                self._other_installed(installed, "ocr_llm_model", self.ocr_llm_model, [c.name for c in OCR_LLM_MODELS])

    def _other_installed(self, installed: list[str], field: str, current: str, known: list[str]) -> None:
        known_norm = {normalize_model_name(k) for k in known}
        others = [m for m in installed if normalize_model_name(m) not in known_norm]
        if not others:
            return
        value = current if normalize_model_name(current) not in known_norm else None
        ui.select(others, value=value if value in others else None, label="…or use another installed model",
                  on_change=lambda e: e.value and self._set(field, e.value)).props("outlined dense clearable") \
            .classes("w-80")

    def _pick_decision(self, name: str) -> None:
        self._set("decision_model", name)

    def _pick_ocr(self, name: str) -> None:
        self._set("ocr_llm_model", name)

    def _set(self, field: str, value: str) -> None:
        setattr(self, field, value)
        self.on_change(field, value)
        ui.timer(0.01, self.refresh, once=True)
