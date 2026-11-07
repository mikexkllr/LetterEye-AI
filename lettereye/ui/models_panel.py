"""Choosing the models per role – local Ollama models (downloaded with progress) or models of an API provider."""

from __future__ import annotations

import threading
from collections.abc import Callable

from nicegui import run, ui

from ..ai import factory
from ..ai.ollama_service import OLLAMA_DOWNLOAD_URL, OllamaService, normalize_model_name
from ..ai.providers import KINDS, base_url, list_models, needs_key, preset
from ..gpu import DECISION_MODELS, OCR_LLM_MODELS, ModelChoice, recommended_models, total_vram_gb
from ..settings import LOCAL_OLLAMA_ID
from .context import ctx
from .providers_panel import ROLE_FIELDS, AiDraft, privacy_note, provider_dialog

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
            "Local models run through **Ollama** (free) on your GPU (NVIDIA CUDA, AMD ROCm, Apple Metal) – no letter "
            "leaves this computer.\n\n"
            "1. Download and install Ollama for your system.\n"
            "2. Start it (on Windows it runs in the system tray after installation).\n"
            "3. Click *Check again*.\n\n"
            "Or let this model run on an API with your own key: choose it under *Runs on*."
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


ROLE_TEXT = {
    "decision": ("Decision model",
                 "Makes the typed decisions (who is it for, which document type) and reads sender/date/subject."),
    "ocr_llm": ("OCR LLM (only when needed)",
                "A vision model that re-reads the page when the fast OCR text is not good enough for a confident "
                "decision."),
}
SENT = {"decision": "The recognized text of each letter", "ocr_llm": "An image of each page that needs it"}
ADD_PROVIDER = "__add__"


class ModelsPanel:
    """Per role (decision model, OCR LLM): where it runs, and which model. Ollama models can be downloaded here;
    for API providers the model list comes from the provider."""

    def __init__(self, ai: AiDraft, ocr_mode: str = "auto"):
        self.ai = ai
        self.ocr_mode = ocr_mode
        self._models: dict[str, list[str] | str] = {}  # provider id (+ key) -> model ids, or an error message
        self._banner_shown = False
        self.dialogs = ui.element("div")  # dialogs live here: the container below is rebuilt on every change
        self.container = ui.column().classes("w-full gap-5")
        ai.on_change(self.schedule_refresh)
        ui.timer(0.05, self.refresh, once=True)

    def schedule_refresh(self) -> None:
        with self.container:  # not the caller's slot: that may be a dialog or a list that is being rebuilt
            ui.timer(0.01, self.refresh, once=True)

    def roles(self) -> list[str]:
        return ["decision"] + ([] if self.ocr_mode == "fast_only" else ["ocr_llm"])

    async def refresh(self) -> None:
        settings = self.ai.settings()
        servers: dict[str, tuple[str | None, list[str]]] = {}
        for role in self.roles():
            provider = settings.role(role)[0]
            if provider is not None and provider.kind == "ollama" and base_url(provider) not in servers:
                service = OllamaService(base_url(provider), api_key=self.ai.api_key(provider))
                version = await run.io_bound(service.version)
                servers[base_url(provider)] = (version, await run.io_bound(service.list_models) if version else [])
        vram = await run.io_bound(total_vram_gb)
        self.container.clear()
        with self.container:
            local = servers.get(base_url(settings.provider(LOCAL_OLLAMA_ID)))
            if local and local[0]:
                with ui.row().classes("items-center gap-2"):
                    ui.icon("check_circle", color="positive")
                    ui.label(f"Ollama {local[0]} is running").classes("text-sm font-medium")
                    if vram:
                        ui.label(f"· GPU memory: {vram} GB").classes("text-sm le-muted")
            self._banner_shown = False
            for role in self.roles():
                self._section(role, settings, servers, vram)

    def _section(self, role: str, settings, servers: dict, vram: float | None) -> None:
        title, text = ROLE_TEXT[role]
        provider, model = settings.role(role)
        with ui.row().classes("w-full items-end justify-between gap-3 mt-1"):
            with ui.column().classes("gap-0").style("min-width: 0; flex: 1"):
                ui.label(title).classes("le-section-title")
                ui.label(text).classes("le-muted text-sm")
            options = {p.id: p.name for p in self.ai.providers()}
            options[ADD_PROVIDER] = "+ Add an API provider…"
            ui.select(options, value=provider.id if provider else None, label="Runs on",
                      on_change=lambda e, r=role: self._pick_provider(r, e.value, e.sender)) \
                .props("outlined dense").classes("w-64")
        if provider is None:
            ui.label("The provider of this model was removed – choose another one.").classes("text-sm text-amber-700")
            return
        if provider.kind == "ollama":
            self._ollama_models(role, provider, model, servers.get(base_url(provider), (None, [])), vram)
        else:
            self._api_models(role, provider, model)
        privacy_note(provider, SENT[role])

    # ------------------------------------------------------------------ Ollama
    def _ollama_models(self, role: str, provider, model: str, server: tuple[str | None, list[str]],
                       vram: float | None) -> None:
        version, installed = server
        if not version:
            if provider.id == LOCAL_OLLAMA_ID and not self._banner_shown:
                ollama_status_banner(on_retry=self.refresh)
                self._banner_shown = True
            elif provider.id == LOCAL_OLLAMA_ID:
                ui.label("Ollama is not running (see above).").classes("text-sm text-red-600")
            else:
                ui.label(f"Ollama is not reachable at {base_url(provider)}.").classes("text-sm text-red-600")
            return
        rec_decision, rec_ocr = recommended_models(vram)
        choices, recommended = (DECISION_MODELS, rec_decision) if role == "decision" else (OCR_LLM_MODELS, rec_ocr)
        field = ROLE_FIELDS[role][1]
        model_cards(choices, model, installed, recommended, lambda name: self._set(field, name), base_url(provider),
                    vram is not None)
        self._other_installed(installed, field, model, [c.name for c in choices])

    def _other_installed(self, installed: list[str], field: str, current: str, known: list[str]) -> None:
        known_norm = {normalize_model_name(k) for k in known}
        others = [m for m in installed if normalize_model_name(m) not in known_norm]
        if not others:
            return
        value = current if normalize_model_name(current) not in known_norm else None
        ui.select(others, value=value if value in others else None, label="…or use another installed model",
                  on_change=lambda e: e.value and self._set(field, e.value)).props("outlined dense clearable") \
            .classes("w-80")

    # ------------------------------------------------------------------ API providers
    def _api_models(self, role: str, provider, model: str) -> None:
        field = ROLE_FIELDS[role][1]
        key = self.ai.api_key(provider)
        cache_key = f"{provider.id}|{base_url(provider)}|{len(key)}|{key[-4:]}"
        known = self._models.get(cache_key)
        hint = (preset(provider.preset).model_hint if preset(provider.preset) else "") or ""
        with ui.row().classes("w-full items-start gap-2 no-wrap"):
            select = ui.select(sorted(set(known if isinstance(known, list) else []) | ({model} if model else set())),
                               value=model or None, label="Model", with_input=True, new_value_mode="add-unique",
                               on_change=lambda e: self._set(field, (e.value or "").strip())) \
                .props(f'outlined dense clearable hint="{"e.g. " + hint if hint else "Type or pick a model id"}"') \
                .classes("flex-grow").style("max-width: 460px")
            if role == "decision":
                ui.button("Check", icon="science", on_click=lambda: self._check(provider)) \
                    .props("outline color=primary").tooltip("Ask the model one test question")
            ui.button(icon="refresh", on_click=lambda: self._load_models(provider, key, cache_key, select, status,
                                                                          force=True)) \
                .props("flat round dense").tooltip("Reload the model list")
        status = ui.label("").classes("text-xs le-muted -mt-1")
        if needs_key(provider) and not key:
            with ui.row().classes("items-center gap-2"):
                ui.icon("key_off", color="warning", size="18px")
                ui.label(f"No API key for {provider.name} yet.").classes("text-sm text-amber-700")
                ui.button("Add key", on_click=lambda: self._dialog(provider)) \
                    .props("flat dense color=primary no-caps")
        info = KINDS[provider.kind]
        if role == "decision":
            text = ("Probabilities: token probabilities (logprobs) – one forward pass, the most precise. If this model "
                    "returns none, LetterEye switches to structured confidence by itself." if info.logprobs else
                    "Probabilities: the model's own estimate per option (structured output) – this API returns no "
                    "token probabilities. Usually a bit over-confident; watch Insights → calibration.")
            ui.label(text).classes("text-xs le-muted")
        else:
            ui.label("Pick a model that can read images (a vision model).").classes("text-xs le-muted")
        if known is None and (key or not needs_key(provider)):
            ui.timer(0.01, lambda: self._load_models(provider, key, cache_key, select, status), once=True)
        elif isinstance(known, str):
            status.text = known
        elif known:
            status.text = f"{len(known)} models available"

    async def _load_models(self, provider, key: str, cache_key: str, select, status, force: bool = False) -> None:
        if not force and cache_key in self._models:
            return
        status.text = "Loading the model list…"
        try:
            models = await run.io_bound(list_models, provider, key)
            self._models[cache_key] = models
            status.text = f"{len(models)} models available"
        except Exception as exc:
            self._models[cache_key] = str(exc)
            status.text = str(exc)
            return
        current = select.value
        select.set_options(sorted(set(models) | ({current} if current else set())), value=current)

    async def _check(self, provider) -> None:
        settings = self.ai.settings()
        if not settings.decision_model.strip():
            ui.notify("Choose a model first.", type="warning")
            return
        note = ui.notification(f"Asking {settings.decision_model} a test question…", spinner=True, timeout=None)
        try:
            result = await run.io_bound(factory.check_decision_model, settings, self.ai.key_lookup())
        except Exception as exc:
            note.message, note.type, note.spinner, note.timeout = f"Check failed: {exc}", "negative", False, 8
            return
        how = {"logprobs": "token probabilities", "structured": "structured confidence",
               "both": "token probabilities and structured confidence"}.get(result["method"], result["method"])
        verdict = "answered correctly" if result["correct"] else f"answered '{result['choice']}' (expected Alice Example)"
        note.message = f"{settings.decision_model} {verdict} with {result['confidence']:.0%} – using {how}."
        note.type = "positive" if result["correct"] else "warning"
        note.spinner, note.timeout = False, 8

    # ------------------------------------------------------------------ changes
    def _pick_provider(self, role: str, provider_id: str | None, select) -> None:
        provider_field, model_field = ROLE_FIELDS[role]
        current = self.ai.settings().role(role)[0]
        if provider_id == ADD_PROVIDER:
            select.value = current.id if current else None
            self._dialog(None, on_saved=lambda p, r=role: self._pick_provider(r, p.id, None))
            return
        if not provider_id or (current and current.id == provider_id):
            return
        provider = self.ai.provider(provider_id)
        if provider is None:
            return
        if provider.kind == "ollama":
            rec_decision, rec_ocr = recommended_models(total_vram_gb())
            model = rec_decision if role == "decision" else rec_ocr
        else:
            p = preset(provider.preset)
            model = p.model_hint if p and role == "decision" else ""
        self.ai.set(model_field, model, notify=False)
        self.ai.set(provider_field, provider_id)  # notifies: this panel and the provider list refresh

    def _dialog(self, provider, on_saved=None) -> None:
        with self.dialogs:
            provider_dialog(self.ai, provider, on_saved=on_saved)

    def _set(self, field: str, value: str) -> None:
        self.ai.set(field, value)  # notifies: the panel refreshes
