"""AI providers in the UI: the local Ollama plus APIs you bring your own key for.

`AiDraft` is the AI part of a page's unsaved settings. API keys typed into the provider dialog stay in
`pending` until the page saves; only then are they written to the secret store (never to settings.json).
"""

from __future__ import annotations

import secrets as stdlib_secrets
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from nicegui import run, ui

from ..ai.providers import (
    KINDS,
    PRESETS,
    ProviderError,
    api_key_for,
    base_url,
    env_key,
    is_local,
    list_models,
    needs_key,
    preset,
)
from ..api_keys import SecretStore, mask
from ..settings import LOCAL_OLLAMA_ID, ProviderConfig, Settings
from .context import ctx

ROLE_FIELDS = {"decision": ("decision_provider", "decision_model"),
               "extraction": ("extraction_provider", "extraction_model"),
               "ocr_llm": ("ocr_llm_provider", "ocr_llm_model")}


class AiDraft:
    """Providers, role choices and pending API keys of a settings draft."""

    def __init__(self, values: dict[str, Any], set_value: Callable[[str, Any], None], keys: SecretStore | None = None):
        self.values = values
        self._set_value = set_value
        self.keys = keys if keys is not None else ctx().engine.keys
        self.pending: dict[str, str] = {}  # provider id -> new key ("" = remove the stored key)
        self.listeners: list[Callable[[], None]] = []

    # ------------------------------------------------------------------ reading
    def settings(self) -> Settings:
        """The draft as Settings (pages with a partial draft are completed from the saved settings)."""
        return Settings.model_validate({**ctx().store.get().model_dump(), **self.values})

    def providers(self) -> list[ProviderConfig]:
        return self.settings().all_providers()

    def provider(self, provider_id: str) -> ProviderConfig | None:
        return self.settings().provider(provider_id)

    def api_key(self, provider: ProviderConfig) -> str:
        if provider.id in self.pending:
            return self.pending[provider.id] or env_key(provider)
        return api_key_for(provider, self.keys)

    def key_lookup(self) -> Callable[[str], str]:
        """For code that resolves keys by provider id (the factory): pending keys first."""
        return lambda provider_id: self.pending[provider_id] if provider_id in self.pending else self.keys.get(provider_id)

    def key_source(self, provider: ProviderConfig) -> str:
        """'stored', 'env' or '' – where the key for this provider would come from."""
        stored = self.pending[provider.id] if provider.id in self.pending else self.keys.get(provider.id)
        if stored:
            return "stored"
        return "env" if env_key(provider) else ""

    # ------------------------------------------------------------------ changing
    def set(self, key: str, value: Any, notify: bool = True) -> None:
        changed = self.values.get(key) != value
        self._set_value(key, value)  # the page's setter (marks the draft as changed)
        self.values[key] = value
        if notify and changed:
            self._changed()

    def _custom(self) -> list[dict[str, Any]]:
        return [p.model_dump() if isinstance(p, ProviderConfig) else dict(p) for p in self.values.get("providers", [])
                if (p.id if isinstance(p, ProviderConfig) else p.get("id")) != LOCAL_OLLAMA_ID]

    def save_provider(self, provider: ProviderConfig, new_key: str | None) -> None:
        """Add or replace a provider. new_key: None = unchanged, "" = remove the stored key."""
        if new_key is not None:
            self.pending[provider.id] = new_key.strip()
            provider = provider.model_copy(update={"key_hint": mask(new_key) or ("set" if new_key.strip() else "")})
        others = [p for p in self._custom() if p["id"] != provider.id]
        existing = next((i for i, p in enumerate(self._custom()) if p["id"] == provider.id), None)
        items = others[:existing] + [provider.model_dump()] + others[existing:] if existing is not None \
            else [*others, provider.model_dump()]
        self.set("providers", items, notify=False)
        self._changed()

    def remove_provider(self, provider_id: str) -> list[str]:
        """Remove a provider; roles that used it go back to the local Ollama. Returns those roles."""
        from ..gpu import recommended_models, total_vram_gb

        before = self.settings().model_dump()
        self.set("providers", [p for p in self._custom() if p["id"] != provider_id], notify=False)
        self.pending[provider_id] = ""
        moved = []
        rec_decision, rec_ocr = recommended_models(total_vram_gb())
        for role, (provider_field, model_field) in ROLE_FIELDS.items():
            if before.get(provider_field) == provider_id:
                moved.append(role)
                if role == "extraction":
                    self.set(provider_field, "", notify=False)
                    self.set(model_field, "", notify=False)
                else:
                    self.set(provider_field, LOCAL_OLLAMA_ID, notify=False)
                    self.set(model_field, rec_decision if role == "decision" else rec_ocr, notify=False)
        self._changed()
        return moved

    def apply_keys(self) -> bool:
        """Write pending keys to the secret store; drop keys of removed providers. True if anything changed."""
        changed = bool(self.pending)
        for provider_id, key in self.pending.items():
            self.keys.set(provider_id, key)
        self.pending.clear()
        self.keys.prune({p.id for p in self.providers()})
        return changed

    def on_change(self, listener: Callable[[], None]) -> None:
        self.listeners.append(listener)

    def _changed(self) -> None:
        for listener in list(self.listeners):
            listener()


def new_provider_id(name: str) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")[:24] or "provider"
    return f"{slug}-{stdlib_secrets.token_hex(3)}"


def host_of(provider: ProviderConfig) -> str:
    return urlparse(base_url(provider)).netloc or base_url(provider)


def privacy_note(provider: ProviderConfig | None, what: str) -> None:
    """Make it obvious when letters leave this computer."""
    if provider is None:
        return
    if is_local(provider):
        if provider.id != LOCAL_OLLAMA_ID:
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.icon("lan", size="18px").classes("text-green-600")
                ui.label(f"Runs in your network ({host_of(provider)}) – letters stay with you.").classes("text-xs le-muted")
        return
    with ui.row().classes("w-full items-start gap-2 no-wrap p-3 rounded-xl").style(
            "background: rgba(245,158,11,.08); border: 1px solid rgba(245,158,11,.3)"):
        ui.icon("cloud_upload", size="18px").classes("text-amber-600 mt-[1px]")
        ui.label(f"{what} is sent to {provider.name} ({host_of(provider)}). Check that its data-processing terms fit "
                 "your letters (e.g. a data-processing agreement / zero data retention).").classes("text-xs")


def provider_dialog(ai: AiDraft, existing: ProviderConfig | None = None,
                    on_saved: Callable[[ProviderConfig], None] | None = None) -> None:
    """Add a provider (pick a preset, paste the key) or edit one."""
    state: dict[str, Any] = {"preset": existing.preset if existing else "openai", "remove_key": False}

    def current_preset():
        return preset(state["preset"]) if state["preset"] else None

    def kind() -> str:
        if existing:
            return existing.kind
        p = current_preset()
        return p.kind if p else "openai"

    with ui.dialog() as dialog, ui.card().classes("le-card p-6 gap-3").style("width: 600px; max-width: 96vw"):
        ui.label("Add an AI provider" if existing is None else f"Edit {existing.name}").classes("text-lg font-semibold")
        ui.label("Bring your own API key – or point LetterEye at a model server in your network. All model calls "
                 "go through LangChain, so any OpenAI-, Anthropic- or Google-compatible API works.") \
            .classes("le-muted text-sm")
        if existing is None:
            ui.select({p.key: p.name for p in PRESETS}, value=state["preset"], label="Provider",
                      on_change=lambda e: apply_preset(e.value)).props("outlined dense").classes("w-full")
        kind_label = ui.label().classes("le-kbd self-start")
        name = ui.input("Name", value=existing.name if existing else "").props("outlined dense").classes("w-full")
        url = ui.input("API address", value=existing.base_url if existing else "").props("outlined dense") \
            .classes("w-full")
        key = ui.input("API key", password=True, password_toggle_button=True).props(
            "outlined dense autocomplete=off").classes("w-full")
        key_info = ui.row().classes("w-full items-center gap-2 -mt-2")
        note = ui.label().classes("text-xs le-muted")
        result = ui.row().classes("w-full items-center gap-2")

        def draft_provider() -> ProviderConfig:
            p = current_preset()
            return ProviderConfig(
                id=existing.id if existing else "draft", name=(name.value or "").strip() or (p.name if p else "API"),
                kind=kind(), base_url=(url.value or "").strip(),
                api_key_env=existing.api_key_env if existing else (p.api_key_env if p else ""),
                key_hint=existing.key_hint if existing else "", preset=state["preset"] or "")

        def entered_key(provider: ProviderConfig) -> str:
            if (key.value or "").strip():
                return key.value.strip()
            if state["remove_key"] or existing is None:
                return env_key(provider)
            return ai.api_key(existing)

        def render_key_info() -> None:
            key_info.clear()
            provider = draft_provider()
            with key_info:
                if existing and not state["remove_key"] and ai.key_source(existing) == "stored":
                    hint = f"••••{existing.key_hint}" if existing.key_hint not in ("", "set") else "a key"
                    ui.label(f"Stored: {hint}. Leave the field empty to keep it.").classes("text-xs le-muted")
                    ui.button("Remove key", on_click=lambda: (state.update(remove_key=True), render_key_info())) \
                        .props("flat dense size=sm color=negative no-caps")
                elif provider.api_key_env:
                    ui.label(f"…or set the environment variable {provider.api_key_env}.").classes("text-xs le-muted")
                p = current_preset()
                if p and p.key_url:
                    ui.link("Get an API key", p.key_url, new_tab=True).classes("text-xs text-primary")
                if not needs_key(provider):
                    ui.label("Optional for this address.").classes("text-xs le-muted")

        def apply_preset(key_name: str) -> None:
            state["preset"] = key_name
            p = current_preset()
            if p is None:
                return
            name.value = p.name if p.key not in ("custom_openai", "custom_anthropic", "custom_google") else ""
            url.value = p.base_url
            refresh_labels()

        def refresh_labels() -> None:
            p = current_preset()
            info = KINDS[kind()]
            kind_label.text = f"{info.title} · LangChain {info.package} · " + (
                "token probabilities" if info.logprobs else "structured confidence")
            url.props(f'placeholder="{info.default_url}"')
            note.text = (p.note if p and p.note else "")
            render_key_info()

        async def test() -> None:
            provider = draft_provider()
            api_key = entered_key(provider)
            result.clear()
            with result:
                ui.spinner(size="18px")
                ui.label("Connecting…").classes("text-sm le-muted")
            try:
                models = await run.io_bound(list_models, provider, api_key)
            except ProviderError as exc:
                show_result(False, str(exc))
                return
            except Exception as exc:  # unexpected (e.g. malformed address)
                show_result(False, f"{exc.__class__.__name__}: {exc}")
                return
            show_result(True, f"Connected – {len(models)} model{'s' if len(models) != 1 else ''} available.")

        def show_result(ok: bool, message: str) -> None:
            result.clear()
            with result:
                ui.icon("check_circle" if ok else "error", color="positive" if ok else "negative")
                ui.label(message).classes("text-sm break-words").style("min-width: 0; flex: 1")

        def save() -> None:
            provider = draft_provider()
            p = current_preset()
            if not (name.value or "").strip():
                ui.notify("Give the provider a name.", type="warning")
                return
            if not provider.base_url and kind() != "google" and not (p and p.base_url):
                ui.notify("Enter the API address.", type="warning")
                return
            if existing is None:
                provider = provider.model_copy(update={"id": new_provider_id(provider.name)})
            new_key: str | None = key.value.strip() if (key.value or "").strip() else ("" if state["remove_key"] else None)
            ai.save_provider(provider, new_key)
            dialog.close()
            if on_saved:
                on_saved(provider)

        if existing is None:
            apply_preset(state["preset"])  # fill name and address of the default preset
        else:
            refresh_labels()
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Test connection", icon="lan", on_click=test).props("outline color=primary")
            ui.space()
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Save", icon="save", on_click=save).props("unelevated color=primary")
    dialog.open()


class ProvidersPanel:
    """The list of providers with add / edit / remove (Settings → AI models)."""

    def __init__(self, ai: AiDraft):
        self.ai = ai
        self.dialogs = ui.element("div")  # dialogs live here: the list below is rebuilt on every change
        self.container = ui.column().classes("w-full gap-2")
        ai.on_change(self.render)
        self.render()

    def _dialog(self, provider: ProviderConfig | None = None) -> None:
        with self.dialogs:
            provider_dialog(self.ai, provider)

    def render(self) -> None:
        self.container.clear()
        settings = self.ai.settings()
        used = {pid for pid in (settings.decision_provider, settings.effective_extraction_provider,
                                settings.ocr_llm_provider)}
        with self.container:
            for provider in self.ai.providers():
                self._row(provider, provider.id in used)
            with ui.row().classes("gap-2 mt-1"):
                ui.button("Add provider", icon="add", on_click=lambda: self._dialog()) \
                    .props("outline color=primary")

    def _row(self, provider: ProviderConfig, used: bool) -> None:
        local = is_local(provider)
        with ui.row().classes("w-full items-center gap-3 p-3 rounded-2xl no-wrap").style(
                "border: 1px solid var(--le-border)"):
            ui.icon("computer" if provider.id == LOCAL_OLLAMA_ID else "lan" if local else "cloud", size="22px") \
                .classes("le-gradient-text")
            with ui.column().classes("gap-0 flex-grow").style("min-width: 0"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(provider.name).classes("font-semibold text-sm")
                    ui.label(KINDS[provider.kind].title).classes("le-kbd")
                    if used:
                        ui.chip("in use", color="primary", text_color="white").props("dense").classes("text-xs")
                details = [base_url(provider)]
                if provider.kind != "ollama":
                    source = self.ai.key_source(provider)
                    if source == "stored":
                        hint = provider.key_hint
                        details.append(f"key ••••{hint}" if hint and hint != "set" else "key stored")
                    elif source == "env":
                        details.append(f"key from {provider.api_key_env}")
                    elif needs_key(provider):
                        details.append("no API key yet")
                details.append("this computer" if provider.id == LOCAL_OLLAMA_ID else
                               "your network" if local else "cloud – letters leave this computer")
                ui.label(" · ".join(details)).classes("le-muted text-xs truncate")
            if provider.id != LOCAL_OLLAMA_ID:
                ui.button(icon="edit", on_click=lambda p=provider: self._dialog(p)) \
                    .props("flat round dense").tooltip("Edit")
                ui.button(icon="delete_outline", on_click=lambda p=provider: self._remove(p)) \
                    .props("flat round dense color=grey").tooltip("Remove")

    def _remove(self, provider: ProviderConfig) -> None:
        with self.dialogs, ui.dialog() as dialog, ui.card().classes("le-card p-5 gap-3"):
            ui.label(f"Remove {provider.name}?").classes("font-semibold")
            ui.label("Its stored API key is deleted when you save. Models that use it switch back to Ollama.") \
                .classes("text-sm le-muted")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def confirm() -> None:
                    moved = self.ai.remove_provider(provider.id)
                    dialog.close()
                    if moved:
                        ui.notify(f"{', '.join(moved).replace('_', ' ')} now use Ollama – check the model choice.",
                                  type="warning")

                ui.button("Remove", on_click=confirm).props("unelevated color=negative")
        dialog.open()
