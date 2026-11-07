"""User settings, edited in the UI and stored as JSON in the app-data folder."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from . import paths

log = logging.getLogger(__name__)


class DocumentType(BaseModel):
    name: str
    description: str = ""


DEFAULT_DOCUMENT_TYPES: dict[str, list[DocumentType]] = {
    "de": [
        DocumentType(name="Rechnung", description="Rechnung oder Abrechnung mit Zahlungsaufforderung"),
        DocumentType(name="Mahnung", description="Zahlungserinnerung oder Mahnung zu einer offenen Forderung"),
        DocumentType(name="Bescheid", description="Bescheid oder Schreiben einer Behörde, eines Amts oder Gerichts"),
        DocumentType(name="Vertrag", description="Vertrag, Vertragsänderung oder Auftragsbestätigung"),
        DocumentType(name="Kündigung", description="Kündigung oder Kündigungsbestätigung"),
        DocumentType(name="Versicherung", description="Schreiben einer Versicherung"),
        DocumentType(name="Bank", description="Kontoauszug oder Schreiben einer Bank"),
        DocumentType(name="Werbung", description="Werbung, Angebot oder Newsletter"),
        DocumentType(name="Sonstiges", description="Alles andere"),
    ],
    "en": [
        DocumentType(name="Invoice", description="Invoice or bill requesting payment"),
        DocumentType(name="Reminder", description="Payment reminder or dunning letter about an open amount"),
        DocumentType(name="Official notice", description="Letter from an authority, government office or court"),
        DocumentType(name="Contract", description="Contract, amendment or order confirmation"),
        DocumentType(name="Termination", description="Termination or cancellation (confirmation)"),
        DocumentType(name="Insurance", description="Letter from an insurance company"),
        DocumentType(name="Bank", description="Bank statement or letter from a bank"),
        DocumentType(name="Advertising", description="Advertising, offer or newsletter"),
        DocumentType(name="Other", description="Anything else"),
    ],
}

LANGUAGES = {"de": "Deutsch", "en": "English", "fr": "Français", "es": "Español", "it": "Italiano", "nl": "Nederlands"}
LANGUAGE_NAMES_EN = {"de": "German", "en": "English", "fr": "French", "es": "Spanish", "it": "Italian", "nl": "Dutch"}


def default_document_types(language: str) -> list[DocumentType]:
    return [t.model_copy() for t in DEFAULT_DOCUMENT_TYPES.get(language, DEFAULT_DOCUMENT_TYPES["en"])]


ProviderKind = Literal["ollama", "openai", "anthropic", "google"]
Role = Literal["decision", "extraction", "ocr_llm"]
LOCAL_OLLAMA_ID = "ollama"
ROLE_TITLES: dict[str, str] = {"decision": "decision model", "extraction": "extraction model", "ocr_llm": "OCR LLM"}


class ProviderConfig(BaseModel):
    """Where models run: the local Ollama server, or an API you bring your own key for.

    `kind` is the API dialect, so one kind covers many services: "openai" is any OpenAI-compatible API
    (OpenAI, OpenRouter, Groq, Mistral, DeepSeek, Together, LM Studio, vLLM, llama.cpp, …), "anthropic" any
    Anthropic-compatible one, "google" the Gemini API or a compatible gateway. API keys are never stored in
    the settings file; they live in the secret store (see api_keys.py), keyed by `id`.
    """

    id: str
    name: str
    kind: ProviderKind
    base_url: str = ""  # empty = the official endpoint of that kind
    api_key_env: str = ""  # optional: environment variable to read the key from when none is stored
    key_hint: str = ""  # last characters of the stored key, only for display ("••••a1B2")
    preset: str = ""  # the preset it was created from (notes, links and model hints in the UI)


def local_ollama(url: str) -> ProviderConfig:
    return ProviderConfig(id=LOCAL_OLLAMA_ID, name="Ollama (this computer)", kind="ollama", base_url=url)


class Settings(BaseModel):
    setup_completed: bool = False
    autostart: bool = True

    # Workflow: "approve" = a human accepts or rejects every letter before it is filed (human in the loop),
    # "automatic" = confident letters are filed directly, only unsure ones wait for a human.
    workflow_mode: Literal["approve", "automatic"] = "approve"

    # Folders
    inbox_folder: str = ""
    output_folder: str = ""
    review_folder_name: str = "_Review"
    failed_folder_name: str = "_Failed"
    rejected_folder_name: str = "_Rejected"
    process_existing_on_start: bool = True
    move_files: bool = True
    recursive_watch: bool = False

    # Language of the letters (OCR hint, subject language, default document types)
    language: str = "de"

    # AI models. Every model call goes through LangChain. Each role runs on a provider: "ollama" is the
    # built-in local Ollama server at `ollama_url`; `providers` holds the APIs added with your own key.
    ollama_url: str = "http://127.0.0.1:11434"
    providers: list[ProviderConfig] = Field(default_factory=list)
    decision_provider: str = LOCAL_OLLAMA_ID
    decision_model: str = "qwen3.5:4b"
    extraction_provider: str = ""  # empty = the decision model's provider
    extraction_model: str = ""  # empty = use the decision model
    ocr_llm_provider: str = LOCAL_OLLAMA_ID
    ocr_llm_model: str = "glm-ocr"
    keep_alive: str = "15m"  # Ollama only
    num_ctx: int = 8192  # Ollama only (also sizes the decision prompt)
    # How the decision model reports probabilities: "logprobs" reads token probabilities (one forward pass,
    # best calibrated), "structured" asks for a probability per option as structured output (for APIs without
    # token probabilities, e.g. Anthropic), "auto" uses logprobs when the API returns them, else structured.
    decision_method: Literal["auto", "logprobs", "structured"] = "auto"

    # OCR
    ocr_mode: Literal["auto", "fast_only", "llm_always"] = "auto"
    fast_ocr_size: Literal["tiny", "small", "medium"] = "small"
    fast_ocr_device: Literal["auto", "gpu", "cpu"] = "auto"
    max_pages: int = 2
    render_dpi: int = 200
    min_text_layer_chars: int = 120

    # Decisions
    confidence_threshold: float = 0.80
    verify_recipient: bool = True
    debias_order: bool = True
    route_unknown_by_description: bool = True
    auto_create_recipients: bool = False

    # Filing
    folder_template: str = "{worker}/{recipient}"
    filename_template: str = "{date}_{sender}_{type}"
    document_types: list[DocumentType] = Field(default_factory=lambda: default_document_types("de"))

    # UI
    theme: Literal["auto", "light", "dark"] = "auto"

    # Updates (installed builds only): "" = the channel this build came from
    update_channel: Literal["", "stable", "dev"] = ""
    auto_update: bool = True
    update_check_minutes: int = 0  # 0 = automatic (dev: every 5 minutes, stable: hourly)
    last_version: str = ""  # to announce "updated to …" after an automatic update

    @property
    def effective_extraction_model(self) -> str:
        return self.extraction_model or self.decision_model

    @property
    def effective_extraction_provider(self) -> str:
        return self.extraction_provider or self.decision_provider

    def all_providers(self) -> list[ProviderConfig]:
        """The built-in local Ollama first, then the providers the user added."""
        return [local_ollama(self.ollama_url), *(p for p in self.providers if p.id != LOCAL_OLLAMA_ID)]

    def provider(self, provider_id: str) -> ProviderConfig | None:
        return next((p for p in self.all_providers() if p.id == provider_id), None)

    def role(self, role: Role) -> tuple[ProviderConfig | None, str]:
        """(provider, model) a role runs on."""
        if role == "decision":
            return self.provider(self.decision_provider), self.decision_model
        if role == "extraction":
            return self.provider(self.effective_extraction_provider), self.effective_extraction_model
        return self.provider(self.ocr_llm_provider), self.ocr_llm_model

    def active_roles(self) -> list[Role]:
        roles: list[Role] = ["decision", "extraction"]
        if self.ocr_mode != "fast_only" and self.ocr_llm_model:
            roles.append("ocr_llm")
        return roles

    def model_label(self, role: Role) -> str:
        """'qwen3.5:4b' for the local Ollama, 'gpt-4.1-mini · OpenAI' for other providers."""
        provider, model = self.role(role)
        if provider is None or provider.id == LOCAL_OLLAMA_ID:
            return model
        return f"{model} · {provider.name}"

    @property
    def review_dir(self) -> Path:
        return Path(self.output_folder) / self.review_folder_name

    @property
    def failed_dir(self) -> Path:
        return Path(self.output_folder) / self.failed_folder_name

    @property
    def rejected_dir(self) -> Path:
        return Path(self.output_folder) / self.rejected_folder_name

    def problems(self) -> list[str]:
        """Things that prevent the watcher from starting, in plain language."""
        issues = []
        if not self.inbox_folder:
            issues.append("No inbox folder selected.")
        elif not Path(self.inbox_folder).is_dir():
            issues.append(f"Inbox folder does not exist: {self.inbox_folder}")
        if not self.output_folder:
            issues.append("No output folder selected.")
        if self.inbox_folder and self.output_folder:
            inbox, output = Path(self.inbox_folder).resolve(), Path(self.output_folder).resolve()
            if inbox == output:
                issues.append("Inbox and output folder must be different.")
            elif self.recursive_watch and output.is_relative_to(inbox):
                issues.append("The output folder is inside the inbox; turn off 'watch subfolders'.")
        if not self.document_types:
            issues.append("Add at least one document type.")
        for role in self.active_roles():
            provider, model = self.role(role)
            title = ROLE_TITLES[role]
            if provider is None:
                issues.append(f"The provider of the {title} was removed. Choose one in Settings → AI models.")
            elif not model.strip():
                issues.append(f"No model chosen for the {title} on {provider.name} (Settings → AI models).")
        return issues


class SettingsStore:
    """Thread-safe holder of the current settings with atomic JSON persistence."""

    def __init__(self, path: Path | None = None):
        self._path = path or paths.settings_file()
        self._lock = threading.RLock()
        self._listeners: list[Callable[[Settings], None]] = []
        self._settings = self._load()

    def _load(self) -> Settings:
        if self._path.exists():
            try:
                return Settings.model_validate(json.loads(self._path.read_text(encoding="utf-8")))
            except (OSError, ValueError, ValidationError) as exc:
                backup = self._path.with_suffix(".broken.json")
                log.error("Could not read %s (%s); starting with defaults, old file kept as %s", self._path, exc, backup)
                try:
                    os.replace(self._path, backup)
                except OSError:
                    pass
        return Settings()

    def get(self) -> Settings:
        with self._lock:
            return self._settings.model_copy(deep=True)

    def save(self, settings: Settings) -> None:
        with self._lock:
            self._settings = settings.model_copy(deep=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(self._settings.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(self.get())
            except Exception:  # a broken listener must not break saving
                log.exception("Settings listener failed")

    def update(self, **changes) -> Settings:
        with self._lock:
            new = self._settings.model_copy(update=changes, deep=True)
            new = Settings.model_validate(new.model_dump())
            self.save(new)
            return self.get()

    def on_change(self, listener: Callable[[Settings], None]) -> None:
        self._listeners.append(listener)
