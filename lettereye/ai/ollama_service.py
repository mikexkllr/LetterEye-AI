"""Talking to the local Ollama server: health, installed models, downloads, GPU placement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx
import ollama

OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"


@dataclass(frozen=True)
class LoadedModel:
    name: str
    size_bytes: int
    vram_bytes: int

    @property
    def gpu_share(self) -> float:
        return self.vram_bytes / self.size_bytes if self.size_bytes else 0.0

    @property
    def placement(self) -> str:
        share = self.gpu_share
        if share >= 0.99:
            return "100% GPU"
        if share <= 0.01:
            return "CPU only"
        return f"{share:.0%} GPU / {1 - share:.0%} CPU"


def normalize_model_name(name: str) -> str:
    name = name.strip()
    return name if ":" in name else f"{name}:latest"


class OllamaService:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = ollama.Client(host=self.base_url, timeout=None)

    def version(self) -> str | None:
        """Ollama version, or None if the server is not reachable."""
        try:
            response = httpx.get(f"{self.base_url}/api/version", timeout=2.5)
            response.raise_for_status()
            return response.json().get("version", "unknown")
        except (httpx.HTTPError, ValueError):
            return None

    def list_models(self) -> list[str]:
        try:
            return sorted(m.model for m in self._client.list().models if m.model)
        except Exception:
            return []

    def has_model(self, name: str) -> bool:
        wanted = normalize_model_name(name)
        return any(normalize_model_name(m) == wanted for m in self.list_models())

    def loaded_models(self) -> list[LoadedModel]:
        try:
            return [LoadedModel(m.model or m.name or "?", m.size or 0, m.size_vram or 0) for m in self._client.ps().models]
        except Exception:
            return []

    def capabilities(self, name: str) -> list[str]:
        try:
            return list(self._client.show(name).capabilities or [])
        except Exception:
            return []

    def pull(self, name: str, on_progress: Callable[[str, float | None], None] | None = None) -> None:
        """Download a model. on_progress(status_text, fraction 0..1 or None)."""
        for part in self._client.pull(name, stream=True):
            if on_progress:
                total, done = part.total or 0, part.completed or 0
                on_progress(part.status or "", (done / total) if total else None)

    def warm_up(self, name: str, keep_alive: str = "15m") -> None:
        """Load a model into (GPU) memory so the first letter is not slow."""
        self._client.generate(model=name, prompt="", keep_alive=keep_alive)
