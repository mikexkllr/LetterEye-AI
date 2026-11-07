"""Model providers: every model call in LetterEye goes through LangChain, wherever the model runs.

A provider is an API dialect plus an address (and, for cloud APIs, your own key):

    kind        LangChain adapter          covers
    ollama      langchain-ollama           the local Ollama server, Ollama on another computer
    openai      langchain-openai           OpenAI and every OpenAI-compatible API: OpenRouter, Groq, Mistral,
                                           DeepSeek, Together, LM Studio, vLLM, llama.cpp, LocalAI, …
    anthropic   langchain-anthropic        Anthropic (Claude) and Anthropic-compatible APIs
    google      langchain-google-genai     Google Gemini and Google-compatible gateways

The decision model needs more than text from a model: it reads *token probabilities* (logprobs). Ollama
and OpenAI-compatible APIs return them through LangChain; Anthropic does not, and the Gemini adapter does
not pass them on. For those the decision engine asks for a probability per option as structured output
instead (see decision.py). `AdaptiveModel` takes care of API differences at runtime: when an API rejects an
option (no thinking mode, fixed temperature, fewer top logprobs, no tool calling) it rebuilds the model
without it and tries again.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from ..settings import LOCAL_OLLAMA_ID, ProviderConfig, ProviderKind, Role, Settings

log = logging.getLogger(__name__)

NO_KEY = "not-needed"  # sent to servers that do not check keys (the SDKs refuse to start without one)


class ProviderError(RuntimeError):
    """A provider could not be reached or rejected the request, in plain language."""


class LogprobsUnavailable(RuntimeError):
    """The API (or this model) does not return token probabilities."""


@dataclass(frozen=True)
class KindInfo:
    title: str
    logprobs: bool  # token probabilities reach LangChain's response_metadata
    structured_method: str  # with_structured_output(method=…)
    default_url: str
    package: str


KINDS: dict[str, KindInfo] = {
    "ollama": KindInfo("Ollama", True, "json_schema", "http://127.0.0.1:11434", "langchain-ollama"),
    "openai": KindInfo("OpenAI-compatible", True, "function_calling", "https://api.openai.com/v1", "langchain-openai"),
    "anthropic": KindInfo("Anthropic-compatible", False, "function_calling", "https://api.anthropic.com",
                          "langchain-anthropic"),
    "google": KindInfo("Google Gemini", False, "json_schema", "https://generativelanguage.googleapis.com",
                       "langchain-google-genai"),
}


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    kind: ProviderKind
    base_url: str
    api_key_env: str = ""
    needs_key: bool = True
    model_hint: str = ""
    note: str = ""
    key_url: str = ""


PRESETS: list[Preset] = [
    Preset("openai", "OpenAI", "openai", "https://api.openai.com/v1", "OPENAI_API_KEY", model_hint="gpt-4.1-mini",
           note="Token probabilities with chat models such as gpt-4.1 and gpt-4o. Reasoning models do not return "
                "them and use structured confidence instead.",
           key_url="https://platform.openai.com/api-keys"),
    Preset("anthropic", "Anthropic (Claude)", "anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY",
           model_hint="claude-haiku-5-5",
           note="The API returns no token probabilities, so decisions use structured confidence.",
           key_url="https://console.anthropic.com/settings/keys"),
    Preset("google", "Google Gemini", "google", "", "GEMINI_API_KEY", model_hint="gemini-2.5-flash",
           note="Decisions use structured confidence (LangChain's Gemini adapter does not pass token "
                "probabilities on).",
           key_url="https://aistudio.google.com/apikey"),
    Preset("openrouter", "OpenRouter", "openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
           model_hint="openai/gpt-4.1-mini",
           note="Hundreds of models behind one key. Whether token probabilities are returned depends on the model.",
           key_url="https://openrouter.ai/keys"),
    Preset("groq", "Groq", "openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
           key_url="https://console.groq.com/keys"),
    Preset("mistral", "Mistral AI", "openai", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
           key_url="https://console.mistral.ai/api-keys"),
    Preset("deepseek", "DeepSeek", "openai", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY",
           model_hint="deepseek-chat", key_url="https://platform.deepseek.com/api_keys"),
    Preset("together", "Together AI", "openai", "https://api.together.xyz/v1", "TOGETHER_API_KEY",
           key_url="https://api.together.ai/settings/api-keys"),
    Preset("lmstudio", "LM Studio (local)", "openai", "http://127.0.0.1:1234/v1", needs_key=False,
           note="Start LM Studio's local server first."),
    Preset("vllm", "vLLM / llama.cpp / LocalAI (local)", "openai", "http://127.0.0.1:8000/v1", needs_key=False,
           note="Any OpenAI-compatible server – also the place to serve a decision model you fine-tuned on "
                "LetterEye's exported training data. Token probabilities included."),
    Preset("ollama_remote", "Ollama on another computer", "ollama", "http://192.168.1.10:11434", needs_key=False,
           note="A stronger GPU in your network. An API key is only needed behind an authenticating proxy."),
    Preset("custom_openai", "Other OpenAI-compatible API", "openai", ""),
    Preset("custom_anthropic", "Other Anthropic-compatible API", "anthropic", ""),
    Preset("custom_google", "Google-compatible gateway", "google", ""),
]


def preset(key: str) -> Preset | None:
    return next((p for p in PRESETS if p.key == key), None)


# ---------------------------------------------------------------------- addresses and keys
def base_url(provider: ProviderConfig) -> str:
    url = (provider.base_url or KINDS[provider.kind].default_url).strip().rstrip("/")
    if provider.kind == "anthropic" and url.endswith("/v1"):
        url = url[:-3]  # the Anthropic SDK adds /v1 itself
    return url


def is_local_url(url: str) -> bool:
    """True for this computer and the local network: letters sent there do not leave your network."""
    host = (urlparse(url if "//" in url else f"http://{url}").hostname or "").lower()
    if not host:
        return False
    if host == "localhost" or host.endswith((".localhost", ".local", ".lan", ".home.arpa")) or "." not in host:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


def is_local(provider: ProviderConfig) -> bool:
    return is_local_url(base_url(provider))


def remote_providers(settings: Settings) -> list[ProviderConfig]:
    """Providers outside your network that the active roles use – letters are sent there."""
    found: dict[str, ProviderConfig] = {}
    for role in settings.active_roles():
        provider = settings.role(role)[0]
        if provider is not None and not is_local(provider):
            found[provider.id] = provider
    return list(found.values())


def needs_key(provider: ProviderConfig) -> bool:
    if provider.kind == "ollama":
        return False
    return provider.kind in ("anthropic", "google") or not is_local(provider)


KeyLookup = Callable[[str], str]


def env_key(provider: ProviderConfig) -> str:
    """The key from the provider's own environment variable (if it names one)."""
    return os.environ.get(provider.api_key_env, "").strip() if provider.api_key_env else ""


def api_key_for(provider: ProviderConfig, keys: Any) -> str:
    """The stored key, else the provider's own environment variable. Never another provider's key."""
    lookup: KeyLookup = keys.get if hasattr(keys, "get") else keys
    stored = lookup(provider.id) if keys is not None else ""
    return stored or env_key(provider)


@dataclass(frozen=True)
class ModelTarget:
    """One model on one provider, with the key to use – everything needed to build a LangChain model."""
    provider: ProviderConfig
    model: str
    api_key: str = ""

    @property
    def kind(self) -> str:
        return self.provider.kind

    @property
    def info(self) -> KindInfo:
        return KINDS[self.provider.kind]

    @property
    def url(self) -> str:
        return base_url(self.provider)

    @property
    def local(self) -> bool:
        return is_local(self.provider)

    @property
    def label(self) -> str:
        return self.model if self.provider.id == LOCAL_OLLAMA_ID else f"{self.model} · {self.provider.name}"


def resolve_target(settings: Settings, role: Role, keys: Any) -> ModelTarget:
    provider, model = settings.role(role)
    if provider is None:
        raise ProviderError(f"The provider for the {role.replace('_', ' ')} model no longer exists.")
    return ModelTarget(provider, model.strip(), api_key_for(provider, keys))


# ---------------------------------------------------------------------- LangChain models
def build_chat_model(target: ModelTarget, *, temperature: float | None = 0.0, max_tokens: int | None = None,
                     logprobs: int | None = None, reasoning: bool | None = False, num_ctx: int | None = None,
                     keep_alive: str | int | None = None, timeout: float | None = None,
                     ollama_options: dict[str, Any] | None = None) -> Any:
    """A LangChain chat model for `target`.

    logprobs: number of top alternatives to return per token (None = no token probabilities).
    reasoning: False = thinking off (the first token must be the answer), None = the model's default.
    num_ctx, keep_alive and ollama_options only apply to Ollama.
    """
    kind, key = target.kind, target.api_key
    if kind == "ollama":
        from langchain_ollama import ChatOllama

        params: dict[str, Any] = dict(model=target.model, base_url=target.url, reasoning=reasoning,
                                      num_ctx=num_ctx, keep_alive=keep_alive, **(ollama_options or {}))
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["num_predict"] = max_tokens
        if logprobs:
            params.update(logprobs=True, top_logprobs=logprobs)
        if key:
            params["client_kwargs"] = {"headers": {"Authorization": f"Bearer {key}"}}
        return ChatOllama(**{k: v for k, v in params.items() if v is not None or k == "reasoning"})

    if kind == "openai":
        from langchain_openai import ChatOpenAI

        params = dict(model=target.model, base_url=target.url, api_key=key or NO_KEY, max_retries=2, timeout=timeout,
                      temperature=temperature, max_tokens=max_tokens)
        if logprobs:
            params.update(logprobs=True, top_logprobs=logprobs)
        return ChatOpenAI(**{k: v for k, v in params.items() if v is not None})

    if kind == "anthropic":
        from langchain_anthropic import ChatAnthropic

        params = dict(model=target.model, base_url=target.url, api_key=key or NO_KEY, max_retries=2, timeout=timeout,
                      temperature=temperature, max_tokens=max_tokens or 1024)
        return ChatAnthropic(**{k: v for k, v in params.items() if v is not None})

    if kind == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        params = dict(model=target.model, api_key=key or NO_KEY, retries=2, timeout=timeout,
                      temperature=temperature, max_output_tokens=max_tokens,
                      thinking_budget=0 if reasoning is False else None)
        if target.provider.base_url:
            params["base_url"] = target.url
        return ChatGoogleGenerativeAI(**{k: v for k, v in params.items() if v is not None})

    raise ProviderError(f"Unknown provider kind: {kind}")


def relax_options(options: dict[str, Any], structured_method: str | None, error: Exception
                  ) -> tuple[dict[str, Any], str | None] | None:
    """Options without the one the API just rejected, or None when the error is not about an option.

    Raises LogprobsUnavailable when the API cannot return token probabilities at all.
    """
    message = str(error).lower()
    relaxed = dict(options)
    if "logprob" in message and relaxed.get("logprobs"):
        if "top_logprobs" in message and relaxed["logprobs"] > 5:
            relaxed["logprobs"] = 5  # some APIs return at most 5 alternatives
            return relaxed, structured_method
        raise LogprobsUnavailable(str(error)) from error
    if "think" in message and relaxed.get("reasoning") is not None:
        relaxed["reasoning"] = None  # no thinking mode (Ollama) or thinking cannot be switched off (Gemini)
        return relaxed, structured_method
    if "temperature" in message and relaxed.get("temperature") is not None:
        relaxed["temperature"] = None  # models with a fixed temperature
        return relaxed, structured_method
    if any(word in message for word in ("max_tokens", "max_completion_tokens", "max_output_tokens")) \
            and relaxed.get("max_tokens") is not None:
        relaxed["max_tokens"] = None
        return relaxed, structured_method
    if structured_method == "function_calling" and ("tool" in message or "function" in message):
        return relaxed, "json_schema"  # an OpenAI-compatible server without tool calling
    return None


class AdaptiveModel:
    """A LangChain chat model (optionally with structured output) that adapts to what the API accepts.

    `invoke` and `stream` behave like the LangChain model's. If the API rejects one of the options, the model
    is rebuilt without it and the call is repeated – once per option, so a real error still surfaces.
    """

    def __init__(self, target: ModelTarget, options: dict[str, Any] | None = None, *,
                 schema: type[BaseModel] | None = None, structured_method: str | None = None,
                 build: Callable[..., Any] = build_chat_model):
        self.target = target
        self.options = dict(options or {})
        self.schema = schema
        self.structured_method = structured_method or (target.info.structured_method if schema else None)
        self._build = build
        self._runnable = self._make()

    def _make(self) -> Any:
        model = self._build(self.target, **self.options)
        if self.schema is not None:
            return model.with_structured_output(self.schema, method=self.structured_method)
        return model

    def _adapt(self, error: Exception) -> bool:
        relaxed = relax_options(self.options, self.structured_method, error)
        if relaxed is None:
            return False
        options, method = relaxed
        if options == self.options and method == self.structured_method:
            return False
        log.info("%s rejected an option (%s); retrying with %s", self.target.label, str(error)[:200],
                 {k: v for k, v in options.items() if self.options.get(k) != v} or {"structured": method})
        self.options, self.structured_method = options, method
        self._runnable = self._make()
        return True

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        for _ in range(6):
            try:
                return self._runnable.invoke(messages, **kwargs)
            except Exception as exc:
                if not self._adapt(exc):
                    raise
        return self._runnable.invoke(messages, **kwargs)

    def stream(self, messages: Any, **kwargs: Any) -> Iterator[Any]:
        for _ in range(6):
            try:
                chunks = iter(self._runnable.stream(messages, **kwargs))
                first = next(chunks, None)
                break
            except Exception as exc:  # options are rejected before the first chunk arrives
                if not self._adapt(exc):
                    raise
        else:
            chunks, first = iter(self._runnable.stream(messages, **kwargs)), None
        if first is not None:
            yield first
        yield from chunks


# ---------------------------------------------------------------------- talking to the API (no model calls)
def list_models(provider: ProviderConfig, api_key: str = "", *, client: httpx.Client | None = None,
                timeout: float = 10.0) -> list[str]:
    """The models a provider offers (for the model picker and to test a key). Free: no tokens are used."""
    url = base_url(provider)
    if provider.kind == "ollama":
        from .ollama_service import OllamaService

        service = OllamaService(url, api_key=api_key)
        if service.version() is None:
            raise ProviderError(f"Ollama is not reachable at {url}.")
        return service.list_models()
    http = client or httpx.Client(timeout=timeout)
    try:
        if provider.kind == "openai":
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            data = _get_json(http, f"{url}/models", headers, provider)
            return sorted({str(m.get("id")) for m in data.get("data", []) if m.get("id")})
        if provider.kind == "anthropic":
            headers = {"x-api-key": api_key, "Authorization": f"Bearer {api_key}", "anthropic-version": "2023-06-01"}
            data = _get_json(http, f"{url}/v1/models?limit=1000", headers, provider)
            return [str(m["id"]) for m in data.get("data", []) if m.get("id")]
        if provider.kind == "google":
            names: list[str] = []
            page = ""
            for _ in range(10):
                data = _get_json(http, f"{url}/v1beta/models?pageSize=1000{'&pageToken=' + page if page else ''}",
                                 {"x-goog-api-key": api_key}, provider)
                names += [str(m["name"]).removeprefix("models/") for m in data.get("models", [])
                          if "generateContent" in (m.get("supportedGenerationMethods") or ["generateContent"])]
                page = data.get("nextPageToken") or ""
                if not page:
                    break
            return sorted(names)
    finally:
        if client is None:
            http.close()
    raise ProviderError(f"Unknown provider kind: {provider.kind}")


def _get_json(http: httpx.Client, url: str, headers: dict[str, str], provider: ProviderConfig) -> dict:
    try:
        response = http.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise ProviderError(f"Cannot reach {provider.name} at {base_url(provider)} ({exc.__class__.__name__}).") from exc
    if response.status_code in (401, 403):
        raise ProviderError(f"{provider.name} rejected the API key (HTTP {response.status_code}).")
    if response.status_code == 404:
        raise ProviderError(f"{provider.name}: no model list at {url} – check the address.")
    if response.status_code >= 400:
        raise ProviderError(f"{provider.name} answered HTTP {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(f"{provider.name} did not answer with JSON – check the address.") from exc
