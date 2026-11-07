"""Providers: LangChain models for every API kind, key handling, adapting to what an API accepts."""

import sys

import httpx
import pytest
from langchain_core.messages import AIMessage

from lettereye.ai import factory
from lettereye.ai.decision import (
    DecisionEngine,
    DecisionError,
    option_schema,
    structured_distribution,
    token_logprobs,
)
from lettereye.ai.providers import (
    PRESETS,
    AdaptiveModel,
    LogprobsUnavailable,
    ModelTarget,
    ProviderError,
    api_key_for,
    base_url,
    build_chat_model,
    is_local_url,
    list_models,
    needs_key,
    relax_options,
)
from lettereye.api_keys import SecretStore, mask
from lettereye.settings import ProviderConfig, Settings

from .conftest import ScriptedLLM, logprob_response

OPENAI = ProviderConfig(id="p-openai", name="OpenAI", kind="openai", base_url="https://api.openai.com/v1",
                        api_key_env="LETTEREYE_TEST_OPENAI_KEY")
CLAUDE = ProviderConfig(id="p-claude", name="Anthropic", kind="anthropic", base_url="https://api.anthropic.com/v1")
GEMINI = ProviderConfig(id="p-gemini", name="Gemini", kind="google")
LMSTUDIO = ProviderConfig(id="p-lm", name="LM Studio", kind="openai", base_url="http://127.0.0.1:1234/v1")


def settings_with(provider: ProviderConfig, model: str, **extra) -> Settings:
    return Settings(providers=[provider], decision_provider=provider.id, decision_model=model,
                    ocr_llm_provider=provider.id, ocr_llm_model=model, **extra)


# ---------------------------------------------------------------------- LangChain models per kind
def test_openai_compatible_model_asks_for_logprobs():
    model = build_chat_model(ModelTarget(LMSTUDIO, "my-decider"), max_tokens=12, logprobs=20)
    assert type(model).__name__ == "ChatOpenAI"
    assert model.openai_api_base == "http://127.0.0.1:1234/v1"
    assert model.logprobs is True and model.top_logprobs == 20 and model.max_tokens == 12
    assert model.temperature == 0.0
    assert model.openai_api_key.get_secret_value() == "not-needed"  # local servers need no key


def test_anthropic_model_uses_the_key_and_the_base_url_without_v1():
    model = build_chat_model(ModelTarget(CLAUDE, "claude-haiku-5-5", "sk-ant-secret"))
    assert type(model).__name__ == "ChatAnthropic"
    assert model.anthropic_api_url == "https://api.anthropic.com"
    assert model.anthropic_api_key.get_secret_value() == "sk-ant-secret"
    assert model.max_tokens == 1024  # required by the API


def test_google_model_switches_thinking_off_for_decisions():
    model = build_chat_model(ModelTarget(GEMINI, "gemini-2.5-flash", "AIza-key"), max_tokens=50)
    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert model.thinking_budget == 0 and model.max_output_tokens == 50


def test_ollama_model_keeps_local_options():
    target = ModelTarget(Settings().provider("ollama"), "qwen3.5:4b")
    model = build_chat_model(target, max_tokens=12, logprobs=20, num_ctx=4096, keep_alive="5m")
    assert type(model).__name__ == "ChatOllama"
    assert (model.num_predict, model.num_ctx, model.top_logprobs, model.reasoning) == (12, 4096, 20, False)


def test_factory_builds_every_role_for_a_cloud_provider_without_network():
    settings = settings_with(OPENAI, "gpt-4.1-mini")
    keys = {"p-openai": "sk-test"}.get
    engine = factory.build_decision_engine(settings, keys)
    assert engine.active_method == "logprobs" and engine.model == "gpt-4.1-mini · OpenAI"
    assert factory.build_extractor(settings, keys) is not None
    ocr = factory.build_llm_ocr(settings, keys)
    assert ocr.device == "OpenAI" and ocr.model == "gpt-4.1-mini"


def test_anthropic_decisions_use_structured_answers():
    engine = factory.build_decision_engine(settings_with(CLAUDE, "claude-haiku-5-5"), lambda _id: "k")
    assert engine.active_method == "structured"
    with pytest.raises(ProviderError):
        factory.build_decision_engine(settings_with(CLAUDE, "claude-haiku-5-5", decision_method="logprobs"),
                                      lambda _id: "k")


# ---------------------------------------------------------------------- addresses and keys
@pytest.mark.parametrize("url,local", [
    ("http://127.0.0.1:11434", True), ("http://localhost:1234/v1", True), ("http://192.168.1.10:11434", True),
    ("http://gpu-box:8000/v1", True), ("http://studio.local:1234", True), ("https://api.openai.com/v1", False),
    ("https://openrouter.ai/api/v1", False), ("http://8.8.8.8", False),
])
def test_local_addresses(url, local):
    assert is_local_url(url) is local


def test_keys_are_needed_for_cloud_apis_only():
    assert needs_key(OPENAI) and needs_key(CLAUDE) and needs_key(GEMINI)
    assert not needs_key(LMSTUDIO)
    assert not needs_key(Settings().provider("ollama"))


def test_key_lookup_never_borrows_another_providers_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-from-env")
    openrouter = ProviderConfig(id="p-or", name="OpenRouter", kind="openai", base_url="https://openrouter.ai/api/v1")
    assert api_key_for(openrouter, lambda _id: "") == ""  # OPENAI_API_KEY is OpenAI's, not OpenRouter's
    monkeypatch.setenv("LETTEREYE_TEST_OPENAI_KEY", "sk-from-its-own-env")
    assert api_key_for(OPENAI, lambda _id: "") == "sk-from-its-own-env"
    assert api_key_for(OPENAI, lambda _id: "sk-stored") == "sk-stored"  # a stored key wins


def test_presets_are_complete():
    assert {p.kind for p in PRESETS} == {"ollama", "openai", "anthropic", "google"}
    assert len({p.key for p in PRESETS}) == len(PRESETS)
    assert base_url(GEMINI) == "https://generativelanguage.googleapis.com"


# ---------------------------------------------------------------------- the secret store
def test_secret_store_round_trip(tmp_path):
    store = SecretStore(tmp_path / "secrets.json")
    store.set("p-openai", "  sk-very-secret-value  ")
    assert store.get("p-openai") == "sk-very-secret-value"
    assert SecretStore(tmp_path / "secrets.json").get("p-openai") == "sk-very-secret-value"  # persisted
    raw = (tmp_path / "secrets.json").read_text(encoding="utf-8")
    if sys.platform == "win32":
        assert "sk-very-secret-value" not in raw  # DPAPI-encrypted
    else:
        assert (tmp_path / "secrets.json").stat().st_mode & 0o777 == 0o600
    store.set("p-claude", "sk-ant")
    store.prune({"p-claude"})
    assert store.get("p-openai") == "" and store.names() == ["p-claude"]
    store.delete("p-claude")
    assert store.names() == []


def test_secret_store_survives_a_broken_file(tmp_path):
    path = tmp_path / "secrets.json"
    path.write_text("{not json", encoding="utf-8")
    store = SecretStore(path)
    assert store.get("anything") == ""
    store.set("p", "value")
    assert store.get("p") == "value"


def test_keys_are_never_written_to_the_settings_file():
    assert "api_key" not in ProviderConfig.model_fields
    assert mask("sk-proj-abcdefghijkl1234") == "1234" and mask("short") == ""


# ---------------------------------------------------------------------- adapting to the API
def test_relax_options_per_error():
    base = {"temperature": 0.0, "max_tokens": 12, "logprobs": 20, "reasoning": False}
    assert relax_options(base, None, ValueError('"think" value is not supported'))[0]["reasoning"] is None
    assert relax_options(base, None, ValueError("Unsupported value: 'temperature'"))[0]["temperature"] is None
    assert relax_options(base, None, ValueError("top_logprobs must be <= 5"))[0]["logprobs"] == 5
    with pytest.raises(LogprobsUnavailable):
        relax_options(base, None, ValueError("logprobs is not supported with this model"))
    assert relax_options(base, "function_calling", ValueError("tool_choice is not supported"))[1] == "json_schema"
    assert relax_options(base, None, ConnectionError("connection refused")) is None


def test_adaptive_model_retries_without_the_rejected_option():
    built = []

    class Model:
        def __init__(self, options):
            self.options = options

        def invoke(self, messages):
            if self.options.get("reasoning") is False:
                raise ValueError("registry.ollama.ai/library/gemma does not support thinking")
            return AIMessage(content="ok")

    def build(target, **options):
        built.append(options)
        return Model(options)

    model = AdaptiveModel(ModelTarget(LMSTUDIO, "m"), {"reasoning": False, "temperature": 0.0}, build=build)
    assert model.invoke([]).content == "ok"
    assert built[-1] == {"reasoning": None, "temperature": 0.0}


def test_adaptive_model_does_not_hide_real_errors():
    def build(target, **options):
        class Broken:
            def invoke(self, messages):
                raise ConnectionError("connection refused")
        return Broken()

    with pytest.raises(ConnectionError):
        AdaptiveModel(ModelTarget(LMSTUDIO, "m"), {"temperature": 0.0}, build=build).invoke([])


# ---------------------------------------------------------------------- decisions on any API
def openai_logprobs(probabilities: dict[str, float]) -> AIMessage:
    """An AIMessage shaped like ChatOpenAI's reply with logprobs=True (OpenAI-compatible APIs)."""
    import math

    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top = [{"token": t, "logprob": math.log(p), "bytes": list(t.encode())} for t, p in ranked]
    entry = {"token": ranked[0][0], "logprob": math.log(ranked[0][1]), "bytes": [], "top_logprobs": top}
    return AIMessage(content=ranked[0][0], response_metadata={"logprobs": {"content": [entry], "refusal": None}})


def test_openai_logprobs_are_read_like_ollamas():
    assert token_logprobs(openai_logprobs({"B": 0.8, "A": 0.2}))[0]["token"] == "B"
    assert token_logprobs(logprob_response({"A": 1.0}))[0]["token"] == "A"
    assert token_logprobs(AIMessage(content="A")) is None

    class OpenAiLike:
        def invoke(self, messages):
            return openai_logprobs({"B": 0.7, "A": 0.3})

    result = DecisionEngine(OpenAiLike(), debias=False).choice("state", "Which?", ["first", "second"])
    assert result.choice == "second" and result.confidence == pytest.approx(0.7)
    assert result.method == "logprobs"


class StructuredFake:
    """Fake chat model with structured output: answers with a probability per option letter."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.schemas = []

    def factory(self, schema):
        self.schemas.append(schema)
        fake = self

        class Runnable:
            def invoke(self, messages):
                prompt = messages[-1].content
                options = [line[3:] for line in prompt.split("OPTIONS:\n", 1)[1].splitlines() if line[1:3] == ") "]
                weights = [fake.scorer(prompt, o) for o in options]
                total = sum(weights)
                labels = list(schema.model_fields)[1:]
                best = labels[weights.index(max(weights))]
                return schema(answer=best, **{label: w / total for label, w in zip(labels, weights, strict=False)})
        return Runnable()


def test_structured_decisions_for_apis_without_logprobs():
    fake = StructuredFake(lambda prompt, option: 9.0 if option == "Bob Smith" else 1.0)
    engine = DecisionEngine(structured=fake.factory, method="structured", debias=True)
    result = engine.choice("Dear Bob Smith", "Who?", ["Alice Johnson", "Bob Smith"])
    assert result.choice == "Bob Smith" and result.confidence == pytest.approx(0.9)
    assert result.method == "structured"
    assert set(fake.schemas[0].model_fields) == {"answer", "A", "B"}


def test_auto_switches_to_structured_when_the_api_returns_no_logprobs():
    class NoLogprobs:
        calls = 0

        def invoke(self, messages):
            NoLogprobs.calls += 1
            return AIMessage(content="B")  # e.g. an OpenAI-compatible server that ignores `logprobs`

    fake = StructuredFake(lambda prompt, option: 3.0 if option == "second" else 1.0)
    engine = DecisionEngine(NoLogprobs(), structured=fake.factory, method="auto", debias=False)
    assert engine.choice("s", "Which?", ["first", "second"]).method == "structured"
    assert engine.active_method == "structured"
    engine.choice("s", "Which?", ["first", "second"])
    assert NoLogprobs.calls == 1  # not asked again

    strict = DecisionEngine(NoLogprobs(), structured=fake.factory, method="logprobs", debias=False)
    with pytest.raises(DecisionError):
        strict.choice("s", "Which?", ["first", "second"])


def test_auto_answers_one_question_structured_when_the_model_writes_prose():
    class Chatty:
        def invoke(self, messages):
            return logprob_response({"The": 0.9, "A": 0.1})

    fake = StructuredFake(lambda prompt, option: 1.0)
    engine = DecisionEngine(Chatty(), structured=fake.factory, method="auto", debias=False)
    assert engine.choice("s", "Which?", ["first", "second"]).method == "structured"
    assert engine.active_method == "logprobs"  # the API does return logprobs; keep using them


def test_structured_distribution_is_robust():
    assert structured_distribution({"answer": "B", "A": 10, "B": 90}, 2) == pytest.approx([0.1, 0.9])
    assert structured_distribution({"answer": "B", "A": 0, "B": 0}, 2) == [0.0, 1.0]
    assert structured_distribution(option_schema(3)(answer="C", A=0.2, B=0.2, C=0.6), 3)[2] == pytest.approx(0.6)
    with pytest.raises(DecisionError):
        structured_distribution(None, 2)


def test_scripted_llm_still_works_through_the_new_engine():
    llm = ScriptedLLM(lambda state, q, option: 10.0 if option in state else 1.0)
    assert DecisionEngine(llm, method="auto").choice("Dear Carol", "Who?", ["Alice", "Carol"]).choice == "Carol"


# ---------------------------------------------------------------------- model lists (no tokens used)
def test_list_models_per_kind():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openai.com":
            assert request.headers["authorization"] == "Bearer sk-x"
            return httpx.Response(200, json={"data": [{"id": "gpt-b"}, {"id": "gpt-a"}]})
        if request.url.host == "api.anthropic.com":
            assert request.url.path == "/v1/models" and request.headers["x-api-key"] == "sk-ant"
            return httpx.Response(200, json={"data": [{"id": "claude-haiku-5-5"}]})
        if request.url.host == "generativelanguage.googleapis.com":
            return httpx.Response(200, json={"models": [
                {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}]})
        return httpx.Response(401)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert list_models(OPENAI, "sk-x", client=client) == ["gpt-a", "gpt-b"]
    assert list_models(CLAUDE, "sk-ant", client=client) == ["claude-haiku-5-5"]
    assert list_models(GEMINI, "AIza", client=client) == ["gemini-2.5-flash"]
    with pytest.raises(ProviderError, match="rejected the API key"):
        list_models(LMSTUDIO, "", client=client)


# ---------------------------------------------------------------------- settings and preflight
def test_old_settings_files_keep_working():
    old = Settings.model_validate({"ollama_url": "http://127.0.0.1:11434", "decision_model": "qwen3.5:4b"})
    assert old.decision_provider == "ollama" and old.role("decision")[0].kind == "ollama"
    assert old.model_label("decision") == "qwen3.5:4b"
    assert old.decision_method == "auto"


def test_settings_report_a_removed_provider():
    settings = Settings(decision_provider="gone", inbox_folder="x", output_folder="y")
    assert any("provider of the decision model was removed" in p for p in settings.problems())


def test_preflight_for_cloud_providers_needs_a_key_but_no_ollama(db, tmp_path):
    from lettereye.events import ActivityFeed
    from lettereye.services.engine import Engine
    from lettereye.settings import SettingsStore

    store = SettingsStore(tmp_path / "settings.json")
    (tmp_path / "in").mkdir()
    store.save(settings_with(OPENAI, "gpt-4.1-mini", inbox_folder=str(tmp_path / "in"),
                             output_folder=str(tmp_path / "out"), ollama_url="http://127.0.0.1:9"))
    keys = SecretStore(tmp_path / "secrets.json")
    engine = Engine(db, store, ActivityFeed(), keys=keys)
    assert engine.preflight() == ["No API key for OpenAI yet (Settings → AI models)."]
    keys.set("p-openai", "sk-test")
    assert engine.preflight() == []  # Ollama (unreachable here) is not needed
