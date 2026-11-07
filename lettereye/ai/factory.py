"""Builds the AI parts of the pipeline from the settings: each role on its provider, all through LangChain."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from ..settings import Settings
from .decision import MAX_OPTIONS, DecisionEngine
from .extraction import FactExtractor, LetterFacts
from .providers import AdaptiveModel, ModelTarget, ProviderError, resolve_target


def _ollama(settings: Settings) -> dict[str, Any]:
    return {"num_ctx": settings.num_ctx, "keep_alive": settings.keep_alive}


def _local_options(target: ModelTarget, settings: Settings) -> dict[str, Any]:
    return _ollama(settings) if target.kind == "ollama" else {}


def build_decision_engine(settings: Settings, keys: Any) -> DecisionEngine:
    target = resolve_target(settings, "decision", keys)
    method = settings.decision_method
    if method == "logprobs" and not target.info.logprobs:
        raise ProviderError(f"{target.provider.name} ({target.info.title}) returns no token probabilities. "
                            "Set the decision method to Auto or Structured (Settings → Decisions).")
    use_logprobs = method == "logprobs" or (method == "auto" and target.info.logprobs)
    llm = None
    if use_logprobs:
        llm = AdaptiveModel(target, {"temperature": 0.0, "max_tokens": 12, "logprobs": MAX_OPTIONS,
                                     "reasoning": False, **_local_options(target, settings)})
    # A structured answer is a short tool call / JSON object. OpenAI and Gemini reasoning models count their
    # thinking against the token limit, so only Ollama and Anthropic (where it is required) get one.
    structured_budget = {"ollama": 600, "anthropic": 1024}.get(target.kind)

    def structured(schema):
        return AdaptiveModel(target, {"temperature": 0.0, "max_tokens": structured_budget, "reasoning": False,
                                      **_local_options(target, settings)}, schema=schema)

    return DecisionEngine(llm, structured=structured, method=method, debias=settings.debias_order,
                          num_ctx=settings.num_ctx, model=target.label)


def build_extractor(settings: Settings, keys: Any) -> FactExtractor:
    target = resolve_target(settings, "extraction", keys)
    budget = {"ollama": 400, "anthropic": 1024}.get(target.kind)
    chain = AdaptiveModel(target, {"temperature": 0.0, "max_tokens": budget, "reasoning": False,
                                   **_local_options(target, settings)}, schema=LetterFacts)
    return FactExtractor(chain, language=settings.language)


def build_llm_ocr(settings: Settings, keys: Any):
    from ..ocr.llm_ocr import LlmOCR

    target = resolve_target(settings, "ocr_llm", keys)
    options: dict[str, Any] = {"temperature": 0.0, "max_tokens": 3072, "reasoning": None}
    if target.kind == "ollama":
        options.update(num_ctx=16384, keep_alive=settings.keep_alive, ollama_options={"repeat_penalty": 1.05})
    elif target.kind == "anthropic":
        options["max_tokens"] = 4096
    device = "Ollama" if target.provider.id == "ollama" else target.provider.name
    return LlmOCR(AdaptiveModel(target, options), target.model, device=device)


def warm_up(settings: Settings, keys: Any) -> list[str]:
    """Load the Ollama models of the active roles into (GPU) memory, so the first letter is not slow.

    Cloud models need no warm-up (and a call would cost money), so they are skipped.
    """
    warmed: list[str] = []
    for role in ("decision", "extraction"):
        target = resolve_target(settings, role, keys)
        if target.kind != "ollama" or target.label in warmed:
            continue
        model = AdaptiveModel(target, {"max_tokens": 1, "reasoning": False, "keep_alive": settings.keep_alive})
        model.invoke([HumanMessage("Hi")])
        warmed.append(target.label)
    return warmed


def check_decision_model(settings: Settings, keys: Any) -> dict[str, Any]:
    """One small decision as a live test: does the model answer, and how are probabilities obtained?"""
    engine = build_decision_engine(settings, keys)
    state = "Dear Ms Alice Example,\nplease find enclosed your invoice no. 4711 for March.\nKind regards"
    result = engine.choice(state, "Who is the addressee (recipient) of this letter?",
                           ["Bob Sample", "Alice Example", "Someone else (not in this list)"], debias=False)
    return {"choice": result.choice, "confidence": result.confidence, "method": result.method,
            "correct": result.choice == "Alice Example"}
