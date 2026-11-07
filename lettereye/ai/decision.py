"""A "System One" decision model in the style of TypeSafe's Jev, on any provider through LangChain.

The model never writes free text. Each question gets a list of options labelled A, B, C, ...; we run one
forward pass, read the model's log-probabilities for the *option letters only* and normalize them. The
answer can therefore only ever be one of the given options, and it comes with a probability.

How the probabilities are obtained (`method`):
* "logprobs": token probabilities, read from LangChain's response metadata – Ollama and OpenAI-compatible
  APIs (OpenAI, vLLM, llama.cpp, LM Studio, OpenRouter, …). One forward pass, best calibrated.
* "structured": for APIs without token probabilities (Anthropic, Gemini, OpenAI reasoning models) the model
  returns a probability for every option letter as structured output (tool call / JSON schema). The answer
  is still restricted to the options; the probabilities are the model's own estimate.
* "auto": logprobs while the API returns them, structured from the first answer without them.

Primitives (same as Jev):
* choice(state, question, options)  -> which option, with a probability for every option
* noul(state, statement)            -> probability that a statement is true
* score(state, question, levels)    -> an ordinal level, with probabilities

Robustness:
* order debiasing: every question is asked twice (original and reversed option order) and averaged,
  which cancels the position bias small models have for "A";
* token variants ("A", " A", "A)") are summed into the same option;
* more options than letters (> 20) are decided in a tournament of groups.

Following Jev's guidance, callers should ask small atomic questions and combine the answers in code
(e.g. "who is the letter addressed to?" then look up that person's worker), rather than one big question.
"""

from __future__ import annotations

import logging
import math
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, create_model

from .providers import LogprobsUnavailable

log = logging.getLogger(__name__)

LABELS = string.ascii_uppercase[:20]  # Ollama returns at most 20 alternatives per token (top_logprobs <= 20)
MAX_OPTIONS = len(LABELS)
_STRIP = " \t\r\n*_`'\"()[]{}.:;,-"
_SKIPPABLE = {"", "<think>", "</think>", "think", "answer", "Answer"}

SYSTEM_PROMPT = (
    "You are a decision model inside a document-sorting application. You never explain and never write "
    "prose. Read the STATE, then answer the QUESTION by replying with exactly one option letter."
)
STRUCTURED_SYSTEM_PROMPT = (
    "You are a decision model inside a document-sorting application. You never explain and never write "
    "prose. Read the STATE, then answer the QUESTION with the structured output: the letter of the best option "
    "and, for every option letter, the probability that it is the correct answer."
)
Method = Literal["auto", "logprobs", "structured"]


class DecisionError(RuntimeError):
    """The model did not answer with one of the option letters."""


@dataclass
class ChoiceResult:
    question: str
    options: list[str]
    probabilities: list[float]
    method: str = "logprobs"  # how the probabilities were obtained: logprobs | structured | both

    @property
    def index(self) -> int:
        return max(range(len(self.probabilities)), key=self.probabilities.__getitem__)

    @property
    def choice(self) -> str:
        return self.options[self.index]

    @property
    def confidence(self) -> float:
        return self.probabilities[self.index]

    @property
    def margin(self) -> float:
        ranked = sorted(self.probabilities, reverse=True)
        return ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)

    def probability_of(self, option: str) -> float:
        return self.probabilities[self.options.index(option)]

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(zip(self.options, self.probabilities, strict=True), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "choice": self.choice,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "options": [{"option": o, "p": round(p, 4)} for o, p in zip(self.options, self.probabilities, strict=True)],
        }


def token_logprobs(response: Any) -> list[dict[str, Any]] | None:
    """Per-token logprobs from a LangChain chat response, in one shape for every provider, or None.

    Ollama puts a list into response_metadata["logprobs"]; OpenAI-compatible APIs a dict whose "content"
    holds the same entries ({"token", "logprob", "top_logprobs": [...]}).
    """
    metadata = getattr(response, "response_metadata", None) or {}
    logprobs = metadata.get("logprobs")
    if isinstance(logprobs, dict):
        logprobs = logprobs.get("content")
    if logprobs is None:
        return None
    return [dict(p) if not isinstance(p, dict) else p for p in logprobs]


def option_distribution(logprobs: Sequence[dict[str, Any]] | None, n_options: int) -> list[float]:
    """Turn Ollama per-token logprobs into a probability per option label.

    Uses the first generated token that is an option letter. Leading tokens that carry no answer
    (whitespace, markdown, empty think tags) are skipped. Raises DecisionError if the model starts
    writing something else.
    """
    labels = tuple(LABELS[:n_options])  # a tuple, not a str: '' and 'AB' must not count as labels
    for position in logprobs or []:
        token = str(position.get("token", "")).strip(_STRIP)
        if token in labels:
            mass = dict.fromkeys(labels, 0.0)
            for alt in position.get("top_logprobs") or [{"token": position["token"], "logprob": position["logprob"]}]:
                key = str(alt.get("token", "")).strip(_STRIP)
                if key in mass:
                    mass[key] += math.exp(float(alt.get("logprob", -1e9)))
            total = sum(mass.values())
            if total <= 0:
                break
            return [mass[label] / total for label in labels]
        if token in _SKIPPABLE or token.startswith("<"):
            continue
        raise DecisionError(f"The model answered with text ({position.get('token')!r}) instead of an option letter.")
    raise DecisionError("The model returned no option letter.")


def option_schema(n_options: int) -> type[BaseModel]:
    """Structured-output schema for one question: the answer letter plus a probability per option letter."""
    labels = tuple(LABELS[:n_options])
    fields: dict[str, Any] = {
        "answer": (Literal[labels], Field(description="Letter of the most likely option.")),  # type: ignore[valid-type]
    }
    for label in labels:
        fields[label] = (float, Field(description=f"Probability (0 to 1) that option {label} is the correct answer."))
    return create_model(f"Decision{n_options}", **fields)


def structured_distribution(result: Any, n_options: int) -> list[float]:
    """Turn a structured answer ({"answer": "B", "A": 0.1, "B": 0.9}) into a probability per option."""
    if result is None:
        raise DecisionError("The model gave no structured answer.")
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    labels = LABELS[:n_options]
    values = []
    for label in labels:
        try:
            value = float(data.get(label) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if 1.0 < value <= 100.0:  # a percentage
            value /= 100.0
        values.append(min(max(value, 0.0), 1.0))
    total = sum(values)
    if total <= 0:
        answer = str(data.get("answer", "")).strip(_STRIP).upper()
        if answer not in labels:
            raise DecisionError("The structured answer named no option.")
        return [1.0 if label == answer else 0.0 for label in labels]
    return [v / total for v in values]


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.65)
    tail = max_chars - head
    return f"{text[:head]}\n[…]\n{text[-tail:]}"


class DecisionEngine:
    """Typed decisions on top of a LangChain chat model.

    llm:        chat model that answers with one letter and returns token logprobs (method logprobs/auto)
    structured: factory schema -> runnable with structured output (method structured, or auto's fallback)
    """

    def __init__(
        self,
        llm: Any | None = None,
        *,
        structured: Callable[[type[BaseModel]], Any] | None = None,
        method: Method = "logprobs",
        debias: bool = True,
        num_ctx: int = 8192,
        model: str = "",
    ):
        if method not in ("auto", "logprobs", "structured"):
            raise ValueError(f"Unknown decision method: {method}")
        self.model = model
        self.method = method
        self.debias = debias
        self.num_ctx = num_ctx
        self._llm = llm
        self._structured_factory = structured
        self._structured: dict[int, Any] = {}
        self.active_method = "structured" if method == "structured" or llm is None else "logprobs"
        self._used: set[str] = set()

    # ------------------------------------------------------------------ public API
    def choice(self, state: str, question: str, options: Sequence[str], *, instructions: str = "",
               none_option: str | None = None, debias: bool | None = None) -> ChoiceResult:
        """Pick one of `options`. `none_option` (if one of the options) is kept in every tournament group."""
        options = list(options)
        if not options:
            raise ValueError("choice() needs at least one option")
        if len(options) == 1:
            return ChoiceResult(question, options, [1.0], "")
        debias = self.debias if debias is None else debias
        self._used = set()
        if len(options) <= MAX_OPTIONS:
            probs = self._debiased(state, question, options, instructions, debias)
        else:
            probs = self._tournament(state, question, options, instructions, none_option, debias)
        method = "both" if len(self._used) > 1 else next(iter(self._used), self.active_method)
        return ChoiceResult(question, options, probs, method)

    def noul(self, state: str, statement: str, *, instructions: str = "", debias: bool | None = None) -> float:
        """Probability (0..1) that `statement` is true for the state."""
        result = self.choice(state, f"Is the following statement true? \"{statement}\"", ["Yes", "No"],
                             instructions=instructions, debias=debias)
        return result.probability_of("Yes")

    def score(self, state: str, question: str, levels: Sequence[str], *, instructions: str = "") -> ChoiceResult:
        """Rate the state on ordered levels (lowest first). Asked in natural order; levels are not shuffled."""
        return self.choice(state, question, levels, instructions=instructions, debias=False)

    # ------------------------------------------------------------------ internals
    def _debiased(self, state: str, question: str, options: list[str], instructions: str, debias: bool) -> list[float]:
        forward = self._distribution(state, question, options, instructions)
        if not debias:
            return forward
        backward = self._distribution(state, question, options[::-1], instructions)[::-1]
        return [(a + b) / 2 for a, b in zip(forward, backward, strict=True)]

    def _tournament(self, state: str, question: str, options: list[str], instructions: str,
                    none_option: str | None, debias: bool) -> list[float]:
        real = [o for o in options if o != none_option]
        group_size = MAX_OPTIONS - (1 if none_option else 0)
        finalists: list[tuple[str, float]] = []
        for start in range(0, len(real), group_size):
            group = real[start:start + group_size] + ([none_option] if none_option else [])
            probs = self._debiased(state, question, group, instructions, debias)
            ranked = sorted((p, o) for o, p in zip(group, probs, strict=True) if o != none_option)[::-1]
            finalists += [(o, p) for p, o in ranked[:3] if p >= 0.05] or [(ranked[0][1], ranked[0][0])]
        finalists = sorted(finalists, key=lambda x: x[1], reverse=True)[:group_size]
        final_options = [o for o, _ in finalists] + ([none_option] if none_option else [])
        final = self._debiased(state, question, final_options, instructions, debias)
        lookup = dict(zip(final_options, final, strict=True))
        return [lookup.get(o, 0.0) for o in options]

    def render_prompt(self, state: str, question: str, options: Sequence[str], instructions: str = "",
                      structured: bool = False) -> str:
        labels = tuple(LABELS[:len(options)])
        option_lines = "\n".join(f"{label}) {option}" for label, option in zip(labels, options, strict=True))
        budget = max(2000, int(self.num_ctx * 3) - len(option_lines) - len(question) - 600)
        parts = [f"STATE:\n<<<\n{_truncate_middle(state.strip(), budget)}\n>>>"]
        if instructions:
            parts.append(instructions.strip())
        parts.append(f"QUESTION: {question}\nOPTIONS:\n{option_lines}")
        if structured:
            parts.append(f"Give the letter of the best option ({labels[0]}-{labels[-1]}) as 'answer', and for every "
                         "option letter the probability that it is correct. The probabilities add up to 1.")
        else:
            parts.append(f"Answer with exactly one letter ({labels[0]}-{labels[-1]}).")
        return "\n\n".join(parts)

    def _distribution(self, state: str, question: str, options: list[str], instructions: str) -> list[float]:
        if self.active_method == "logprobs":
            try:
                probs = self._from_logprobs(state, question, options, instructions)
                self._used.add("logprobs")
                return probs
            except LogprobsUnavailable as exc:
                if self.method != "auto" or self._structured_factory is None:
                    raise DecisionError("This model or API returns no token probabilities. Set the decision "
                                        "method to Auto or Structured (Settings → Decisions).") from exc
                log.warning("%s returns no token probabilities (%s); using structured answers from now on.",
                            self.model or "The decision model", str(exc)[:200])
                self.active_method = "structured"
            except DecisionError:
                if self.method != "auto" or self._structured_factory is None:
                    raise
                log.info("No option letter from %s; asking this question with structured output.", self.model)
        probs = self._from_structured(state, question, options, instructions)
        self._used.add("structured")
        return probs

    def _from_logprobs(self, state: str, question: str, options: list[str], instructions: str) -> list[float]:
        if self._llm is None:
            raise LogprobsUnavailable("No model for token probabilities configured.")
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(self.render_prompt(state, question, options, instructions))]
        response = self._llm.invoke(messages)
        positions = token_logprobs(response)
        if positions is None:
            raise LogprobsUnavailable("The API returned no token probabilities.")
        return option_distribution(positions, len(options))

    def _from_structured(self, state: str, question: str, options: list[str], instructions: str) -> list[float]:
        if self._structured_factory is None:
            raise DecisionError("No model for structured answers configured.")
        n = len(options)
        if n not in self._structured:
            self._structured[n] = self._structured_factory(option_schema(n))
        messages = [SystemMessage(STRUCTURED_SYSTEM_PROMPT),
                    HumanMessage(self.render_prompt(state, question, options, instructions, structured=True))]
        return structured_distribution(self._structured[n].invoke(messages), n)
