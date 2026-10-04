"""A local "System One" decision model in the style of TypeSafe's Jev, running on Ollama via LangChain.

The model never writes free text. Each question gets a list of options labelled A, B, C, ...; we run one
forward pass, read the model's log-probabilities for the *option letters only* and normalize them. The
answer can therefore only ever be one of the given options, and it comes with a probability.

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

import math
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

LABELS = string.ascii_uppercase[:20]  # Ollama returns at most 20 alternatives per token (top_logprobs <= 20)
MAX_OPTIONS = len(LABELS)
_STRIP = " \t\r\n*_`'\"()[]{}.:;,-"
_SKIPPABLE = {"", "<think>", "</think>", "think", "answer", "Answer"}

SYSTEM_PROMPT = (
    "You are a decision model inside a document-sorting application. You never explain and never write "
    "prose. Read the STATE, then answer the QUESTION by replying with exactly one option letter."
)


class DecisionError(RuntimeError):
    """The model did not answer with one of the option letters."""


@dataclass
class ChoiceResult:
    question: str
    options: list[str]
    probabilities: list[float]

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
            "options": [{"option": o, "p": round(p, 4)} for o, p in zip(self.options, self.probabilities, strict=True)],
        }


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


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.65)
    tail = max_chars - head
    return f"{text[:head]}\n[…]\n{text[-tail:]}"


class DecisionEngine:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:4b",
        *,
        num_ctx: int = 8192,
        keep_alive: str | int = "15m",
        debias: bool = True,
        llm: Any | None = None,
    ):
        self.model = model
        self.debias = debias
        self.num_ctx = num_ctx
        self._llm = llm or self._make_llm(base_url, model, num_ctx, keep_alive, reasoning=False)
        self._fallback_factory: Callable[[], Any] | None = (
            None if llm is not None else lambda: self._make_llm(base_url, model, num_ctx, keep_alive, reasoning=None)
        )

    @staticmethod
    def _make_llm(base_url: str, model: str, num_ctx: int, keep_alive: str | int, reasoning: bool | None) -> Any:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
            reasoning=reasoning,  # thinking off: the first token must be the answer
            logprobs=True,
            top_logprobs=MAX_OPTIONS,
            num_predict=12,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
        )

    # ------------------------------------------------------------------ public API
    def choice(self, state: str, question: str, options: Sequence[str], *, instructions: str = "",
               none_option: str | None = None, debias: bool | None = None) -> ChoiceResult:
        """Pick one of `options`. `none_option` (if one of the options) is kept in every tournament group."""
        options = list(options)
        if not options:
            raise ValueError("choice() needs at least one option")
        if len(options) == 1:
            return ChoiceResult(question, options, [1.0])
        debias = self.debias if debias is None else debias
        if len(options) <= MAX_OPTIONS:
            probs = self._debiased(state, question, options, instructions, debias)
        else:
            probs = self._tournament(state, question, options, instructions, none_option, debias)
        return ChoiceResult(question, options, probs)

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

    def render_prompt(self, state: str, question: str, options: Sequence[str], instructions: str = "") -> str:
        labels = tuple(LABELS[:len(options)])
        option_lines = "\n".join(f"{label}) {option}" for label, option in zip(labels, options, strict=True))
        budget = max(2000, int(self.num_ctx * 3) - len(option_lines) - len(question) - 600)
        parts = [f"STATE:\n<<<\n{_truncate_middle(state.strip(), budget)}\n>>>"]
        if instructions:
            parts.append(instructions.strip())
        parts.append(f"QUESTION: {question}\nOPTIONS:\n{option_lines}")
        parts.append(f"Answer with exactly one letter ({labels[0]}-{labels[-1]}).")
        return "\n\n".join(parts)

    def _distribution(self, state: str, question: str, options: list[str], instructions: str) -> list[float]:
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(self.render_prompt(state, question, options, instructions))]
        try:
            response = self._llm.invoke(messages)
        except Exception as exc:
            # Models without a thinking mode reject `think: false`; retry once with the model default.
            if self._fallback_factory and "think" in str(exc).lower():
                self._llm = self._fallback_factory()
                self._fallback_factory = None
                response = self._llm.invoke(messages)
            else:
                raise
        return option_distribution(response.response_metadata.get("logprobs"), len(options))
