import pytest

from lettereye.ai.decision import MAX_OPTIONS, DecisionEngine, DecisionError, option_distribution

from .conftest import ScriptedLLM, logprob_response


def test_distribution_sums_token_variants():
    positions = [{"token": "A", "logprob": -0.1, "top_logprobs": [
        {"token": "A", "logprob": -0.105},   # ~0.90
        {"token": "B", "logprob": -2.996},   # ~0.05
        {"token": " A", "logprob": -2.996},  # ~0.05, same answer with a leading space
    ]}]
    probs = option_distribution(positions, 2)
    assert probs[0] == pytest.approx(0.95, abs=0.01)
    assert sum(probs) == pytest.approx(1.0)


def test_distribution_skips_think_tags_and_whitespace():
    message = logprob_response({"B": 0.8, "A": 0.2}, prefix_tokens=("<think>", "</think>", "\n\n"))
    probs = option_distribution(message.response_metadata["logprobs"], 2)
    assert probs == pytest.approx([0.2, 0.8])


def test_distribution_rejects_prose():
    message = logprob_response({"The": 0.9, "A": 0.1})
    with pytest.raises(DecisionError):
        option_distribution(message.response_metadata["logprobs"], 2)


def test_answers_can_only_be_given_options():
    # the model would love to answer "C", but only two options exist
    message = logprob_response({"C": 0.6, "A": 0.3, "B": 0.1}, prefix_tokens=())
    positions = message.response_metadata["logprobs"]
    positions[0]["token"] = "A"
    assert option_distribution(positions, 2) == pytest.approx([0.75, 0.25])


def test_choice_debiasing_cancels_position_bias():
    # a model that always prefers whatever is labelled "A"
    class AlwaysA:
        def invoke(self, messages):
            return logprob_response({"A": 0.9, "B": 0.1})

    engine = DecisionEngine(llm=AlwaysA(), debias=True)
    result = engine.choice("state", "Which?", ["first", "second"])
    assert result.probabilities == pytest.approx([0.5, 0.5])

    biased = DecisionEngine(llm=AlwaysA(), debias=False).choice("state", "Which?", ["first", "second"])
    assert biased.choice == "first" and biased.confidence == pytest.approx(0.9)


def test_choice_follows_content():
    llm = ScriptedLLM(lambda state, q, option: 10.0 if option in state else 1.0)
    result = DecisionEngine(llm=llm).choice("Dear Bob Smith", "Who?", ["Alice Johnson", "Bob Smith", "Carol"])
    assert result.choice == "Bob Smith"
    assert result.confidence > 0.75
    assert len(llm.calls) == 2  # forward + reversed


def test_noul_is_probability_of_yes():
    llm = ScriptedLLM(lambda state, q, option: 9.0 if option == "Yes" else 1.0)
    assert DecisionEngine(llm=llm).noul("anything", "It is a letter") == pytest.approx(0.9)


def test_tournament_handles_more_options_than_letters():
    names = [f"Person {i:02d}" for i in range(45)]
    llm = ScriptedLLM(lambda state, q, option: 25.0 if option in state else (2.0 if option.startswith("Nobody") else 1.0))
    engine = DecisionEngine(llm=llm)
    result = engine.choice("Letter for Person 37.", "Who?", [*names, "Nobody"], none_option="Nobody")
    assert result.choice == "Person 37"
    assert all(len(options) <= MAX_OPTIONS for _, options in llm.calls)
    assert all("Nobody" in options for _, options in llm.calls)


def test_single_option_needs_no_model_call():
    llm = ScriptedLLM(lambda *a: 1.0)
    assert DecisionEngine(llm=llm).choice("x", "q", ["only"]).confidence == 1.0
    assert llm.calls == []


def test_prompt_truncates_long_state_in_the_middle():
    engine = DecisionEngine(llm=ScriptedLLM(lambda *a: 1.0), num_ctx=1000)
    prompt = engine.render_prompt("HEAD " + "x" * 50_000 + " TAIL", "Q?", ["a", "b"])
    assert "HEAD" in prompt and "TAIL" in prompt and "[…]" in prompt
    assert len(prompt) < 10_000
