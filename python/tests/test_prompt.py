from __future__ import annotations

import pytest

from affect_kernel.affect import TurnShapePolicy, turn_shape_directive
from affect_kernel.models import CompanionState, MemoryCandidate, Message, PADMood, RankedMemory
from affect_kernel.prompt import (
    PromptInputs,
    PromptPolicy,
    build_system_prompt,
    build_untrusted_context,
)


def test_prompt_is_caller_owned_and_deterministically_assembled() -> None:
    policy = PromptPolicy(
        affect_rule="Let state influence cadence without naming internal labels.",
        high_memory_heading="Reliable context:",
        medium_memory_heading="Tentative context:",
        emotion_heading="Current response tendencies:",
        emotion_instructions={"joy": "Use a lighter cadence.", "fatigue": "Keep it compact."},
        carried_thought_prefix="A prior thread remains:",
    )
    state = CompanionState(
        mood=PADMood(valence=0.2, arousal=-0.5, dominance=0.0), carried_thought="unfinished idea"
    )
    inputs = PromptInputs(
        memories=(
            RankedMemory(
                candidate=MemoryCandidate(id="high", text="stable fact", distance=0.0), score=0.8
            ),
            RankedMemory(
                candidate=MemoryCandidate(id="mid", text="uncertain fact", distance=0.0), score=0.6
            ),
            RankedMemory(
                candidate=MemoryCandidate(id="low", text="noise", distance=0.0), score=0.4
            ),
        ),
        emotions={"joy": 0.6, "fatigue": 0.31, "unknown": 0.9},
        turn_shape="Prefer one compact observation.",
        surface_carried_thought=True,
    )

    prompt = build_system_prompt("Synthetic companion instructions.", state, inputs, policy=policy)

    assert prompt.startswith("Synthetic companion instructions.")
    assert "Let state influence cadence" in prompt
    assert "Use a lighter cadence." in prompt
    assert "Keep it compact." in prompt
    assert "stable fact" not in prompt and "uncertain fact" not in prompt
    assert "noise" not in prompt and "unknown" not in prompt
    assert "unfinished idea" not in prompt
    assert prompt.endswith("Prefer one compact observation.")


def test_prompt_omits_empty_optional_sections() -> None:
    prompt = build_system_prompt(
        "  Base instructions.  ",
        CompanionState(),
        PromptInputs(),
        policy=PromptPolicy(affect_rule=""),
    )
    assert prompt == "Base instructions."


def test_carried_thought_is_explicitly_first_turn_only() -> None:
    state = CompanionState(carried_thought="carry this")
    policy = PromptPolicy(carried_thought_prefix="Prior thread:")
    hidden = build_system_prompt(
        "Base", state, PromptInputs(surface_carried_thought=False), policy=policy
    )
    shown = build_system_prompt(
        "Base", state, PromptInputs(surface_carried_thought=True), policy=policy
    )
    assert "carry this" not in hidden
    assert "carry this" not in shown


def test_untrusted_context_is_typed_bounded_and_separate() -> None:
    memories = tuple(
        RankedMemory(
            candidate=MemoryCandidate(id=str(index), text="x" * 4_096, distance=0.0),
            score=0.8,
        )
        for index in range(4)
    )
    context = build_untrusted_context(
        CompanionState(carried_thought="y" * 4_096),
        memories,
        surface_carried_thought=True,
    )

    assert len(context.memory_texts) == 4
    assert all(len(text) <= 2_000 for text in context.memory_texts)
    assert context.carried_thought is not None
    assert sum(map(len, context.memory_texts)) + len(context.carried_thought) <= 8_000


def test_policy_and_prompt_input_mappings_are_immutable_defensive_copies() -> None:
    instructions = {"joy": "light"}
    emotions = {"joy": 0.4}
    policy = PromptPolicy(emotion_instructions=instructions)
    inputs = PromptInputs(emotions=emotions)
    instructions["joy"] = "changed"
    emotions["joy"] = 0.9

    assert policy.emotion_instructions["joy"] == "light"
    assert inputs.emotions["joy"] == 0.4
    with pytest.raises(TypeError):
        policy.emotion_instructions["joy"] = "mutate"  # type: ignore[index]
    with pytest.raises(TypeError):
        inputs.emotions["joy"] = 0.8  # type: ignore[index]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"emotion_floor": -0.1}, "emotion_floor"),
        ({"emotion_instructions": {"joy": ""}}, "instruction"),
        ({"emotion_instructions": {"joy": 3}}, "instruction"),
    ],
)
def test_prompt_policy_validates_mapping_and_thresholds(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        PromptPolicy(**kwargs)  # type: ignore[arg-type]


def test_prompt_inputs_validate_emotion_values() -> None:
    with pytest.raises(ValueError, match="emotions"):
        PromptInputs(emotions={"joy": 1.1})


def test_turn_shape_copy_is_caller_configurable() -> None:
    policy = TurnShapePolicy(
        heading="Shape:",
        statement_close="Prefer a statement close.",
        no_question_close="Do not repeat a question close.",
        one_idea="Use one idea.",
        mirror_length="Match the input length.",
        ambivalence="Keep opposing signals unresolved.",
        octant_cues={"relaxed": "Keep an easy pace."},
    )
    directive = turn_shape_directive(
        PADMood(0.5, -0.5, 0.5),
        history=(Message("assistant", "Still there?"),),
        emotions={"joy": 0.5, "distress": 0.3},
        policy=policy,
    )
    assert directive.splitlines() == [
        "Shape:",
        "- Do not repeat a question close.",
        "- Use one idea.",
        "- Match the input length.",
        "- Keep an easy pace.",
        "- Keep opposing signals unresolved.",
    ]
