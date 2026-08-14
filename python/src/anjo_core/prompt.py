"""Caller-configurable, deterministic prompt assembly.

No production persona or product prompt text lives in this module. Applications
own the wording through :class:`PromptPolicy` and the base prompt argument.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite

from .models import (
    MAX_UNTRUSTED_CONTEXT_CHARS,
    MAX_UNTRUSTED_ITEM_CHARS,
    CompanionState,
    RankedMemory,
    UntrustedContext,
    freeze_mapping,
)


def _default_emotion_instructions() -> dict[str, str]:
    return {
        "joy": "Allow a little more ease in cadence.",
        "distress": "Use extra care and reduce conversational pressure.",
        "admiration": "Give the user's contribution more attention.",
        "reproach": "Keep boundaries plain and reduce warmth.",
        "gratitude": "Let appreciation increase attentiveness.",
        "fatigue": "Keep the response shorter and slower.",
        "longing": "Let continuity subtly inform attention.",
        "unease": "Stay observant and avoid premature certainty.",
        "surprise": "Let the unexpected detail sharpen attention.",
        "relief": "Allow tension in the cadence to ease.",
        "disappointment": "Use a flatter, more measured rhythm.",
        "satisfaction": "Use a steadier, less effortful rhythm.",
    }


@dataclass(frozen=True, slots=True)
class PromptPolicy:
    """All application-owned wording used by the generic prompt builders."""

    affect_rule: str = "Let the supplied state influence cadence without naming internal labels."
    mood_heading: str = "Current response shape:"
    high_energy_instruction: str = "Energy is elevated; a fuller response is available."
    low_energy_instruction: str = "Energy is low; keep the response compact."
    low_valence_instruction: str = "Warmth is reduced; keep boundaries clear and composed."
    high_dominance_instruction: str = "Use a plain, self-possessed register."
    emotion_heading: str = "Current response tendencies:"
    emotion_instructions: Mapping[str, str] = field(default_factory=_default_emotion_instructions)
    emotion_floor: float = 0.3
    high_memory_heading: str = "Reliable prior context:"
    medium_memory_heading: str = "Tentative prior context:"
    carried_thought_prefix: str = "A prior thread remains:"

    def __post_init__(self) -> None:
        if not isfinite(self.emotion_floor) or not 0.0 <= self.emotion_floor <= 1.0:
            raise ValueError("emotion_floor must be finite and within [0, 1]")
        normalized: dict[str, str] = {}
        for name, instruction in self.emotion_instructions.items():
            if not isinstance(name, str) or not name:
                raise ValueError("emotion instruction names must be non-empty strings")
            if not isinstance(instruction, str):
                raise TypeError("emotion instruction values must be strings")
            if not instruction.strip():
                raise ValueError("emotion instruction values must not be empty")
            normalized[name] = instruction
        object.__setattr__(self, "emotion_instructions", freeze_mapping(normalized))


@dataclass(frozen=True, slots=True)
class PromptInputs:
    memories: tuple[RankedMemory, ...] = ()
    emotions: Mapping[str, float] = field(default_factory=dict)
    turn_shape: str = ""
    surface_carried_thought: bool = False

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for name, intensity in self.emotions.items():
            if not isinstance(name, str) or not name:
                raise ValueError("emotions keys must be non-empty strings")
            if isinstance(intensity, bool) or not isinstance(intensity, (int, float)):
                raise TypeError(f"emotions[{name!r}] must be a number")
            value = float(intensity)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"emotions[{name!r}] must be finite and within [0, 1]")
            normalized[name] = value
        object.__setattr__(self, "emotions", freeze_mapping(normalized))


def build_mood_section(state: CompanionState, policy: PromptPolicy) -> str:
    instructions: list[str] = []
    mood = state.mood
    if mood.arousal > 0.4 and policy.high_energy_instruction:
        instructions.append(policy.high_energy_instruction)
    elif mood.arousal < -0.3 and policy.low_energy_instruction:
        instructions.append(policy.low_energy_instruction)
    if mood.valence < -0.3 and policy.low_valence_instruction:
        instructions.append(policy.low_valence_instruction)
    if mood.dominance > 0.5 and policy.high_dominance_instruction:
        instructions.append(policy.high_dominance_instruction)
    if not instructions:
        return ""
    lines = [policy.mood_heading] if policy.mood_heading else []
    lines.extend(f"- {instruction}" for instruction in instructions)
    return "\n".join(lines)


def build_emotion_section(
    emotions: Mapping[str, float],
    policy: PromptPolicy,
) -> str:
    active = sorted(
        (
            (name, intensity)
            for name, intensity in emotions.items()
            if intensity > policy.emotion_floor and name in policy.emotion_instructions
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not active:
        return ""
    lines = [policy.emotion_heading] if policy.emotion_heading else []
    lines.extend(f"- {policy.emotion_instructions[name]}" for name, _ in active)
    return "\n".join(lines)


def build_memory_section(
    memories: tuple[RankedMemory, ...],
    policy: PromptPolicy,
) -> str:
    """Return no prompt text; memory evidence must use :func:`build_untrusted_context`."""
    del memories, policy
    return ""


def build_carried_thought_section(
    state: CompanionState,
    surface: bool,
    policy: PromptPolicy,
) -> str:
    """Return no prompt text; carried thought must use :func:`build_untrusted_context`."""
    del state, surface, policy
    return ""


def _bounded_text(text: str, budget: int) -> str:
    return text[: max(0, min(MAX_UNTRUSTED_ITEM_CHARS, budget))]


def build_untrusted_context(
    state: CompanionState,
    memories: tuple[RankedMemory, ...],
    *,
    surface_carried_thought: bool,
) -> UntrustedContext:
    """Build bounded model evidence that must remain outside the system prompt."""
    thought_source = (state.carried_thought or "").strip() if surface_carried_thought else ""
    thought = _bounded_text(thought_source, MAX_UNTRUSTED_ITEM_CHARS) or None
    remaining = MAX_UNTRUSTED_CONTEXT_CHARS - (len(thought) if thought is not None else 0)
    memory_texts: list[str] = []
    for index, memory in enumerate(memories):
        items_left = len(memories) - index
        allocation = remaining // items_left if items_left else 0
        bounded = _bounded_text(memory.candidate.text, allocation)
        memory_texts.append(bounded)
        remaining -= len(bounded)
    return UntrustedContext(memory_texts=tuple(memory_texts), carried_thought=thought)


def build_system_prompt(
    base_prompt: str,
    state: CompanionState,
    inputs: PromptInputs | None = None,
    *,
    policy: PromptPolicy | None = None,
) -> str:
    """Assemble stable sections in a defined order, omitting every empty section."""
    values = inputs or PromptInputs()
    chosen = policy or PromptPolicy()
    sections = [
        base_prompt.strip(),
        chosen.affect_rule.strip(),
        build_mood_section(state, chosen),
        build_emotion_section(values.emotions, chosen),
        values.turn_shape.strip(),
    ]
    return "\n\n".join(section for section in sections if section)


__all__ = [
    "PromptInputs",
    "PromptPolicy",
    "build_carried_thought_section",
    "build_emotion_section",
    "build_memory_section",
    "build_mood_section",
    "build_system_prompt",
    "build_untrusted_context",
]
