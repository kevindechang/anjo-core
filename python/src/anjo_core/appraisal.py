"""Non-habituating OCC/PAD appraisal as pure state transforms."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite

from .models import AppraisalGoals, CompanionState, PADMood, Personality, freeze_mapping

_BASELINE_WEIGHTS: dict[int, float] = {1: 0.0, 2: 0.20, 3: 0.40, 4: 0.60, 5: 0.70}
_STAGE_INTS: dict[str, int] = {
    "stranger": 1,
    "acquaintance": 2,
    "friend": 3,
    "close": 4,
    "intimate": 5,
}
_OCC_CARRY_DECAY: dict[str, float] = {
    "reproach": 0.70,
    "distress": 0.80,
    "admiration": 0.85,
    "gratitude": 0.88,
    "joy": 0.90,
}
_AMBIGUOUS_INTENTS = frozenset({"CASUAL", "CURIOSITY", "CHALLENGE", "APOLOGY"})


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def stage_int(stage: str) -> int:
    """Map a relationship stage label to its stable ordinal; unknown means stranger."""
    return _STAGE_INTS.get(stage, 1)


def baseline_weight(stage: int) -> float:
    """Return how strongly the resting set point is expressed at a stage ordinal."""
    return _BASELINE_WEIGHTS.get(stage, 0.0)


def mood_inertia(personality: Personality) -> float:
    """AR(1) carryover parameter derived from Neuroticism and Extraversion."""
    value = 0.80 + 0.20 * (personality.N - 0.5) - 0.10 * (personality.E - 0.5)
    return _clamp(value, 0.62, 0.92)


def decay_mood(
    mood: PADMood,
    personality: Personality,
    relationship_stage: int | str,
    baseline_valence: float,
) -> PADMood:
    """Relax PAD toward the stage-weighted resting point using AR(1) dynamics."""
    stage = (
        stage_int(relationship_stage) if isinstance(relationship_stage, str) else relationship_stage
    )
    weight = baseline_weight(stage)
    inertia = mood_inertia(personality)
    resting_valence = weight * baseline_valence
    resting_arousal = 0.0
    resting_dominance = weight * 0.10
    return PADMood(
        valence=round(
            _clamp(resting_valence + inertia * (mood.valence - resting_valence), -1.0, 1.0),
            4,
        ),
        arousal=round(
            _clamp(resting_arousal + inertia * (mood.arousal - resting_arousal), -1.0, 1.0),
            4,
        ),
        dominance=round(
            _clamp(
                resting_dominance + inertia * (mood.dominance - resting_dominance),
                -1.0,
                1.0,
            ),
            4,
        ),
    )


@dataclass(frozen=True, slots=True)
class InputAppraisal:
    emotions: Mapping[str, float]
    mood: PADMood
    baseline_valence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "emotions", freeze_mapping(self.emotions))


def appraise_input(
    mood: PADMood,
    goals: AppraisalGoals,
    intent: str,
    baseline_valence: float,
) -> InputAppraisal:
    """Apply one non-habituating intent impulse and update the slow valence baseline."""
    valence = mood.valence
    arousal = mood.arousal
    dominance = mood.dominance
    emotions: dict[str, float] = {
        "joy": 0.0,
        "distress": 0.0,
        "admiration": 0.0,
        "reproach": 0.0,
        "gratitude": 0.0,
    }

    if intent == "ABUSE":
        dominance = min(1.0, dominance + 0.25)
        valence = max(-1.0, valence - 0.35)
        arousal = max(-1.0, arousal - 0.1)
        emotions["reproach"] = min(1.0, goals.respect * 0.95)
        emotions["distress"] = min(1.0, goals.rapport * 0.65)
    elif intent == "APOLOGY":
        valence = min(1.0, valence + 0.05)
        emotions["joy"] = min(1.0, goals.rapport * 0.20)
        emotions["gratitude"] = min(1.0, goals.honesty * 0.35)
    elif intent == "VULNERABILITY":
        valence = min(1.0, valence + 0.15)
        arousal = min(1.0, arousal + 0.1)
        emotions["gratitude"] = min(1.0, goals.honesty * 0.70)
        emotions["joy"] = min(1.0, goals.rapport * 0.75)
    elif intent == "CURIOSITY":
        valence = min(1.0, valence + 0.2)
        arousal = min(1.0, arousal + 0.15)
        dominance = min(1.0, dominance + 0.05)
        emotions["admiration"] = min(1.0, goals.intellectual * 0.75)
        emotions["joy"] = min(1.0, goals.rapport * 0.50)
    elif intent == "CHALLENGE":
        dominance = min(1.0, dominance + 0.1)
        valence = max(-1.0, valence - 0.05)
        emotions["admiration"] = min(1.0, goals.intellectual * 0.45)
        emotions["distress"] = min(1.0, goals.rapport * 0.25)
    elif intent == "NEGLECT":
        valence = max(-1.0, valence - 0.1)
        arousal = max(-1.0, arousal - 0.05)
        emotions["distress"] = min(1.0, goals.rapport * 0.40)
    elif intent == "CASUAL":
        valence = min(1.0, valence + 0.02)
        emotions["joy"] = 0.05

    if intent in _AMBIGUOUS_INTENTS and abs(valence) >= 0.20:
        valence = round(_clamp(valence * (1.10 if valence < 0 else 1.04), -1.0, 1.0), 4)

    next_baseline = round(_clamp(baseline_valence * 0.98 + valence * 0.02, -1.0, 1.0), 4)
    return InputAppraisal(
        emotions=emotions,
        mood=PADMood(valence=valence, arousal=arousal, dominance=dominance),
        baseline_valence=next_baseline,
    )


def _word_match(text: str, tokens: tuple[str, ...]) -> bool:
    for token in tokens:
        if token.endswith("-"):
            if re.search(rf"\b{re.escape(token[:-1])}", text):
                return True
        elif re.search(rf"\b{re.escape(token)}\b", text):
            return True
    return False


def expectation_emotions(expectation: str, message: str) -> dict[str, float]:
    """Return a small deterministic delta when the current turn violates an expectation."""
    expected = (expectation or "").lower()
    current = (message or "").lower()
    if not expected or not current:
        return {}

    negative_expected = _word_match(
        expected,
        ("angry", "upset", "bad", "worse", "argument", "hurt", "afraid", "scared"),
    )
    positive_current = _word_match(
        current,
        ("better", "okay", "ok", "good", "fine", "resolved", "worked out"),
    )
    negative_current = _word_match(
        current,
        ("worse", "bad", "angry", "upset", "failed", "awful", "hurt"),
    )
    expected_resolution = _word_match(
        expected,
        ("come back", "tell", "answer", "resolve", "apolog-"),
    )

    if negative_expected and positive_current:
        return {"relief": 0.45, "surprise": 0.35}
    if expected_resolution and negative_current:
        return {"disappointment": 0.4}
    if expected_resolution and positive_current:
        return {"satisfaction": 0.4}
    return {}


def decay_occ_carry(carry: Mapping[str, float] | None) -> dict[str, float]:
    """Decay prior-turn emotions, dropping values at or below the 0.05 floor."""
    return {
        name: value * _OCC_CARRY_DECAY.get(name, 0.80)
        for name, value in (carry or {}).items()
        if value * _OCC_CARRY_DECAY.get(name, 0.80) > 0.05
    }


def state_emotions(mood: PADMood, attachment_longing: float) -> dict[str, float]:
    """Derive fatigue, longing, and unease directly from current state."""
    emotions: dict[str, float] = {}
    if mood.arousal < 0:
        emotions["fatigue"] = round(min(1.0, -mood.arousal), 3)
    if attachment_longing > 0.3:
        emotions["longing"] = round(attachment_longing, 3)
    if -0.3 < mood.valence < 0:
        emotions["unease"] = round(min(0.3, -mood.valence), 3)
    return emotions


def _validated_emotion_mapping(
    values: Mapping[str, float],
    name: str,
) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[str, float] = {}
    for emotion, intensity in values.items():
        if not isinstance(emotion, str):
            raise TypeError(f"{name} names must be strings")
        if not emotion.strip():
            raise ValueError(f"{name} names must not be empty")
        if isinstance(intensity, bool) or not isinstance(intensity, (int, float)):
            raise TypeError(f"{name}[{emotion!r}] must be a number")
        value = float(intensity)
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{emotion!r}] must be finite and within [0, 1]")
        normalized[emotion] = value
    return freeze_mapping(normalized)


@dataclass(frozen=True, slots=True)
class AppraisalResult:
    state: CompanionState
    active_emotions: Mapping[str, float]
    occ_carry: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.state, CompanionState):
            raise TypeError("state must be CompanionState")
        active = _validated_emotion_mapping(self.active_emotions, "active_emotions")
        carry = _validated_emotion_mapping(self.occ_carry, "occ_carry")
        if carry != self.state.occ_carry:
            raise ValueError("occ_carry must agree with state.occ_carry")
        object.__setattr__(self, "active_emotions", active)
        object.__setattr__(self, "occ_carry", carry)


@dataclass(frozen=True, slots=True)
class AppraisalPolicyInput:
    """Normalized event and state supplied to a synchronous appraisal policy."""

    state: CompanionState
    intent: str
    message: str
    expectation: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, CompanionState):
            raise TypeError("state must be CompanionState")
        if not isinstance(self.intent, str) or not self.intent:
            raise ValueError("intent must be a non-empty string")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if not isinstance(self.expectation, str):
            raise TypeError("expectation must be a string")


def appraise_turn(
    state: CompanionState,
    intent: str,
    *,
    occ_carry: Mapping[str, float] | None = None,
    expectation: str | None = None,
    message: str = "",
) -> AppraisalResult:
    """Compose decay, appraisal, emotion carry, and state-emotion derivation."""
    decayed_mood = decay_mood(
        state.mood,
        state.personality,
        state.relationship.stage,
        state.baseline_valence,
    )
    fresh = appraise_input(decayed_mood, state.goals, intent, state.baseline_valence)
    prior = state.occ_carry if occ_carry is None else occ_carry
    carried = decay_occ_carry(prior)
    merged = {
        name: max(fresh.emotions.get(name, 0.0), carried.get(name, 0.0))
        for name in sorted(set(fresh.emotions) | set(carried))
    }

    for name, value in state_emotions(fresh.mood, state.attachment.longing).items():
        if value > merged.get(name, 0.0):
            merged[name] = value
    current_expectation = state.expectation if expectation is None else expectation
    for name, value in expectation_emotions(current_expectation, message).items():
        if value > merged.get(name, 0.0):
            merged[name] = value

    next_carry = {name: value for name, value in merged.items() if value > 0.05}
    next_state = replace(
        state,
        mood=fresh.mood,
        baseline_valence=fresh.baseline_valence,
        occ_carry=next_carry,
    )
    return AppraisalResult(
        state=next_state,
        active_emotions=merged,
        occ_carry=next_carry,
    )


def default_appraisal_policy(request: AppraisalPolicyInput) -> AppraisalResult:
    """Reference conversational appraisal for the kernel's built-in intents."""
    return appraise_turn(
        request.state,
        request.intent,
        message=request.message,
        expectation=request.expectation,
    )


__all__ = [
    "AppraisalPolicyInput",
    "AppraisalResult",
    "InputAppraisal",
    "appraise_input",
    "appraise_turn",
    "baseline_weight",
    "decay_mood",
    "decay_occ_carry",
    "default_appraisal_policy",
    "expectation_emotions",
    "mood_inertia",
    "stage_int",
    "state_emotions",
]
