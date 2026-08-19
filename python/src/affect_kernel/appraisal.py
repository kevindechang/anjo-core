"""Non-habituating OCC/PAD appraisal as pure state transforms."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from math import isfinite

from .models import AppraisalGoals, CompanionState, PADMood, Personality, freeze_mapping

DEFAULT_STAGES: tuple[str, ...] = (
    "stranger",
    "acquaintance",
    "friend",
    "close",
    "intimate",
)
DEFAULT_STAGE_WEIGHTS: tuple[float, ...] = (0.0, 0.20, 0.40, 0.60, 0.70)


class UnknownStageError(KeyError):
    """Raised when a strict :class:`StageLadder` receives a stage it does not define."""


@dataclass(frozen=True, slots=True)
class StageLadder:
    """An ordered relationship ladder and the resting-point weight of each rung.

    The default rungs are the reference conversational ladder. Domains with a
    different progression -- a game faction track, a tutoring competence ladder,
    a support-case lifecycle -- should supply their own ``stages`` and
    ``weights`` rather than reusing these labels.

    ``strict`` controls what happens when a stage is not on the ladder. The
    default (``False``) floors the stage to the first rung, which matches the
    pinned cross-runtime contract. Set ``strict=True`` to raise
    :class:`UnknownStageError` instead; a typo'd or unmapped stage otherwise
    resolves to weight ``0.0`` and silently removes the resting set point.
    """

    stages: tuple[str, ...] = DEFAULT_STAGES
    weights: tuple[float, ...] = DEFAULT_STAGE_WEIGHTS
    strict: bool = False
    _ordinals: Mapping[str, int] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        weights = tuple(float(weight) for weight in self.weights)
        if not stages:
            raise ValueError("stages must not be empty")
        if len(stages) != len(weights):
            raise ValueError("stages and weights must have equal length")
        if len(set(stages)) != len(stages):
            raise ValueError("stages must be unique")
        for stage in stages:
            if not isinstance(stage, str) or not stage:
                raise ValueError("stage names must be non-empty strings")
        for weight in weights:
            if not isfinite(weight) or not 0.0 <= weight <= 1.0:
                raise ValueError("stage weights must be finite and within [0, 1]")
        if not isinstance(self.strict, bool):
            raise TypeError("strict must be a boolean")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "_ordinals",
            freeze_mapping({stage: index + 1 for index, stage in enumerate(stages)}),
        )

    def knows(self, stage: str) -> bool:
        """Whether ``stage`` is a defined rung, for validation at an app boundary."""
        return stage in self._ordinals

    def ordinal(self, stage: str) -> int:
        """Return the 1-based rung for ``stage``; see ``strict`` for unknown stages."""
        try:
            return self._ordinals[stage]
        except KeyError:
            if self.strict:
                raise UnknownStageError(
                    f"stage {stage!r} is not on the ladder {self.stages!r}"
                ) from None
            return 1

    def weight_for_ordinal(self, ordinal: int) -> float:
        """Return the resting-point weight of a 1-based rung; out of range means 0.0."""
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise TypeError("ordinal must be an integer")
        if 1 <= ordinal <= len(self.weights):
            return self.weights[ordinal - 1]
        return 0.0

    def weight(self, stage: str) -> float:
        """Return the resting-point weight for a stage label."""
        return self.weight_for_ordinal(self.ordinal(stage))


DEFAULT_STAGE_LADDER = StageLadder()

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


def stage_int(stage: str, ladder: StageLadder | None = None) -> int:
    """Map a stage label to its stable ordinal on ``ladder`` (default: conversational).

    With the default non-strict ladder an unknown stage floors to the first rung.
    Pass ``StageLadder(strict=True)`` to surface unmapped stages instead.
    """
    return (ladder or DEFAULT_STAGE_LADDER).ordinal(stage)


def baseline_weight(stage: int, ladder: StageLadder | None = None) -> float:
    """Return how strongly the resting set point is expressed at a stage ordinal."""
    return (ladder or DEFAULT_STAGE_LADDER).weight_for_ordinal(stage)


def mood_inertia(personality: Personality) -> float:
    """AR(1) carryover parameter derived from Neuroticism and Extraversion."""
    value = 0.80 + 0.20 * (personality.N - 0.5) - 0.10 * (personality.E - 0.5)
    return _clamp(value, 0.62, 0.92)


def decay_mood(
    mood: PADMood,
    personality: Personality,
    relationship_stage: int | str,
    baseline_valence: float,
    *,
    ladder: StageLadder | None = None,
) -> PADMood:
    """Relax PAD toward the stage-weighted resting point using AR(1) dynamics."""
    chosen = ladder or DEFAULT_STAGE_LADDER
    stage = (
        chosen.ordinal(relationship_stage)
        if isinstance(relationship_stage, str)
        else relationship_stage
    )
    weight = chosen.weight_for_ordinal(stage)
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


@dataclass(frozen=True, slots=True)
class ExpectationCues:
    """The lexical cues used to detect a violated expectation.

    These defaults are an **English conversational preset**, not domain-neutral
    behavior. Supply your own tuples to localize the kernel or to express
    expectation violation in a non-conversational vocabulary. A trailing ``-``
    marks a stem match (``"apolog-"`` matches ``apologize`` and ``apology``).
    """

    negative_expected: tuple[str, ...] = (
        "angry",
        "upset",
        "bad",
        "worse",
        "argument",
        "hurt",
        "afraid",
        "scared",
    )
    positive_current: tuple[str, ...] = (
        "better",
        "okay",
        "ok",
        "good",
        "fine",
        "resolved",
        "worked out",
    )
    negative_current: tuple[str, ...] = (
        "worse",
        "bad",
        "angry",
        "upset",
        "failed",
        "awful",
        "hurt",
    )
    expected_resolution: tuple[str, ...] = ("come back", "tell", "answer", "resolve", "apolog-")

    def __post_init__(self) -> None:
        for name in (
            "negative_expected",
            "positive_current",
            "negative_current",
            "expected_resolution",
        ):
            tokens = tuple(getattr(self, name))
            for token in tokens:
                if not isinstance(token, str):
                    raise TypeError(f"{name} tokens must be strings")
                if not token.strip():
                    raise ValueError(f"{name} tokens must not be empty")
            object.__setattr__(self, name, tokens)


DEFAULT_EXPECTATION_CUES = ExpectationCues()


def expectation_emotions(
    expectation: str,
    message: str,
    *,
    cues: ExpectationCues | None = None,
) -> dict[str, float]:
    """Return a small deterministic delta when the current turn violates an expectation."""
    chosen = cues or DEFAULT_EXPECTATION_CUES
    expected = (expectation or "").lower()
    current = (message or "").lower()
    if not expected or not current:
        return {}

    negative_expected = _word_match(expected, chosen.negative_expected)
    positive_current = _word_match(current, chosen.positive_current)
    negative_current = _word_match(current, chosen.negative_current)
    expected_resolution = _word_match(expected, chosen.expected_resolution)

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
    ladder: StageLadder | None = None,
    cues: ExpectationCues | None = None,
) -> AppraisalResult:
    """Compose decay, appraisal, emotion carry, and state-emotion derivation."""
    decayed_mood = decay_mood(
        state.mood,
        state.personality,
        state.relationship.stage,
        state.baseline_valence,
        ladder=ladder,
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
    for name, value in expectation_emotions(current_expectation, message, cues=cues).items():
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


def conversational_appraisal_policy(
    *,
    ladder: StageLadder | None = None,
    cues: ExpectationCues | None = None,
) -> Callable[[AppraisalPolicyInput], AppraisalResult]:
    """Build a reference-shaped policy bound to a custom ladder and cue set.

    Use this when the reference intent vocabulary fits but the progression or
    the language does not. Domains whose events are not conversational should
    write a policy directly against :class:`AppraisalPolicyInput` instead.
    """

    def policy(request: AppraisalPolicyInput) -> AppraisalResult:
        return appraise_turn(
            request.state,
            request.intent,
            message=request.message,
            expectation=request.expectation,
            ladder=ladder,
            cues=cues,
        )

    return policy


def default_appraisal_policy(request: AppraisalPolicyInput) -> AppraisalResult:
    """Reference conversational appraisal for the kernel's built-in intents."""
    return appraise_turn(
        request.state,
        request.intent,
        message=request.message,
        expectation=request.expectation,
    )


__all__ = [
    "DEFAULT_EXPECTATION_CUES",
    "DEFAULT_STAGES",
    "DEFAULT_STAGE_LADDER",
    "DEFAULT_STAGE_WEIGHTS",
    "AppraisalPolicyInput",
    "AppraisalResult",
    "ExpectationCues",
    "InputAppraisal",
    "StageLadder",
    "UnknownStageError",
    "appraise_input",
    "appraise_turn",
    "baseline_weight",
    "conversational_appraisal_policy",
    "decay_mood",
    "decay_occ_carry",
    "default_appraisal_policy",
    "expectation_emotions",
    "mood_inertia",
    "stage_int",
    "state_emotions",
]
