"""Deterministic affect controls derived from a PAD mood.

The octant partition is ALMA's (Gebhard 2005). The decoder controls have no
literature behind them at all; see ``docs/foundations.md`` sections 1 and 8.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .models import DecodingParams, Message, PADMood, freeze_mapping

_OCTANTS: dict[tuple[int, int, int], str] = {
    (1, 1, 1): "exuberant",
    (1, 1, -1): "dependent",
    (1, -1, 1): "relaxed",
    (1, -1, -1): "docile",
    (-1, 1, 1): "hostile",
    (-1, 1, -1): "anxious",
    (-1, -1, 1): "disdainful",
    (-1, -1, -1): "bored",
}

_POSITIVE_EMOTIONS = frozenset({"joy", "admiration", "gratitude", "relief", "satisfaction", "hope"})
_NEGATIVE_EMOTIONS = frozenset(
    {"distress", "reproach", "fear", "disappointment", "unease", "anger", "sadness"}
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sign(value: float) -> int:
    return 1 if value >= 0 else -1


def mood_octant(valence: float, arousal: float, dominance: float) -> str:
    """Return the ALMA PAD octant (Gebhard 2005), with a deadband around neutral.

    The deadband is an addition of ours, so that a near-zero mood does not flip
    labels on rounding noise.
    """
    if abs(valence) < 0.15 and abs(arousal) < 0.15 and abs(dominance) < 0.15:
        return "neutral"
    return _OCTANTS[(_sign(valence), _sign(arousal), _sign(dominance))]


def decoding_params(mood: PADMood | None) -> DecodingParams:
    """Map arousal to the dependency-free sampling envelope.

    No published work licenses mapping arousal onto a softmax temperature. This
    is an engineering convention, bounded so that it cannot make a model
    incoherent. See ``docs/foundations.md`` section 8.
    """
    if mood is None:
        return DecodingParams(temperature=1.0, top_p=None)
    return DecodingParams(
        temperature=_clamp(1.0 + 0.20 * mood.arousal, 0.72, 1.18),
        top_p=0.97,
    )


def length_factor(mood: PADMood | None) -> float:
    """Return a response-ceiling multiplier; this transform only shortens."""
    if mood is None:
        return 1.0
    if mood.arousal < -0.3:
        return 0.6
    if mood.valence < -0.3:
        return 0.72
    return 1.0


def apply_length_factor(base_tokens: int, mood: PADMood | None) -> int:
    """Shorten a valid budget, using the 180-token floor only when it fits."""
    if isinstance(base_tokens, bool) or not isinstance(base_tokens, int):
        raise TypeError("base_tokens must be an integer")
    if base_tokens <= 0:
        raise ValueError("base_tokens must be positive")
    factor = length_factor(mood)
    if factor >= 1.0:
        return base_tokens
    return min(base_tokens, max(180, round(base_tokens * factor)))


def is_ambivalent(emotions: Mapping[str, float] | None) -> bool:
    """Whether opposing emotion families are both materially active."""
    if not emotions:
        return False
    positive = max(
        (float(value) for name, value in emotions.items() if name in _POSITIVE_EMOTIONS),
        default=0.0,
    )
    negative = max(
        (float(value) for name, value in emotions.items() if name in _NEGATIVE_EMOTIONS),
        default=0.0,
    )
    if positive < 0.15 or negative < 0.15:
        return False
    return min(positive, negative) / max(positive, negative) >= 0.40


def _default_suppressed_octant_cues() -> dict[str, tuple[str, ...]]:
    """Reference rule: do not add upbeat momentum when a user has just been vulnerable.

    This is a conversational preset expressed as data, not kernel behavior.
    Pass ``suppressed_octant_cues={}`` to disable it, or map your own intents.
    """
    return {"VULNERABILITY": ("exuberant",)}


def _default_octant_cues() -> dict[str, str]:
    return {
        "exuberant": "Allow a little more momentum.",
        "dependent": "Keep the cadence warm and compact.",
        "relaxed": "Use an easy, unhurried pace.",
        "docile": "Keep the response quiet and spare.",
        "hostile": "Use a composed, plain register.",
        "anxious": "Keep the response shorter and less resolved.",
        "disdainful": "Use a cool, economical register.",
        "bored": "Keep the response minimal and unforced.",
    }


@dataclass(frozen=True, slots=True)
class TurnShapePolicy:
    """Caller-owned language for deterministic response-shape selection."""

    heading: str = "Turn shape:"
    statement_close: str = "Prefer a statement close unless a question is genuinely useful."
    no_question_close: str = "Do not end on another question after the previous reply did."
    one_idea: str = "Let one main idea land before adding another."
    mirror_length: str = "Keep the response proportionate to the user's message."
    ambivalence: str = "Preserve meaningful opposing signals instead of forcing a verdict."
    octant_cues: Mapping[str, str] = field(default_factory=_default_octant_cues)
    suppressed_octant_cues: Mapping[str, tuple[str, ...]] = field(
        default_factory=_default_suppressed_octant_cues
    )

    def __post_init__(self) -> None:
        normalized: dict[str, str] = {}
        for octant, cue in self.octant_cues.items():
            if not isinstance(octant, str) or not octant:
                raise ValueError("octant cue names must be non-empty strings")
            if not isinstance(cue, str):
                raise TypeError("octant cue values must be strings")
            if not cue.strip():
                raise ValueError("octant cue values must not be empty")
            normalized[octant] = cue
        object.__setattr__(self, "octant_cues", freeze_mapping(normalized))

        suppressed: dict[str, tuple[str, ...]] = {}
        for intent, octants in self.suppressed_octant_cues.items():
            if not isinstance(intent, str) or not intent:
                raise ValueError("suppressed cue intents must be non-empty strings")
            if isinstance(octants, str):
                raise TypeError("suppressed cue values must be a sequence of octant names")
            names = tuple(octants)
            for octant in names:
                if not isinstance(octant, str) or not octant:
                    raise ValueError("suppressed octant names must be non-empty strings")
            suppressed[intent.upper()] = names
        object.__setattr__(self, "suppressed_octant_cues", freeze_mapping(suppressed))


def _last_assistant_text(history: Sequence[Message | Mapping[str, object]] | None) -> str:
    for message in reversed(history or ()):
        if isinstance(message, Message):
            if message.role == "assistant":
                return message.content
            continue
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        return content if isinstance(content, str) else ""
    return ""


def turn_shape_directive(
    mood: PADMood | None,
    *,
    intent: str = "",
    history: Sequence[Message | Mapping[str, object]] | None = None,
    emotions: Mapping[str, float] | None = None,
    policy: TurnShapePolicy | None = None,
) -> str:
    """Select structural cues while leaving every word of those cues caller-configurable."""
    chosen = policy or TurnShapePolicy()
    previous_asked = _last_assistant_text(history).rstrip().endswith("?")
    rules = [
        chosen.no_question_close if previous_asked else chosen.statement_close,
        chosen.one_idea,
        chosen.mirror_length,
    ]
    if mood is not None:
        octant = mood_octant(mood.valence, mood.arousal, mood.dominance)
        suppressed = chosen.suppressed_octant_cues.get(intent.upper(), ())
        cue = "" if octant in suppressed else chosen.octant_cues.get(octant, "")
        if cue:
            rules.append(cue)
    if is_ambivalent(emotions) and chosen.ambivalence:
        rules.append(chosen.ambivalence)
    rules = [rule for rule in rules if rule]
    if not rules:
        return ""
    lines = [chosen.heading] if chosen.heading else []
    lines.extend(f"- {rule}" for rule in rules)
    return "\n".join(lines)


__all__ = [
    "TurnShapePolicy",
    "apply_length_factor",
    "decoding_params",
    "is_ambivalent",
    "length_factor",
    "mood_octant",
    "turn_shape_directive",
]
