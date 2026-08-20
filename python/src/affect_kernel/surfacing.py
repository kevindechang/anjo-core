"""Pure state-to-presence transforms with all external cognition injected."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .models import (
    AffectState,
    CognitionState,
    PresenceAffect,
    PresenceRelationship,
    PresenceVector,
)


def _smart_trim(text: object, max_chars: int) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    window = cleaned[:max_chars]
    sentence_end = max(window.rfind(mark) for mark in (".", "!", "?"))
    if sentence_end >= max_chars * 0.5:
        return window[: sentence_end + 1].rstrip()
    last_space = window.rfind(" ")
    if last_space == -1:
        return window.rstrip()
    trimmed = window[:last_space].rstrip().rstrip(",;:—-")
    return f"{trimmed}…" if trimmed else window.rstrip()


def clean_text(value: object, max_len: int) -> str | None:
    """Normalize a possible surfaced string, then shorten it at a word boundary."""
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().strip('"').split())
    if not cleaned:
        return None
    return _smart_trim(cleaned, max_len)


@dataclass(frozen=True, slots=True)
class PresenceLabels:
    """The rendered wording of the presence surface.

    The defaults are the reference conversational phrasing. They are English and
    companion-shaped on purpose; a game, tutoring, or support domain should pass
    its own labels rather than surfacing "here with you" to its users.
    """

    reflecting: str = "still reflecting"
    due_intention: str = "holding a follow-up"
    carried_thought: str = "carrying a thread"
    open_thread: str = "holding a pattern"
    idle: str = "here with you"
    reflecting_mode: str = "reflecting"
    idle_mode: str = "quiet"

    def __post_init__(self) -> None:
        for name in (
            "reflecting",
            "due_intention",
            "carried_thought",
            "open_thread",
            "idle",
            "reflecting_mode",
            "idle_mode",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")


DEFAULT_PRESENCE_LABELS = PresenceLabels()


def presence_line(
    *,
    reflection_pending: bool,
    due_intention: bool,
    carried_thought: bool,
    open_thread: bool,
    labels: PresenceLabels | None = None,
) -> str:
    """Select one compact presence label using a fixed priority cascade."""
    chosen = labels or DEFAULT_PRESENCE_LABELS
    if reflection_pending:
        return chosen.reflecting
    if due_intention:
        return chosen.due_intention
    if carried_thought:
        return chosen.carried_thought
    if open_thread:
        return chosen.open_thread
    return chosen.idle


def build_presence_vector(
    state: AffectState,
    cognition: CognitionState | None = None,
    *,
    labels: PresenceLabels | None = None,
) -> PresenceVector:
    """Build a UI-ready presence snapshot without reading persistence or services."""
    chosen = labels or DEFAULT_PRESENCE_LABELS
    supplied = cognition or CognitionState()
    carried = bool(clean_text(state.carried_thought, 300))
    resolved_cognition = replace(supplied, carried_thought=carried)
    line = presence_line(
        reflection_pending=resolved_cognition.reflection_pending,
        due_intention=resolved_cognition.due_intention,
        carried_thought=resolved_cognition.carried_thought,
        open_thread=resolved_cognition.open_thread,
        labels=chosen,
    )
    trust = round(state.relationship.trust, 4)
    valence = round(state.mood.valence, 4)
    arousal = round(state.mood.arousal, 4)
    longing = round(state.attachment.longing, 4)
    return PresenceVector(
        trust=trust,
        valence=valence,
        arousal=arousal,
        longing=longing,
        awaiting=resolved_cognition.reflection_pending,
        mode=chosen.reflecting_mode if resolved_cognition.reflection_pending else chosen.idle_mode,
        line=line,
        affect=PresenceAffect(
            valence=valence,
            arousal=arousal,
            dominance=round(state.mood.dominance, 4),
        ),
        relationship=PresenceRelationship(
            stage=state.relationship.stage,
            trust=trust,
            longing=longing,
            comfort=round(state.attachment.comfort, 4),
        ),
        cognition=resolved_cognition,
    )


__all__ = [
    "DEFAULT_PRESENCE_LABELS",
    "PresenceLabels",
    "build_presence_vector",
    "clean_text",
    "presence_line",
]
