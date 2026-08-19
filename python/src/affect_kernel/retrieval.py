"""Pure memory scoring and ranking, independent of storage or embeddings.

The relevance x recency x salience decomposition follows Generative Agents
(Park et al. 2023), which sums those factors where this module multiplies them.
Constant provenance is recorded in ``docs/foundations.md`` sections 6-7.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime

from .models import MemoryCandidate, RankedMemory


def _require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _validate_distance(distance: float) -> float:
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        raise TypeError("distance must be a number")
    if not math.isfinite(distance) or not 0.0 <= distance <= 2.0:
        raise ValueError("distance must be finite and within [0, 2]")
    return distance


def recency_weight(timestamp: str, *, now: datetime | None = None) -> float:
    """Return a linear freshness weight with a 0.4 floor and 0.7 parse fallback.

    Linear-to-a-floor is the least defensible curve in the module: human
    forgetting is better described by a power law (Wixted & Ebbesen 1991) and
    the closest published comparable decays exponentially. It is kept because
    it is trivially inspectable and because the floor dominates in practice.
    See ``docs/foundations.md`` section 6.
    """
    reference = _require_aware(now or datetime.now(UTC), "now")
    try:
        parsed = datetime.fromisoformat(timestamp)
        _require_aware(parsed, "timestamp")
        days_ago = max(0.0, (reference - parsed).total_seconds() / 86_400)
        return max(0.4, min(1.0, 1.0 - days_ago / 60.0))
    except (TypeError, ValueError, OverflowError):
        return 0.7


def mood_congruence_factor(
    mem_valence: float,
    mood_valence: float,
    congruence_on: bool,
) -> float:
    """Return the small asymmetric multiplier for same-sign memory and mood valence.

    Mood-congruent recall is Bower (1981); the threshold, the magnitudes, and
    the negative/positive asymmetry are production-tuned and unsupported by any
    citation in ``docs/foundations.md`` section 7.
    """
    if not congruence_on or mem_valence == 0.0:
        return 1.0
    if (mem_valence > 0.0) == (mood_valence > 0.0):
        return 1.06 if mood_valence < 0.0 else 1.03
    return 1.0


def similarity_from_distance(distance: float) -> float:
    """Convert cosine distance from the production convention into similarity."""
    return 1.0 - _validate_distance(distance) / 2.0


def candidate_score(
    distance: float,
    days_ago: float,
    *,
    episode: bool,
    significance: float,
    recall_count: int,
) -> float:
    """Score a worked candidate from distance, age, salience, and rehearsal."""
    _validate_distance(distance)
    if not math.isfinite(days_ago):
        raise ValueError("days_ago must be finite")
    if not math.isfinite(significance):
        raise ValueError("significance must be finite")
    if isinstance(recall_count, bool) or not isinstance(recall_count, int):
        raise TypeError("recall_count must be an integer")
    if recall_count < 0:
        raise ValueError("recall_count must be non-negative")
    similarity = similarity_from_distance(distance)
    recency = max(0.4, min(1.0, 1.0 - max(0.0, days_ago) / 60.0))
    bounded_significance = max(0.0, min(1.0, significance))
    salience = 1.0 + bounded_significance * 0.03 + min(0.025, math.log1p(recall_count) * 0.006)
    return similarity * recency * salience + (0.05 if episode else 0.0)


def score_candidate(
    candidate: MemoryCandidate,
    *,
    now: datetime | None = None,
    mood_valence: float = 0.0,
    mood_congruence: bool = True,
) -> float:
    """Score a candidate carrying an ISO timestamp and optional emotional valence."""
    reference = _require_aware(now or datetime.now(UTC), "now")
    similarity = similarity_from_distance(candidate.distance)
    recency = recency_weight(candidate.timestamp or "", now=reference)
    significance = max(0.0, min(1.0, candidate.significance))
    salience = (
        1.0
        + significance * 0.03
        + min(
            0.025,
            math.log1p(max(0, candidate.recall_count)) * 0.006,
        )
    )
    score = similarity * recency * salience + (0.05 if candidate.episode else 0.0)
    congruence_on = mood_congruence and abs(mood_valence) >= 0.20
    return score * mood_congruence_factor(
        candidate.emotional_valence,
        mood_valence,
        congruence_on,
    )


def rank_candidates(
    candidates: Iterable[MemoryCandidate],
    *,
    limit: int = 4,
    now: datetime | None = None,
    mood_valence: float = 0.0,
    mood_congruence: bool = True,
) -> tuple[RankedMemory, ...]:
    """Deduplicate by id, retain the best score, and return a stable descending rank."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    reference = _require_aware(now or datetime.now(UTC), "now")
    best: dict[str, RankedMemory] = {}
    for candidate in candidates:
        ranked = RankedMemory(
            candidate=candidate,
            score=score_candidate(
                candidate,
                now=reference,
                mood_valence=mood_valence,
                mood_congruence=mood_congruence,
            ),
        )
        previous = best.get(candidate.id)
        if previous is None or ranked.score > previous.score:
            best[candidate.id] = ranked
    ordered = sorted(best.values(), key=lambda item: (-item.score, item.candidate.id))
    return tuple(ordered[:limit])


__all__ = [
    "candidate_score",
    "mood_congruence_factor",
    "rank_candidates",
    "recency_weight",
    "score_candidate",
    "similarity_from_distance",
]
