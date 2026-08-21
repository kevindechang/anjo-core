"""Pure memory scoring and ranking, independent of storage or embeddings.

The relevance x recency x salience decomposition follows Generative Agents
(Park et al. 2023), which sums those factors where this module multiplies them.
Constant provenance is recorded in ``docs/foundations.md`` sections 6-7.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from .models import MemoryCandidate, RankedMemory


def _bounded_number(name: str, value: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be finite and within [{low}, {high}]")
    return number


@dataclass(frozen=True, slots=True)
class RetrievalWeights:
    """The numeric parameters of the retrieval scorer.

    Defaults reproduce the pinned cross-runtime contract. These are magnitudes
    only: the *shape* of the curves -- linear recency, multiplicative
    composition, log-compressed rehearsal -- is fixed here, and
    ``docs/foundations.md`` section 6 records why each shape was chosen and what
    the closest published comparable does instead.
    """

    recency_horizon_days: float = 60.0
    recency_floor: float = 0.40
    recency_fallback: float = 0.70
    significance_weight: float = 0.03
    rehearsal_weight: float = 0.006
    rehearsal_cap: float = 0.025
    episode_bonus: float = 0.05
    congruence_threshold: float = 0.20
    congruence_negative_mood: float = 1.06
    congruence_positive_mood: float = 1.03

    def __post_init__(self) -> None:
        horizon = _bounded_number(
            "recency_horizon_days", self.recency_horizon_days, 0.0, 3_650_000.0
        )
        if horizon <= 0.0:
            raise ValueError("recency_horizon_days must be positive")
        for name in (
            "recency_floor",
            "recency_fallback",
            "significance_weight",
            "rehearsal_weight",
            "rehearsal_cap",
            "episode_bonus",
            "congruence_threshold",
        ):
            _bounded_number(name, getattr(self, name), 0.0, 1.0)
        for name in ("congruence_negative_mood", "congruence_positive_mood"):
            _bounded_number(name, getattr(self, name), 0.0, 10.0)


DEFAULT_RETRIEVAL_WEIGHTS = RetrievalWeights()


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


def recency_weight(
    timestamp: str,
    *,
    now: datetime | None = None,
    weights: RetrievalWeights | None = None,
) -> float:
    """Return a linear freshness weight with a 0.4 floor and 0.7 parse fallback.

    Human forgetting is better described by a power law (Wixted & Ebbesen 1991)
    and the closest published comparable decays exponentially, so this curve is
    the one most obviously open to challenge. It survived that challenge: at a
    matched 30-day half-life it out-ranked both alternatives in ``bench/``. See
    ``docs/foundations.md`` section 6.
    """
    reference = _require_aware(now or datetime.now(UTC), "now")
    tuning = weights or DEFAULT_RETRIEVAL_WEIGHTS
    try:
        parsed = datetime.fromisoformat(timestamp)
        _require_aware(parsed, "timestamp")
        days_ago = max(0.0, (reference - parsed).total_seconds() / 86_400)
        return max(
            tuning.recency_floor,
            min(1.0, 1.0 - days_ago / tuning.recency_horizon_days),
        )
    except (TypeError, ValueError, OverflowError):
        return tuning.recency_fallback


def mood_congruence_factor(
    mem_valence: float,
    mood_valence: float,
    congruence_on: bool,
    *,
    weights: RetrievalWeights | None = None,
) -> float:
    """Return the small asymmetric multiplier for same-sign memory and mood valence.

    Mood-congruent recall is Bower (1981); the threshold, the magnitudes, and
    the negative/positive asymmetry are production-tuned and unsupported by any
    citation in ``docs/foundations.md`` section 7.
    """
    if not congruence_on or mem_valence == 0.0:
        return 1.0
    tuning = weights or DEFAULT_RETRIEVAL_WEIGHTS
    if (mem_valence > 0.0) == (mood_valence > 0.0):
        return (
            tuning.congruence_negative_mood
            if mood_valence < 0.0
            else tuning.congruence_positive_mood
        )
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
    weights: RetrievalWeights | None = None,
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
    tuning = weights or DEFAULT_RETRIEVAL_WEIGHTS
    similarity = similarity_from_distance(distance)
    recency = max(
        tuning.recency_floor,
        min(1.0, 1.0 - max(0.0, days_ago) / tuning.recency_horizon_days),
    )
    bounded_significance = max(0.0, min(1.0, significance))
    salience = (
        1.0
        + bounded_significance * tuning.significance_weight
        + min(tuning.rehearsal_cap, math.log1p(recall_count) * tuning.rehearsal_weight)
    )
    return similarity * recency * salience + (tuning.episode_bonus if episode else 0.0)


def score_candidate(
    candidate: MemoryCandidate,
    *,
    now: datetime | None = None,
    mood_valence: float = 0.0,
    mood_congruence: bool = True,
    weights: RetrievalWeights | None = None,
) -> float:
    """Score a candidate carrying an ISO timestamp and optional emotional valence."""
    reference = _require_aware(now or datetime.now(UTC), "now")
    tuning = weights or DEFAULT_RETRIEVAL_WEIGHTS
    similarity = similarity_from_distance(candidate.distance)
    recency = recency_weight(candidate.timestamp or "", now=reference, weights=tuning)
    significance = max(0.0, min(1.0, candidate.significance))
    salience = (
        1.0
        + significance * tuning.significance_weight
        + min(
            tuning.rehearsal_cap,
            math.log1p(max(0, candidate.recall_count)) * tuning.rehearsal_weight,
        )
    )
    score = similarity * recency * salience + (tuning.episode_bonus if candidate.episode else 0.0)
    congruence_on = mood_congruence and abs(mood_valence) >= tuning.congruence_threshold
    return score * mood_congruence_factor(
        candidate.emotional_valence,
        mood_valence,
        congruence_on,
        weights=tuning,
    )


def rank_candidates(
    candidates: Iterable[MemoryCandidate],
    *,
    limit: int = 4,
    now: datetime | None = None,
    mood_valence: float = 0.0,
    mood_congruence: bool = True,
    weights: RetrievalWeights | None = None,
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
                weights=weights,
            ),
        )
        previous = best.get(candidate.id)
        if previous is None or ranked.score > previous.score:
            best[candidate.id] = ranked
    ordered = sorted(best.values(), key=lambda item: (-item.score, item.candidate.id))
    return tuple(ordered[:limit])


__all__ = [
    "DEFAULT_RETRIEVAL_WEIGHTS",
    "RetrievalWeights",
    "candidate_score",
    "mood_congruence_factor",
    "rank_candidates",
    "recency_weight",
    "score_candidate",
    "similarity_from_distance",
]
