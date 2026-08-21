"""Ranking functions compared in the retrieval benchmark.

Every alternative is implemented here rather than imported, so that the
comparison is against a written-down formula a reader can check, not against
another part of this library. Only ``kernel_default`` and ``kernel_no_salience``
call into ``affect_kernel``.

The recency curves are deliberately given a **common half-life** (30 days), so
the shootout compares the *shape* of forgetting rather than an arbitrary
difference in scale. The kernel's linear curve reaches 0.5 at 30 days by
construction (``1 - 30/60``); the exponential and power-law curves are solved
to match it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from affect_kernel import DEFAULT_RETRIEVAL_WEIGHTS, MemoryCandidate, candidate_score

HALF_LIFE_DAYS = 30.0
_POWER_LAW_BETA = math.log(2.0) / math.log(1.0 + HALF_LIFE_DAYS)


def similarity(candidate: MemoryCandidate) -> float:
    """Cosine distance in [0, 2] mapped to similarity in [0, 1]."""
    return 1.0 - candidate.distance / 2.0


def linear_recency(days_ago: float) -> float:
    """The kernel's curve: linear to a floor. Half-life 30 days by construction."""
    return max(
        DEFAULT_RETRIEVAL_WEIGHTS.recency_floor,
        min(1.0, 1.0 - days_ago / DEFAULT_RETRIEVAL_WEIGHTS.recency_horizon_days),
    )


def exponential_recency(days_ago: float) -> float:
    """Generative Agents' shape, rescaled to the same half-life."""
    return 0.5 ** (max(0.0, days_ago) / HALF_LIFE_DAYS)


def power_law_recency(days_ago: float) -> float:
    """The shape human forgetting actually follows, at the same half-life."""
    return (1.0 + max(0.0, days_ago)) ** -_POWER_LAW_BETA


def _salience(candidate: MemoryCandidate) -> float:
    weights = DEFAULT_RETRIEVAL_WEIGHTS
    return (
        1.0
        + min(1.0, max(0.0, candidate.significance)) * weights.significance_weight
        + min(
            weights.rehearsal_cap,
            math.log1p(candidate.recall_count) * weights.rehearsal_weight,
        )
    )


def _days_ago(candidate: MemoryCandidate, ages: dict[str, float]) -> float:
    return ages[candidate.id]


Scorer = Callable[[MemoryCandidate, dict[str, float], float], float]


def similarity_only(
    candidate: MemoryCandidate, ages: dict[str, float], mood: float
) -> float:
    """Ablation: no recency, no salience, no episode bonus, no congruence."""
    return similarity(candidate)


def kernel_default(
    candidate: MemoryCandidate, ages: dict[str, float], mood: float
) -> float:
    """The library's own scorer, called through its public entry point."""
    return candidate_score(
        candidate.distance,
        _days_ago(candidate, ages),
        episode=candidate.episode,
        significance=candidate.significance,
        recall_count=candidate.recall_count,
    )


def kernel_no_salience(
    candidate: MemoryCandidate, ages: dict[str, float], mood: float
) -> float:
    """Ablation: keep multiplicative recency, drop salience and the episode bonus."""
    return similarity(candidate) * linear_recency(_days_ago(candidate, ages))


def multiplicative_exponential(
    candidate: MemoryCandidate, ages: dict[str, float], mood: float
) -> float:
    """The kernel's composition with an exponential recency curve."""
    return similarity(candidate) * exponential_recency(
        _days_ago(candidate, ages)
    ) * _salience(candidate) + (
        DEFAULT_RETRIEVAL_WEIGHTS.episode_bonus if candidate.episode else 0.0
    )


def multiplicative_power_law(
    candidate: MemoryCandidate, ages: dict[str, float], mood: float
) -> float:
    """The kernel's composition with a power-law recency curve."""
    return similarity(candidate) * power_law_recency(
        _days_ago(candidate, ages)
    ) * _salience(candidate) + (
        DEFAULT_RETRIEVAL_WEIGHTS.episode_bonus if candidate.episode else 0.0
    )


def additive_park(
    candidate: MemoryCandidate, ages: dict[str, float], mood: float
) -> float:
    """Generative Agents' shape: an equally weighted sum of the three factors.

    Park et al. use alpha_recency = alpha_importance = alpha_relevance = 1 over
    min-max normalized components. Components here are already in [0, 1], so the
    sum is taken directly and the ordering is unaffected by the missing rescale.
    """
    return (
        exponential_recency(_days_ago(candidate, ages))
        + min(1.0, max(0.0, candidate.significance))
        + similarity(candidate)
    )


def kernel_with_congruence(
    candidate: MemoryCandidate, ages: dict[str, float], mood: float
) -> float:
    """The kernel's scorer plus the mood-congruence multiplier."""
    weights = DEFAULT_RETRIEVAL_WEIGHTS
    score = kernel_default(candidate, ages, mood)
    if abs(mood) < weights.congruence_threshold or candidate.emotional_valence == 0.0:
        return score
    if (candidate.emotional_valence > 0.0) == (mood > 0.0):
        return score * (
            weights.congruence_negative_mood
            if mood < 0.0
            else weights.congruence_positive_mood
        )
    return score


@dataclass(frozen=True, slots=True)
class NamedScorer:
    name: str
    note: str
    fn: Scorer
