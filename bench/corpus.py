"""A seeded synthetic retrieval corpus with an explicit generative model.

The point of the design is that **relevance is defined independently of every
scoring formula under test**. For each query exactly one memory is the gold
answer, and gold status is decided first. Similarity, age, significance,
rehearsal count, episode status, and emotional valence are then drawn from
distributions that are *conditioned on* gold status only to the degree the
regime says they should be.

That separation is what makes an unflattering result possible. In the
``uncorrelated`` regime the nuisance signals carry no information about which
memory is correct, so any scorer that weights them is adding noise and should
lose to plain similarity. A benchmark whose generator quietly encodes the
kernel's own assumptions could not produce that outcome.

The similarity distributions overlap on purpose. With separable distributions
every scorer scores 1.0 and the benchmark measures nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from affect_kernel import MemoryCandidate

Regime = Literal[
    "uncorrelated",
    "correlated",
    "recency_informative",
    "both_informative",
    "mood_informative",
]

# Overlapping beta-ish draws: gold is usually nearer, but not always.
_GOLD_DISTANCE = (0.30, 0.22)  # mean, spread
_DISTRACTOR_DISTANCE = (0.62, 0.26)


@dataclass(frozen=True, slots=True)
class Query:
    """One retrieval episode: a candidate pool with exactly one correct memory."""

    gold_id: str
    candidates: tuple[MemoryCandidate, ...]
    ages: dict[str, float]
    mood_valence: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _distance(rng: random.Random, gold: bool) -> float:
    mean, spread = _GOLD_DISTANCE if gold else _DISTRACTOR_DISTANCE
    return _clamp(rng.gauss(mean, spread), 0.0, 2.0)


def build_queries(
    *,
    regime: Regime,
    query_count: int = 400,
    pool_size: int = 20,
    seed: int = 20260820,
) -> list[Query]:
    """Generate a deterministic corpus for one regime."""
    rng = random.Random(seed)
    queries: list[Query] = []
    for q in range(query_count):
        gold_index = rng.randrange(pool_size)
        gold_id = f"q{q}-m{gold_index}"
        mood = rng.choice([-0.6, -0.3, 0.3, 0.6])
        candidates: list[MemoryCandidate] = []
        ages: dict[str, float] = {}
        for m in range(pool_size):
            is_gold = m == gold_index
            memory_id = f"q{q}-m{m}"

            if regime in ("recency_informative", "both_informative"):
                # Age carries real signal: the answer is usually the fresher item.
                age = rng.uniform(0.0, 20.0) if is_gold else rng.uniform(0.0, 180.0)
            else:
                age = rng.uniform(0.0, 180.0)

            if regime in ("correlated", "both_informative"):
                # Important, rehearsed, episodic memories really are likelier answers.
                significance = _clamp(
                    rng.gauss(0.80 if is_gold else 0.35, 0.18), 0.0, 1.0
                )
                recall_count = rng.randrange(8, 40) if is_gold else rng.randrange(0, 6)
                episode = rng.random() < (0.75 if is_gold else 0.15)
            else:
                significance = _clamp(rng.gauss(0.5, 0.2), 0.0, 1.0)
                recall_count = rng.randrange(0, 40)
                episode = rng.random() < 0.3

            if regime == "mood_informative":
                # The answer usually shares the sign of the current mood.
                same_sign = rng.random() < 0.75 if is_gold else rng.random() < 0.5
                magnitude = rng.uniform(0.2, 1.0)
                sign = 1.0 if (mood > 0) == same_sign else -1.0
                valence = sign * magnitude
            else:
                valence = rng.uniform(-1.0, 1.0)

            ages[memory_id] = age
            candidates.append(
                MemoryCandidate(
                    id=memory_id,
                    text=f"memory {memory_id}",
                    distance=_distance(rng, is_gold),
                    episode=episode,
                    significance=significance,
                    recall_count=recall_count,
                    emotional_valence=valence,
                )
            )
        queries.append(
            Query(
                gold_id=gold_id,
                candidates=tuple(candidates),
                ages=ages,
                mood_valence=mood,
            )
        )
    return queries
