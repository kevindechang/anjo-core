"""The numeric seam: every coefficient is caller-owned data, not kernel behavior.

The kernel already lets a domain replace every word it emits. These tests pin
the same promise for the numbers, and — just as importantly — pin that the
*defaults* still reproduce the cross-runtime contract exactly, so exposing the
knobs did not quietly move the baseline.
"""

from __future__ import annotations

import copy
import pickle
from datetime import UTC, datetime

import pytest

from affect_kernel import (
    DEFAULT_AFFECT_DYNAMICS,
    DEFAULT_RETRIEVAL_WEIGHTS,
    AffectDynamics,
    AppraisalGoals,
    CompanionState,
    MemoryCandidate,
    PADMood,
    Personality,
    RetrievalWeights,
    appraise_input,
    appraise_turn,
    decay_mood,
    decay_occ_carry,
    mood_congruence_factor,
    mood_inertia,
    rank_candidates,
    recency_weight,
    score_candidate,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


class TestDefaultsAreTheContract:
    def test_explicit_defaults_match_omitting_them(self) -> None:
        state = CompanionState(
            mood=PADMood(valence=0.4, arousal=0.2, dominance=0.1),
            baseline_valence=0.3,
        )
        implicit = appraise_turn(state, "CURIOSITY")
        explicit = appraise_turn(state, "CURIOSITY", dynamics=AffectDynamics())
        assert implicit.state.mood == explicit.state.mood
        assert implicit.state.baseline_valence == explicit.state.baseline_valence
        assert dict(implicit.active_emotions) == dict(explicit.active_emotions)

    def test_retrieval_defaults_match_omitting_them(self) -> None:
        candidate = MemoryCandidate(
            id="m", text="t", distance=0.6, significance=0.9, recall_count=5, episode=True
        )
        assert score_candidate(candidate, now=NOW) == score_candidate(
            candidate, now=NOW, weights=RetrievalWeights()
        )

    def test_module_defaults_are_equal_to_freshly_constructed(self) -> None:
        assert AffectDynamics() == DEFAULT_AFFECT_DYNAMICS
        assert RetrievalWeights() == DEFAULT_RETRIEVAL_WEIGHTS


class TestAffectDynamics:
    def test_inertia_terms_are_caller_owned(self) -> None:
        flat = AffectDynamics(inertia_neuroticism=0.0, inertia_extraversion=0.0)
        anxious = Personality(N=1.0, E=0.0)
        calm = Personality(N=0.0, E=1.0)
        assert mood_inertia(anxious, dynamics=flat) == mood_inertia(calm, dynamics=flat)
        assert mood_inertia(anxious) > mood_inertia(calm)

    def test_inertia_clamp_is_caller_owned(self) -> None:
        pinned = AffectDynamics(inertia_min=0.5, inertia_max=0.5)
        assert mood_inertia(Personality(N=1.0), dynamics=pinned) == 0.5

    def test_no_inertia_collapses_mood_to_the_resting_point(self) -> None:
        frozen = AffectDynamics(inertia_base=0.0, inertia_min=0.0, inertia_max=0.0)
        decayed = decay_mood(
            PADMood(valence=0.9, arousal=0.9, dominance=0.9),
            Personality(),
            "stranger",
            0.0,
            dynamics=frozen,
        )
        assert decayed == PADMood(valence=0.0, arousal=0.0, dominance=0.0)

    def test_resting_dominance_coefficient_is_caller_owned(self) -> None:
        raised = AffectDynamics(resting_dominance=1.0)
        default_mood = decay_mood(PADMood(), Personality(), "intimate", 0.5)
        raised_mood = decay_mood(PADMood(), Personality(), "intimate", 0.5, dynamics=raised)
        assert raised_mood.dominance > default_mood.dominance

    def test_baseline_blend_is_caller_owned(self) -> None:
        instant = AffectDynamics(baseline_retention=0.0, baseline_intake=1.0)
        result = appraise_input(PADMood(), AppraisalGoals(), "CURIOSITY", 0.9, dynamics=instant)
        assert result.baseline_valence == pytest.approx(result.mood.valence)

    def test_ambiguity_amplification_is_caller_owned(self) -> None:
        off = AffectDynamics(ambiguity_negative_gain=1.0, ambiguity_positive_gain=1.0)
        amplified = appraise_input(PADMood(valence=0.30), AppraisalGoals(), "CASUAL", 0.0)
        plain = appraise_input(PADMood(valence=0.30), AppraisalGoals(), "CASUAL", 0.0, dynamics=off)
        assert amplified.mood.valence == pytest.approx(0.3328)
        assert plain.mood.valence == pytest.approx(0.32)

    def test_carry_decay_rates_and_fallback_are_caller_owned(self) -> None:
        tuning = AffectDynamics(carry_decay={"joy": 0.5}, carry_decay_default=0.25)
        decayed = decay_occ_carry({"joy": 1.0, "admiration": 1.0}, dynamics=tuning)
        assert decayed["joy"] == pytest.approx(0.5)
        assert decayed["admiration"] == pytest.approx(0.25)

    def test_carry_floor_is_caller_owned(self) -> None:
        loose = AffectDynamics(carry_floor=0.0)
        strict = AffectDynamics(carry_floor=0.9)
        assert decay_occ_carry({"joy": 0.06}, dynamics=loose)
        assert decay_occ_carry({"joy": 0.06}, dynamics=strict) == {}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"inertia_base": 1.5},
            {"inertia_base": float("nan")},
            {"inertia_base": "high"},
            {"inertia_base": True},
            {"inertia_min": 0.9, "inertia_max": 0.1},
            {"resting_dominance": -2.0},
            {"ambiguity_negative_gain": -1.0},
            {"carry_decay": {"joy": 1.5}},
            {"carry_decay": {"": 0.5}},
            {"carry_decay": "not-a-mapping"},
            {"carry_floor": 2.0},
        ],
    )
    def test_invalid_parameters_are_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises((TypeError, ValueError)):
            AffectDynamics(**kwargs)  # type: ignore[arg-type]

    def test_is_immutable_copyable_and_picklable(self) -> None:
        tuning = AffectDynamics(carry_decay={"joy": 0.5})
        with pytest.raises((AttributeError, TypeError)):
            tuning.inertia_base = 0.1  # type: ignore[misc]
        with pytest.raises(TypeError):
            tuning.carry_decay["joy"] = 0.9  # type: ignore[index]
        assert pickle.loads(pickle.dumps(tuning)) == tuning
        assert copy.deepcopy(tuning) == tuning


class TestRetrievalWeights:
    def test_recency_horizon_is_caller_owned(self) -> None:
        short = RetrievalWeights(recency_horizon_days=10.0)
        stamp = "2025-12-17T00:00:00+00:00"  # 15 days before NOW
        assert recency_weight(stamp, now=NOW) == pytest.approx(0.75)
        assert recency_weight(stamp, now=NOW, weights=short) == pytest.approx(0.4)

    def test_recency_floor_and_fallback_are_caller_owned(self) -> None:
        tuning = RetrievalWeights(recency_floor=0.1, recency_fallback=0.2)
        assert recency_weight("1900-01-01T00:00:00+00:00", now=NOW, weights=tuning) == 0.1
        assert recency_weight("not-a-timestamp", now=NOW, weights=tuning) == 0.2

    def test_episode_bonus_is_caller_owned(self) -> None:
        episodic = MemoryCandidate(id="m", text="t", distance=0.5, episode=True)
        plain = MemoryCandidate(id="m", text="t", distance=0.5)
        none = RetrievalWeights(episode_bonus=0.0)
        assert score_candidate(episodic, now=NOW) > score_candidate(plain, now=NOW)
        assert score_candidate(episodic, now=NOW, weights=none) == pytest.approx(
            score_candidate(plain, now=NOW, weights=none)
        )

    def test_congruence_threshold_and_magnitudes_are_caller_owned(self) -> None:
        eager = RetrievalWeights(congruence_threshold=0.0, congruence_negative_mood=2.0)
        candidate = MemoryCandidate(id="m", text="t", distance=0.5, emotional_valence=-0.5)
        base = score_candidate(candidate, now=NOW)
        assert score_candidate(candidate, now=NOW, mood_valence=-0.05) == pytest.approx(base)
        assert score_candidate(
            candidate, now=NOW, mood_valence=-0.05, weights=eager
        ) == pytest.approx(base * 2.0)

    def test_weights_reach_the_ranking_entry_point(self) -> None:
        candidates = [
            MemoryCandidate(
                id="old", text="t", distance=0.4, timestamp="2025-11-01T00:00:00+00:00"
            ),
            MemoryCandidate(
                id="new", text="t", distance=0.5, timestamp="2025-12-31T00:00:00+00:00"
            ),
        ]
        patient = RetrievalWeights(recency_horizon_days=100_000.0)
        assert [m.candidate.id for m in rank_candidates(candidates, now=NOW)] == ["new", "old"]
        assert [m.candidate.id for m in rank_candidates(candidates, now=NOW, weights=patient)] == [
            "old",
            "new",
        ]

    def test_congruence_factor_accepts_custom_magnitudes(self) -> None:
        tuning = RetrievalWeights(congruence_negative_mood=3.0, congruence_positive_mood=2.0)
        assert mood_congruence_factor(-0.5, -0.5, True, weights=tuning) == 3.0
        assert mood_congruence_factor(0.5, 0.5, True, weights=tuning) == 2.0
        assert mood_congruence_factor(0.5, -0.5, True, weights=tuning) == 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"recency_horizon_days": 0.0},
            {"recency_horizon_days": -1.0},
            {"recency_floor": 1.5},
            {"rehearsal_cap": float("inf")},
            {"episode_bonus": "big"},
            {"congruence_negative_mood": -1.0},
            {"congruence_threshold": True},
        ],
    )
    def test_invalid_weights_are_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises((TypeError, ValueError)):
            RetrievalWeights(**kwargs)  # type: ignore[arg-type]
