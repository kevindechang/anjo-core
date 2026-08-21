"""Seeded property and fuzz tests over the kernel's stated invariants.

Deliberately built on ``random.Random(seed)`` rather than a property-testing
dependency: the repository's whole claim is that its behavior is reproducible
without pulling anything in, and a fixed seed makes any failure re-runnable by
anyone. Each test names the invariant it defends.
"""

from __future__ import annotations

import copy
import pickle
import random

import pytest

from affect_kernel import (
    BUILTIN_INTENTS,
    AffectState,
    AppraisalGoals,
    AttachmentState,
    MemoryCandidate,
    PADMood,
    Personality,
    RelationshipState,
    apply_length_factor,
    appraise_turn,
    build_presence_vector,
    clean_text,
    decoding_params,
    rank_candidates,
)

SEED = 20260820
CASES = 2_000
STAGES = ("stranger", "acquaintance", "friend", "close", "intimate")
INTENTS = sorted(BUILTIN_INTENTS)

# Codepoints that historically break naive text handling: a combining acute, a
# zero-width space and joiner, an RTL override, NEL, an ideographic space, a
# BOM, an astral-plane emoji, the last valid codepoint, and quoting characters.
NASTY_CHARS = "\u0301\u200b\u200d\u202e\u0085\u3000\ufeff\U0001f600\U0010ffff\t\n\r \"'\\/<>&abc."

BLANK_TEXTS = ["", "   ", "\u200b", '""', "\ufeff", "\u0085", "\u3000"]


def _rng() -> random.Random:
    return random.Random(SEED)


def _random_state(rng: random.Random) -> AffectState:
    return AffectState(
        mood=PADMood(
            valence=rng.uniform(-1, 1), arousal=rng.uniform(-1, 1), dominance=rng.uniform(-1, 1)
        ),
        personality=Personality(
            O=rng.random(), C=rng.random(), E=rng.random(), A=rng.random(), N=rng.random()
        ),
        goals=AppraisalGoals(
            rapport=rng.random(),
            intellectual=rng.random(),
            autonomy=rng.random(),
            respect=rng.random(),
            honesty=rng.random(),
        ),
        relationship=RelationshipState(
            stage=rng.choice(STAGES),
            trust=rng.random(),
            session_count=rng.randrange(0, 500),
            prior_session_valence=rng.uniform(-1, 1),
        ),
        attachment=AttachmentState(weight=rng.random(), longing=rng.random(), comfort=rng.random()),
        baseline_valence=rng.uniform(-1, 1),
        occ_carry={name: rng.random() for name in ("joy", "distress", "reproach")},
    )


def _nasty_text(rng: random.Random) -> str:
    return "".join(rng.choice(NASTY_CHARS) for _ in range(rng.randrange(0, 80)))


def _assert_state_in_range(state: AffectState) -> None:
    for axis in (state.mood.valence, state.mood.arousal, state.mood.dominance):
        assert -1.0 <= axis <= 1.0
    assert -1.0 <= state.baseline_valence <= 1.0
    for value in state.occ_carry.values():
        assert 0.0 <= value <= 1.0


class TestAppraisalStaysBounded:
    def test_one_turn_never_leaves_the_declared_domains(self) -> None:
        rng = _rng()
        for _ in range(CASES):
            result = appraise_turn(_random_state(rng), rng.choice(INTENTS))
            _assert_state_in_range(result.state)
            for value in result.active_emotions.values():
                assert 0.0 <= value <= 1.0

    def test_a_long_adversarial_walk_never_diverges(self) -> None:
        """200 turns of worst-case intent choice must not escape the domain."""
        rng = _rng()
        for _ in range(40):
            state = _random_state(rng)
            for _ in range(200):
                # Bias hard toward the strongest impulses in both directions.
                intent = rng.choice(["ABUSE", "CURIOSITY", "VULNERABILITY", "ABUSE"])
                state = appraise_turn(state, intent).state
                _assert_state_in_range(state)

    def test_repeating_the_same_input_gives_the_same_output(self) -> None:
        """Guards against set or dict iteration order leaking into results."""
        rng = _rng()
        for _ in range(200):
            state = _random_state(rng)
            intent = rng.choice(INTENTS)
            first = appraise_turn(state, intent)
            second = appraise_turn(state, intent)
            assert first.state == second.state
            assert dict(first.active_emotions) == dict(second.active_emotions)

    def test_appraisal_never_mutates_the_state_it_was_given(self) -> None:
        rng = _rng()
        for _ in range(200):
            state = _random_state(rng)
            before = copy.deepcopy(state)
            appraise_turn(state, rng.choice(INTENTS))
            assert state == before


class TestAffectControlsStayBounded:
    def test_length_factor_can_only_shorten_a_budget(self) -> None:
        rng = _rng()
        for _ in range(CASES):
            budget = rng.randrange(1, 100_000)
            mood = PADMood(
                valence=rng.uniform(-1, 1),
                arousal=rng.uniform(-1, 1),
                dominance=rng.uniform(-1, 1),
            )
            assert 0 < apply_length_factor(budget, mood) <= budget

    def test_decoding_stays_inside_the_published_envelope(self) -> None:
        rng = _rng()
        for _ in range(CASES):
            params = decoding_params(
                PADMood(
                    valence=rng.uniform(-1, 1),
                    arousal=rng.uniform(-1, 1),
                    dominance=rng.uniform(-1, 1),
                )
            )
            assert 0.72 <= params.temperature <= 1.18
            assert params.top_p == 0.97


class TestRankingIsATotalOrder:
    @staticmethod
    def _candidates(rng: random.Random, count: int) -> list[MemoryCandidate]:
        return [
            MemoryCandidate(
                id=f"m{rng.randrange(0, count)}",
                text="t",
                distance=rng.uniform(0, 2),
                episode=rng.random() < 0.3,
                significance=rng.random(),
                recall_count=rng.randrange(0, 1_000),
                emotional_valence=rng.uniform(-1, 1),
            )
            for _ in range(count)
        ]

    def test_ranking_dedupes_sorts_and_respects_the_limit(self) -> None:
        rng = _rng()
        for _ in range(500):
            limit = rng.randrange(0, 10)
            candidates = self._candidates(rng, rng.randrange(1, 25))
            ranked = rank_candidates(candidates, limit=limit, mood_valence=rng.uniform(-1, 1))
            ids = [item.candidate.id for item in ranked]
            assert len(ids) == len(set(ids)), "duplicate ids survived ranking"
            assert len(ids) <= limit
            scores = [item.score for item in ranked]
            assert scores == sorted(scores, reverse=True)

    def test_shuffling_the_input_does_not_change_the_ranking(self) -> None:
        rng = _rng()
        for _ in range(300):
            unique = {c.id: c for c in self._candidates(rng, 20)}
            expected = [m.candidate.id for m in rank_candidates(unique.values(), limit=5)]
            shuffled = list(unique.values())
            rng.shuffle(shuffled)
            assert [m.candidate.id for m in rank_candidates(shuffled, limit=5)] == expected


class TestUnicodeAndSerialization:
    def test_clean_text_never_exceeds_its_limit_and_survives_a_utf8_round_trip(self) -> None:
        rng = _rng()
        for _ in range(CASES):
            limit = rng.randrange(1, 40)
            cleaned = clean_text(_nasty_text(rng), limit)
            if cleaned is None:
                continue
            assert len(cleaned) <= limit
            assert cleaned.encode("utf-8").decode("utf-8") == cleaned

    def test_states_holding_nasty_text_pickle_and_deepcopy_intact(self) -> None:
        rng = _rng()
        for _ in range(300):
            base = _random_state(rng)
            state = AffectState(
                mood=base.mood,
                personality=base.personality,
                goals=base.goals,
                relationship=base.relationship,
                attachment=base.attachment,
                baseline_valence=base.baseline_valence,
                carried_thought=_nasty_text(rng) or None,
                occ_carry=base.occ_carry,
                expectation=_nasty_text(rng),
            )
            assert pickle.loads(pickle.dumps(state)) == state
            assert copy.deepcopy(state) == state

    def test_presence_surfacing_stays_in_range_for_any_state(self) -> None:
        rng = _rng()
        for _ in range(300):
            vector = build_presence_vector(_random_state(rng))
            assert -1.0 <= vector.affect.valence <= 1.0
            assert isinstance(vector.line, str)


@pytest.mark.parametrize("text", BLANK_TEXTS)
def test_effectively_empty_text_cleans_without_leaking_whitespace(text: str) -> None:
    cleaned = clean_text(text, 20)
    assert cleaned is None or cleaned == cleaned.strip()
