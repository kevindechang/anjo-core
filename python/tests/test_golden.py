from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from anjo_core.affect import (
    apply_length_factor,
    decoding_params,
    is_ambivalent,
    length_factor,
    mood_octant,
)
from anjo_core.appraisal import (
    appraise_input,
    appraise_turn,
    baseline_weight,
    decay_mood,
    decay_occ_carry,
    expectation_emotions,
    mood_inertia,
    stage_int,
    state_emotions,
)
from anjo_core.models import (
    AppraisalGoals,
    AttachmentState,
    CognitionState,
    CompanionState,
    PADMood,
    Personality,
    RelationshipState,
)
from anjo_core.retrieval import candidate_score, mood_congruence_factor, recency_weight
from anjo_core.surfacing import build_presence_vector, clean_text, presence_line

_REPOSITORY_GOLDEN = (
    Path(__file__).resolve().parents[2] / "shared" / "golden" / "kernel_golden.json"
)
_SDIST_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "kernel_golden.json"
GOLDEN_PATH = _REPOSITORY_GOLDEN if _REPOSITORY_GOLDEN.is_file() else _SDIST_GOLDEN


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    assert GOLDEN_PATH.is_file(), f"shared parity fixture is missing: {GOLDEN_PATH}"
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _state(values: dict[str, Any]) -> CompanionState:
    return CompanionState(
        mood=PADMood(
            values.get("valence", 0.0),
            values.get("arousal", 0.0),
            values.get("dominance", 0.0),
        ),
        personality=Personality(
            O=values.get("O", 0.8),
            C=values.get("C", 0.72),
            E=values.get("E", 0.45),
            A=values.get("A", 0.72),
            N=values.get("N", 0.15),
        ),
        goals=AppraisalGoals(
            rapport=values.get("rapport", 0.8),
            intellectual=values.get("intellectual", 0.8),
            autonomy=values.get("autonomy", 0.7),
            respect=values.get("respect", 0.85),
            honesty=values.get("honesty", 0.9),
        ),
        relationship=RelationshipState(
            stage={1: "stranger", 2: "acquaintance", 3: "friend", 4: "close", 5: "intimate"}.get(
                values.get("stage_int"), values.get("stage", "stranger")
            ),
            trust=values.get("trust_score", 0.0),
            session_count=values.get("session_count", 0),
            prior_session_valence=values.get("prior_session_valence", 0.0),
        ),
        attachment=AttachmentState(
            longing=values.get("longing", 0.0),
            comfort=values.get("comfort", 0.0),
        ),
        baseline_valence=values.get("baseline_valence", 0.0),
        carried_thought=values.get("carried_thought"),
    )


def _assert_nested_approx(actual: Any, expected: Any) -> None:
    if isinstance(expected, bool):
        assert actual is expected
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-9)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_nested_approx(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected, strict=True):
            _assert_nested_approx(got, want)
    else:
        assert actual == expected


def test_shared_fixture_has_only_the_public_deterministic_surfaces(golden: dict[str, Any]) -> None:
    assert {key for key in golden if not key.startswith("_")} == {
        "affect_control",
        "retrieval",
        "appraisal",
        "surfacing",
    }
    assert "turn_shape_directive" not in golden["affect_control"]


def test_affect_golden(golden: dict[str, Any]) -> None:
    cases = golden["affect_control"]
    for case in cases["mood_octant"]:
        values = case["in"]
        assert mood_octant(**values) == case["out"]
    for case in cases["decoding_params"]:
        values = case["in"]
        result = decoding_params(None if values is None else PADMood(**values))
        _assert_nested_approx(asdict(result), case["out"])
    for case in cases["length_factor"]:
        values = case["in"]
        result = length_factor(None if values is None else PADMood(**values))
        _assert_nested_approx(result, case["out"])
    for case in cases["apply_length_factor"]:
        values = case["in"]
        mood = PADMood(values["valence"], values["arousal"], values["dominance"])
        assert apply_length_factor(values["base_tokens"], mood) == case["out"]
    for case in cases["is_ambivalent"]:
        assert is_ambivalent(case["in"]) is case["out"]


def test_retrieval_golden(golden: dict[str, Any]) -> None:
    cases = golden["retrieval"]
    now = datetime.now(UTC)
    for case in cases["recency_weight"]:
        values = case["in"]
        if "timestamp" in values:
            actual = recency_weight(values["timestamp"], now=now)
        else:
            timestamp = (now - timedelta(days=values["days_ago"])).isoformat()
            actual = recency_weight(timestamp, now=now)
        _assert_nested_approx(actual, case["out"])
    for case in cases["mood_congruence_factor"]:
        assert mood_congruence_factor(**case["in"]) == pytest.approx(case["out"], abs=1e-9)
    for case in cases["candidate_score"]:
        assert candidate_score(**case["in"]) == pytest.approx(case["out"], abs=1e-9)


def test_appraisal_golden(golden: dict[str, Any]) -> None:
    cases = golden["appraisal"]
    for case in cases["mood_inertia"]:
        assert mood_inertia(Personality(**case["in"])) == pytest.approx(case["out"], abs=1e-9)
    for case in cases["stage_int"]:
        assert stage_int(case["in"]["stage"]) == case["out"]
    for case in cases["baseline_weight"]:
        assert baseline_weight(case["in"]["stage_int"]) == pytest.approx(case["out"], abs=1e-9)
    for case in cases["decay_mood"]:
        values = case["in"]
        actual = decay_mood(
            PADMood(values["valence"], values["arousal"], values["dominance"]),
            Personality(O=values["O"], C=values["C"], E=values["E"], A=values["A"], N=values["N"]),
            values["stage_int"],
            values["baseline_valence"],
        )
        _assert_nested_approx(asdict(actual), case["out"])
    for case in cases["appraise_input"]:
        values = case["in"]
        actual = appraise_input(
            PADMood(values["valence"], values["arousal"], values["dominance"]),
            AppraisalGoals(
                rapport=values["rapport"],
                intellectual=values["intellectual"],
                autonomy=values["autonomy"],
                respect=values["respect"],
                honesty=values["honesty"],
            ),
            values["intent"],
            values["baseline_valence"],
        )
        _assert_nested_approx(
            {
                "emotions": dict(actual.emotions),
                "mood": asdict(actual.mood),
                "baseline_valence": actual.baseline_valence,
            },
            case["out"],
        )
    for case in cases["expectation_emotions"]:
        values = case["in"]
        _assert_nested_approx(
            expectation_emotions(values["expectation"], values["message"]), case["out"]
        )
    for case in cases["occ_carry_decay"]:
        _assert_nested_approx(decay_occ_carry(case["in"]), case["out"])
    for case in cases["state_emotions"]:
        values = case["in"]
        mood = PADMood(values["valence"], values["arousal"], 0.0)
        _assert_nested_approx(state_emotions(mood, values["longing"]), case["out"])
    for case in cases["appraise_turn"]:
        values = case["in"]
        actual = appraise_turn(
            _state(values),
            values["intent"],
            occ_carry=values["occ_carry"],
            expectation=values["expectation"],
            message=values["message"],
        )
        _assert_nested_approx(
            {
                "mood": asdict(actual.state.mood),
                "active_emotions": dict(actual.active_emotions),
                "occ_carry": dict(actual.occ_carry),
                "baseline_valence": actual.state.baseline_valence,
            },
            case["out"],
        )


def test_surfacing_golden(golden: dict[str, Any]) -> None:
    cases = golden["surfacing"]
    for case in cases["clean_text"]:
        values = case["in"]
        assert clean_text(values["value"], values["max_len"]) == case["out"]
    for case in cases["presence_line"]:
        assert presence_line(**case["in"]) == case["out"]
    for case in cases["presence_vector"]:
        values = case["in"]
        cognition = CognitionState(**values["cognition"])
        actual = build_presence_vector(_state(values), cognition).to_dict()
        _assert_nested_approx(actual, case["out"])


@pytest.mark.parametrize("base_tokens", [0, -1, 2.5, True])
def test_length_control_rejects_invalid_budgets(base_tokens: object) -> None:
    with pytest.raises((TypeError, ValueError), match="base_tokens"):
        apply_length_factor(base_tokens, PADMood(arousal=-1.0))  # type: ignore[arg-type]


def test_length_control_never_increases_a_small_valid_budget() -> None:
    assert apply_length_factor(100, PADMood(arousal=-1.0)) == 100
