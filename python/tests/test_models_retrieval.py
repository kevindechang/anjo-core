from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pytest

from anjo_core.affect import TurnShapePolicy
from anjo_core.models import CompanionState, MemoryCandidate, RelationshipState
from anjo_core.retrieval import (
    candidate_score,
    recency_weight,
    similarity_from_distance,
)


def test_companion_state_mappings_are_validated_and_immutable() -> None:
    source = {"joy": 0.4}
    state = CompanionState(occ_carry=source)
    source["joy"] = 0.9

    assert state.occ_carry["joy"] == 0.4
    assert json.loads(json.dumps(asdict(state)))["occ_carry"] == {"joy": 0.4}
    with pytest.raises(TypeError):
        state.occ_carry["joy"] = 0.8  # type: ignore[index]
    with pytest.raises(TypeError):
        state.occ_carry.clear()  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="occ_carry"):
        CompanionState(occ_carry={"joy": float("nan")})


@pytest.mark.parametrize("stage", [3, True, ["friend"]])
def test_relationship_stage_must_be_a_string(stage: object) -> None:
    with pytest.raises(TypeError, match="stage"):
        RelationshipState(stage=stage)  # type: ignore[arg-type]


@pytest.mark.parametrize("stage", ["", "x" * 65])
def test_relationship_stage_has_a_small_non_empty_boundary(stage: str) -> None:
    with pytest.raises(ValueError, match="stage"):
        RelationshipState(stage=stage)


@pytest.mark.parametrize("session_count", [True, 1.5, "1"])
def test_relationship_session_count_must_be_an_integer(session_count: object) -> None:
    with pytest.raises(TypeError, match="session_count"):
        RelationshipState(session_count=session_count)  # type: ignore[arg-type]


def test_companion_expectation_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="expectation"):
        CompanionState(expectation=3)  # type: ignore[arg-type]


def test_turn_shape_policy_mapping_is_validated_and_immutable() -> None:
    cues = {"relaxed": "easy"}
    policy = TurnShapePolicy(octant_cues=cues)
    cues["relaxed"] = "changed"

    assert policy.octant_cues["relaxed"] == "easy"
    with pytest.raises(TypeError):
        policy.octant_cues["relaxed"] = "mutate"  # type: ignore[index]
    with pytest.raises(ValueError, match="octant cue"):
        TurnShapePolicy(octant_cues={"relaxed": ""})


@pytest.mark.parametrize("distance", [-0.01, 2.01, float("nan"), float("inf"), True])
def test_cosine_distance_is_consistently_restricted(distance: object) -> None:
    with pytest.raises((TypeError, ValueError), match="distance"):
        MemoryCandidate(id="m", text="safe", distance=distance)
    with pytest.raises((TypeError, ValueError), match="distance"):
        similarity_from_distance(distance)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError), match="distance"):
        candidate_score(
            distance,  # type: ignore[arg-type]
            0,
            episode=False,
            significance=0.5,
            recall_count=0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": 3},
        {"significance": True},
        {"recall_count": True},
        {"recall_count": 1.5},
        {"episode": 1},
    ],
)
def test_memory_candidate_rejects_malformed_adapter_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {"id": "m", "text": "safe", "distance": 0.5}
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        MemoryCandidate(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("significance", [0.0, 1.0])
def test_memory_candidate_accepts_significance_boundaries(significance: float) -> None:
    assert (
        MemoryCandidate(id="m", text="safe", distance=0.5, significance=significance).significance
        == significance
    )


@pytest.mark.parametrize("significance", [-0.000_001, 1.000_001])
def test_memory_candidate_rejects_significance_outside_unit_interval(
    significance: float,
) -> None:
    with pytest.raises(ValueError, match=r"significance.*\[0\.0, 1\.0\]"):
        MemoryCandidate(id="m", text="safe", distance=0.5, significance=significance)


def test_standalone_candidate_score_keeps_significance_clamping() -> None:
    common = {
        "distance": 0.5,
        "days_ago": 0,
        "episode": False,
        "recall_count": 0,
    }
    assert candidate_score(**common, significance=-2.0) == candidate_score(
        **common, significance=0.0
    )
    assert candidate_score(**common, significance=2.0) == candidate_score(
        **common, significance=1.0
    )


def test_timestamps_require_timezone_and_future_values_are_bounded() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    naive = "2026-08-13T00:00:00"
    future = (now + timedelta(days=30)).isoformat()

    with pytest.raises(ValueError, match="timezone-aware"):
        MemoryCandidate(id="m", text="safe", distance=0.5, timestamp=naive)
    assert recency_weight(naive, now=now) == 0.7
    assert recency_weight(future, now=now) == 1.0
    with pytest.raises(ValueError, match="timezone-aware"):
        recency_weight(future, now=now.replace(tzinfo=None))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", "x" * 4_097),
        ("carried_thought", "x" * 4_097),
    ],
)
def test_untrusted_source_fields_are_bounded(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="4,096"):
        if field == "text":
            MemoryCandidate(id="m", text=value, distance=0.5)
        else:
            CompanionState(carried_thought=value)
