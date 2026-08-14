from __future__ import annotations

from dataclasses import replace

import pytest

from anjo_core import AppraisalResult, CompanionState


def test_appraisal_result_emotion_mappings_are_defensive_and_immutable() -> None:
    active = {"joy": 0.6}
    carry = {"joy": 0.5}
    result = AppraisalResult(
        state=CompanionState(occ_carry=carry),
        active_emotions=active,
        occ_carry=carry,
    )
    active["joy"] = 0.1
    carry["joy"] = 0.1

    assert result.active_emotions == {"joy": 0.6}
    assert result.occ_carry == {"joy": 0.5}
    with pytest.raises(TypeError):
        result.active_emotions["joy"] = 0.2  # type: ignore[index]
    with pytest.raises(TypeError):
        result.occ_carry["joy"] = 0.2  # type: ignore[index]


@pytest.mark.parametrize("field", ["active_emotions", "occ_carry"])
@pytest.mark.parametrize(
    "mapping",
    [
        {"": 0.5},
        {"   ": 0.5},
        {3: 0.5},
        {"joy": True},
        {"joy": "high"},
        {"joy": float("nan")},
        {"joy": float("inf")},
        {"joy": -0.000_001},
        {"joy": 1.000_001},
    ],
)
def test_appraisal_result_rejects_invalid_emotion_mappings(
    field: str,
    mapping: dict[object, object],
) -> None:
    kwargs: dict[str, object] = {
        "state": CompanionState(),
        "active_emotions": {},
        "occ_carry": {},
    }
    kwargs[field] = mapping
    with pytest.raises((TypeError, ValueError)):
        AppraisalResult(**kwargs)  # type: ignore[arg-type]


def test_appraisal_result_occ_carry_must_match_next_state() -> None:
    state = CompanionState(occ_carry={"joy": 0.5})
    with pytest.raises(ValueError, match=r"agree with state\.occ_carry"):
        AppraisalResult(
            state=state,
            active_emotions={"joy": 0.5},
            occ_carry={"joy": 0.4},
        )

    matching = AppraisalResult(
        state=replace(state, occ_carry={"joy": 0.4}),
        active_emotions={"joy": 0.5},
        occ_carry={"joy": 0.4},
    )
    assert matching.occ_carry == matching.state.occ_carry
