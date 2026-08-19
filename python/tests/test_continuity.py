from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from affect_kernel import (
    AppraisalPolicyInput,
    CompanionState,
    PADMood,
    Personality,
    appraise_turn,
    default_appraisal_policy,
)

_REPOSITORY_TRACES = (
    Path(__file__).resolve().parents[2] / "shared" / "golden" / "continuity_traces.json"
)
_SDIST_TRACES = Path(__file__).resolve().parent / "fixtures" / "continuity_traces.json"
CONTINUITY_PATH = _REPOSITORY_TRACES if _REPOSITORY_TRACES.is_file() else _SDIST_TRACES


def _assert_float_mapping(
    actual: Mapping[str, float],
    expected: Mapping[str, float],
) -> None:
    assert set(actual) == set(expected)
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, abs=1e-9), name


def _assert_mood(actual: PADMood, expected: Mapping[str, float], label: str) -> None:
    assert actual.valence == pytest.approx(expected["valence"], abs=1e-9), label
    assert actual.arousal == pytest.approx(expected["arousal"], abs=1e-9), label
    assert actual.dominance == pytest.approx(expected["dominance"], abs=1e-9), label


def test_default_appraisal_policy_matches_every_continuity_trace_step() -> None:
    assert CONTINUITY_PATH.is_file(), f"continuity fixture is missing: {CONTINUITY_PATH}"
    fixture: dict[str, Any] = json.loads(CONTINUITY_PATH.read_text(encoding="utf-8"))
    assert fixture["_meta"]["traces"] == 3

    for trace in fixture["traces"]:
        initial = trace["initial"]
        state = CompanionState(
            mood=PADMood(**initial["mood"]),
            personality=Personality(**initial["personality"]),
            baseline_valence=initial["baseline_valence"],
        )
        for index, step in enumerate(trace["steps"]):
            event = step["event"]
            request = AppraisalPolicyInput(
                state=state,
                intent=event["intent"],
                message=event["message"],
                expectation=state.expectation,
            )
            result = default_appraisal_policy(request)
            assert result == appraise_turn(
                state,
                event["intent"],
                message=event["message"],
                expectation=state.expectation,
            )

            expected = step["expected"]
            label = f"{trace['name']} step {index + 1}"
            _assert_mood(result.state.mood, expected["mood"], label)
            assert result.state.baseline_valence == pytest.approx(
                expected["baseline_valence"], abs=1e-9
            ), label
            _assert_float_mapping(result.active_emotions, expected["active_emotions"])
            _assert_float_mapping(result.occ_carry, expected["occ_carry"])
            state = result.state


def test_sdist_declares_continuity_trace_fallback() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        '"../shared/golden/continuity_traces.json" = "tests/fixtures/continuity_traces.json"'
    ) in pyproject
