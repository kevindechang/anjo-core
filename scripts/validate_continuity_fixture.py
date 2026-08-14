#!/usr/bin/env python3
"""Strictly validate the public longitudinal continuity traces."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "shared/golden/continuity_traces.json"
EXPECTED_META = {
    "purpose": "Synthetic longitudinal affect-continuity traces",
    "traces": 3,
    "float_round": 9,
}
EXPECTED_TRACE_EVENTS = {
    "engagement_builds_activation": (
        ("CURIOSITY", "tell me about the stars"),
        ("CURIOSITY", "tell me more"),
        ("CASUAL", "okay"),
    ),
    "conflict_then_repair": (
        ("ABUSE", "you're useless"),
        ("APOLOGY", "sorry about that"),
        ("CASUAL", "hey"),
    ),
    "vulnerability_then_decay": (
        ("VULNERABILITY", "things are hard"),
        ("CASUAL", "okay"),
        ("CASUAL", "okay"),
    ),
}
PAD_FIELDS = frozenset({"valence", "arousal", "dominance"})
PERSONALITY_FIELDS = frozenset({"O", "C", "E", "A", "N"})
EMOTION_FIELDS = frozenset(
    {
        "admiration",
        "anger",
        "disappointment",
        "distress",
        "fatigue",
        "fear",
        "gratitude",
        "hope",
        "joy",
        "longing",
        "relief",
        "reproach",
        "sadness",
        "satisfaction",
        "surprise",
        "unease",
    }
)


def _object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    if frozenset(value) != fields:
        raise ValueError(f"{label} has unexpected fields")
    return value


def _number(value: Any, low: float, high: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{label} must be finite and within [{low}, {high}]")
    return number


def _pad(value: Any, label: str) -> None:
    mood = _object(value, PAD_FIELDS, label)
    for field in PAD_FIELDS:
        _number(mood[field], -1.0, 1.0, f"{label}.{field}")


def _personality(value: Any, label: str) -> None:
    personality = _object(value, PERSONALITY_FIELDS, label)
    for field in PERSONALITY_FIELDS:
        _number(personality[field], 0.0, 1.0, f"{label}.{field}")


def _emotions(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not set(value) <= EMOTION_FIELDS:
        raise ValueError(f"{label} contains unexpected emotion fields")
    for name, intensity in value.items():
        _number(intensity, 0.0, 1.0, f"{label}.{name}")


def validate_continuity_fixture(payload: Any) -> None:
    document = _object(payload, frozenset({"_meta", "traces"}), "continuity fixture")
    if document["_meta"] != EXPECTED_META:
        raise ValueError(
            "continuity fixture metadata does not match the release contract"
        )
    traces = document["traces"]
    if not isinstance(traces, list) or len(traces) != len(EXPECTED_TRACE_EVENTS):
        raise ValueError("continuity fixture must contain exactly three traces")

    seen: set[str] = set()
    for trace_index, raw_trace in enumerate(traces):
        label = f"continuity trace {trace_index}"
        trace = _object(raw_trace, frozenset({"name", "initial", "steps"}), label)
        name = trace["name"]
        if (
            not isinstance(name, str)
            or name not in EXPECTED_TRACE_EVENTS
            or name in seen
        ):
            raise ValueError(f"{label} has an unexpected or duplicate name")
        seen.add(name)

        initial = _object(
            trace["initial"],
            frozenset({"mood", "personality", "baseline_valence"}),
            f"{label}.initial",
        )
        _pad(initial["mood"], f"{label}.initial.mood")
        _personality(initial["personality"], f"{label}.initial.personality")
        _number(
            initial["baseline_valence"],
            -1.0,
            1.0,
            f"{label}.initial.baseline_valence",
        )

        steps = trace["steps"]
        expected_events = EXPECTED_TRACE_EVENTS[name]
        if not isinstance(steps, list) or len(steps) != len(expected_events):
            raise ValueError(f"{label} has an unexpected step count")
        for step_index, raw_step in enumerate(steps):
            expected_event = expected_events[step_index]
            step_label = f"{label}.steps[{step_index}]"
            step = _object(raw_step, frozenset({"event", "expected"}), step_label)
            event = _object(
                step["event"], frozenset({"intent", "message"}), f"{step_label}.event"
            )
            if (event["intent"], event["message"]) != expected_event:
                raise ValueError(f"{step_label} contains non-public event prose")

            expected = _object(
                step["expected"],
                frozenset({"mood", "baseline_valence", "active_emotions", "occ_carry"}),
                f"{step_label}.expected",
            )
            _pad(expected["mood"], f"{step_label}.expected.mood")
            _number(
                expected["baseline_valence"],
                -1.0,
                1.0,
                f"{step_label}.expected.baseline_valence",
            )
            _emotions(
                expected["active_emotions"], f"{step_label}.expected.active_emotions"
            )
            _emotions(expected["occ_carry"], f"{step_label}.expected.occ_carry")

    if seen != set(EXPECTED_TRACE_EVENTS):
        raise ValueError("continuity fixture is missing a required trace")


def main() -> int:
    path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else TRACE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_continuity_fixture(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"CONTINUITY FIXTURE: FAIL: {exc}")
        return 1
    print("CONTINUITY FIXTURE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
