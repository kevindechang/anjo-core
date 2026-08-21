from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.build_kernel_fixture import validate_public_fixture
from scripts.validate_continuity_fixture import validate_continuity_fixture
from scripts.verify_public_boundary import verify

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "shared" / "golden" / "kernel_golden.json"
CONTINUITY_PATH = ROOT / "shared" / "golden" / "continuity_traces.json"


@pytest.fixture
def fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def continuity_fixture() -> dict[str, Any]:
    return json.loads(CONTINUITY_PATH.read_text(encoding="utf-8"))


def _write_minimal_candidate(
    root: Path,
    fixture: dict[str, Any],
    continuity_fixture: dict[str, Any],
) -> None:
    destination = root / "shared" / "golden" / "kernel_golden.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(fixture), encoding="utf-8")
    (destination.parent / "continuity_traces.json").write_text(
        json.dumps(continuity_fixture), encoding="utf-8"
    )


def test_checked_fixture_satisfies_strict_public_schema(
    fixture: dict[str, Any],
) -> None:
    validate_public_fixture(fixture)


def test_checked_continuity_fixture_satisfies_strict_public_schema(
    continuity_fixture: dict[str, Any],
) -> None:
    validate_continuity_fixture(continuity_fixture)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["traces"][0]["steps"][0]["event"].update(
            {"message": "paste a private instruction here"}
        ),
        lambda value: value["traces"][0]["steps"][0]["expected"].update(
            {"private_state": 0.5}
        ),
        lambda value: value["traces"][0]["initial"]["mood"].update(
            {"valence": float("nan")}
        ),
        lambda value: value["_meta"].update({"traces": 4}),
    ],
)
def test_continuity_fixture_rejects_prose_fields_numbers_and_metadata(
    continuity_fixture: dict[str, Any], mutation: Any
) -> None:
    mutation(continuity_fixture)
    with pytest.raises((TypeError, ValueError)):
        validate_continuity_fixture(continuity_fixture)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["appraisal"]["appraise_turn"][0].update(
            {"private_prompt_text": "hidden"}
        ),
        lambda value: value["appraisal"]["appraise_turn"][0]["in"].update(
            {"message": "copy an internal instruction into this allowed field"}
        ),
        lambda value: value["retrieval"]["candidate_score"][0]["in"].update(
            {"distance": "not-a-number"}
        ),
        lambda value: value["affect_control"]["mood_octant"].append(
            deepcopy(value["affect_control"]["mood_octant"][0])
        ),
        lambda value: value["_meta"].update({"cases": 226}),
    ],
)
def test_fixture_rejects_unknown_fields_prose_types_counts_and_metadata(
    fixture: dict[str, Any], mutation: Any
) -> None:
    mutation(fixture)
    with pytest.raises((TypeError, ValueError)):
        validate_public_fixture(fixture)


def test_boundary_rejects_binary_payload_in_an_allowed_directory(
    tmp_path: Path,
    fixture: dict[str, Any],
    continuity_fixture: dict[str, Any],
) -> None:
    _write_minimal_candidate(tmp_path, fixture, continuity_fixture)
    payload = tmp_path / "docs" / "payload.md"
    payload.parent.mkdir()
    payload.write_bytes(b"text\0hidden")

    errors = verify(tmp_path)

    assert any("binary file is not allowed" in error for error in errors)


def test_boundary_allows_reviewed_support_policy(
    tmp_path: Path,
    fixture: dict[str, Any],
    continuity_fixture: dict[str, Any],
) -> None:
    _write_minimal_candidate(tmp_path, fixture, continuity_fixture)
    (tmp_path / "SUPPORT.md").write_text("# Support\n", encoding="utf-8")

    errors = verify(tmp_path)

    assert not any("unexpected top-level entries" in error for error in errors)


def test_boundary_scans_force_tracked_ignored_paths(
    tmp_path: Path,
    fixture: dict[str, Any],
    continuity_fixture: dict[str, Any],
) -> None:
    _write_minimal_candidate(tmp_path, fixture, continuity_fixture)
    payload = tmp_path / "typescript" / "node_modules" / "forced.js"
    payload.parent.mkdir(parents=True)
    payload.write_text("export const forced = true;\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "add", "shared/golden/kernel_golden.json"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "add", "-f", "typescript/node_modules/forced.js"],
        cwd=tmp_path,
        check=True,
    )

    errors = verify(tmp_path)

    assert any(
        "generated/ignored path must not be tracked" in error for error in errors
    )
