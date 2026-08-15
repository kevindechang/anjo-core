"""State objects must survive the round trips a real StateStore performs.

``FrozenMapping`` refuses mutation, and the default ``dict`` pickle protocol
restores items by mutating a fresh instance. Without an explicit reduction every
state object holding one is unpicklable, which breaks any store that serializes
with ``pickle`` and any use across a process boundary.
"""

from __future__ import annotations

import copy
import pickle

import pytest

from anjo_core import (
    AppraisalGoals,
    AttachmentState,
    CompanionState,
    ExpectationCues,
    PADMood,
    Personality,
    PresenceLabels,
    PromptPolicy,
    RelationshipState,
    StageLadder,
    TurnShapePolicy,
)
from anjo_core.models import FrozenMapping

ROUND_TRIP_CASES = [
    pytest.param(FrozenMapping({"joy": 0.5}), id="frozen-mapping"),
    pytest.param(CompanionState(), id="default-state"),
    pytest.param(
        CompanionState(
            mood=PADMood(0.3, -0.2, 0.1),
            personality=Personality(O=0.4, C=0.5, E=0.6, A=0.7, N=0.8),
            goals=AppraisalGoals(rapport=0.5),
            relationship=RelationshipState(stage="friend", trust=0.4, session_count=9),
            attachment=AttachmentState(weight=0.2, longing=0.4, comfort=0.6),
            baseline_valence=0.15,
            carried_thought="an unfinished thread",
            occ_carry={"joy": 0.5, "distress": 0.2},
            expectation="she said she would come back",
        ),
        id="populated-state",
    ),
    pytest.param(TurnShapePolicy(), id="turn-shape-policy"),
    pytest.param(PromptPolicy(), id="prompt-policy"),
    pytest.param(StageLadder(), id="default-ladder"),
    pytest.param(
        StageLadder(stages=("hostile", "sworn"), weights=(0.0, 0.85), strict=True),
        id="domain-ladder",
    ),
    pytest.param(ExpectationCues(), id="expectation-cues"),
    pytest.param(PresenceLabels(), id="presence-labels"),
]


@pytest.mark.parametrize("value", ROUND_TRIP_CASES)
def test_state_objects_survive_a_pickle_round_trip(value: object) -> None:
    restored = pickle.loads(pickle.dumps(value))
    assert restored == value


@pytest.mark.parametrize("value", ROUND_TRIP_CASES)
def test_state_objects_survive_copy_and_deepcopy(value: object) -> None:
    assert copy.copy(value) == value
    assert copy.deepcopy(value) == value


def test_a_restored_mapping_is_still_immutable() -> None:
    restored = pickle.loads(pickle.dumps(FrozenMapping({"joy": 0.5})))
    assert isinstance(restored, FrozenMapping)
    with pytest.raises(TypeError):
        restored["joy"] = 0.9  # type: ignore[index]
    with pytest.raises(TypeError):
        restored.update({"joy": 0.9})  # type: ignore[arg-type]


def test_a_restored_ladder_still_resolves_and_enforces_strictness() -> None:
    ladder = StageLadder(stages=("hostile", "sworn"), weights=(0.0, 0.85), strict=True)
    restored = pickle.loads(pickle.dumps(ladder))
    assert restored.ordinal("sworn") == 2
    assert restored.weight("sworn") == 0.85
    with pytest.raises(KeyError):
        restored.ordinal("swron")
