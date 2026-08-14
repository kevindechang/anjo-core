"""The seams that let a non-companion domain reuse the kernel.

Each test here pins a promise the README makes about generalization: that a
domain can supply its own progression, its own expectation vocabulary, its own
response-shaping rules, and its own presence wording without forking the kernel.
"""

from __future__ import annotations

import pytest

from anjo_core import (
    DEFAULT_STAGE_LADDER,
    CompanionState,
    ExpectationCues,
    PADMood,
    Personality,
    PresenceLabels,
    RelationshipState,
    StageLadder,
    TurnShapePolicy,
    UnknownStageError,
    appraise_turn,
    baseline_weight,
    build_presence_vector,
    conversational_appraisal_policy,
    decay_mood,
    expectation_emotions,
    presence_line,
    stage_int,
    turn_shape_directive,
)
from anjo_core.appraisal import AppraisalPolicyInput

FACTION_LADDER = StageLadder(
    stages=("hostile", "wary", "neutral", "friendly", "sworn"),
    weights=(0.0, 0.15, 0.35, 0.60, 0.85),
)


class TestStageLadder:
    def test_custom_ladder_replaces_the_conversational_rungs(self) -> None:
        assert FACTION_LADDER.ordinal("neutral") == 3
        assert FACTION_LADDER.weight("sworn") == 0.85
        # The reference labels are meaningless on a domain ladder.
        assert FACTION_LADDER.knows("intimate") is False

    def test_default_ladder_floors_unknown_stages_as_pinned(self) -> None:
        # This is the cross-runtime contract in shared/golden: unknown -> 1.
        assert stage_int("unknown") == 1
        assert baseline_weight(stage_int("unknown")) == 0.0
        assert DEFAULT_STAGE_LADDER.strict is False

    def test_strict_ladder_surfaces_a_typo_instead_of_silently_flooring(self) -> None:
        strict = StageLadder(
            stages=FACTION_LADDER.stages, weights=FACTION_LADDER.weights, strict=True
        )
        assert strict.ordinal("friendly") == 4
        with pytest.raises(UnknownStageError):
            strict.ordinal("freindly")

    def test_out_of_range_ordinals_have_no_resting_weight(self) -> None:
        assert FACTION_LADDER.weight_for_ordinal(0) == 0.0
        assert FACTION_LADDER.weight_for_ordinal(99) == 0.0
        with pytest.raises(TypeError):
            FACTION_LADDER.weight_for_ordinal(True)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"stages": (), "weights": ()},
            {"stages": ("a", "b"), "weights": (0.1,)},
            {"stages": ("a", "a"), "weights": (0.1, 0.2)},
            {"stages": ("a", ""), "weights": (0.1, 0.2)},
            {"stages": ("a",), "weights": (1.5,)},
            {"stages": ("a",), "weights": (float("nan"),)},
        ],
    )
    def test_malformed_ladders_are_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises((TypeError, ValueError)):
            StageLadder(**kwargs)  # type: ignore[arg-type]

    def test_ladder_changes_the_resting_point_of_mood_decay(self) -> None:
        mood = PADMood(valence=0.5, arousal=0.0, dominance=0.0)
        personality = Personality()
        # "friendly" is rung 4 on the faction ladder (weight 0.60) but is not on
        # the conversational ladder at all, where it floors to weight 0.0.
        with_faction = decay_mood(mood, personality, "friendly", 0.4, ladder=FACTION_LADDER)
        with_default = decay_mood(mood, personality, "friendly", 0.4)
        assert with_faction.valence != with_default.valence

    def test_appraise_turn_accepts_a_domain_ladder(self) -> None:
        state = CompanionState(
            mood=PADMood(0.3, 0.1, 0.0),
            relationship=RelationshipState(stage="friendly"),
            baseline_valence=0.5,
        )
        domain = appraise_turn(state, "CASUAL", ladder=FACTION_LADDER)
        reference = appraise_turn(state, "CASUAL")
        assert domain.state.mood.valence != reference.state.mood.valence


class TestExpectationCues:
    def test_default_cues_are_the_english_preset(self) -> None:
        assert expectation_emotions("she was angry", "i feel better now") == {
            "relief": 0.45,
            "surprise": 0.35,
        }

    def test_a_domain_can_replace_the_vocabulary_entirely(self) -> None:
        # No English sentiment words; a build-status vocabulary instead.
        cues = ExpectationCues(
            negative_expected=("failing",),
            positive_current=("green",),
            negative_current=("red",),
            expected_resolution=("deploy",),
        )
        assert expectation_emotions("the suite was failing", "build is green", cues=cues) == {
            "relief": 0.45,
            "surprise": 0.35,
        }
        # The English preset finds nothing in the same strings.
        assert expectation_emotions("the suite was failing", "build is green") == {}

    def test_cue_tokens_must_be_non_empty_strings(self) -> None:
        with pytest.raises(ValueError):
            ExpectationCues(negative_expected=("  ",))
        with pytest.raises(TypeError):
            ExpectationCues(positive_current=(3,))  # type: ignore[arg-type]


class TestTurnShapeSuppression:
    def test_reference_policy_still_suppresses_upbeat_after_vulnerability(self) -> None:
        upbeat = PADMood(valence=0.6, arousal=0.5, dominance=0.4)
        directive = turn_shape_directive(upbeat, intent="VULNERABILITY")
        assert "momentum" not in directive
        assert "momentum" in turn_shape_directive(upbeat, intent="CASUAL")

    def test_the_rule_is_data_a_domain_can_drop(self) -> None:
        upbeat = PADMood(valence=0.6, arousal=0.5, dominance=0.4)
        neutral = TurnShapePolicy(suppressed_octant_cues={})
        assert "momentum" in turn_shape_directive(upbeat, intent="VULNERABILITY", policy=neutral)

    def test_a_domain_can_suppress_its_own_intents(self) -> None:
        upbeat = PADMood(valence=0.6, arousal=0.5, dominance=0.4)
        policy = TurnShapePolicy(suppressed_octant_cues={"player_died": ("exuberant",)})
        # Intent matching is case-insensitive on the policy side.
        assert "momentum" not in turn_shape_directive(upbeat, intent="PLAYER_DIED", policy=policy)


class TestPresenceLabels:
    def test_default_labels_are_the_conversational_phrasing(self) -> None:
        assert (
            presence_line(
                reflection_pending=False,
                due_intention=False,
                carried_thought=False,
                open_thread=False,
            )
            == "here with you"
        )

    def test_a_domain_renders_its_own_presence_wording(self) -> None:
        labels = PresenceLabels(idle="on watch", idle_mode="posted")
        vector = build_presence_vector(CompanionState(), labels=labels)
        assert vector.line == "on watch"
        assert vector.mode == "posted"

    def test_labels_must_not_be_blank(self) -> None:
        with pytest.raises(ValueError):
            PresenceLabels(idle="   ")


def test_conversational_policy_factory_binds_a_domain_ladder() -> None:
    policy = conversational_appraisal_policy(ladder=FACTION_LADDER)
    state = CompanionState(
        mood=PADMood(0.3, 0.1, 0.0),
        relationship=RelationshipState(stage="friendly"),
        baseline_valence=0.5,
    )
    request = AppraisalPolicyInput(state=state, intent="CASUAL", message="", expectation="")
    bound = policy(request)
    assert (
        bound.state.mood.valence
        == appraise_turn(state, "CASUAL", ladder=FACTION_LADDER).state.mood.valence
    )
