"""A non-conversational domain driving the same kernel: a game NPC's disposition.

Nothing in this example is a chat turn. The events are world events -- the player
attacked, healed, kept or broke a promise -- and the "stages" are faction standing,
not relationship intimacy. The kernel still supplies the affect dynamics, the
response shaping, and the presence surface.

Everything here uses the published API only. There is no private import, no model
call, and no network access. Run it with:

    python examples/game-npc/main.py
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from affect_kernel import (
    AppraisalPolicyInput,
    AppraisalResult,
    CompanionEngine,
    CompanionState,
    GateResult,
    PADMood,
    Personality,
    PresenceLabels,
    RelationshipState,
    StageLadder,
    TurnShapePolicy,
    decay_mood,
    decay_occ_carry,
)
from affect_kernel.adapters import InMemoryStateStore, ScriptedModelAdapter

# A faction-standing ladder. These rungs have nothing to do with the reference
# conversational ladder, and strict mode means a typo'd standing raises instead
# of silently resolving to the bottom rung.
FACTION_LADDER = StageLadder(
    stages=("hostile", "wary", "neutral", "friendly", "sworn"),
    weights=(0.0, 0.15, 0.35, 0.60, 0.85),
    strict=True,
)

# World events, not conversational intents. Each maps to a bounded PAD delta and
# the emotions the NPC carries forward.
WORLD_EVENTS: dict[str, tuple[tuple[float, float, float], dict[str, float]]] = {
    "PLAYER_ATTACKED": ((-0.45, 0.30, 0.20), {"reproach": 0.85, "distress": 0.55}),
    "PLAYER_HEALED": ((0.30, -0.05, -0.05), {"gratitude": 0.70, "joy": 0.35}),
    "QUEST_COMPLETED": ((0.35, 0.20, 0.10), {"admiration": 0.75, "joy": 0.50}),
    "PROMISE_BROKEN": (
        (-0.30, 0.10, -0.10),
        {"disappointment": 0.65, "distress": 0.30},
    ),
    "GIFT_GIVEN": ((0.20, 0.05, 0.00), {"gratitude": 0.45, "joy": 0.30}),
}

# The NPC speaks in barks, not paragraphs. Every word of this is ours, not the
# kernel's -- and nothing suppresses cues, because "vulnerability" is not a
# concept in this domain.
BARK_POLICY = TurnShapePolicy(
    heading="Bark shape:",
    statement_close="End on a statement; guards do not ask questions.",
    no_question_close="End on a statement; guards do not ask questions.",
    one_idea="One clause only.",
    mirror_length="Never exceed a single line.",
    ambivalence="Let the conflict show rather than resolving it.",
    octant_cues={
        "hostile": "Clipped and cold.",
        "anxious": "Guarded, watching the exits.",
        "exuberant": "Openly pleased.",
        "relaxed": "Easy, unhurried.",
        "bored": "Barely engaged.",
        "docile": "Deferential and quiet.",
        "dependent": "Warm, seeking approval.",
        "disdainful": "Dismissive.",
    },
    suppressed_octant_cues={},
)

# The presence surface is rendered wording too. A guard does not say
# "here with you" -- that is the reference companion phrasing, not kernel truth.
GUARD_PRESENCE = PresenceLabels(
    reflecting="sizing you up",
    due_intention="waiting on a debt",
    carried_thought="chewing on something",
    open_thread="keeping a tally",
    idle="on watch",
    reflecting_mode="alert",
    idle_mode="posted",
)


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def faction_appraisal_policy(request: AppraisalPolicyInput) -> AppraisalResult:
    """Translate a world event into a deterministic disposition change.

    This is a complete replacement for the reference conversational appraisal.
    It never inspects message text, so it is language-independent by construction.
    """
    state = request.state
    decayed = decay_mood(
        state.mood,
        state.personality,
        state.relationship.stage,
        state.baseline_valence,
        ladder=FACTION_LADDER,
    )
    (d_valence, d_arousal, d_dominance), fresh = WORLD_EVENTS.get(
        request.intent, ((0.0, 0.0, 0.0), {})
    )
    mood = PADMood(
        valence=round(_clamp(decayed.valence + d_valence), 4),
        arousal=round(_clamp(decayed.arousal + d_arousal), 4),
        dominance=round(_clamp(decayed.dominance + d_dominance), 4),
    )
    carried = decay_occ_carry(state.occ_carry)
    merged = {
        name: max(fresh.get(name, 0.0), carried.get(name, 0.0))
        for name in sorted(set(fresh) | set(carried))
    }
    next_carry = {name: value for name, value in merged.items() if value > 0.05}
    next_state = replace(
        state,
        mood=mood,
        baseline_valence=round(
            _clamp(state.baseline_valence * 0.95 + mood.valence * 0.05), 4
        ),
        occ_carry=next_carry,
    )
    return AppraisalResult(
        state=next_state, active_emotions=merged, occ_carry=next_carry
    )


def _npc_state() -> CompanionState:
    return CompanionState(
        mood=PADMood(0.0, 0.0, 0.1),
        # A gruff, disagreeable guard: low Agreeableness, high Neuroticism means
        # mood swings harder and settles slower.
        personality=Personality(O=0.3, C=0.8, E=0.35, A=0.25, N=0.7),
        relationship=RelationshipState(stage="wary", trust=0.2, session_count=4),
    )


async def main() -> None:
    npc_id = "gate-guard-varic"
    store = InMemoryStateStore(states={npc_id: _npc_state()})
    events = (
        "QUEST_COMPLETED",
        "PROMISE_BROKEN",
        "PLAYER_ATTACKED",
        "PLAYER_HEALED",
    )
    model = ScriptedModelAdapter(
        gates=[
            GateResult(intent=event, should_respond=True, should_retrieve=False)
            for event in events
        ],
        responses=[
            ("The road's safer for it. I'll remember that.",),
            ("You said you'd come back before dark.",),
            ("Draw on me again and you'll not walk out.",),
            ("...That was more than I'd have done. Go on, then.",),
        ],
    )
    engine = CompanionEngine(
        model=model,
        store=store,
        conversation_id=npc_id,
        base_prompt="You are a gate guard in a walled town. Speak in single lines.",
        appraisal_policy=faction_appraisal_policy,
        turn_shape_policy=BARK_POLICY,
        presence_labels=GUARD_PRESENCE,
        custom_intents=tuple(WORLD_EVENTS),
        state_factory=_npc_state,
    )

    print(f"{'world event':<18} {'valence':>8} {'arousal':>8} {'dominance':>10}   bark")
    print("-" * 86)
    observed: list[float] = []
    for event in events:
        result = await engine.turn(event)
        mood = result.mood
        observed.append(mood.valence)
        print(
            f"{event:<18} {mood.valence:>8.4f} {mood.arousal:>8.4f} "
            f"{mood.dominance:>10.4f}   {result.text}"
        )

    presence = await engine.presence()
    print(
        f"\ndisposition: {presence.relationship.stage} | "
        f"presence: {presence.line} ({presence.mode})"
    )

    # The example doubles as a test: betrayal and violence must move the guard
    # negative, and care must recover him, with no conversational text anywhere.
    assert observed[0] > 0, "a completed quest should raise valence"
    assert observed[2] < observed[1] < observed[0], "betrayal then violence should fall"
    assert observed[3] > observed[2], "being healed should begin recovery"
    print("invariants held: disposition tracked the world events")


if __name__ == "__main__":
    asyncio.run(main())
