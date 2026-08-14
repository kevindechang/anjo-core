# Game NPC example

The kernel with **no conversation in it**. A gate guard's disposition moves in
response to world events -- a quest completed, a promise broken, a blade drawn,
a wound healed -- and the same affect dynamics that drive a companion drive him.

```bash
python examples/game-npc/main.py
```

```text
world event         valence  arousal  dominance   bark
--------------------------------------------------------------------------------------
QUEST_COMPLETED      0.3500   0.2000     0.1877   The road's safer for it. I'll remember that.
PROMISE_BROKEN      -0.0004   0.2710     0.0627   You said you'd come back before dark.
PLAYER_ATTACKED     -0.4500   0.5317     0.2558   Draw on me again and you'll not walk out.
PLAYER_HEALED       -0.0849   0.4046     0.1709   ...That was more than I'd have done. Go on, then.

disposition: wary | presence: on watch (posted)
invariants held: disposition tracked the world events
```

The bark lines come from a scripted adapter -- they are fixed strings, not model
output. The numbers are the kernel.

## What this demonstrates

Nothing here reuses the reference conversational vocabulary:

| Seam | Reference preset | This example |
|---|---|---|
| `StageLadder` | stranger → intimate | hostile → wary → neutral → friendly → sworn, `strict=True` |
| `AppraisalPolicy` | English intent appraisal | a table of world events → PAD deltas |
| `ExpectationCues` | English sentiment words | unused; no text is inspected at all |
| `TurnShapePolicy` | companion cadence rules | guard barks, no cue suppression |
| `PresenceLabels` | "here with you" | "on watch" |

The appraisal policy never reads message text, so it is language-independent by
construction. Because `strict=True`, a misspelled faction standing raises
`UnknownStageError` instead of silently resolving to the bottom rung.

The example asserts its own invariants and runs in CI, so the generalization
claim is checked rather than asserted.

## Using it as a starting point

Copy `main.py` and replace three things: the ladder rungs, the `WORLD_EVENTS`
table, and the wording objects. The kernel imports stay the same.
