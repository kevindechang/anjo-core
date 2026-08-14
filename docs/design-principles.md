# Design principles

## Make hidden state explicit

If a character's mood, relationship state, or remembered evidence affects a
reply, represent it as data that can be inspected and tested. Do not rely on the
model to recreate continuity from prose alone.

## Keep model judgment behind an interface

Classification and generation can be model-driven. Reflection remains
application-owned in the current release. The kernel does not pretend these
behaviors are deterministic; applications must evaluate their quality separately.

## Bound every state transition

Appraisal and reflection should move state in small, clamped increments. A single
turn should not rewrite a long-lived character.

## Treat memory as evidence

Retrieved text is candidate evidence, not truth. A model should receive certainty
and provenance where available, and applications should test false-memory
behavior explicitly.

## Surface state intentionally

Internal state is useful only when it produces a justified external moment. Keep
state-to-surface transforms small and auditable.

## Prefer behavioral contracts over shared prose

Two runtimes can share expected inputs and outputs without sharing a provider,
database, or branded prompt. The shared corpus is a versioned compatibility
contract, not a claim that model replies are identical.
