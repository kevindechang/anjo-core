# Architecture

Anjo Core is a deterministic state-to-surface kernel inside an injected
orchestration shell. The kernel is the product boundary; the engine is a
credential-free reference integration.

```text
                         application-owned
                  ┌──────────────────────────┐
event ───────────▶│ interpreter / model gate │
                  └────────────┬─────────────┘
                               │ normalized event + decisions
                  ┌────────────▼─────────────┐
                  │ appraisal policy         │
                  └────────────┬─────────────┘
                               │ affect transition
                  ┌────────────▼─────────────┐
                  │ deterministic kernel     │
                  │                          │
                  │ retrieval scoring        │
                  │ OCC/PAD appraisal        │
                  │ affect controls          │
                  │ presence surfacing       │
                  │ prompt composition       │
                  └────────────┬─────────────┘
                               │ grounded context + controls
                  ┌────────────▼─────────────┐
                  │ model generation adapter │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │ application store        │
                  └──────────────────────────┘
```

## State

The reference state includes OCEAN personality values, PAD mood, appraisal goals,
relationship metadata, attachment signals, and a small amount of session carry.
The current kernel evolves PAD mood, baseline valence, and OCC carry. Relationship,
attachment, and personality values are persisted and surfaced but are not
automatically evolved. Applications can wrap or extend the state, but kernel
inputs remain serializable values.

## Appraisal policy

The default policy maps a small set of conversational intent labels and English
expectation cues into the reference affect transition. It is a preset, not a
claim that those labels are domain-neutral. Both engines accept an injected,
synchronous appraisal policy so an application can translate game, tutoring,
coaching, or other events into its own deterministic transition.

## Adapters

The engine depends on protocols rather than provider SDKs:

- a model adapter decides intent/respond/retrieve and generates text
- an appraisal policy converts the normalized event to an affect transition
- a store serializes each conversation and atomically commits state plus transcript
- a retriever supplies bounded candidate memories

The reference adapters are in-memory and scripted. They exist to prove the
orchestration without network access. Retrieved text and carried thoughts are
bounded, typed as untrusted evidence, and excluded from the trusted system prompt.
Adapter or policy failure before commit leaves the turn unchanged.
Both reference engines also enforce configurable ceilings before forwarding
messages, prompts, retrieved evidence, or buffered model output across adapter
boundaries.

## Runtime parity

Python and TypeScript consume one sanitized 225-case component fixture and three
synthetic longitudinal traces. Product prompt text and model output are
deliberately not part of either fixture. The deterministic behavior is aligned;
language-level orchestration APIs follow each ecosystem and are not byte-for-byte
identical.
