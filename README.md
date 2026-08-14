# Anjo Core

[![CI](https://github.com/kevindechang/anjo-core/actions/workflows/ci.yml/badge.svg)](https://github.com/kevindechang/anjo-core/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](python/)
[![Node](https://img.shields.io/badge/node-20%2B-blue.svg)](typescript/)

A deterministic affect-state engine for long-lived AI characters, in Python and
TypeScript, with no runtime dependencies.

It turns appraised events into bounded mood, memory ranking, response shape, and
presence signals — instead of asking a language model to improvise continuity
from scratch on every turn. Extracted from [Anjo](https://anjo.love), a
production companion app, and generalized.

## See it move

Every number below is the kernel. Run it yourself with no API key:

```bash
python examples/python-headless/main.py
```

```text
Turn 1 — User: I picked up my sketchbook again.
State: intent=CURIOSITY,      PAD=(0.2844, 0.1868, 0.0500), evidence=['demo-memory']

Turn 2 — User: The first page looks terrible.
State: intent=VULNERABILITY,  PAD=(0.3590, 0.2373, 0.0368), evidence=none

Turn 3 — User: Still, I want to try again tomorrow.
State: intent=CURIOSITY,      PAD=(0.4825, 0.3244, 0.0770), evidence=none
```

> The reply text in these examples comes from a **scripted adapter** — fixed
> strings, not model output. The examples inject a scripted model and an
> in-memory store precisely so you can watch the state change without a
> provider. The PAD numbers are what the library computes.

The same kernel with **no conversation in it** — a game NPC whose disposition
tracks world events, with its own progression ladder and its own vocabulary
([full example](examples/game-npc/)):

```bash
python examples/game-npc/main.py
```

```text
world event         valence  arousal  dominance
QUEST_COMPLETED      0.3500   0.2000     0.1877
PROMISE_BROKEN      -0.0004   0.2710     0.0627
PLAYER_ATTACKED     -0.4500   0.5317     0.2558
PLAYER_HEALED       -0.0849   0.4046     0.1709

disposition: wary | presence: on watch (posted)
```

## Why this exists

Memory libraries answer *what should this agent recall?* Agent frameworks answer
*how should this agent run?* Anjo Core covers the layer between them: *how should
an experience change the character, and what part of that change should become
perceptible?*

| | Focus | Relationship to this project |
|---|---|---|
| [mem0](https://github.com/mem0ai/mem0), [Zep](https://github.com/getzep/zep) | Extracting, storing, and recalling memory | Complementary — plug one in behind `MemoryRetriever`; this library scores and ranks what they return |
| [Letta / MemGPT](https://github.com/letta-ai/letta) | A stateful agent server with self-editing memory | Overlapping ambition, opposite shape: Letta is a service you run, this is a dependency-free library you embed |
| [LangGraph](https://github.com/langchain-ai/langgraph), [Agno](https://github.com/agno-agi/agno) | Orchestrating steps, tools, and control flow | Complementary — this is one deterministic node inside whatever graph you already have |
| Prompt-only "personality" | A persona paragraph in the system prompt | The thing this replaces: a paragraph cannot decay, carry, or accumulate |

The distinguishing bet: **the parts that can be deterministic should be**. Mood
dynamics, appraisal, ranking, and surfacing are ordinary math with pinned
behavior, not a model call you hope stays consistent.

```text
application event
  → application-owned interpreter
  → appraisal policy
  → deterministic affect transition
  → mood-aware candidate ranking
  → response controls + semantic presence
  → model adapter
  → atomic persistence
```

## Install

Neither runtime has a production dependency.

```bash
# Python 3.11+
pip install anjo-core

# Node.js 20+
npm install @anjo-ai/core
```

From a checkout:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e "./python[dev]"
pytest python/tests

npm ci --prefix typescript
npm test --prefix typescript
```

## Core contracts

Both runtimes expose the same conceptual seams:

- `ModelAdapter` — classify/gate and generate
- `AppraisalPolicy` — translate a normalized event into an affect transition
- `StateStore` — load/save state and transcript, with an atomic commit
- `MemoryRetriever` — return grounded candidate memories
- `CompanionEngine` — orchestrate one turn without choosing a provider or database

Every piece of domain vocabulary is data you can replace, not behavior baked into
the kernel:

| Seam | Reference preset | Replace it with |
|---|---|---|
| `StageLadder` | stranger → intimate | your own rungs and resting weights; `strict=True` to reject unmapped stages |
| `AppraisalPolicy` | English conversational intents | any synchronous event → transition function |
| `ExpectationCues` | English sentiment words | your own tokens, or nothing |
| `TurnShapePolicy` | companion cadence rules | your own wording and cue-suppression rules |
| `PresenceLabels` | "here with you" | your own surface wording |
| `PromptPolicy` | neutral section headings | your own prompt language |

## What is public

- OCC-inspired intent appraisal and PAD mood dynamics
- Big Five N/E-conditioned affect inertia
- an injected appraisal-policy seam with an English conversational preset
- bounded memory relevance, recency, salience, and mood-congruence scoring
- compact presence/state surfacing
- configurable prompt composition with a neutral reference policy
- injected model, store, and retriever contracts
- headless Python and TypeScript engines
- **225 synthetic cross-runtime vectors** covering the generalizable math and
  state transforms — derived from the deterministic behavior of the production
  app, with product prose and policy excluded (see [provenance](docs/provenance.md))
- **3 synthetic longitudinal traces** covering engagement, conflict/recovery,
  and affect decay
- credential-free scripted examples for a companion and a game NPC

## What is deliberately not here

- Anjo's identity, static persona, product-tuned prompt text, or private policies
- hosted application, mobile UI, accounts, billing, analytics, or deployment
- production user data, memory databases, operational documents, or git history
- claims that two language models will produce identical conversations
- production safety, clinical, or dependency-prevention policy
- automatic relationship, attachment, or personality evolution

Applications must supply and evaluate those layers for their own domain. Read
[limitations](docs/limitations.md) before making product claims.

## Runtime parity

The deterministic public contract is intentionally narrower than the engine:

| Surface | Shared parity | Notes |
|---|---:|---|
| Affect math | Yes | Mood octant, decoding envelope, length control, ambivalence |
| OCC/PAD appraisal | Yes | Non-habituating contract |
| Retrieval scoring | Yes | Ranking math, not embeddings or a vector store |
| Presence surfacing | Yes | Pure state-to-payload transform |
| Longitudinal affect | Yes | Three synthetic three-step traces |
| Prompt composition | Independently tested | Public policy is configurable and neutral |
| Gate / generation / reflection | No | Gate/generation are adapters; reflection is not included |

See the [algorithm and invariants](docs/algorithm.md) and the
[parity contract](docs/parity-contract.md).

## Repository layout

```text
shared/golden/       cross-runtime behavioral fixture
python/              Python package and tests
typescript/          TypeScript package and tests
examples/            credential-free reference programs
docs/                architecture, boundaries, and design principles
scripts/             public-boundary and repository checks
```

## Contributing

Good first contributions: new storage/model adapters, adversarial kernel cases,
appraisal presets for other domains, serialization fixtures, additional language
ports, and evaluation tools. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
the [roadmap](ROADMAP.md).

Run everything the way CI does:

```bash
./scripts/check.sh
```

## Security and privacy

The core has no network client and does not persist or transmit anything by
itself; its reference stores retain state and transcripts only in process memory.
Your adapters define the real privacy boundary. Retrieved text and carried
thoughts are typed as untrusted evidence and are structurally excluded from the
system prompt.

Never put conversation data, credentials, model weights, or production
configuration in a contribution. Report vulnerabilities through the process in
[SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
