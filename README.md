# Anjo Core

Anjo Core is a deterministic affect-state engine for long-lived AI characters.
It turns appraised events into bounded mood, memory-ranking, response-shape, and
presence signals instead of asking a language model to improvise continuity from
scratch on every turn.

It separates the parts that can be deterministic and testable—affect, appraisal,
memory ranking, state surfacing, and prompt composition—from the parts that must
be supplied by an application, such as model inference, persistence, embeddings,
safety policy, and reflection.

Bring your own model, memory, persistence, safety policy, and event interpreter.
The repository contains behaviorally aligned Python and TypeScript kernels, a
shared behavioral corpus, and credential-free examples.

> Status: experimental public core. The deterministic surfaces are parity-tested;
> orchestration APIs may still change before 1.0. Semantic retrieval, model
> output, production safety policy, and long-term reflection remain
> application-owned.

## Why this exists

Memory libraries answer *what should this agent recall?* Full agent frameworks
answer *how should this agent run?* Anjo Core focuses on the missing layer
between them: *how should an experience change the character, and what part of
that change should become perceptible?* It makes that smaller set of state
transitions explicit:

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

The included English conversational appraisal is a reference preset. Applications
can replace it with a synchronous `AppraisalPolicy`, which makes the same state
kernel useful for companions, game characters, tutoring systems, coaching tools,
interactive fiction, and other long-lived conversational characters.

## What is public

- OCC-inspired intent appraisal and PAD mood dynamics
- Big Five N/E-conditioned affect inertia
- an injected appraisal-policy seam with an English conversational preset
- bounded memory relevance, recency, salience, and mood-congruence scoring
- compact presence/state surfacing
- configurable prompt composition with neutral reference policy
- injected model, store, and retriever contracts
- headless Python and TypeScript engines
- **225 production-derived cross-runtime cases** for the generalizable math and
  state transforms
- **3 synthetic longitudinal traces** covering engagement, conflict/recovery,
  and affect decay
- credential-free scripted examples

## What is deliberately not here

- Anjo's identity, static persona, product-tuned prompt text, or private policies
- hosted application, mobile UI, accounts, billing, analytics, or deployment
- production user data, memory databases, operational documents, or git history
- claims that two language models will produce identical conversations
- production safety, clinical, or dependency-prevention policy
- automatic relationship, attachment, or personality evolution

Applications must supply and evaluate those layers for their own domain.

## Quick start

### Python

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e "./python[dev]"
pytest python/tests
python examples/python-headless/main.py
```

### TypeScript

Requires Node.js 20+.

```bash
npm ci --prefix typescript
npm test --prefix typescript
npm run example --prefix typescript
```

Neither example uses a network connection or API key. Each injects a scripted
model and an in-memory store so the orchestration and state changes are visible.

## Core contracts

Both runtimes expose the same conceptual seams:

- `ModelAdapter`: classify/gate and generate
- `AppraisalPolicy`: translate a normalized event into an affect transition
- `StateStore`: load/save state and transcript
- `MemoryRetriever`: return grounded candidate memories
- `CompanionEngine`: orchestrate one turn without choosing a provider or database

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

See the [algorithm and invariants](docs/algorithm.md),
[parity contract](docs/parity-contract.md), and
[limitations](docs/limitations.md) before making product claims.

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

Good first contribution areas include new storage/model adapters, adversarial
kernel cases, serialization fixtures, additional language ports, and evaluation
tools. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
[roadmap](ROADMAP.md).

## Security and privacy

The core has no network client and does not persist or transmit anything by
itself; its reference stores retain state and transcripts only in process memory.
Your adapters define the real privacy boundary. Never put conversation data,
credentials, model weights, or production configuration in a contribution.
Report vulnerabilities through the process in [SECURITY.md](SECURITY.md).
The clean-extraction boundary and pre-publication ownership check are documented
in [provenance](docs/provenance.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
