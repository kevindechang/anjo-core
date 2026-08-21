# Affect Kernel

[![CI](https://github.com/kevindechang/affect-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/kevindechang/affect-kernel/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/affect-kernel.svg)](https://pypi.org/project/affect-kernel/)
[![npm](https://img.shields.io/npm/v/affect-kernel.svg)](https://www.npmjs.com/package/affect-kernel)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](python/)
[![Node](https://img.shields.io/badge/node-20%2B-blue.svg)](typescript/)

**Deterministic emotional continuity for AI characters.**

Affect Kernel is a dependency-free state layer for companions, NPCs, tutors,
and other long-lived characters. It turns appraised events into bounded mood,
emotion carry, memory scores, response controls, and UI-ready presence signals.

You bring the model, memory backend, and persistence. The kernel gives you the
part that should be inspectable and replayable, with Python and TypeScript
pinned to the same behavioral contract.

It was extracted from the deterministic state machinery behind
[Anjo](https://anjo.love), then stripped of product prose and private policy and
generalized behind replaceable contracts.

```text
appraised event ──→ PAD/OCC state ──→ response controls + presence
memory candidates ──→ bounded ranking ────────────────┘
```

| What you get | What you keep control of |
|---|---|
| Bounded, non-mutating state transitions | Model provider and prompts |
| Mood-aware candidate scoring | Embeddings and vector database |
| Response and decoding controls | Persistence and transactions |
| Semantic presence for UI surfaces | Domain vocabulary and safety policy |
| Cross-runtime behavioral fixtures | Relationship and personality evolution |

## Start in 60 seconds

Install either runtime. Both packages have zero runtime dependencies.

```bash
python -m pip install affect-kernel  # import affect_kernel
npm install affect-kernel
```

Appraise one event and turn the resulting state into a UI-ready presence
payload:

```python
from affect_kernel import AffectState, appraise_turn, build_presence_vector

state = AffectState()
step = appraise_turn(
    state,
    "CURIOSITY",
    message="I picked up my sketchbook again.",
)

state = step.state
print(build_presence_vector(state).to_dict()["affect"])
# {'valence': 0.208, 'arousal': 0.15, 'dominance': 0.05}
```

That call is a pure state transition: no model, network, global state, or hidden
persistence. For complete turn orchestration, start with the credential-free
[Python engine example](examples/python-headless/main.py) or
[TypeScript engine example](examples/typescript-headless/src/main.ts).

## Watch state accumulate

The headless example injects an in-memory store, one synthetic memory, and a
scripted model. The text is fixed; the state changes are computed by the kernel.
The state portion of its output is:

```bash
python examples/python-headless/main.py
```

```text
Turn 1 — User: I picked up my sketchbook again.
State: intent=CURIOSITY, PAD=(0.2844, 0.1868, 0.0500), evidence=['demo-memory']

Turn 2 — User: The first page looks terrible.
State: intent=VULNERABILITY, PAD=(0.3590, 0.2373, 0.0368), evidence=none

Turn 3 — User: Still, I want to try again tomorrow.
State: intent=CURIOSITY, PAD=(0.4825, 0.3244, 0.0770), evidence=none
```

The same machinery can track world events instead of conversation. The
[game NPC example](examples/game-npc/main.py) replaces the progression ladder,
appraisal policy, and surface vocabulary:

```text
world event         valence  arousal  dominance
QUEST_COMPLETED      0.3500   0.2000     0.1877
PROMISE_BROKEN      -0.0004   0.2710     0.0627
PLAYER_ATTACKED     -0.4500   0.5317     0.2558
PLAYER_HEALED       -0.0849   0.4046     0.1709

disposition: wary | presence: on watch (posted)
```

## Where it fits

Memory systems answer **what happened?** Agent frameworks answer **what runs
next?** Affect Kernel answers **how should this event change the character, and
what should become perceptible?**

```text
message or world event
        │
        ▼
ModelAdapter.gate or your own event interpreter
        │ normalized intent
        ▼
MemoryRetriever → deterministic ranking
        │
        ▼
AppraisalPolicy → bounded affect transition
        │
        ▼
response controls + untrusted evidence → ModelAdapter.generate
        │
        ▼
StateStore atomic commit + presence payload
```

It is designed to sit beside tools such as
[mem0](https://github.com/mem0ai/mem0) or [Zep](https://github.com/getzep/zep)
for memory, and inside orchestration such as
[LangGraph](https://github.com/langchain-ai/langgraph) or
[Agno](https://github.com/agno-agi/agno). Unlike a hosted agent server, it is a
small library you embed and own.

A persona prompt describes a character. Affect Kernel represents what changed
after an event. It does **not** yet prove that this produces better long-horizon
responses than a prompt-only persona; that comparison remains an open
evaluation problem.

## Choose your integration level

Use the pure functions when your application already owns classification and
control flow. Use `AffectEngine` when you want the reference turn transaction.

| Job | Python | TypeScript |
|---|---|---|
| Create or validate state | `AffectState` | `createAffectState` |
| Appraise one event | `appraise_turn` | `appraiseTurn` |
| Rank memory candidates | `rank_candidates` | `rankCandidates` |
| Derive response controls | `decoding_params` | `decodingParams` |
| Build presence payload | `build_presence_vector` | `buildPresenceVector` |
| Orchestrate a complete turn | `AffectEngine` | `AffectEngine` |

The engine coordinates this sequence:

1. Load state and gate the input.
2. Optionally retrieve and deterministically rank evidence.
3. Apply the appraisal policy and derive response controls.
4. Ask the injected model adapter to generate.
5. Atomically commit the next state and messages.

Its four boundary contracts are deliberately small:

- `ModelAdapter` classifies/gates and generates.
- `MemoryRetriever` returns candidate memories or evidence.
- `AppraisalPolicy` maps a normalized event to a state transition.
- `StateStore` provides per-conversation transactions and atomic commits.

See the [Python package guide](python/README.md),
[TypeScript package guide](typescript/README.md), and
[architecture](docs/architecture.md) for signatures and adapter requirements.

## State model

`AffectState` is serializable and caller-owned. It contains:

- PAD mood: valence, arousal, and dominance
- OCC-inspired emotion carry between turns
- OCEAN (Big Five) personality and appraisal goals
- relationship stage, trust, session count, and prior-session valence
- attachment, baseline valence, expectation, and a carried thought

The reference appraisal updates **mood, baseline valence, and OCC carry**. It
does not automatically evolve personality, relationship stage, trust, or
attachment. Those are inputs your application may update through its own
policy, reflection process, or progression system.

All numeric state is validated and bounded. Python state objects are frozen
dataclasses; TypeScript inputs are normalized into defensively copied,
statically read-only structures.

## Customize without forking

Domain vocabulary is configuration, not hard-coded behavior:

| Seam | Reference preset | Replace it with |
|---|---|---|
| `StageLadder` | stranger → intimate | NPC reputation, tutoring levels, game factions |
| `AppraisalPolicy` | conversational intents | any synchronous event → transition function |
| `ExpectationCues` | English sentiment cues | another language, domain tokens, or nothing |
| `TurnShapePolicy` | companion cadence | your own response-shaping rules |
| `PresenceLabels` | “here with you” | UI text for your product or world |
| `PromptPolicy` | neutral prompt sections | your own prompt language and structure |

The coefficients are configurable too. `AffectDynamics` exposes inertia,
resting dominance, baseline blending, and per-emotion decay;
`RetrievalWeights` exposes recency, salience, episode, and mood-congruence
weights.

```python
from affect_kernel import AffectDynamics, AffectState, appraise_turn

state = AffectState()

# A character whose mood carries less strongly between events.
volatile = AffectDynamics(inertia_base=0.30, inertia_min=0.0, inertia_max=0.5)
step = appraise_turn(state, "CURIOSITY", dynamics=volatile)
```

Defaults reproduce the pinned cross-runtime fixtures. Custom values
intentionally take you off that behavioral contract. Every default constant is
tagged in [foundations](docs/foundations.md) as literature-grounded,
production-tuned, or a bounded design choice.

## Evidence and limits

The deterministic contract is checked against **225 shared vectors** and
**3 longitudinal traces** in both runtimes. The vectors cover affect math,
appraisal, ranking, presence, and state transforms; the traces cover engagement,
conflict/recovery, and decay. See the [parity contract](docs/parity-contract.md)
and [provenance](docs/provenance.md).

The seeded retrieval benchmark is intentionally adversarial. It compares the
current scorer with plain similarity and the additive form used by Generative
Agents across five synthetic regimes:

| Finding | MRR result |
|---|---:|
| Assumptions hold | **+0.415** vs similarity-only |
| Assumptions are violated | **−0.175** vs similarity-only |
| Additive form vs current multiplicative form | **+0.111** against the current scorer |
| `significance_weight` from `0.03` to `1.0` | `0.858` → `0.960` |
| Mood congruence in a favorable regime | **+0.012** |

Run it with `python bench/run.py`. These are synthetic corpora with
machine-assigned ground truth, not evidence of better conversations or user
outcomes. Read the [benchmark limitations](bench/README.md#limitations--read-before-quoting-any-number)
before quoting a result.

This project is also **not** a complete agent, an emotion detector, a vector
database, a reflection engine, or a production safety layer. It must not be
treated as a clinical model or as evidence that a system feels emotions. The
full boundary is documented in [limitations](docs/limitations.md).

## Security and privacy boundary

The core has no network client and does not persist or transmit data by itself.
Reference stores are in-memory only. Retrieved text and carried thoughts are
typed as untrusted evidence and kept structurally separate from the system
prompt.

Your adapters define the real privacy and security boundary. A model adapter
can defeat that separation if it inserts retrieved text into privileged
instructions; a store adapter can persist sensitive data insecurely. Read the
[threat model](docs/threat-model.md) before a production integration and report
vulnerabilities through [SECURITY.md](SECURITY.md).

## Documentation

| Read this | When you need |
|---|---|
| [Architecture](docs/architecture.md) | data flow, components, and transaction semantics |
| [Algorithm and invariants](docs/algorithm.md) | equations, bounds, and edge-case behavior |
| [Foundations](docs/foundations.md) | sources and provenance for every constant |
| [Design principles](docs/design-principles.md) | why the public seams have this shape |
| [Parity contract](docs/parity-contract.md) | what Python and TypeScript guarantee in common |
| [Threat model](docs/threat-model.md) | trust boundaries and adapter obligations |
| [Limitations](docs/limitations.md) | unsupported claims and deliberately excluded layers |

## Contributing

Good first contributions include storage or model adapters, adversarial cases,
domain-specific appraisal presets, serialization fixtures, language ports, and
evaluation tools. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
[roadmap](ROADMAP.md).

Use [Discussions](https://github.com/kevindechang/affect-kernel/discussions) for
integration questions and design ideas, or
[Issues](https://github.com/kevindechang/affect-kernel/issues) for reproducible
bugs and scoped feature requests.

```bash
git clone https://github.com/kevindechang/affect-kernel.git
cd affect-kernel
./scripts/setup.sh
./scripts/check.sh
```

`./scripts/check.sh` runs the same repository checks as CI.

## Citation

Machine-readable metadata is in [CITATION.cff](CITATION.cff), validated against
CFF schema 1.2.0.

```bibtex
@software{chang_affect_kernel_2026,
  author  = {Chang, Chia Wei},
  title   = {affect-kernel: a deterministic affect-state kernel for
             long-lived AI characters},
  version = {0.1.0},
  year    = {2026},
  license = {Apache-2.0},
  url     = {https://github.com/kevindechang/affect-kernel}
}
```

If you cite the ideas rather than this implementation, cite the primary sources
in [foundations](docs/foundations.md); the library implements only a subset and
documents where it departs from them.

## License

Apache License 2.0. See [LICENSE](LICENSE).
