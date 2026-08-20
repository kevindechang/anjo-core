# Roadmap

## 0.1 — useful extraction

- Stable Python and TypeScript state/model/store/retriever contracts
- Injectable appraisal policy with a reference conversational preset
- Configurable stage ladder, expectation cues, turn-shape rules, and presence
  labels, so no transform branches on product-specific vocabulary
- Credential-free headless engines and examples, including a non-conversational
  game-NPC domain that runs in CI
- Shared affect/appraisal/retrieval/surfacing corpus
- Cross-runtime longitudinal affect traces
- Reproducible CI, packaging, and publication-boundary checks

## Next

- Re-tune `significance_weight`: `bench/` shows the salience term is
  underpowered, and that raising it closes almost the whole gap to the additive
  baseline (MRR 0.858 → 0.960). Needs a fixture change and a reviewed decision.
- Time-aware mood decay, so two turns a minute apart and two turns a week apart
  stop decaying identically — falsification item (1) in `docs/foundations.md`.
- A persona-consistency evaluation against a prompt-only baseline, which is the
  README's headline claim and is currently untested.
- JSON Schema for portable state snapshots
- Reference SQLite and file-backed stores
- Optional provider adapters in separate packages
- A generic, pluggable reflection-delta contract and drift evaluation
- Packaged appraisal presets for tutoring and coaching, alongside the game
  example
- Embedding-provider interface, and a retrieval benchmark over real embeddings
  rather than the drawn similarities used in `bench/`
- A TypeScript port of the game-NPC example
- Generic naming for the top-level state type, which still reads `AffectState`
- Additional language runtime driven by the same behavioral contract

## Explicitly outside the core

Hosted accounts, billing, branded personality text, product analytics, engagement
mechanics, production moderation policy, and application UI are not planned for
this repository.
