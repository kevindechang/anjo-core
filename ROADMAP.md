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

- JSON Schema for portable state snapshots
- Reference SQLite and file-backed stores
- Optional provider adapters in separate packages
- A generic, pluggable reflection-delta contract and drift evaluation
- Packaged appraisal presets for tutoring and coaching, alongside the game
  example
- Embedding-provider interface plus retrieval benchmark fixtures
- A TypeScript port of the game-NPC example
- Generic naming for the top-level state type, which still reads `CompanionState`
- Additional language runtime driven by the same behavioral contract
- Property-based and fuzz tests around clamping, Unicode, and serialization

## Explicitly outside the core

Hosted accounts, billing, branded personality text, product analytics, engagement
mechanics, production moderation policy, and application UI are not planned for
this repository.
