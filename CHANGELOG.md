# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until `1.0.0`, orchestration APIs may change in a minor release. The
parity-tested deterministic surfaces listed in the
[parity contract](docs/parity-contract.md) will not change silently: any change
to a pinned vector is called out here.

## [Unreleased]

### Added

- `AffectDynamics` and `RetrievalWeights`: the numeric coefficients are now
  caller-owned data, on the same principle that already made stage names,
  expectation cues, turn-shape rules, and presence labels replaceable. Inertia
  terms, the resting-dominance coefficient, the baseline blend, per-emotion
  carry decay and floor, the recency horizon and floor, the episode bonus, and
  the mood-congruence threshold and asymmetry can all be changed without
  forking. Defaults reproduce the pinned fixture exactly, so the 225 shared
  vectors are unchanged.
- `docs/foundations.md`: per-constant provenance — literature, production-tuned,
  or arbitrary-but-bounded — with the departures from the cited work stated,
  and a list of results that would falsify the current choices.
- `CITATION.cff`, validated against CFF schema 1.2.0.
- `bench/`: a seeded, dependency-free retrieval benchmark over five regimes,
  comparing the scorer against plain similarity and against the additive form
  used by Generative Agents. `bench/RESULTS.md` is generated and drift-checked
  in CI, so no document can quote a stale number. It reports results against the
  current design, including that the additive form wins wherever salience
  carries signal and that mood congruence is worth +0.012 MRR in a regime built
  to favour it.

### Fixed

- Retracted a claim in `docs/foundations.md` and in the `recency_weight`
  docstring that linear-to-a-floor recency was "the least defensible" curve in
  the module. At a matched 30-day half-life it out-ranks both the exponential
  and the power-law curve.

### Changed

- Renamed from `anjo-core` / `@anjo-ai/core` to `affect-kernel` on both
  registries, and the Python module from `anjo_core` to `affect_kernel`. No
  version was ever tagged or published under the old name.

### Documentation

- `docs/algorithm.md` now specifies the ambiguous-intent valence amplification
  (`x1.10` negative, `x1.04` positive above `|v| >= 0.20`), which was
  implemented but undocumented.

## [0.1.0]

First public release: the deterministic kernel extracted from
[Anjo](https://anjo.love) and generalized beyond conversation.

The repository was briefly public as `anjo-core` before this release and was
renamed to `affect-kernel` to name the library by what it does rather than by
the application it came from. No version was ever tagged or published under the
old name, so no installed artifact is affected.

### Added

- Behaviorally aligned Python and TypeScript kernels with no runtime
  dependencies, both published as `affect-kernel`.
- OCC-inspired appraisal, PAD mood dynamics, and Big Five N/E-conditioned affect
  inertia.
- Bounded memory relevance, recency, salience, and mood-congruence scoring.
- Presence surfacing, configurable prompt composition, and injected
  `ModelAdapter` / `AppraisalPolicy` / `StateStore` / `MemoryRetriever` contracts.
- `StageLadder`: a configurable progression with its own rungs and resting
  weights. `strict=True` raises `UnknownStageError` instead of silently flooring
  an unmapped stage to the bottom rung.
- `ExpectationCues`: the English expectation-violation vocabulary is now data a
  caller can replace, not behavior compiled into the kernel.
- `PresenceLabels`: the presence surface wording is caller-owned; the
  conversational phrasing is a default, not a fixture of the library.
- `TurnShapePolicy.suppressed_octant_cues` / `suppressedMoodCues`: the reference
  "no upbeat cue right after vulnerability" rule is expressed as policy data.
  The kernel no longer special-cases any intent label.
- `conversational_appraisal_policy(...)`: build a reference-shaped policy bound
  to a custom ladder and cue set.
- A game-NPC example (`examples/game-npc/`) that drives the same kernel from
  world events with its own ladder, vocabulary, and presence wording. It asserts
  its own invariants and runs in CI.
- 225 synthetic cross-runtime vectors and 3 synthetic longitudinal traces shared
  by both runtimes.
- Public-boundary verification, pinned Gitleaks history scan, and reproducible
  packaging checks.

### Fixed

- `FrozenMapping` is picklable. The default `dict` pickle protocol restores
  items by mutating a fresh instance, which the class refuses, so every state
  object holding one — `CompanionState` with a non-empty `occ_carry`,
  `TurnShapePolicy`, `PromptPolicy` — raised `TypeError` on `pickle.dumps`.
  That broke any `StateStore` serializing with `pickle` and any use across a
  process boundary. Restored values remain immutable.
- Suppressed turn-shape cues resolve declared keys case-insensitively in
  TypeScript, matching Python. Python normalizes declared keys when the frozen
  policy is constructed; TypeScript policies are plain object literals with no
  construction step, so a lower-case declaration silently failed to suppress
  there while working in Python.

[Unreleased]: https://github.com/kevindechang/affect-kernel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kevindechang/affect-kernel/releases/tag/v0.1.0
