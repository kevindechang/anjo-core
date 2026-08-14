# Contributing to Anjo Core

Thank you for helping make long-lived character systems more inspectable and
portable.

## Before opening a pull request

1. Open an issue for behavior or API changes that affect both runtimes.
2. Keep product-specific personas, prompts, credentials, and user data out of the
   repository.
3. Add a failing test before changing behavior.
4. When a parity-covered behavior changes, update both runtimes and the shared
   fixture in one pull request.
5. Run `./scripts/check.sh` from the repository root.

Read the [parity fixture workflow](docs/parity-contract.md#changing-the-public-fixtures)
before adding or changing behavioral vectors.

## Development setup

```bash
./scripts/setup.sh
./scripts/check.sh
```

The examples must remain credential-free and deterministic.

## Pull-request expectations

- Explain the user-facing or integrator-facing behavior being changed.
- Name the authoritative implementation and tests.
- Document whether the change affects the shared parity contract.
- Add error-path and boundary cases, not only a happy path.
- Keep provider SDKs in optional adapters; the kernel must retain zero runtime
  dependencies.
- Keep domain event interpretation in an injected appraisal policy. The default
  English conversational policy is a preset, not the universal event model.
- Do not branch on a label only one product uses. Relationship rungs, intent
  names, cue words, and surface wording belong in a caller-supplied value object
  with the preset as its default. See
  [design principles](docs/design-principles.md#domain-vocabulary-is-data-not-behavior).
- Update public documentation when a claim or limitation changes.

## Cross-runtime changes

The shared fixture covers generalizable deterministic behavior only. Do not add
product-tuned prose to it. A cross-runtime change is complete when:

- Python tests pass.
- TypeScript tests pass.
- the shared fixture diff is intentional and reviewed.
- neither implementation quietly adds behavior absent from the other.

## Commit and review style

Use focused commits and descriptive messages such as `feat: add sqlite store
adapter` or `fix: preserve negative-zero rounding parity`. Pull requests should
be small enough to review behaviorally.

By contributing, you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE).
