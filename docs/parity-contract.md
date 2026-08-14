# Cross-runtime parity contract

The shared fixture in `shared/golden/kernel_golden.json` is generated from
production-derived deterministic behavior, then sanitized to exclude branded or
product-tuned prompt text.

It contains 225 cases across:

- affect math, excluding prose-producing turn-shape directives
- OCC/PAD appraisal with affect dynamics on and habituation off
- retrieval scoring math
- presence/state surfacing

`shared/golden/continuity_traces.json` adds three synthetic three-step traces for
engagement, conflict/recovery, and affect decay. They exercise the default
conversational appraisal policy longitudinally rather than as isolated scalars.

Both runtimes must reproduce those expected outputs within the documented float
tolerance. A behavior change requires one reviewed fixture change and matching
runtime updates.

## What “parity” means

Parity means the exercised deterministic input produces the same normalized
output. It does not mean:

- complete branch coverage
- identical prompts or replies
- equivalent model providers
- equivalent embeddings
- production policy, safety, or reflection parity
- native-device correctness
- identical Python and TypeScript method signatures or error objects

## Changing the public fixtures

The checked-in fixtures are authoritative; contributors do not need access to
the private application corpus. A fixture change must:

1. State the invariant or bug that the case demonstrates.
2. Contain only synthetic, bounded inputs—never production prompts or user data.
3. Update the strict public schema and exact case/trace counts.
4. Pass in both runtimes in the same pull request.
5. Explain whether the result is backwards-compatible and update the version as
   required below.

Expected values are reviewed behavior, not snapshots to regenerate blindly. If
the runtimes disagree, resolve the specification before updating the fixture.

## Versioning

A backwards-incompatible state field, enum, or calculation change requires a
minor version before 1.0 and a major version after 1.0. Adding cases without
changing expected behavior is a patch change.
