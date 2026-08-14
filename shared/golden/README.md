# Shared behavioral corpus

`kernel_golden.json` is the cross-runtime compatibility contract for the
deterministic kernel. It contains 225 synthetic input/output vectors covering
affect control, non-habituating OCC/PAD appraisal, retrieval scoring, and
presence surfacing.

It intentionally excludes prompt prose, turn-shape directives, model output,
embeddings, persistence, reflection, and application safety policy. The Python
and TypeScript suites consume this same file.

The checked-in fixture is authoritative for this repository. Maintainers can
rebuild it from the private upstream corpus with `scripts/build_kernel_fixture.py`;
contributors do not need access to that upstream source.

`continuity_traces.json` contains three fully synthetic, three-step sequences.
They pin longitudinal behavior for engagement, conflict/recovery, and decay. A
strict public validator allowlists every event string and rejects extra fields,
unbounded values, or altered metadata before either runtime consumes the file.
