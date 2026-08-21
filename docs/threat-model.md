# Threat model

What this library defends against, what it does not, and where the boundary
moves to your code. The short version: **the kernel has no network client, no
persistence, and no authority.** Almost every real security property is defined
by the adapters you inject.

## Assets

| Asset | Where it lives | Who protects it |
|---|---|---|
| Character state (mood, traits, relationship) | your `StateStore` | you |
| Conversation transcript | your `StateStore` | you |
| Retrieved memory text | your `MemoryRetriever` | you |
| Model credentials | your `ModelAdapter` | you |
| The system prompt sent to a model | composed here, sent by you | shared |

The kernel holds all of these in process memory for the duration of a turn and
writes none of them anywhere. Its reference `StateStore` and `MemoryRetriever`
are in-memory and are for examples and tests, not production.

## The one boundary the kernel does enforce

**Retrieved memory and carried thoughts are untrusted input.** They usually
originate from a user, sometimes from a model, and they are the natural vector
for prompt injection against a system that recalls things.

The kernel treats them as a separate, lower-trust channel:

- `UntrustedContext` is a distinct type. Memory text and carried thoughts cannot
  be passed where trusted derived context is expected.
- They are **structurally excluded from the default system prompt**. There is no
  formatting option that places them there — the exclusion is a property of the
  type, not a policy string.
- They are bounded before they reach a model: 4,096 characters per source field,
  2,000 per surfaced item, 8,000 per assembled context
  (`MAX_UNTRUSTED_SOURCE_CHARS`, `MAX_UNTRUSTED_ITEM_CHARS`,
  `MAX_UNTRUSTED_CONTEXT_CHARS`).
- `AffectEngine` applies its own ceilings to message, history, prompt, and
  output size (`EngineLimits`), so a hostile transcript cannot grow a request
  without bound.

Both runtimes have tests asserting that untrusted text never reaches the system
prompt. **An adapter that flattens the channel on the wire defeats this**, and
that is the most likely way to lose the property in practice: if your transport
concatenates trusted and untrusted context into one string before sending, the
distinction is gone. Preserve the separation end to end.

## What the kernel does not defend against

- **Prompt injection in general.** Bounding and channel-separating untrusted
  text raises the cost of an attack; it does not stop a model from obeying
  instructions inside a memory it was shown. Evaluate your own model's behavior.
- **False memories.** The scorer ranks candidates; it has no notion of whether a
  candidate is true. `docs/limitations.md` says this and it belongs here too.
- **Authentication, authorization, multi-tenancy.** None exist. Two users' state
  is only isolated if your `StateStore` isolates it.
- **Encryption at rest or in transit.** Not present, by design — there is
  nothing to encrypt until your adapter persists something.
- **Denial of service.** `EngineLimits` bounds a single turn. Rate limiting,
  concurrency control, and cost ceilings are yours.
- **Malicious `AppraisalPolicy`, `StateStore`, or `ModelAdapter`.** Injected
  code runs with your process's privileges. The kernel calls what you give it.
- **Side channels.** Response length and sampling temperature vary with affect
  state (`docs/foundations.md` section 8). An observer who can measure replies
  can infer something about the character's state. This is intended behavior —
  the point of the library is that state becomes perceptible — but if that state
  is derived from sensitive user history, the inference reaches the user.

## Concurrency and integrity

A turn commits through a store-owned transaction. Any failure before commit
leaves state and transcript unchanged, and one conversation is serialized across
engine instances by the store. **That guarantee is only as good as your store's
transaction.** An implementation that returns a no-op transaction silently
converts the atomicity claim into nothing.

## Supply chain

- Zero runtime dependencies in both runtimes, so there is no transitive runtime
  surface to audit.
- Dev dependencies are hash-pinned (`python/requirements-dev.lock`,
  `typescript/package-lock.json`); CI installs with `--require-hashes` and
  `npm ci --ignore-scripts`.
- GitHub Actions are pinned to commit SHAs, not tags.
- CI scans the full Git history with a pinned Gitleaks version whose archive
  checksum is verified before extraction.
- `scripts/verify_public_boundary.py` fails closed on credentials, private
  paths, symlinks, binaries, non-UTF-8 files, public IP literals, SSH targets,
  and absolute deployment paths.
- Releases publish via PyPI Trusted Publishing (no stored token) and npm with
  provenance attestation.

## Reporting

See [SECURITY.md](../SECURITY.md). Please do not open a public issue for a
vulnerability.
