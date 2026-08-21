# `affect-kernel`

A zero-runtime-dependency TypeScript affect-state kernel for long-lived AI
characters. It provides deterministic affect, appraisal, memory scoring, presence
surfacing, and an engine with injected model, appraisal, storage, and retrieval
adapters. Application instructions and prompt wording remain caller-owned.

```sh
npm install affect-kernel
```

The [repository README](https://github.com/kevindechang/affect-kernel#readme)
contains runnable headless examples, benchmark results, limitations, and the
cross-runtime contract.

The engine runtime-validates unknown gate output, normalizes built-in or explicitly
registered custom intents, and propagates gate errors unless `gateFallback` is supplied.
`AppraisalPolicy` is a synchronous injection seam; `DEFAULT_APPRAISAL_POLICY` preserves
the reference conversational OCC/PAD mapping. Model evidence is supplied separately as
bounded `GenerateInput.untrustedContext`, not added to the system prompt. `PromptPolicy`
sections receive only trusted instruction and derived affect/decoding controls; raw messages,
history, memories, and carried thoughts are excluded even for untyped runtime callers.

Stores implement `transaction(operation)` so a full turn commits state and both messages
atomically and serializes across engine instances. Direct reads must expose the last committed
snapshot without waiting for a transaction's private staged work; direct writes serialize behind
that transaction. `EngineLimits` bounds messages, loaded history, retrieval candidates, ranked
memories, evidence, prompts, output, and queued turns.
`turn()` also accepts an AbortSignal-compatible `signal`, a `deadline`, and an async
`onToken` callback.

```sh
npm test
npm run build
npm run example
npm run test:coverage
```

`npm run example` packs the package, installs that tarball into a clean headless
consumer, typechecks and runs three turns, then removes the generated consumer
artifacts. Internal helpers are not package exports.

Licensed under Apache-2.0. Source and issues are hosted at
<https://github.com/kevindechang/affect-kernel>.
