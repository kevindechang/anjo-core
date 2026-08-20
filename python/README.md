# affect-kernel for Python

This directory contains the zero-runtime-dependency Python implementation of the
deterministic affect-state kernel for long-lived AI characters. Install it with
`python -m pip install -e .`; run its tests with `python -m pytest`.

The package deliberately leaves model inference, persistence, and retrieval I/O
behind injected protocols. Its cross-runtime guarantee covers deterministic,
non-habituating affect, appraisal, memory scoring, and presence transforms only.
Prompt wording is supplied by the caller and is not part of production parity.

## Adapter security and transaction contract

`AffectEngine` sends trusted instructions in `GenerateInput.system_prompt` and
bounded retrieved/carried evidence in `GenerateInput.untrusted_context`. Model
adapters must place `untrusted_context` in a user/tool-data channel and obey its
evidence-only rule; they must never concatenate it, `GenerateInput.state`, or
retriever output into the system prompt. Gate, retrieval, and generation request
states have `carried_thought=None`, so the carried text is available only through
the explicit untrusted field on generation.

Gate failures and unsupported intents raise by default. Applications may opt in
to `gate_error_mode="respond"` or `"silent"`. Built-in intents are normalized to
uppercase. A caller can register `custom_intents`; custom labels are also
normalized, must match `[A-Z][A-Z0-9_]{0,63}`, and intentionally receive no
built-in appraisal impulse.

Domain kernels can inject a synchronous `AppraisalPolicy` into
`AffectEngine`. It receives an `AppraisalPolicyInput` containing the current
state, normalized intent/event, message, and expectation, and must return an
`AppraisalResult`. `default_appraisal_policy` explicitly preserves the reference
English conversational mapping implemented by `appraise_turn`.

`EngineLimits` bounds each message (16,000 characters), loaded history (200
messages / 128,000 characters), assembled system prompt (128,000 characters),
and buffered model output (32,000 characters) by default. Pass a replacement
`EngineLimits` value to tighten those ceilings for your deployment. The engine
checks an offending output chunk before invoking its token callback or committing
the turn.

Stores implement `StateStore.transaction(conversation_id)`. That context must
serialize the complete turn for the conversation across every engine instance,
and its `ConversationTransaction.commit(...)` must atomically persist the state
update and message batch. If gate, retrieval, generation, a token callback, or
cancellation fails before commit, the store must leave both state and transcript
unchanged. `InMemoryStateStore` is the dependency-free reference implementation.
