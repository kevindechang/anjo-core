# Limitations

- The kernel is not a complete companion product.
- Gate and generation quality depend on injected adapters; reflection is not
  included in the current public runtime.
- The shared appraisal contract excludes production-only habituation behavior.
- The default appraisal policy, stage ladder, expectation cues, and presence
  labels are English conversational presets. They are replaceable values rather
  than kernel behavior, but a non-conversational or non-English domain must
  actually supply its own; the defaults will otherwise leak companion phrasing
  into an unrelated product. See `examples/game-npc/` for a worked replacement.
- The reference engine evolves mood, baseline valence, and OCC carry; it does not
  evolve relationship stage, attachment, or personality.
- Retrieval code scores candidates; it does not create embeddings or operate a
  vector database.
- Memory grounding and false-memory refusal must be evaluated by the application.
- The neutral prompt policy is an example, not a production safety policy.
- There is no authentication, encryption, synchronization, or multi-user
  isolation in the core.
- Python and TypeScript share deterministic behavior, fixtures, and conceptual
  seams, but their orchestration APIs are idiomatic rather than identical.
- State models are behavioral abstractions, not claims of emotion or
  consciousness.
- The project is not designed for clinical diagnosis, therapy, crisis response,
  or other high-stakes decisions.

Use the project as a transparent foundation, then add domain-appropriate safety,
privacy, observability, and evaluation at the application boundary.
