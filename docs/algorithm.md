# Algorithm and invariants

The checked-in fixtures are the behavioral authority. Neither language runtime
is the oracle by itself. `kernel_golden.json` contains 101 affect, 16 retrieval,
69 appraisal, and 39 surfacing cases; `continuity_traces.json` adds three
three-step sequences.

## State domains

- PAD mood and baseline valence: `[-1, 1]`
- OCEAN traits, appraisal goals, trust, and attachment values: `[0, 1]`
- OCC carry values: `[0, 1]`
- relationship stage: `stranger`, `acquaintance`, `friend`, `close`, or `intimate`

Only Neuroticism and Extraversion currently affect inertia. Relationship stage
affects the resting point, and attachment longing can add a state emotion. The
default engine does not evolve personality, goals, relationship, or attachment.

## Affect controls

A PAD value is `neutral` when all three absolute components are below `0.15`.
Otherwise the signs map to the standard eight labels:

| PAD signs | Label | PAD signs | Label |
|---|---|---|---|
| `+++` | exuberant | `-++` | hostile |
| `++-` | dependent | `-+-` | anxious |
| `+-+` | relaxed | `--+` | disdainful |
| `+--` | docile | `---` | bored |

Sampling temperature is `clamp(1 + 0.20 × arousal, 0.72, 1.18)` with
`top_p=0.97`. A missing mood produces temperature `1` and no top-p override.

The response-budget factor is `0.60` below arousal `-0.30`, otherwise `0.72`
below valence `-0.30`, otherwise `1`. Applying it can never increase the
caller's positive integer budget.

Ambivalence requires the strongest positive and negative emotion to each be at
least `0.15`, with `min/max >= 0.40`.

## Default conversational appraisal

The included intent and English expectation mapping is a non-habituating
reference policy. It is not a universal event ontology. Applications can inject
a synchronous appraisal policy while retaining the same transaction and state
boundaries.

Relationship stages map to ordinals 1–5. Their resting-point weights are
`0, 0.20, 0.40, 0.60, 0.70`; unknown labels behave like `stranger`.

Mood inertia is:

```text
phi = clamp(0.80 + 0.20 × (N - 0.5) - 0.10 × (E - 0.5), 0.62, 0.92)
rest = (stage_weight × baseline_valence, 0, stage_weight × 0.10)
next_mood = round4(clamp(rest + phi × (current_mood - rest), -1, 1))
```

The reference intent impulses are:

| Intent | PAD change | Fresh emotion signal |
|---|---|---|
| `ABUSE` | `V-.35, A-.10, D+.25` | reproach `.95×respect`, distress `.65×rapport` |
| `APOLOGY` | `V+.05` | joy `.20×rapport`, gratitude `.35×honesty` |
| `VULNERABILITY` | `V+.15, A+.10` | gratitude `.70×honesty`, joy `.75×rapport` |
| `CURIOSITY` | `V+.20, A+.15, D+.05` | admiration `.75×intellectual`, joy `.50×rapport` |
| `CHALLENGE` | `V-.05, D+.10` | admiration `.45×intellectual`, distress `.25×rapport` |
| `NEGLECT` | `V-.10, A-.05` | distress `.40×rapport` |
| `CASUAL` | `V+.02` | joy `.05` |

Slow baseline valence becomes
`round4(clamp(0.98 × old + 0.02 × appraised_valence, -1, 1))`.

Prior OCC signals decay by `0.70` for reproach, `0.80` for distress, `0.85`
for admiration, `0.88` for gratitude, `0.90` for joy, and `0.80` otherwise.
Values at or below `0.05` are dropped. Fresh, carried, state-derived, and
expectation-derived signals merge by maximum.

## Retrieval scoring

The scorer uses cosine distance in `[0, 2]`:

```text
similarity = 1 - distance / 2
age_days = max(0, now - timestamp)
recency = clamp(1 - age_days / 60, 0.40, 1)
salience = 1 + 0.03 × clamp(significance, 0, 1)
             + min(0.025, 0.006 × ln(1 + recall_count))
score = similarity × recency × salience + (0.05 when episode else 0)
```

Future timestamps receive maximum freshness. Missing, invalid, or unusable
timestamps receive the `0.70` fallback. Candidate timestamps must be zoned ISO
values at adapter boundaries.

Mood congruence applies only when enabled and absolute mood valence is at least
`0.20`. A same-sign memory receives `1.06` in a negative mood or `1.03` in a
positive mood. Ranking deduplicates by ID, retains the best duplicate, then sorts
by descending score and ascending Unicode code-point ID.

## Presence surfacing

Presence is a pure state-to-payload transform. The compact line uses this fixed
priority:

```text
reflection pending → still reflecting
due intention      → holding a follow-up
carried thought    → carrying a thread
open thread        → holding a pattern
otherwise          → here with you
```

External cognition is injected. Numeric display values are rounded to four
decimals, and carried-thought presence is derived from cleaned state rather than
trusted from a caller flag.

## Engine and trust invariants

- Responding turns run gate → optional retrieval → ranking → appraisal policy →
  response controls → generation → one atomic commit.
- Silent turns persist the user message without appraising or changing state.
- Gate errors and malformed outputs propagate unless a caller explicitly opts
  into a respond/silent fallback.
- Intents are trimmed, uppercased, and allowlisted. Custom labels must match
  `[A-Z][A-Z0-9_]{0,63}`.
- A store-owned transaction serializes one conversation across engine instances.
  Any pre-commit failure leaves state and transcript unchanged.
- Raw memory and carried-thought text never enters the default system prompt.
  It is bounded as untrusted evidence—2,000 characters per item and 8,000 total
  by default—and adapters must preserve that lower-trust channel on the wire.

Prompt prose, model output, embeddings, custom policies, application safety,
reflection, and language-level API shapes are outside cross-runtime parity.
