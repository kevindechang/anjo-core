# Retrieval benchmark

```bash
python bench/run.py            # print the report
python bench/run.py --write    # regenerate RESULTS.md
python bench/run.py --check    # fail if RESULTS.md is stale (CI runs this)
```

Deterministic, dependency-free, seeded. [RESULTS.md](RESULTS.md) is generated,
and CI fails if it drifts from what the code produces, so the numbers quoted in
the documentation cannot go stale.

## The question

`docs/foundations.md` labels most of the retrieval constants **P** —
production-tuned, never ablated. This benchmark asks the first three questions
from that document's falsification list:

1. Does the kernel's extra machinery — recency, salience, the episode bonus —
   actually beat plain similarity?
2. Does the multiplicative composition beat the additive weighted sum used by
   Generative Agents [Park et al. 2023], the closest published comparable?
3. Is linear-to-a-floor recency worse than the exponential and power-law curves,
   as `foundations.md` asserted?

And one more it raises: does the mood-congruence multiplier do anything at all?

## Method

Five regimes, each 400 queries × 20 candidates, exactly one correct memory per
query. **Relevance is assigned before any feature is drawn**, so ground truth is
independent of every formula under test. Similarity distributions for correct
and incorrect memories overlap deliberately; separable distributions would make
every scorer perfect and measure nothing.

| Regime | Salience predicts relevance | Age predicts relevance |
|---|---|---|
| A `uncorrelated` | no | no |
| B `correlated` | yes | no |
| C `recency_informative` | no | yes |
| D `both_informative` | yes | yes |
| E `mood_informative` | valence matches mood 75% of the time | no |

Regime A exists so the benchmark can embarrass the kernel: when the nuisance
signals carry no information, anything that weights them is adding noise and
should lose to plain similarity. Regime D is the only one in which all of the
kernel's assumptions hold at once, and is therefore the fairest test of the
design as intended.

All three recency curves are pinned to a **common 30-day half-life**, so
experiment C compares the shape of forgetting rather than an arbitrary
difference in scale.

## What the results say

Read [RESULTS.md](RESULTS.md) for the tables. The findings:

**The machinery is not free, and it is not always worth it.** In regime A the
kernel loses to plain similarity by 0.175 MRR. That is the correct behavior for
a scorer built on assumptions that regime deliberately violates, but it means
"add the kernel's scorer" is not unconditionally good advice. Applications whose
memory salience is uninformative should turn the salience and recency terms down
or off — which the `RetrievalWeights` seam now makes possible without forking.

**Where its assumptions hold, it helps a great deal.** In regime D the kernel
beats similarity-only by 0.415 MRR (0.858 vs 0.442), and dropping the salience
term costs 0.052 MRR. The design is doing real work.

**The additive form beats the multiplicative form everywhere signal exists** —
by 0.200 MRR in regime B and 0.111 in regime D. This is a result against the
current design, and it confirms falsification item (3) in `foundations.md`.

**But the composition is not the cause.** Experiment F separates the two
explanations. Significance enters the kernel's score at weight `0.03` and enters
Park et al.'s sum at `1.0`. Raising that single parameter — with the
multiplicative composition untouched — lifts regime-D MRR from 0.858 to 0.960,
against the additive form's 0.968. **The kernel's salience term is
underpowered, not misshapen.** Its 3% ceiling cannot express an informative
salience signal, while its recency multiplier ranges over 0.4–1.0 and can
inject far more noise than salience can remove.

**The linear recency curve is not the weak point.** `foundations.md` called
linear-to-a-floor "the least defensible curve in the module". At matched
half-life it beats the exponential curve by 0.009 MRR and the power-law curve by
0.044. That claim was wrong under this test and has been corrected.

**Mood congruence barely registers.** In a regime built to favour it — correct
memories share the mood's sign 75% of the time — it is worth +0.012 MRR. It does
not hurt, and it does not earn its unexplained asymmetry.

## Limitations — read before quoting any number

- **The corpora are synthetic and the ground truth is machine-assigned.** These
  experiments measure whether a ranking function recovers a signal under a
  stated generative model. They are not evidence about real user memories,
  real embeddings, or real conversation.
- **The generative model is ours.** Relevance is assigned independently of the
  scoring formulas, which is what makes an unflattering result possible, but the
  distributions themselves are chosen. A different similarity overlap or age
  distribution could move the margins, and experiment C in particular depends on
  the age distribution used.
- **Similarity is drawn, not embedded.** No embedding model is involved, so
  nothing here speaks to retrieval quality end to end.
- **One correct memory per query.** Real retrieval has graded, multiple
  relevance; recall@k and MRR under a single gold are a simplification.
- **No language model is involved anywhere.** This benchmark says nothing about
  response quality, persona consistency, or long-horizon character drift. Those
  remain unevaluated; see the open items below.

## Still unevaluated

Falsification items (1) and (5) from `docs/foundations.md` are untouched:
whether wall-clock decay beats per-turn decay, and whether Openness,
Conscientiousness, or Agreeableness carry signal if wired into inertia. Both
need affect trajectories with external ground truth, which this repository does
not have. The headline claim in the README — that a deterministic kernel gives
more consistent character state than a prompt-only persona — is **not tested
here and remains unsupported.**

## References

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein,
M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST
'23*. <https://doi.org/10.1145/3586183.3606763>

Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting.
*Psychological Science, 2*(6), 409–415.
<https://doi.org/10.1111/j.1467-9280.1991.tb00175.x>
