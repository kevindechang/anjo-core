# Foundations: where the numbers come from

Every mechanism in this kernel is either taken from published work, fitted
against a production deployment, or chosen arbitrarily inside a bound. This
document says which is which, for every constant, without flattering the
project.

Read it as a provenance record, not as a validation study. Nothing here has
been tested against human affect data. A citation next to a mechanism means
"this is the idea we implemented," never "this implementation has been shown to
reproduce that result."

## Provenance classes

Every constant in the tables below carries one of three tags.

| Tag | Meaning |
|---|---|
| **L** — literature | The *form* or the *sign* of the relationship is taken from published work, cited inline. The specific magnitude usually is not. |
| **P** — production-tuned | Hand-tuned against the [Anjo](https://anjo.love) deployment until behavior looked right to its maintainer. Not fitted to a dataset, not ablated, not externally validated. |
| **B** — bounded choice | Arbitrary. The only load-bearing property is that the value is finite, stable, and inside a stated bound. A different value in the same range would be equally defensible. |

There are more **P** and **B** rows than **L** rows. That is the honest state of
the artifact, and it is the reason [the evaluation](../bench/README.md) matters
more than the citation list. Three of the five falsification items below have
now been run; two have not, and are marked as such.

## 1. The state space is PAD, not a discrete emotion set

Mood is a point in three bounded dimensions — pleasure/valence, arousal, and
dominance — rather than one of *n* named emotions. This follows Mehrabian's PAD
temperament model [Mehrabian 1996] and, for the two-dimensional core, Russell's
circumplex [Russell 1980].

Octant labels (`exuberant`, `dependent`, `relaxed`, `docile`, `hostile`,
`anxious`, `disdainful`, `bored`) are the sign-partition naming used by ALMA
[Gebhard 2005], which is also where the idea of layering a fast *emotion* signal
over a slow *mood* point comes from.

| Constant | Value | Tag | Note |
|---|---|---|---|
| PAD bounds | `[-1, 1]` per axis | **L** | Mehrabian's dimensions are bipolar and bounded. |
| Octant label set | 8 names | **L** | Gebhard 2005, ALMA. |
| Neutral deadband | `0.15` on all three axes | **B** | ALMA has no deadband. Ours exists so a near-zero mood does not flip labels on rounding noise. Any small value works. |

**Departure.** ALMA derives a character's *default* mood point from Big Five
traits using Mehrabian's regression equations. This kernel does not: the
resting point comes from a relationship-stage weight and a slow valence
baseline instead (§3). Trait-to-default-mood mapping is unimplemented, not
rejected.

## 2. Mood relaxes toward an attractor (AR(1))

```text
rest       = (stage_weight × baseline_valence, 0, stage_weight × 0.10)
next_mood  = clamp(rest + phi × (current_mood − rest), −1, 1)
```

This is a first-order autoregressive pull toward a set point. It corresponds
directly to the DynAffect model of core affect [Kuppens, Oravecz & Tuerlinckx
2010], in which affect is described by a *home base*, an *attractor strength*
pulling back to it, and variability around it. Our `rest` is their home base and
our `phi` is `1 −` their attractor strength. Exponential decay of mood toward a
baseline is also how ALMA [Gebhard 2005] and WASABI [Becker-Asano & Wachsmuth
2010] move mood between events.

| Constant | Value | Tag | Note |
|---|---|---|---|
| Update form | AR(1) toward `rest` | **L** | Kuppens et al. 2010; Gebhard 2005. |
| Resting arousal | `0` | **B** | Assumes a calm home base. Not measured. |
| Dominance resting coefficient | `0.10 × stage_weight` | **P** | Encodes "familiarity raises baseline dominance a little." Magnitude is arbitrary. |
| Rounding | 4 decimals | **B** | Cross-runtime determinism, not a modeling claim. |

**Departure.** DynAffect fits its parameters per person from experience-sampling
data. Ours are fixed constants. The kernel borrows the shape of the model and
none of its estimation.

## 3. Personality conditions inertia — only N and E

```text
phi = clamp(0.80 + 0.20 × (N − 0.5) − 0.10 × (E − 0.5), 0.62, 0.92)
```

Higher Neuroticism ⇒ higher `phi` ⇒ mood carries further. Higher Extraversion ⇒
lower `phi` ⇒ mood returns to baseline faster.

The **direction of both effects** is literature-supported. Emotional inertia —
the autocorrelation of affect over time — is elevated in people with lower
psychological adjustment and higher negative-affect tendencies [Kuppens, Allen &
Sheeber 2010]. Extraversion is associated with greater reactivity to positive
affect induction [Larsen & Ketelaar 1991], which in an AR(1) formulation
corresponds to a weaker carryover term.

The **magnitudes are not**. `0.20` and `0.10` are hand-chosen, and no study
licenses that N's effect is exactly twice E's.

| Constant | Value | Tag | Note |
|---|---|---|---|
| Sign of the N term | positive | **L** | Kuppens, Allen & Sheeber 2010. |
| Sign of the E term | negative | **L** | Larsen & Ketelaar 1991. |
| Base inertia | `0.80` | **P** | Sets the per-turn half-life. At the default personality (N `0.15`, E `0.45`) `phi` is `0.735`, a mood half-life of 2.25 turns. |
| N coefficient | `0.20` | **P** | |
| E coefficient | `0.10` | **P** | |
| Clamp | `[0.62, 0.92]` | **B** | Keeps mood neither frozen nor amnesiac at extreme traits. |

**Departure.** Openness, Conscientiousness, and Agreeableness are accepted,
validated, stored, and then ignored by every transform. They are present for
callers and future work. This is a modeling gap, not a finding.

**Second departure.** `phi` is applied per *turn*, not per unit of *time*. Two
turns a minute apart and two turns a week apart decay identically. Real affect
dynamics are continuous-time. A time-aware `phi` is the single most defensible
improvement available to this file.

## 4. Appraisal is OCC-shaped but skips OCC's appraisal variables

The emotion vocabulary — joy, distress, admiration, reproach, gratitude — is a
subset of the OCC taxonomy [Ortony, Clore & Collins 1988]. Gratitude is
correctly treated as an OCC compound (approval of another's act plus a desirable
outcome), and reproach/admiration correctly sit on the praiseworthiness branch.

**This is where the kernel departs most sharply from the literature it names.**
OCC generates emotions by evaluating events against *appraisal variables* —
desirability, praiseworthiness, appealingness. Computational OCC systems such as
EMA [Marsella & Gratch 2009] and FAtiMA [Dias, Mascarenhas & Paiva 2014] compute
those variables from an explicit representation of goals, plans, and standards.

This kernel does none of that. It takes a **pre-classified intent label** —
supplied by an application, typically by a language model — and looks up a fixed
PAD impulse and a fixed set of emotion coefficients. The `AppraisalGoals` weights
(`rapport`, `respect`, `honesty`, `intellectual`) scale those coefficients, which
is a thin gesture at OCC's goal structure, not an implementation of it.

The honest description is: **an OCC-flavored lookup table with goal-weighted
intensities.** It is not an appraisal engine. Callers who need real appraisal
should inject their own `AppraisalPolicy` and treat the bundled one as a
reference shape.

| Constant | Value | Tag | Note |
|---|---|---|---|
| Emotion names | OCC subset | **L** | Ortony, Clore & Collins 1988. |
| Intent → PAD impulse table | 7 intents | **P** | Every delta is hand-tuned. See [algorithm.md](algorithm.md) for the table. |
| Goal coefficients (`.95`, `.75`, `.70`, …) | per intent | **P** | |
| Ambiguous-intent amplification | `×1.10` negative, `×1.04` positive, above `|v| ≥ 0.20` | **P** | Undocumented before this release; widens already-polarized valence on ambiguous intents. |
| Baseline valence blend | `0.98 × old + 0.02 × new` | **P** | ≈ 34-turn half-life. A slow trait-like drift under a fast state. |

## 5. Emotion carry decays per-emotion

```text
carry[e] ← carry[e] × rate[e],  dropped at or below 0.05
```

Per-emotion decay rates, with negative social emotions fading fastest:

| Emotion | Rate | Turns to half | Tag |
|---|---:|---:|---|
| reproach | `0.70` | 1.9 | **P** |
| distress | `0.80` | 3.1 | **P** |
| admiration | `0.85` | 4.3 | **P** |
| gratitude | `0.88` | 5.4 | **P** |
| joy | `0.90` | 6.6 | **P** |
| anything else | `0.80` | 3.1 | **B** |

Decaying emotion faster than mood is the two-layer structure ALMA [Gebhard 2005]
and WASABI [Becker-Asano & Wachsmuth 2010] both use, and that part is **L**.
The rate *ordering* — that a companion should let reproach go before it lets joy
go — is a product value judgment made at Anjo, not a psychological finding. It
is stated here so that anyone who disagrees can see exactly what they are
changing.

| Constant | Value | Tag | Note |
|---|---|---|---|
| Two-layer emotion/mood split | — | **L** | Gebhard 2005; Becker-Asano & Wachsmuth 2010. |
| Drop floor | `0.05` | **B** | Keeps the carry map from filling with dust. |
| Rate ordering | reproach < … < joy | **P** | A design stance, not evidence. |

## 6. Retrieval scoring

```text
similarity = 1 − distance / 2
recency    = clamp(1 − age_days / 60, 0.40, 1)
salience   = 1 + 0.03 × significance + min(0.025, 0.006 × ln(1 + recall_count))
score      = similarity × recency × salience + 0.05·[episode]
```

The three-factor shape — relevance × recency × importance — is the same
decomposition used by Generative Agents [Park et al. 2023], which is the closest
published comparable.

**Two departures from that paper, both deliberate:**

1. **Multiplicative, not additive.** Park et al. use a weighted *sum* of
   normalized recency, importance, and relevance. This kernel *multiplies*. A
   product means a memory that is irrelevant cannot be rescued by being recent,
   which we wanted; it also means the factors are not independently
   interpretable, which is a real cost.
2. **Linear recency, not exponential.** Park et al. decay recency
   exponentially (`0.995^hours`). Human forgetting is better described by a
   power law than by either shape [Wixted & Ebbesen 1991]. Ours is linear to a
   floor, chosen because it is trivially inspectable and because the `0.40`
   floor matters more in practice than the curve between.

   An earlier revision of this document called the linear curve "the least
   defensible of the three". [The benchmark](../bench/README.md) does not
   support that: at a matched 30-day half-life it beats the exponential curve by
   0.009 MRR and the power-law curve by 0.044. The claim was retracted rather
   than quietly softened.

The rehearsal term is motivated by the testing effect — retrieval practice
strengthens later retrieval [Roediger & Karpicke 2006] — but `0.006 × ln(1+n)`
capped at `0.025` is a token gesture at that literature, not a fit to it. At its
cap it moves a score by 2.5%.

| Constant | Value | Tag | Note |
|---|---|---|---|
| relevance × recency × importance | — | **L** | Park et al. 2023 (shape only; they sum). |
| `similarity = 1 − d/2` | — | **B** | Maps the `[0, 2]` cosine-distance convention to `[0, 1]`. |
| Recency horizon | `60` days | **P** | |
| Recency floor | `0.40` | **P** | Old memories stay reachable rather than vanishing. |
| Unparseable-timestamp fallback | `0.70` | **B** | Deliberately better than the floor: a broken timestamp should not be treated as ancient. |
| Significance weight | `0.03` | **P** | |
| Rehearsal weight / cap | `0.006 × ln(1+n)`, cap `0.025` | **P** | Roediger & Karpicke 2006 motivates the sign only. The cap binds at 64 recalls. |
| Episode bonus | `0.05` | **P** | Additive, so it can outweigh the entire salience term. Known wart. |
| Default `limit` | `4` | **P** | |

## 7. Mood-congruent retrieval

```text
if congruence enabled and |mood_valence| ≥ 0.20 and sign(memory) == sign(mood):
    score ×= 1.06 in a negative mood, 1.03 in a positive mood
```

Mood-congruent recall — that affect biases which memories come back — is Bower's
[Bower 1981]. That the mechanism *exists* is **L**.

Everything else about it here is **P**. The `0.20` activation threshold, the
`1.06`/`1.03` magnitudes, and in particular the **asymmetry** — a stronger pull
in a negative mood than a positive one — are product choices. No citation in
this document supports that asymmetry.

It is easy to opt out, and easy to opt into by accident. `candidate_score()`
has no congruence term at all. `score_candidate()` accepts one but defaults
`mood_valence` to `0.0`, so a caller who never passes a mood gets a factor of
exactly `1.0`; the term only engages once a caller supplies `|mood_valence| ≥
0.20`.

## 8. Decoder controls have no literature behind them at all

```text
temperature = clamp(1 + 0.20 × arousal, 0.72, 1.18),  top_p = 0.97
length factor = 0.60 if arousal < −0.30 else 0.72 if valence < −0.30 else 1
```

There is no published basis for mapping arousal onto a softmax temperature.
This is an engineering convention: it makes an aroused character sample a little
more loosely and a withdrawn one answer a little more briefly. It is included
because it is the point where affect state becomes observable in output, and it
is bounded so that it can never make a model incoherent.

| Constant | Value | Tag |
|---|---|---|
| Temperature slope `0.20`, clamp `[0.72, 1.18]` | — | **B** |
| `top_p = 0.97` | — | **B** |
| Length factors `0.60` / `0.72`, thresholds `−0.30` | — | **P** |
| Token floor `180` | — | **P** |
| Ambivalence thresholds `0.15`, ratio `0.40` | — | **B** |

## 9. Relationship stages and presence

Stage weights `(0, 0.20, 0.40, 0.60, 0.70)`, the five rung names, the presence
priority cascade, and every surfaced string are **P** — they come from one
product's design. The kernel treats them as replaceable data precisely because
they carry no general claim; see
[design principles](design-principles.md#domain-vocabulary-is-data-not-behavior)
and `examples/game-npc/` for a full replacement.

## What would falsify these choices

Concrete results that should change the code, not just the prose:

1. **Time-aware decay beats per-turn decay.** If mood trajectories under a
   wall-clock `phi` track human affect ratings better than per-turn `phi`, §3
   is wrong in form, not just in magnitude.
2. **Power-law recency beats linear-with-floor.** A retrieval evaluation where
   `recency = (1 + age)^−β` outperforms the current curve would make §6.2 a bug.
3. **Additive beats multiplicative scoring.** Reproducing Park et al.'s weighted
   sum and winning would remove our main departure from the closest comparable.
4. **The congruence asymmetry does nothing.** If ablating the `1.06`/`1.03`
   split changes no downstream metric, it should be deleted rather than kept as
   an unexplained constant.
5. **O, C, A carry signal.** If any of the three ignored traits improves a
   consistency metric when wired into inertia, §3's restriction to N and E is a
   loss, not a simplification.

Items (2), (3), and (4) have been run — see [the benchmark](../bench/README.md)
and its [results](../bench/RESULTS.md):

- **(2) is answered, against the prediction.** Linear-to-a-floor recency is not
  the weak point; at matched half-life it edges out both alternatives.
- **(3) is answered, against the current design.** Park et al.'s additive form
  beats the multiplicative form wherever salience carries signal — but the
  composition is not the cause. Significance enters this scorer at weight
  `0.03` and Park's at `1.0`; raising that single parameter closes almost the
  whole gap (MRR 0.858 → 0.960 against 0.968) with the multiplicative form
  untouched. **The salience term is underpowered, not misshapen**, and
  `significance_weight` is the constant with the strongest case for changing.
- **(4) is answered weakly.** In a regime built to favour it, mood congruence is
  worth +0.012 MRR. It does not hurt and it does not earn its asymmetry.

Items (1) and (5) have **not** been run: both need affect trajectories with
external ground truth this repository does not have. Treat every **P** row not
covered above as an unfalsified design choice rather than a result.

## References

Becker-Asano, C., & Wachsmuth, I. (2010). Affective computing with primary and
secondary emotions in a virtual human. *Autonomous Agents and Multi-Agent
Systems, 20*(1), 32–49. <https://doi.org/10.1007/s10458-009-9094-9>

Bower, G. H. (1981). Mood and memory. *American Psychologist, 36*(2), 129–148.
<https://doi.org/10.1037/0003-066X.36.2.129>

Dias, J., Mascarenhas, S., & Paiva, A. (2014). FAtiMA Modular: Towards an agent
architecture with a generic appraisal framework. In *Emotion Modeling* (LNCS
8750, pp. 44–56). <https://doi.org/10.1007/978-3-319-12973-0_3>

Gebhard, P. (2005). ALMA: A layered model of affect. In *Proceedings of the
Fourth International Joint Conference on Autonomous Agents and Multiagent
Systems (AAMAS '05)* (pp. 29–36). <https://doi.org/10.1145/1082473.1082478>

Kuppens, P., Allen, N. B., & Sheeber, L. B. (2010). Emotional inertia and
psychological maladjustment. *Psychological Science, 21*(7), 984–991.
<https://doi.org/10.1177/0956797610372634>

Kuppens, P., Oravecz, Z., & Tuerlinckx, F. (2010). Feelings change: Accounting
for individual differences in the temporal dynamics of affect. *Journal of
Personality and Social Psychology, 99*(6), 1042–1060.
<https://doi.org/10.1037/a0020962>

Larsen, R. J., & Ketelaar, T. (1991). Personality and susceptibility to positive
and negative emotional states. *Journal of Personality and Social Psychology,
61*(1), 132–140. <https://doi.org/10.1037/0022-3514.61.1.132>

Marsella, S. C., & Gratch, J. (2009). EMA: A process model of appraisal
dynamics. *Cognitive Systems Research, 10*(1), 70–90.
<https://doi.org/10.1016/j.cogsys.2008.03.005>

Mehrabian, A. (1996). Pleasure-arousal-dominance: A general framework for
describing and measuring individual differences in temperament. *Current
Psychology, 14*(4), 261–292. <https://doi.org/10.1007/BF02686918>

Ortony, A., Clore, G. L., & Collins, A. (1988). *The cognitive structure of
emotions*. Cambridge University Press.
<https://doi.org/10.1017/CBO9780511571299>

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein,
M. S. (2023). Generative agents: Interactive simulacra of human behavior. In
*Proceedings of the 36th Annual ACM Symposium on User Interface Software and
Technology (UIST '23)* (pp. 1–22).
<https://doi.org/10.1145/3586183.3606763> · <https://arxiv.org/abs/2304.03442>

Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory
tests improves long-term retention. *Psychological Science, 17*(3), 249–255.
<https://doi.org/10.1111/j.1467-9280.2006.01693.x>

Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and
Social Psychology, 39*(6), 1161–1178. <https://doi.org/10.1037/h0077714>

Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting.
*Psychological Science, 2*(6), 409–415.
<https://doi.org/10.1111/j.1467-9280.1991.tb00175.x>

### Related systems referenced elsewhere in this repository

Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., &
Gonzalez, J. E. (2023). MemGPT: Towards LLMs as operating systems.
<https://arxiv.org/abs/2310.08560>

Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., & Yu, D. (2024).
LongMemEval: Benchmarking chat assistants on long-term interactive memory.
<https://arxiv.org/abs/2410.10813>
