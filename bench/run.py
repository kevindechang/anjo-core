#!/usr/bin/env python3
"""Retrieval-scoring benchmark: does the kernel's extra machinery earn its place?

Run it:

    python bench/run.py                # print the report
    python bench/run.py --write        # regenerate bench/RESULTS.md
    python bench/run.py --check        # fail if bench/RESULTS.md is stale

Deterministic, dependency-free, and seeded. Read bench/README.md for what the
numbers can and cannot support.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from corpus import Query, Regime, build_queries
from scorers import (
    NamedScorer,
    additive_park,
    kernel_default,
    kernel_no_salience,
    kernel_with_congruence,
    multiplicative_exponential,
    multiplicative_power_law,
    similarity_only,
)

RESULTS = Path(__file__).resolve().parent / "RESULTS.md"
SEED = 20260820
QUERY_COUNT = 400
POOL_SIZE = 20


@dataclass(frozen=True, slots=True)
class Metrics:
    recall_at_1: float
    recall_at_5: float
    mrr: float

    def row(self) -> str:
        return f"{self.recall_at_1:.3f} | {self.recall_at_5:.3f} | {self.mrr:.3f}"


def evaluate(scorer: NamedScorer, queries: list[Query]) -> Metrics:
    hits_1 = hits_5 = 0
    reciprocal = 0.0
    for query in queries:
        ranked = sorted(
            query.candidates,
            key=lambda c: (-scorer.fn(c, query.ages, query.mood_valence), c.id),
        )
        rank = next(i for i, c in enumerate(ranked, start=1) if c.id == query.gold_id)
        hits_1 += rank == 1
        hits_5 += rank <= 5
        reciprocal += 1.0 / rank
    n = len(queries)
    return Metrics(hits_1 / n, hits_5 / n, reciprocal / n)


def table(scorers: list[NamedScorer], queries: list[Query], baseline: str) -> list[str]:
    rows = [
        "| Scorer | recall@1 | recall@5 | MRR | vs baseline |",
        "|---|---:|---:|---:|---|",
    ]
    scores = {s.name: evaluate(s, queries) for s in scorers}
    base = scores[baseline].mrr
    for scorer in scorers:
        metrics = scores[scorer.name]
        delta = metrics.mrr - base
        verdict = "baseline" if scorer.name == baseline else f"{delta:+.3f} MRR"
        rows.append(f"| `{scorer.name}` | {metrics.row()} | {verdict} |")
    return rows


EXPERIMENTS: list[tuple[str, Regime, str, list[NamedScorer], str]] = [
    (
        "A. Nuisance signals carry no information",
        "uncorrelated",
        """Age, significance, rehearsal, and episode status are drawn independently of
which memory is correct. Any scorer that weights them is adding noise, so plain
similarity should win. This is the experiment that can embarrass the kernel, and
it is reported first for that reason.""",
        [
            NamedScorer("similarity-only", "", similarity_only),
            NamedScorer("kernel-default", "", kernel_default),
            NamedScorer("kernel-no-salience", "", kernel_no_salience),
            NamedScorer("additive-park", "", additive_park),
        ],
        "similarity-only",
    ),
    (
        "B. Salience is informative, recency is not",
        "correlated",
        """Important, rehearsed, episodic memories genuinely are likelier answers — the
assumption the kernel's salience term encodes. Age remains uninformative here, so
this isolates salience.""",
        [
            NamedScorer("similarity-only", "", similarity_only),
            NamedScorer("kernel-default", "", kernel_default),
            NamedScorer("kernel-no-salience", "", kernel_no_salience),
            NamedScorer("additive-park", "", additive_park),
        ],
        "similarity-only",
    ),
    (
        "C. Recency-curve shootout",
        "recency_informative",
        """Age genuinely predicts relevance here, so a recency term should help. All three
curves are pinned to the same 30-day half-life, which makes this a comparison of
shape rather than of scale.""",
        [
            NamedScorer("similarity-only", "", similarity_only),
            NamedScorer("kernel-default (linear to a floor)", "", kernel_default),
            NamedScorer(
                "exponential (Park et al. shape)", "", multiplicative_exponential
            ),
            NamedScorer(
                "power-law (Wixted & Ebbesen shape)", "", multiplicative_power_law
            ),
        ],
        "kernel-default (linear to a floor)",
    ),
    (
        "D. Both signals informative — the kernel's own assumed conditions",
        "both_informative",
        """Salience *and* recency both predict relevance. Experiments A-C each leave one of
the kernel's assumptions violated; this is the only regime in which all of them
hold, so it is the fairest test of the design as intended.""",
        [
            NamedScorer("similarity-only", "", similarity_only),
            NamedScorer("kernel-default", "", kernel_default),
            NamedScorer("kernel-no-salience", "", kernel_no_salience),
            NamedScorer("additive-park", "", additive_park),
        ],
        "similarity-only",
    ),
    (
        "E. Mood-congruence ablation",
        "mood_informative",
        """Memories sharing the sign of the current mood are 75% likely to be the answer — a
regime deliberately favourable to the congruence multiplier. If the 1.06/1.03
asymmetry cannot help here, it cannot help anywhere.""",
        [
            NamedScorer("kernel-default (no congruence)", "", kernel_default),
            NamedScorer("kernel + congruence", "", kernel_with_congruence),
        ],
        "kernel-default (no congruence)",
    ),
]


SWEEP_VALUES = (0.03, 0.1, 0.25, 0.5, 1.0)


def significance_sweep() -> list[str]:
    """Why does the additive form win? Sweep the one weight that differs most."""
    from affect_kernel import RetrievalWeights, candidate_score

    queries = build_queries(
        regime="both_informative",
        query_count=QUERY_COUNT,
        pool_size=POOL_SIZE,
        seed=SEED,
    )
    rows = ["| `significance_weight` | recall@1 | MRR |", "|---|---:|---:|"]
    for value in SWEEP_VALUES:
        weights = RetrievalWeights(significance_weight=value)
        hits = 0
        reciprocal = 0.0
        for query in queries:
            ranked = sorted(
                query.candidates,
                key=lambda c: (
                    -candidate_score(
                        c.distance,
                        query.ages[c.id],
                        episode=c.episode,
                        significance=c.significance,
                        recall_count=c.recall_count,
                        weights=weights,
                    ),
                    c.id,
                ),
            )
            rank = next(
                i for i, c in enumerate(ranked, start=1) if c.id == query.gold_id
            )
            hits += rank == 1
            reciprocal += 1.0 / rank
        marker = " (current default)" if value == 0.03 else ""
        rows.append(
            f"| {value}{marker} | {hits / len(queries):.3f} | {reciprocal / len(queries):.3f} |"
        )
    return rows


def report() -> str:
    lines = [
        "# Retrieval benchmark results",
        "",
        "<!-- Generated by bench/run.py. Do not edit by hand; CI regenerates and",
        "     fails on drift. Read bench/README.md for method and limitations. -->",
        "",
        f"Seed `{SEED}` · {QUERY_COUNT} queries per regime · {POOL_SIZE} candidates per query ·",
        "one correct memory per query.",
        "",
        "**These are synthetic corpora with machine-assigned ground truth.** They test",
        "whether a ranking function recovers a known signal under a stated generative",
        "model. They say nothing about real user memories. See the limitations in",
        "`bench/README.md` before quoting any number here.",
        "",
    ]
    for title, regime, blurb, scorers, baseline in EXPERIMENTS:
        queries = build_queries(
            regime=regime, query_count=QUERY_COUNT, pool_size=POOL_SIZE, seed=SEED
        )
        lines += [f"## {title}", "", blurb, ""]
        lines += table(scorers, queries, baseline)
        lines += [""]

    lines += [
        "## F. Why the additive form wins: a significance-weight sweep",
        "",
        "Experiments B and D show `additive-park` beating the kernel wherever salience",
        "carries signal. The two differ in composition, but they also differ in *weight*:",
        "significance enters the kernel's score at `0.03` (a 3% ceiling) and enters Park",
        "et al.'s sum at `1.0`. Sweeping that one parameter separates the two",
        "explanations. Regime D, everything else held at the default.",
        "",
    ]
    lines += significance_sweep()
    lines += [""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="regenerate bench/RESULTS.md"
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if RESULTS.md is stale"
    )
    args = parser.parse_args()

    rendered = report()
    if args.write:
        RESULTS.write_text(rendered, encoding="utf-8")
        print(f"wrote {RESULTS}")
        return 0
    if args.check:
        if not RESULTS.exists():
            print("bench/RESULTS.md is missing; run: python bench/run.py --write")
            return 1
        if RESULTS.read_text(encoding="utf-8") != rendered:
            print("bench/RESULTS.md is stale; run: python bench/run.py --write")
            return 1
        print("bench/RESULTS.md is current")
        return 0
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
