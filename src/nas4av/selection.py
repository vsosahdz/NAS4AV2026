"""The selection pipeline, with the signal it selects on made into an argument.

The prior campaign's results come out of six stages, and the test split enters at the
first one and never leaves:

```
① f(x) = 0.75·AUC(test) + 0.25·AUC(train)     ~6,000 evaluations per setting
② top-24 per run, by that f(x)                 5 runs → 120 models
③ re-evaluation after full training            same test split
④ top-5 of 120, per feature strategy           → 10 models
⑤ top-5 of C(10,5) = 252 ensembles             ranked on test
⑥ top-3 reported in Table 7                    ranked on test
```

Every stage from ② onward ranks on a quantity computed on the split the result is then
reported on. That is one defect repeated six times rather than six defects, and it is why
the repair is a parameter rather than a rewrite: the stages are correct, the signal they
rank by is not.

So `signal` is an argument everywhere, and the module exists to support two uses at once.
Set it to `test` and this reproduces the submitted Table 7, which is the only way to know
the reimplementation is faithful before trusting what it says under any other setting. Set
it to `train` and it reports what the same pipeline would have chosen without seeing the
test split — the difference between the two being selection optimism, which is the quantity
D3 exists to measure.

One asymmetry is forced by the artifacts rather than chosen. `ensemblesDF*.csv` carries no
fitness column, so training AUC cannot be recovered at stage ⑤ and the clean signal there
degrades to training accuracy. It is a weaker selector and it is the only one available;
the stage records which signal it actually used so that the degradation appears in the
output rather than in a footnote.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .prior.repository import Evaluation, ensembles, fully_trained, search_evaluations

#: Columns Table 7 reports, in its order.
REPORTED = ("test accuracy", "test precision", "test recall", "test fscore", "test auc")

#: How many ensembles Table 7 averages. Three, which is also why the submitted standard
#: deviations are the spread of three selected values rather than a dispersion.
TABLE7_TOP = 3


class UnknownSignal(ValueError):
    """Raised for a selection signal the stage cannot compute from its artifacts."""


def evaluation_signal(evaluation: Evaluation, signal: str) -> float:
    """Rank an architecture by the named signal.

    ``test`` is the submitted pipeline's choice and the one that makes a result an upper
    bound. ``train`` is the rung-B repair: recovered exactly by inverting the fitness, and
    honest but weak, because training AUC prefers architectures that overfit.
    """
    if signal == "test":
        return evaluation.test_auc
    if signal == "train":
        return evaluation.train_auc
    if signal == "fitness":
        return evaluation.fitness
    if signal == "training_accuracy":
        return evaluation.train_accuracy
    raise UnknownSignal(f"no evaluation signal named {signal!r}")


def ensemble_signal(row: dict[str, Any], signal: str) -> float:
    """Rank an ensemble by the named signal.

    Only two exist. The ensemble tables carry no fitness column, so ``train`` cannot be
    recovered here and asking for it is an error rather than a silent fallback — a stage
    that quietly substituted a different signal would make the optimism measurement
    compare two things that were never the same.
    """
    if signal == "test":
        return float(row["test auc"])
    if signal in {"train", "training_accuracy"}:
        return float(row["training accuracy"])
    raise UnknownSignal(f"no ensemble signal named {signal!r}")


def degraded_at_ensembles(signal: str) -> bool:
    """Whether the ensemble stage had to fall back to a weaker signal than was asked for."""
    return signal == "train"


@dataclass
class Selection:
    """What one run of the pipeline chose, and on what."""

    corpus: str
    signal: str
    ensemble_signal_used: str
    degraded: bool
    top_ensembles: list[dict[str, Any]]
    average: dict[str, float]
    spread: dict[str, float]
    pool: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "signal": self.signal,
            "ensemble_signal_used": self.ensemble_signal_used,
            "degraded_at_ensemble_stage": self.degraded,
            "ensembles_considered": self.pool,
            "selected": [row["models"] for row in self.top_ensembles],
            "average": {k: round(v, 6) for k, v in self.average.items()},
            "spread": {k: round(v, 6) for k, v in self.spread.items()},
            "spread_note": (
                "the spread of the selected top-three, not a dispersion over replicates; "
                "it narrows as selection gets more aggressive and must not be read as a "
                "standard error"
            ),
        }


def top_per_run(
    evaluations: Iterable[Evaluation], k: int = 24, signal: str = "test"
) -> list[Evaluation]:
    """Stage ②: the best ``k`` of each run, ranked by ``signal``.

    Per run rather than pooled, which matters more than it looks: pooling would let one
    run contribute all 120 architectures, and the five runs are what the manuscript treats
    as replication.
    """
    by_run: dict[int, list[Evaluation]] = {}
    for evaluation in evaluations:
        by_run.setdefault(evaluation.run, []).append(evaluation)
    chosen: list[Evaluation] = []
    for run in sorted(by_run):
        ranked = sorted(by_run[run], key=lambda e: evaluation_signal(e, signal), reverse=True)
        chosen.extend(ranked[:k])
    return chosen


def top_of_pool(
    evaluations: Iterable[Evaluation], k: int = 5, signal: str = "test"
) -> list[Evaluation]:
    """Stage ④: the best ``k`` of the fully trained pool, ranked by ``signal``."""
    return sorted(evaluations, key=lambda e: evaluation_signal(e, signal), reverse=True)[:k]


def rank_ensembles(
    rows: Sequence[dict[str, Any]], signal: str = "test", k: int = TABLE7_TOP
) -> list[dict[str, Any]]:
    """Stages ⑤ and ⑥: rank the 252 ensembles and keep the top ``k``.

    Ranked explicitly rather than sliced. The stored tables arrive pre-sorted by test
    score, so the published `tables.py` reaches a test-ranked selection by taking rows
    `[0:3]` without sorting anything — which reads like an arbitrary slice and is a
    selection on the reported split. Sorting here by a named signal is the same operation
    when the signal is `test`, and a different one otherwise, which is the point.
    """
    used = "test" if signal == "test" else "training_accuracy"
    ordered = sorted(rows, key=lambda row: ensemble_signal(row, signal), reverse=True)
    return ordered[:k]


def report(corpus: str, signal: str = "test", k: int = TABLE7_TOP) -> Selection:
    """Stage ⑥ for one corpus: the row Table 7 prints."""
    rows = ensembles(corpus)
    top = rank_ensembles(rows, signal=signal, k=k)
    average = {column: statistics.fmean(float(row[column]) for row in top) for column in REPORTED}
    spread = {
        column: (statistics.stdev(float(row[column]) for row in top) if len(top) > 1 else 0.0)
        for column in REPORTED
    }
    return Selection(
        corpus=corpus,
        signal=signal,
        ensemble_signal_used="test" if signal == "test" else "training_accuracy",
        degraded=degraded_at_ensembles(signal),
        top_ensembles=top,
        average=average,
        spread=spread,
        pool=len(rows),
    )


def optimism(corpus: str, k: int = TABLE7_TOP) -> dict[str, Any]:
    """The gap between selecting on test and selecting without it, for one corpus.

    The quantity D3 exists to report. Both arms are evaluated on the same test split and
    differ only in what chose the ensembles, so the difference is attributable to the
    selection signal rather than to the models.

    The clean arm here is not clean in the full sense and must not be reported as though it
    were: stages ② through ④ of the legacy campaign already filtered on test before these
    ensembles existed, so this measures the optimism of the *last* stage over a pool that
    was itself test-selected. It is a lower bound on the total, and the campaign re-run is
    what measures the rest.
    """
    selected_on_test = report(corpus, signal="test", k=k)
    selected_without = report(corpus, signal="train", k=k)
    return {
        "corpus": corpus,
        "selected_on_test": selected_on_test.as_dict(),
        "selected_without_test": selected_without.as_dict(),
        "gap": {
            column: round(selected_on_test.average[column] - selected_without.average[column], 6)
            for column in REPORTED
        },
        "same_ensembles": [row["models"] for row in selected_on_test.top_ensembles]
        == [row["models"] for row in selected_without.top_ensembles],
        "bound": (
            "a lower bound on total selection optimism: stages 2-4 of the legacy campaign "
            "had already selected on test before these ensembles were formed, so only the "
            "final stage's contribution is measured here"
        ),
    }


def legacy_arm(corpus: str, strategy: str, signal: str = "train") -> dict[str, Any]:
    """Stages ② and ④ over the legacy search database, at the rung the audit reached.

    Reported separately from the ensemble stage because the two reach different rungs:
    training AUC is recoverable exactly for every search row and not at all for an
    ensemble, so a single figure spanning both would hide which half is weaker.
    """
    pool = top_per_run(search_evaluations(corpus, strategy), k=24, signal=signal)
    trained = list(fully_trained(corpus))
    finalists = top_of_pool(trained, k=5, signal=signal) if trained else []
    return {
        "corpus": corpus,
        "strategy": strategy,
        "signal": signal,
        "stage_2_pool": len(pool),
        "stage_4_finalists": [
            {
                "encoding": e.encoding,
                "test_auc": round(e.test_auc, 6),
                "train_auc": round(e.train_auc, 6),
            }
            for e in finalists
        ],
        "fully_trained_available": len(trained),
    }
