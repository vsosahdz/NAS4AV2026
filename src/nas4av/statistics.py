"""Comparisons that declare their structure, and intervals that admit their width.

The submitted manuscript ran ten Mann-Whitney tests across five corpora and two comparison
families with no multiplicity correction, over samples of three results that had been
selected for being the best. Every part of that is replaced here, and the replacement is
mostly about saying what is being compared before computing anything.

Three rules the submitted version broke, each enforced rather than remembered.

**Selected results are not a sample.** The top three ensembles are the tip of a
distribution chosen for its tip; their spread is not variability and their mean is not an
estimate. Dispersion comes from a declared replicate structure or it is not reported.

**Repeated tests need a correction.** Pairwise comparisons across corpora multiply, and an
uncorrected family of ten tests at 0.05 has roughly a two-in-five chance of a false
positive somewhere.

**An interval is part of a number.** On thirty test pairs a difference of 0.03 is inside
the noise, and reporting the point estimate alone invites a reader to believe otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import stats

#: Resamples for a bootstrap interval. Enough that the 2.5th and 97.5th percentiles are
#: stable to the third decimal, which is finer than any difference this work reports.
BOOTSTRAP_RESAMPLES = 10_000

#: Blocks below which a Friedman test is reported as underpowered rather than trusted.
#: Five is the conventional floor in the evolutionary-computation literature, and this
#: study has five corpora, so the question of whether settings or corpora are the blocks
#: decides whether any omnibus test here is usable at all.
MINIMUM_BLOCKS = 5


class ComparisonStructureError(ValueError):
    """Raised when a comparison is asked for without a usable structure."""


@dataclass(frozen=True)
class Interval:
    """A point estimate with the width that qualifies it."""

    estimate: float
    low: float
    high: float
    resamples: int
    units: int
    metric: str

    @property
    def width(self) -> float:
        return self.high - self.low

    def covers(self, value: float) -> bool:
        """Whether a reference value lies inside — the no-discrimination point, usually."""
        return self.low <= value <= self.high

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "estimate": round(self.estimate, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
            "width": round(self.width, 4),
            "resamples": self.resamples,
            "evaluation_units": self.units,
            "covers_chance": self.covers(0.5),
        }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([positive, negative]))
    rank_sum = ranks[: positive.size].sum()
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2) / (positive.size * negative.size)
    )


def _balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    predicted = (scores > 0.0).astype(int)
    sensitivity = (
        float(np.mean(predicted[labels == 1] == 1)) if np.any(labels == 1) else float("nan")
    )
    specificity = (
        float(np.mean(predicted[labels == 0] == 0)) if np.any(labels == 0) else float("nan")
    )
    return (sensitivity + specificity) / 2


METRICS = {"roc_auc": _roc_auc, "balanced_accuracy": _balanced_accuracy}


def bootstrap_interval(
    scores: Sequence[float],
    labels: Sequence[int],
    metric: str = "roc_auc",
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 20261006,
    confidence: float = 0.95,
) -> Interval:
    """A percentile interval over the evaluation units, resampled within each class.

    Stratified by class, which is not a refinement. An unstratified resample of thirty
    pairs produces single-class draws often enough to matter, and both metrics here are
    undefined on those — so the unstratified interval would be computed from whichever
    resamples happened to survive, which is a different quantity with the same name.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if score_array.shape != label_array.shape:
        raise ValueError("scores and labels must describe the same evaluation units")

    function = METRICS[metric]
    estimate = function(label_array, score_array)

    positive = np.flatnonzero(label_array == 1)
    negative = np.flatnonzero(label_array == 0)
    if positive.size == 0 or negative.size == 0:
        return Interval(estimate, float("nan"), float("nan"), 0, label_array.size, metric)

    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        picked = np.concatenate(
            [
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            ]
        )
        draws[index] = function(label_array[picked], score_array[picked])

    tail = (1.0 - confidence) / 2
    low, high = np.nanpercentile(draws, [100 * tail, 100 * (1 - tail)])
    return Interval(
        estimate=float(estimate),
        low=float(low),
        high=float(high),
        resamples=resamples,
        units=int(label_array.size),
        metric=metric,
    )


@dataclass
class Comparison:
    """An omnibus test over blocks, and the pairwise results that survive correction."""

    treatments: list[str]
    blocks: list[str]
    metric: str
    mean_ranks: dict[str, float]
    #: None when there are only two treatments: an omnibus over two is the pairwise test
    #: again, and reporting both would double-count one comparison.
    omnibus_statistic: float | None
    omnibus_p: float | None
    pairwise: list[dict[str, object]] = field(default_factory=list)
    correction: str = "holm"
    control: str | None = None

    @property
    def underpowered(self) -> bool:
        return len(self.blocks) < MINIMUM_BLOCKS

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "treatments": self.treatments,
            "blocks": self.blocks,
            "block_count": len(self.blocks),
            "underpowered": self.underpowered,
            "mean_ranks": {name: round(value, 3) for name, value in self.mean_ranks.items()},
            "omnibus": (
                {
                    "test": "friedman",
                    "statistic": round(self.omnibus_statistic, 4),
                    "p": float(f"{self.omnibus_p:.4g}"),
                }
                if self.omnibus_p is not None and self.omnibus_statistic is not None
                else {
                    "test": None,
                    "reason": "two treatments; the paired test is the whole comparison",
                }
            ),
            "correction": self.correction,
            "control": self.control,
            "pairwise": self.pairwise,
        }


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    Step-down rather than Bonferroni: uniformly more powerful at the same family-wise
    error rate, and the standard in the evolutionary-computation literature this work
    sits in. Adjusted values are made monotone, so a later comparison can never be
    reported as stronger than an earlier one it is dominated by.
    """
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = (count - rank) * p_values[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return [float(value) for value in adjusted]


def friedman_with_holm(
    table: dict[str, dict[str, float]],
    metric: str,
    control: str | None = None,
) -> Comparison:
    """Friedman over blocks, then corrected pairwise comparisons.

    ``table`` maps treatment to {block: value}. Every treatment must carry every block,
    because a Friedman test over ragged data silently compares different things.

    ``control`` restricts the pairwise family to comparisons against one treatment. That
    is the right choice when the question is "does anything beat the baseline" — it is a
    smaller family, so the correction costs less power, and it matches how the NAS
    literature frames a random-search control.
    """
    treatments = sorted(table)
    if len(treatments) < 2:
        raise ComparisonStructureError("a comparison needs at least two treatments")

    block_sets = {name: set(values) for name, values in table.items()}
    shared = set.intersection(*block_sets.values())
    if not shared:
        raise ComparisonStructureError("the treatments share no block")
    missing = {
        name: sorted(values - shared) for name, values in block_sets.items() if values - shared
    }
    if missing:
        raise ComparisonStructureError(
            f"every treatment must carry every block; these are ragged: {missing}"
        )

    if control is not None and control not in table:
        raise ComparisonStructureError(f"control {control!r} is not among the treatments")

    blocks = sorted(shared)
    matrix = np.array([[table[name][block] for block in blocks] for name in treatments])

    # Friedman needs three treatments. With two there is one comparison and no
    # multiplicity, so the paired test *is* the comparison and an omnibus would report
    # the same evidence twice under another name.
    if len(treatments) >= 3:
        statistic, p_value = stats.friedmanchisquare(*matrix)
        statistic, p_value = float(statistic), float(p_value)
    else:
        statistic, p_value = None, None

    # Ranked within each block, highest value best, so rank 1 is the winner.
    ranks = np.apply_along_axis(lambda column: stats.rankdata(-column), 0, matrix)
    mean_ranks = {name: float(ranks[index].mean()) for index, name in enumerate(treatments)}

    pairs = (
        [(control, name) for name in treatments if name != control]
        if control is not None
        else list(combinations(treatments, 2))
    )

    raw: list[float] = []
    records: list[dict[str, object]] = []
    for first, second in pairs:
        left = np.array([table[first][block] for block in blocks])
        right = np.array([table[second][block] for block in blocks])
        if np.allclose(left, right):
            # Wilcoxon refuses an all-zero difference; identical treatments are a p of 1,
            # not an error, and this does happen when a metric is coarse.
            pair_p = 1.0
        else:
            pair_p = float(stats.wilcoxon(left, right, zero_method="zsplit").pvalue)
        raw.append(pair_p)
        records.append(
            {
                "treatments": [first, second],
                "median_difference": round(float(np.median(left - right)), 4),
                "blocks_favouring_first": int(np.sum(left > right)),
                "blocks_favouring_second": int(np.sum(left < right)),
                "p_raw": float(f"{pair_p:.4g}"),
            }
        )

    for record, adjusted in zip(records, holm(raw), strict=True):
        record["p_holm"] = float(f"{adjusted:.4g}")
        record["significant_at_05"] = bool(adjusted < 0.05)

    return Comparison(
        treatments=treatments,
        blocks=blocks,
        metric=metric,
        mean_ranks=mean_ranks,
        omnibus_statistic=statistic,
        omnibus_p=p_value,
        pairwise=records,
        control=control,
    )


def describe_difference(interval_a: Interval, interval_b: Interval) -> dict[str, object]:
    """Whether two intervals separate, stated without a test.

    Useful where a test is unavailable or underpowered: overlapping intervals are not
    proof of equivalence, and this says so rather than letting a reader infer it.
    """
    separated = interval_a.low > interval_b.high or interval_b.low > interval_a.high
    return {
        "difference": round(interval_a.estimate - interval_b.estimate, 4),
        "intervals_separate": separated,
        "widths": [round(interval_a.width, 4), round(interval_b.width, 4)],
        "note": (
            "separated intervals indicate a difference; overlapping ones do not indicate "
            "equivalence, only that this evidence does not distinguish them"
        ),
    }


# ── equivalence ────────────────────────────────────────────────────────────────
#
# A non-significant frequentist test is not evidence of equivalence, and here that
# distinction is not academic: the likely finding is that the evolutionary search ties
# uniform random sampling, and "we found no significant difference" would leave the
# paper's central comparison in the weakest form available.
#
# So differences are also reported against a region of practical equivalence, and the
# region has a measured origin rather than a chosen one.


def rope_from_intervals(intervals: Sequence[Interval]) -> float:
    """A region of practical equivalence, derived from sampling noise.

    The evaluator is deterministic — identical encodings return identical scores — so
    there is no seed noise to derive a region from, which is how the sibling project sets
    one. What remains, and what actually limits this study, is *which documents happened
    to be in the evaluation set*. A bootstrap over those documents measures it directly.

    The region is the median half-width of the intervals being compared. A difference
    smaller than that is smaller than the variation resampling the evaluation set would
    produce, so nobody could act on it, which is what practical equivalence means.

    Reported with every comparison that uses it, because a region chosen after seeing the
    differences is not a region.
    """
    half_widths = [interval.width / 2 for interval in intervals if not np.isnan(interval.width)]
    if not half_widths:
        raise ComparisonStructureError("no usable interval to derive a region from")
    return float(np.median(half_widths))


@dataclass(frozen=True)
class Equivalence:
    """Where the difference between two treatments sits relative to the region."""

    treatments: tuple[str, str]
    rope: float
    rope_origin: str
    probability_first_better: float
    probability_equivalent: float
    probability_second_better: float
    median_difference: float
    blocks: int

    @property
    def verdict(self) -> str:
        """The three-way reading, stated rather than left to the reader.

        ``inconclusive`` is a real outcome and the most likely one at these block counts;
        calling it equivalence would be the same error as calling a non-significant test
        equivalence, one level up.
        """
        for probability, label in (
            (self.probability_equivalent, "practically equivalent"),
            (self.probability_first_better, f"{self.treatments[0]} better"),
            (self.probability_second_better, f"{self.treatments[1]} better"),
        ):
            if probability >= 0.95:
                return label
        return "inconclusive"

    def as_dict(self) -> dict[str, object]:
        return {
            "treatments": list(self.treatments),
            "rope": round(self.rope, 5),
            "rope_origin": self.rope_origin,
            "blocks": self.blocks,
            "median_difference": round(self.median_difference, 4),
            "p_first_better": round(self.probability_first_better, 4),
            "p_equivalent": round(self.probability_equivalent, 4),
            "p_second_better": round(self.probability_second_better, 4),
            "verdict": self.verdict,
        }


def equivalence(
    table: dict[str, dict[str, float]],
    first: str,
    second: str,
    rope: float,
    rope_origin: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 20261007,
) -> Equivalence:
    """Resample the paired per-block differences and ask where they fall.

    Paired by block, because the two treatments were measured on the same corpora and
    pairing removes the between-corpus variation that otherwise swamps everything. The
    resample is over blocks rather than over documents: the question is whether the
    difference holds across settings, not within one.
    """
    for name in (first, second):
        if name not in table:
            raise ComparisonStructureError(f"{name!r} is not among the treatments")

    blocks = sorted(set(table[first]) & set(table[second]))
    if not blocks:
        raise ComparisonStructureError(f"{first!r} and {second!r} share no block")

    differences = np.array([table[first][block] - table[second][block] for block in blocks])
    rng = np.random.default_rng(seed)
    means = np.array(
        [
            differences[rng.integers(0, differences.size, differences.size)].mean()
            for _ in range(resamples)
        ]
    )

    return Equivalence(
        treatments=(first, second),
        rope=rope,
        rope_origin=rope_origin,
        probability_first_better=float(np.mean(means > rope)),
        probability_equivalent=float(np.mean(np.abs(means) <= rope)),
        probability_second_better=float(np.mean(means < -rope)),
        median_difference=float(np.median(differences)),
        blocks=len(blocks),
    )


# ── the two blocking structures ────────────────────────────────────────────────


def by_corpus(
    table: dict[str, dict[str, float]], reduce: str = "mean"
) -> dict[str, dict[str, float]]:
    """Collapse per-setting values onto corpora, for the conservative analysis.

    Settings keyed ``corpus/strategy`` become corpora. The two feature strategies of one
    corpus are measured on the same test documents, so treating them as separate blocks
    overstates the independent evidence; collapsing them is the price of blocks that are
    genuinely independent, and the cost is landing at exactly five.
    """
    if reduce not in {"mean", "max"}:
        raise ValueError("reduce must be 'mean' or 'max'")
    collapsed: dict[str, dict[str, list[float]]] = {}
    for treatment, blocks in table.items():
        for setting, value in blocks.items():
            corpus = setting.split("/")[0]
            collapsed.setdefault(treatment, {}).setdefault(corpus, []).append(value)
    return {
        treatment: {
            corpus: float(np.mean(values) if reduce == "mean" else np.max(values))
            for corpus, values in corpora.items()
        }
        for treatment, corpora in collapsed.items()
    }


def both_structures(
    table: dict[str, dict[str, float]],
    metric: str,
    control: str | None = None,
) -> dict[str, object]:
    """Run the comparison over settings and over corpora, and report both.

    Side by side rather than one or the other. Settings give eleven blocks and usable
    power at the cost of non-independence, since the two strategies of a corpus share its
    test documents; corpora give five independent blocks at exactly the conventional floor.

    Agreement between them means the conclusion does not depend on the choice. Disagreement
    is itself the finding, and reporting only the structure that produced the preferred
    answer would be the kind of selection this project exists to remove.
    """
    per_setting = friedman_with_holm(table, metric=metric, control=control)
    per_corpus = friedman_with_holm(by_corpus(table), metric=metric, control=control)

    def ordering(comparison: Comparison) -> list[str]:
        return sorted(comparison.mean_ranks, key=lambda name: comparison.mean_ranks[name])

    return {
        "per_setting": per_setting.as_dict(),
        "per_corpus": per_corpus.as_dict(),
        "orderings_agree": ordering(per_setting) == ordering(per_corpus),
        "note": (
            "settings are not independent — the two feature strategies of a corpus share "
            "its test documents — while corpora are independent but sit at the minimum "
            "block count. Neither is reported alone."
        ),
    }
