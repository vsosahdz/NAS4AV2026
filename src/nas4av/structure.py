"""What the genotype says about performance, attributed to features that mean something.

The submitted manuscript fits a decision-tree regressor from the raw integer encoding to
test AUC, then reads permutation importance off it (Table 9) and split frequencies off it
(Figures 12a–12b). The model is fine; the axis it splits on is not. Integer 8 is
``Linear(2)`` and integer 20 is ``Conv1D(kernel=2, ch=2)``, so a threshold at 14 separates
nothing a reader can name, and an importance computed over that axis attributes to
"position 6" a quantity with no interpretation.

The fix is not simply to one-hot everything. Ordinality is real *within* a family and
absent *between* families: Linear widths are powers of two and genuinely ordered, dropout
rates are ordered, kernel and channel counts are ordered — while Linear, Conv1D and
Dropout have no order at all. So the encoding here is one-hot over the family and over the
activation function, and numeric within a family, which keeps the splits a tree can
actually make meaningful.

Two consequences shape the rest of the module.

**Importance is reported over groups, not columns.** A single position now spans eleven
columns, and the importance of "position 6" is only recoverable by permuting them
together. Column-wise importance on this encoding would understate every feature it split
apart, which is the opposite of the defect it was introduced to fix.

**Nothing here is a recommendation until it clears its base rate.** Linear occupies 13 of
24 alphabet values, so a run of four Linear layers appears at P = 0.085 by chance and six
of the seven Section IV-D recommendations are base rates. Enrichment against the sampled
population is computed alongside every pattern claim, and the module reports the ratio
rather than deciding what survives.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .space.genotype import (
    CATEGORIES,
    CATEGORY_SIZES,
    CONV_OPERATIONS,
    LENGTH,
    category,
    linear_width,
    parse,
)

#: Dropout keep-rates by alphabet value, from ``Model.int_to_mod``.
DROPOUT_RATES: dict[int, float] = {1: 0.9, 2: 0.6, 3: 0.3}

#: Activation functions by alphabet value. Nominal: ReLU is not "less than" TanH, so these
#: are one-hot rather than numbered.
ACTIVATIONS: dict[int, str] = {4: "ReLU", 5: "TanH", 6: "LeakyReLU"}

#: Columns contributed by one position. Named here because the grouping is what makes the
#: importance interpretable, and a grouping that drifts from the encoding is a silent
#: error — the numbers still come out, attributed to the wrong thing.
_PER_POSITION = (
    [f"is_{name}" for name in CATEGORIES]
    + [f"act_{name}" for name in ACTIVATIONS.values()]
    + ["linear_log2_width", "dropout_rate", "conv_kernel", "conv_channels"]
)

#: Sentinel for a numeric parameter that does not apply at this position. Not zero: zero is
#: a real value for ``linear_log2_width`` (``Linear(1)``), and collapsing "absent" onto a
#: real level is exactly the kind of false adjacency this encoding exists to remove. The
#: family indicator tells a tree which rows the sentinel covers, so it can split the
#: question away before splitting on the value.
ABSENT = -1.0


def feature_names() -> list[str]:
    """Column names, in the order :func:`encode` produces them."""
    return [f"p{position:02d}_{name}" for position in range(LENGTH) for name in _PER_POSITION]


def encode(encoding: str | Sequence[int]) -> np.ndarray:
    """One genotype as a feature vector of length ``15 * len(_PER_POSITION)``."""
    genes = parse(encoding)
    row = np.full((LENGTH, len(_PER_POSITION)), ABSENT, dtype=np.float64)
    for position, gene in enumerate(genes):
        family = category(gene)
        for index, name in enumerate(CATEGORIES):
            row[position, index] = 1.0 if name == family else 0.0
        offset = len(CATEGORIES)
        for index, name in enumerate(ACTIVATIONS.values()):
            row[position, offset + index] = 1.0 if ACTIVATIONS.get(gene) == name else 0.0
        offset += len(ACTIVATIONS)
        if family == "Linear":
            row[position, offset] = float(int(math.log2(linear_width(gene))))
        elif family == "Dropout":
            row[position, offset + 1] = DROPOUT_RATES[gene]
        elif family == "Conv1D":
            kernel, channels = CONV_OPERATIONS[gene]
            row[position, offset + 2] = float(kernel)
            row[position, offset + 3] = float(channels)
    return row.reshape(-1)


def encode_many(encodings: Iterable[str | Sequence[int]]) -> np.ndarray:
    return np.vstack([encode(item) for item in encodings]) if encodings else np.empty((0, 0))


def position_groups() -> dict[str, list[int]]:
    """Column indices per position, for importance attributed to depth."""
    width = len(_PER_POSITION)
    return {f"position_{p:02d}": list(range(p * width, (p + 1) * width)) for p in range(LENGTH)}


def family_groups() -> dict[str, list[int]]:
    """Column indices per operation family, pooled across positions.

    The grouping that answers the question the reframed contribution actually asks — which
    *elements* a network needs for authorship verification — rather than which slot in a
    fixed-length string happens to carry them.
    """
    width = len(_PER_POSITION)
    groups: dict[str, list[int]] = {name: [] for name in CATEGORIES}
    for position in range(LENGTH):
        base = position * width
        for index, name in enumerate(CATEGORIES):
            groups[name].append(base + index)
        offset = len(CATEGORIES)
        groups["Activation"].extend(base + offset + i for i in range(len(ACTIVATIONS)))
        groups["Linear"].append(base + offset + len(ACTIVATIONS))
        groups["Dropout"].append(base + offset + len(ACTIVATIONS) + 1)
        groups["Conv1D"].extend(
            (base + offset + len(ACTIVATIONS) + 2, base + offset + len(ACTIVATIONS) + 3)
        )
    return groups


@dataclass
class Importance:
    """One group's permutation importance, with the spread that says whether to believe it."""

    group: str
    mean: float
    std: float
    columns: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "mean_score_drop": round(self.mean, 5),
            "std": round(self.std, 5),
            "columns": self.columns,
        }


def grouped_permutation_importance(
    model: Any,
    features: np.ndarray,
    target: np.ndarray,
    groups: dict[str, list[int]],
    repeats: int = 20,
    seed: int = 20260920,
) -> list[Importance]:
    """Permutation importance with whole groups shuffled together.

    Written out rather than taken from scikit-learn because the library permutes one column
    at a time, and under this encoding a single position spans eleven correlated columns. A
    tree that loses one of them reads the rest and barely drops, so column-wise importance
    would report every feature as unimportant and the fix for the ordinality defect would
    look like evidence that structure does not matter.

    The same row permutation is applied across a group's columns, which keeps each row's
    position internally consistent — a position must stay a valid operation after shuffling
    or the model is scored on inputs the encoding cannot produce.
    """
    rng = np.random.default_rng(seed)
    baseline = float(model.score(features, target))
    results: list[Importance] = []
    for name, columns in groups.items():
        if not columns:
            continue
        drops = np.empty(repeats, dtype=np.float64)
        for repeat in range(repeats):
            shuffled = features.copy()
            order = rng.permutation(features.shape[0])
            shuffled[:, columns] = features[order][:, columns]
            drops[repeat] = baseline - float(model.score(shuffled, target))
        results.append(Importance(name, float(drops.mean()), float(drops.std()), len(columns)))
    results.sort(key=lambda item: item.mean, reverse=True)
    return results


def split_frequency(model: Any, groups: dict[str, list[int]]) -> dict[str, dict[str, float]]:
    """How often, and how usefully, the fitted model splits on each group.

    The replacement for Figures 12a–12b. Reported two ways because they answer different
    questions and the submitted figures conflate them: a count says how often the model
    *asked* about a group, while impurity decrease says how much the answer was worth. A
    group split on constantly for tiny gains looks dominant under the first and negligible
    under the second.
    """
    estimators = getattr(model, "estimators_", None)
    trees = [getattr(e, "tree_", e) for e in np.ravel(estimators)] if estimators is not None else []
    if not trees:
        trees = [model.tree_]

    counts: dict[str, float] = {name: 0.0 for name in groups}
    gains: dict[str, float] = {name: 0.0 for name in groups}
    column_to_group = {
        column: name for name, columns in groups.items() for column in columns
    }
    for tree in trees:
        for node in range(tree.node_count):
            column = int(tree.feature[node])
            if column < 0:
                continue
            name = column_to_group.get(column)
            if name is None:
                continue
            counts[name] += 1.0
            gains[name] += float(tree.weighted_n_node_samples[node]) * float(
                tree.impurity[node]
                - (
                    tree.weighted_n_node_samples[tree.children_left[node]]
                    * tree.impurity[tree.children_left[node]]
                    + tree.weighted_n_node_samples[tree.children_right[node]]
                    * tree.impurity[tree.children_right[node]]
                )
                / max(tree.weighted_n_node_samples[node], 1e-12)
            )
    total_count = sum(counts.values()) or 1.0
    total_gain = sum(gains.values()) or 1.0
    return {
        name: {
            "split_share": round(counts[name] / total_count, 5),
            "gain_share": round(gains[name] / total_gain, 5),
            "splits": int(counts[name]),
        }
        for name in groups
    }


@dataclass
class Enrichment:
    """A pattern's frequency among top performers against its frequency in the population."""

    pattern: str
    top_frequency: float
    population_frequency: float
    null_probability: float
    top_count: int
    population_count: int

    @property
    def ratio(self) -> float:
        """Enrichment over the sampled population, which is the comparison that is sound.

        Not over the null. The database is EA-guided and therefore already biased toward
        whatever the search favoured, so a pattern can be far above its chance rate purely
        because the search converged. Stating the null alongside keeps both readings
        visible, and the ratio against the population is the one a recommendation rests on.
        """
        return self.top_frequency / self.population_frequency if self.population_frequency else 0.0

    @property
    def ratio_over_null(self) -> float:
        return self.top_frequency / self.null_probability if self.null_probability else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "frequency_in_top": round(self.top_frequency, 5),
            "frequency_in_population": round(self.population_frequency, 5),
            "probability_under_null": round(self.null_probability, 5),
            "enrichment_over_population": round(self.ratio, 2),
            "enrichment_over_null": round(self.ratio_over_null, 2),
            "top_count": self.top_count,
            "population_count": self.population_count,
        }


def consecutive_run(genes: Sequence[int], family: str, length: int) -> bool:
    """Whether ``length`` operations of ``family`` appear consecutively."""
    streak = 0
    for gene in genes:
        streak = streak + 1 if category(gene) == family else 0
        if streak >= length:
            return True
    return False


def null_probability_of_run(family: str, length: int) -> float:
    """Chance of a run of that length somewhere in a length-15 encoding, drawn uniformly.

    Computed by recurrence over positions rather than as ``p**length``, which is the
    arithmetic the manuscript's reasoning implies and which answers a different question:
    ``p**length`` is the chance a *given* window matches, while a reader comparing against
    an observed frequency means the chance of the pattern appearing *anywhere*. For four
    consecutive Linear layers that is 0.086 against **0.46** — the pattern Section IV-D
    issues as a design recommendation is expected in close to half of all random encodings,
    and the per-window figure makes it look five times rarer than it is.

    Over the **unconstrained** space, deliberately. The feasible space is what the search
    samples, and its null has no closed form because ``g₁`` and ``g₂`` couple positions; it
    is estimated by rejection sampling instead. The two are close for Linear runs, since
    neither constraint touches Linear, and they diverge for anything involving Conv1D —
    which is where the closed form must not be quoted.
    """
    p = CATEGORY_SIZES[family] / sum(CATEGORY_SIZES.values())
    # states[k] = probability of a current streak of exactly k, having never reached
    # `length`; absorbed is the probability the run has already happened.
    states = [1.0] + [0.0] * (length - 1)
    absorbed = 0.0
    for _ in range(LENGTH):
        nxt = [0.0] * length
        for streak, mass in enumerate(states):
            if streak + 1 >= length:
                absorbed += mass * p
            else:
                nxt[streak + 1] += mass * p
            nxt[0] += mass * (1.0 - p)
        states = nxt
    return absorbed


def enrichment(
    population: Sequence[tuple[str, float]],
    pattern: str,
    predicate: Any,
    null: float,
    top_fraction: float = 0.05,
) -> Enrichment:
    """Frequency of a pattern among the best ``top_fraction`` against the whole population.

    ``population`` is (encoding, score) pairs from one setting. Pooling settings would mix
    corpora whose score scales differ, so the caller decides what a population is.
    """
    ranked = sorted(population, key=lambda item: item[1], reverse=True)
    cut = max(1, int(round(len(ranked) * top_fraction)))
    top = ranked[:cut]
    top_hits = sum(1 for encoding, _ in top if predicate(parse(encoding)))
    all_hits = sum(1 for encoding, _ in ranked if predicate(parse(encoding)))
    return Enrichment(
        pattern=pattern,
        top_frequency=top_hits / len(top),
        population_frequency=all_hits / len(ranked),
        null_probability=null,
        top_count=top_hits,
        population_count=all_hits,
    )
