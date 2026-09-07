"""Reading the prior campaign's published tables.

One loader, because several analyses would otherwise each re-derive the directory
layout and the fitness inversion, and a disagreement between them would be invisible.

The tables live in a clone of ``github.com/luisferro2/NAS_4_AV`` pinned at ``65ee410``.
It is obtained rather than vendored — it belongs to its own authors, and anything
asserted about it should stay checkable against the source.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

#: Weights of the fitness function, read from ``train_predict_evaluate``:
#:
#:      curr_acc = curr_aucte * 0.75 + curr_auctr * 0.25
#:
#: and stored under the column name ``combined accuracy``, which is a fitness value and
#: not an accuracy. That naming is why the training-side score looked absent from the
#: tables when it is in fact recoverable from them.
FITNESS_TEST_WEIGHT = 0.75
FITNESS_TRAIN_WEIGHT = 0.25

#: Every column the prior tables call AUC holds balanced accuracy. This constant exists
#: so that no module can reach for the word without meeting the correction.
#:
#: ``Search_Space.metrics`` binarises predictions with ``logit2int`` (``logit > 0``),
#: builds a confusion matrix from the resulting 0/1 labels, and hands it to ``calc_auc``,
#: which returns ``(specificity + sensitivity) / 2``. That is the mean of the two
#: class-wise recalls at a single operating point — balanced accuracy — and not an area
#: under a ranking curve. No continuous score ever reaches the metric.
#:
#: It matters whenever these values are placed beside results from elsewhere: the
#: CLEF-PAN authorship-verification tracks report ROC-AUC computed over scores, so the
#: two are different measurements that share a name, and neither is a rescaling of the
#: other. Which way the difference runs depends on calibration rather than on quality.
AUC_IS_BALANCED_ACCURACY = True

CORPORA = ("2013", "2014Essay2", "2014Novel2", "2015", "2020")

#: Search-tree directory names. ``semBert`` holds 6,000 evaluations over BERT
#: embeddings and exists for CLEF-PAN 2013 only.
STRATEGIES = ("sem", "synBetter", "semBert")

#: Strategy directory to the ``surrogateFTDF*`` file suffix holding its fully trained
#: counterpart. ``semBert`` has none.
FULLY_TRAINED_SUFFIX: dict[str, str | None] = {
    "sem": "Semantic",
    "synBetter": "Syntax",
    "semBert": None,
}


class PriorRepositoryMissing(RuntimeError):
    """Raised when the pinned clone is not present.

    Its message carries the clone command rather than a path, because the useful
    response to this error is to obtain the repository, not to go looking for it.
    """


def reference_root() -> Path:
    """Where the pinned clone lives. Overridable so tests can point at a fixture."""
    return Path(os.environ.get("NAS4AV_REFERENCE", "./reference/NAS_4_AV")).resolve()


def results_root() -> Path:
    root = reference_root() / (
        "OurResults/SequentialSearchSpaceResults/float64/sequential/STRIX/GA"
    )
    if not root.is_dir():
        raise PriorRepositoryMissing(
            f"the prior campaign's tables are not at {root}.\n"
            "    git clone https://github.com/luisferro2/NAS_4_AV.git reference/NAS_4_AV\n"
            "    git -C reference/NAS_4_AV checkout 65ee41046f8c3417e4159f8f9bc96ca1a7ddf84b"
        )
    return root


@dataclass(frozen=True)
class Evaluation:
    """One architecture as the prior campaign measured it.

    ``train_auc`` is recovered rather than stored. Inverting the fitness gives it
    exactly, and the recovery lands inside [0, 1] for all 66,000 rows with no failures —
    which is the evidence that ``combined accuracy`` really is the fitness and not some
    other blend.
    """

    encoding: str
    corpus: str
    strategy: str
    run: int
    test_auc: float
    train_auc: float
    fitness: float
    train_accuracy: float
    test_accuracy: float
    seconds: float

    @property
    def setting(self) -> str:
        return f"{self.corpus}/{self.strategy}"


def recover_train_auc(fitness: float, test_auc: float) -> float:
    """Recover the training-side score from the fitness and the test-side score.

    The prior tables store ``training accuracy`` but never the training-side score the
    fitness was built from, so a selection signal that avoids the test split looks
    unavailable. It is not: the fitness column carries it, and the inversion is exact on
    all 66,000 rows.

    How much that buys is smaller than it first appears. The score is balanced accuracy
    (see ``AUC_IS_BALANCED_ACCURACY``), which equals plain accuracy whenever the labels
    are balanced — as they are on nine of the eleven settings. On those nine the recovery
    returns a number the tables already carried, and only CLEF-PAN 2020 gains a new one.
    """
    return (fitness - FITNESS_TEST_WEIGHT * test_auc) / FITNESS_TRAIN_WEIGHT


def _run_number(path: Path, prefix: str) -> int:
    match = re.search(rf"{prefix}(\d+)\.csv$", path.name)
    if match is None:
        raise ValueError(f"cannot read a run number from {path.name}")
    return int(match.group(1))


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def search_evaluations(
    corpus: str | None = None, strategy: str | None = None
) -> Iterator[Evaluation]:
    """Every partial-training evaluation, 66,000 of them across eleven settings.

    Note that these arrive **sorted by fitness descending**, not in evaluation order, and
    carry no generation column, so which individuals formed each run's initial population
    cannot be recovered from them.
    """
    root = results_root()
    pattern = f"{corpus or '*'}/{strategy or '*'}/surrogateDF*.csv"
    for path in sorted(root.glob(pattern)):
        this_corpus, this_strategy = path.parts[-3], path.parts[-2]
        run = _run_number(path, "surrogateDF")
        for row in _rows(path):
            fitness = float(row["combined accuracy"])
            test_auc = float(row["test auc"])
            yield Evaluation(
                encoding=row["encoding"],
                corpus=this_corpus,
                strategy=this_strategy,
                run=run,
                test_auc=test_auc,
                train_auc=recover_train_auc(fitness, test_auc),
                fitness=fitness,
                train_accuracy=float(row["training accuracy"]),
                test_accuracy=float(row["test accuracy"]),
                seconds=float(row["training time"]),
            )


def fully_trained(corpus: str | None = None) -> Iterator[Evaluation]:
    """The models trained to completion: 1,010 across ten settings.

    Fewer than the 120 per setting a top-24-from-five-runs selection would give, most
    plausibly because encodings found by more than one run were deduplicated.
    """
    root = results_root()
    for path in sorted(root.glob(f"{corpus or '*'}/surrogateFTDF*.csv")):
        this_corpus = path.parts[-2]
        suffix = "Semantic" if "Semantic" in path.name else "Syntax"
        strategy = next(k for k, v in FULLY_TRAINED_SUFFIX.items() if v == suffix)
        for row in _rows(path):
            fitness = float(row["combined accuracy"])
            test_auc = float(row["test auc"])
            yield Evaluation(
                encoding=row["encoding"],
                corpus=this_corpus,
                strategy=strategy,
                run=0,  # the file does not record which run proposed the architecture
                test_auc=test_auc,
                train_auc=recover_train_auc(fitness, test_auc),
                fitness=fitness,
                train_accuracy=float(row["training accuracy"]),
                test_accuracy=float(row["test accuracy"]),
                seconds=float(row["training time"]),
            )


def ensembles(corpus: str) -> list[dict[str, float | str]]:
    """The 252 five-model ensembles for one corpus, in their stored order.

    Stored **pre-sorted by test score descending**, so taking the first rows of one of
    these files is a selection on the test split rather than an arbitrary slice.

    These carry no fitness column, so the training-side score cannot be recovered here.
    The ensemble stage's only signal that avoids the test split is ``training accuracy``.
    """
    root = results_root()
    paths = sorted((root / corpus).glob("ensemblesDF*.csv"))
    if not paths:
        raise PriorRepositoryMissing(f"no ensemble table for corpus {corpus}")
    out: list[dict[str, float | str]] = []
    for row in _rows(paths[0]):
        record: dict[str, float | str] = {"models": row["models"]}
        for column, value in row.items():
            if column != "models":
                record[column] = float(value)
        out.append(record)
    return out


def convergence(corpus: str, strategy: str) -> dict[int, list[tuple[float, str]]]:
    """Best-so-far fitness per generation, per run. Fifty rows each.

    Monotone non-decreasing and ending exactly at the run's best fitness, so row 0 is the
    best of the initial population — the only surviving trace of that uniform draw.
    """
    root = results_root()
    traces: dict[int, list[tuple[float, str]]] = {}
    for path in sorted((root / corpus / strategy).glob("convergenceDF*.csv")):
        traces[_run_number(path, "convergenceDF")] = [
            (float(row["fitness"]), row["encoding"]) for row in _rows(path)
        ]
    return traces


def settings() -> list[tuple[str, str]]:
    """Every (corpus, strategy) pair the prior campaign actually produced."""
    root = results_root()
    found = {(path.parts[-3], path.parts[-2]) for path in root.glob("*/*/surrogateDF*.csv")}
    return sorted(found)
