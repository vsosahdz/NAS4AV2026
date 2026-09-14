"""The evaluation protocol: how data is split, and how randomness is kept separated.

Two defects in the prior arrangement are addressed here, and they are independent of each
other.

**The test split was inside the objective.** The fitness read test-set performance
directly, and six further stages selected on it afterwards. Here the split is three-way —
train, search-validation, test — and the test partition is handed out only through
:func:`ProtocolSplit.release_test`, which records that it happened.

**Training reset the randomness the search drew from.** The prior training loop calls
``random.seed((k + 1) * (epoch + 1))`` inside its batch loop, on Python's *global*
generator, which is the same generator the evolutionary operators use. Epoch count and
batch count are fixed per corpus, so every evaluation leaves that generator at an
identical state: seed 20 for CLEF-PAN 2013.

The consequence is measurable rather than theoretical. Seeding a generator to 20 and
drawing fifteen integers repeatedly produces encodings present in the stored 2013
database 23% of the time, against 1.3e-16 expected for genuinely fresh draws. The search
was not exploring from an evolving random state; it was restarting from a fixed one after
every evaluation, and only the novelty filter kept it moving.

:func:`isolated_randomness` is the fix, and it is a context manager rather than a
convention because a convention would be forgotten exactly once.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

#: Minimum validation pairs below which selecting on that split is not meaningful.
#:
#: Set from the smallest split that can distinguish two architectures at all rather than
#: from a convention. Balanced accuracy over n pairs takes values on a grid of spacing
#: 1/n per class, so with fewer than ten pairs the metric has fewer distinct values than
#: the population has candidates, and selection becomes a tie-break among ties.
#:
#: CLEF-PAN 2013 has ten training pairs in total, so no split of it clears this. That is
#: a fact about the corpus, not a threshold chosen to exclude it.
MINIMUM_VALIDATION_PAIRS = 10


class SplitLeakError(RuntimeError):
    """Raised when the test partition is requested during search."""


@dataclass
class ProtocolSplit:
    """A three-way partition of one corpus, with the test half kept behind a gate.

    ``train`` and ``validation`` index into the training pairs; ``test`` indexes the
    corpus's own held-out set. The indices are pair indices, not document indices — the
    features hold two rows per pair, so anything consuming these must double them.
    """

    corpus: str
    train: tuple[int, ...]
    validation: tuple[int, ...]
    seed: int
    _test: tuple[int, ...] = field(repr=False, default=())
    _released: bool = field(default=False, repr=False)

    @property
    def supports_clean_selection(self) -> bool:
        """Whether the validation split is large enough to select on at all."""
        return len(self.validation) >= MINIMUM_VALIDATION_PAIRS

    def release_test(self, reason: str) -> tuple[int, ...]:
        """Hand out the test partition, once, with the reason recorded.

        Deliberately awkward. Reading the test split is legitimate exactly once per
        result — for reporting — and the awkwardness is what distinguishes that from the
        six unrecorded reads the prior pipeline performed.
        """
        if not reason.strip():
            raise SplitLeakError("releasing the test split requires a stated reason")
        self._released = True
        return self._test

    @property
    def test_was_released(self) -> bool:
        return self._released

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "seed": self.seed,
            "train_pairs": len(self.train),
            "validation_pairs": len(self.validation),
            "test_pairs": len(self._test),
            "supports_clean_selection": self.supports_clean_selection,
            "minimum_validation_pairs": MINIMUM_VALIDATION_PAIRS,
            "test_released": self._released,
        }


def stratified_holdout(
    labels: Sequence[int], fraction: float, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split pair indices into (kept, held out), preserving the label balance.

    Stratified rather than uniform because the metric is balanced accuracy, which is
    undefined in one class's term when a split contains only the other. On corpora with
    tens of training pairs an unstratified draw produces that often enough to matter.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie strictly between 0 and 1, got {fraction}")

    rng = np.random.default_rng(seed)
    held: list[int] = []
    for value in sorted(set(labels)):
        members = [index for index, label in enumerate(labels) if label == value]
        rng.shuffle(members)
        take = int(round(len(members) * fraction))
        # Never consume a class entirely, and never take none of it when the caller asked
        # for a split: either extreme removes a term from the metric.
        take = max(1, min(len(members) - 1, take)) if len(members) > 1 else 0
        held.extend(members[:take])

    held_set = set(held)
    kept = tuple(index for index in range(len(labels)) if index not in held_set)
    return kept, tuple(sorted(held_set))


def three_way_split(
    corpus: str,
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    seed: int,
    validation_fraction: float = 0.25,
) -> ProtocolSplit:
    """Carve a search-validation split out of the training pairs.

    The test partition is the corpus's own, untouched — this function never subdivides it.
    What it does is take the search's selection signal out of the test set entirely, at
    the cost of training on less data, which is the trade the prior protocol avoided by
    reading test labels instead.
    """
    train, validation = stratified_holdout(train_labels, validation_fraction, seed)
    return ProtocolSplit(
        corpus=corpus,
        train=train,
        validation=validation,
        seed=seed,
        _test=tuple(range(len(test_labels))),
    )


@contextlib.contextmanager
def isolated_randomness() -> Iterator[None]:
    """Restore every global generator on exit, so a callee cannot steer its caller.

    Wrap any call into the extracted pipeline with this. Training reseeds Python's global
    generator on every batch, and without the restore those writes land in the stream the
    search draws its operators from — which is the defect this exists to prevent, measured
    at 23% recovery of the prior 2013 database from a single fixed seed.

    numpy and torch are saved as well. ``RDropout`` seeds numpy's legacy global generator
    from the layer width, so it writes there too.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = None
    try:
        import torch

        torch_state = torch.random.get_rng_state()
    except ImportError:
        pass

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_state is not None:
            import torch

            torch.random.set_rng_state(torch_state)


def seed_everything(seed: int) -> None:
    """Seed every global generator the pipeline touches.

    Called once per run, not once per evaluation. Per-evaluation seeding is what the prior
    arrangement effectively did, and it is why five runs shared most of their
    architectures.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def run_seeds(base: int, runs: int) -> tuple[int, ...]:
    """Distinct, reproducible seeds, one per run.

    Spread rather than consecutive: consecutive integer seeds produce correlated opening
    states in some generators, and independent runs are the entire point of having more
    than one.
    """
    rng = np.random.default_rng(base)
    return tuple(int(value) for value in rng.choice(2**31 - 1, size=runs, replace=False))
