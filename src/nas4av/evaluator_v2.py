"""Evaluating a redesigned-space architecture, under the protocol the campaign already uses.

Satisfies the same `Evaluator` protocol as `PipelineEvaluator`, so `run_search` does not
know which space it is searching and the budget, checkpointing and record-keeping are
shared. That is the point: a comparison between two spaces is about the spaces only if the
driver around them is identical.

Everything outside the architecture is held fixed — same corpora, same three-way split,
same optimiser, learning rate, batch size and epoch count, both metrics recorded for every
evaluation. The one thing that differs is what gets built from the genotype.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .evaluator import BATCH_SIZE, LEARNING_RATE, PARTIAL_EPOCHS
from .prior.dataset import Corpus, Partition, partition
from .protocol import ProtocolSplit
from .search.campaign import Measurement
from .search.strategy import Genotype
from .space.v2 import describe
from .space.v2_model import pair_tensor, train_and_score


def _metrics(logits: Sequence[float], labels: Sequence[int]) -> tuple[float, float]:
    """ROC-AUC over continuous scores and balanced accuracy at a zero threshold.

    Both, for every evaluation. ROC-AUC is what the literature reports; balanced accuracy
    is what the prior campaign measured under the name AUC, and keeping it is what lets the
    two campaigns be compared to each other at all.
    """
    labels_list = list(labels)
    if len(set(labels_list)) < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(labels_list, list(logits)))
    predicted = [1 if value > 0 else 0 for value in logits]
    accuracy = float(balanced_accuracy_score(labels_list, predicted))
    return auc, accuracy


class V2Evaluator:
    """Trains one redesigned-space architecture per call and measures it.

    Stateful only in what it keeps for the record, matching `PipelineEvaluator`: the
    per-document scores of every architecture, so the ranking-versus-threshold question
    can be answered after the campaign rather than requiring another one.
    """

    def __init__(
        self,
        corpus: Corpus,
        split: ProtocolSplit,
        epochs: int = PARTIAL_EPOCHS,
        keep_scores: bool = True,
        seed: int = 20260920,
    ) -> None:
        self.corpus = corpus
        self.split = split
        self.epochs = epochs
        self.keep_scores = keep_scores
        self.seed = seed
        self.scores: dict[str, dict[str, list[float]]] = {}

        # The per-document width, not the pair width. The first space takes the pair width
        # and halves it internally, which has already cost a 2x error once in this project.
        self.per_document_width = corpus.input_shape // 2

        self._train = self._partition(
            corpus.train_features, corpus.train_documents, corpus.train_labels,
            split.train, corpus.train_chunks,
        )
        self._validation = self._partition(
            corpus.train_features, corpus.train_documents, corpus.train_labels,
            split.validation, corpus.train_chunks,
        )
        self._test = self._partition(
            corpus.test_features, corpus.test_documents, corpus.test_labels,
            split.release_test("held for reporting; the evaluator never scores fitness on it"),
            corpus.test_chunks,
        )

    @staticmethod
    def _partition(
        features: torch.Tensor,
        documents: Sequence[str],
        labels: Sequence[int],
        indices: Sequence[int],
        chunks: int,
    ) -> Partition:
        return partition(features, list(documents), list(labels), list(indices), chunks)

    def __call__(self, genotype: Genotype) -> Measurement:
        encoding = ",".join(str(gene) for gene in genotype)
        started = time.perf_counter()
        try:
            produced = train_and_score(
                encoding,
                self.per_document_width,
                self._train,
                {"validation": self._validation, "test": self._test},
                epochs=self.epochs,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                seed=self.seed,
            )
            validation_auc, validation_accuracy = _metrics(
                produced["validation"], self._validation.labels
            )
            test_auc, test_accuracy = _metrics(produced["test"], self._test.labels)
        except Exception as error:  # noqa: BLE001
            # Recorded as a failure with its reason, never as a score of zero. A zero is
            # indistinguishable downstream from an architecture that trained and was bad,
            # and the two need different handling in every analysis.
            return Measurement(
                validation_roc_auc=0.0,
                validation_balanced_accuracy=0.0,
                test_roc_auc=0.0,
                test_balanced_accuracy=0.0,
                seconds=time.perf_counter() - started,
                failed=True,
                detail=f"{type(error).__name__}: {error}"[:300],
            )

        if self.keep_scores:
            self.scores[encoding] = {
                "validation": [float(v) for v in produced["validation"]],
                "test": [float(v) for v in produced["test"]],
            }

        # NaN means the split had one class, which is a property of the corpus rather than
        # of the architecture. Recorded as a failure so it cannot be averaged into a score.
        if any(np.isnan(v) for v in (validation_auc, test_auc)):
            return Measurement(
                validation_roc_auc=0.0,
                validation_balanced_accuracy=0.0,
                test_roc_auc=0.0,
                test_balanced_accuracy=0.0,
                seconds=time.perf_counter() - started,
                failed=True,
                detail="a split carried only one class, so no ranking metric is defined",
            )

        return Measurement(
            validation_roc_auc=validation_auc,
            validation_balanced_accuracy=validation_accuracy,
            test_roc_auc=test_auc,
            test_balanced_accuracy=test_accuracy,
            seconds=time.perf_counter() - started,
        )

    def params(self, genotype: Genotype) -> int:
        """Exact parameter count, derived from the genotype without building anything."""
        return describe(
            ",".join(str(gene) for gene in genotype), self.per_document_width
        ).params

    def partition_sizes(self) -> dict[str, int]:
        """Recorded with every campaign: a metric's meaning depends on how many pairs.

        Reported identically to the first space's evaluator, because these numbers are the
        reader's only way to tell a 0.7 over thirty pairs from a 0.7 over five hundred.
        """
        from .evaluator import _balance

        return {
            "train_pairs": self._train.pairs,
            "validation_pairs": self._validation.pairs,
            "test_pairs": self._test.pairs,
            "validation_label_balance": _balance(self._validation.labels),
            "test_label_balance": _balance(self._test.labels),
        }


def pairs_in(part: Partition) -> int:
    """How many pairs a partition holds, after chunks are folded into documents."""
    return int(pair_tensor(part).shape[0])
