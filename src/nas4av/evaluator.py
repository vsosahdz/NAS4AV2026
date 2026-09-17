"""Training one architecture and measuring it, under the repaired protocol.

Two things distinguish this from the prior evaluation, and both are structural rather
than a matter of care.

**The fitness never sees the test split.** The architecture is trained on the training
partition and scored on the search-validation partition. The test partition is scored too,
because a reader wants the number, but that value travels into the record without passing
through anything the search can read.

**Continuous scores are kept.** The prior pipeline thresholds predictions at ``logit > 0``
inside ``sequential_predict`` and measures the confusion matrix that results, so every
quantity it calls AUC is balanced accuracy at one operating point. Here the model is
called directly for its logits, which makes a real ROC-AUC available and lets both be
reported: ROC-AUC for comparability with published results, balanced accuracy for
continuity with the prior campaign.

Keeping the logits also makes a hypothesis testable that the prior artifacts cannot
answer — that the low-fidelity estimate fails because a thresholded metric is brittle
after ten epochs while a ranking metric would not be. Storing them costs nothing now and
cannot be recovered later.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .original import pipeline as original
from .prior.dataset import Corpus, Partition, partition
from .protocol import ProtocolSplit
from .search.campaign import Measurement
from .search.strategy import Genotype

#: Training hyperparameters, fixed exactly as the prior campaign fixed them.
#:
#: Held constant on purpose. The new campaign differs from the old one in its split and
#: its seeds and in nothing else, which is what makes the difference between them
#: attributable to the protocol rather than to an accumulation of changes.
OPTIMIZER = "Adam"
LEARNING_RATE = 0.001
BATCH_SIZE = 4
PARTIAL_EPOCHS = 10


@dataclass
class Scores:
    """Continuous logits and the labels they were produced against."""

    logits: list[float]
    labels: list[int]

    def roc_auc(self) -> float:
        """Area under the ROC curve over the raw scores.

        Undefined when the labels are one class only — which happens on small validation
        partitions — so it returns the no-discrimination value rather than raising. A
        caller cannot tell those apart from the number alone, which is why the partition
        sizes travel with every result.
        """
        if len(set(self.labels)) < 2:
            return 0.5
        return float(roc_auc_score(self.labels, self.logits))

    def balanced_accuracy(self) -> float:
        """The prior campaign's metric: thresholded at zero, both recalls averaged."""
        if len(set(self.labels)) < 2:
            return 0.5
        predicted = [1 if value > 0.0 else 0 for value in self.logits]
        return float(balanced_accuracy_score(self.labels, predicted))


def predict_scores(model: torch.nn.Module, part: Partition) -> Scores:
    """Raw logits, before any threshold.

    Calls the model rather than ``Sequential_Search_Space.sequential_predict``, which
    applies ``logit2int`` and discards exactly the information a ranking metric needs.
    The extracted pipeline is left untouched — the caller reaches past the helper instead
    of the helper being rewritten.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(part.chunks_per_document, embeddings=part.features)
    return Scores(
        logits=[float(value) for value in outputs.reshape(-1).cpu()],
        labels=list(part.labels),
    )


class PipelineEvaluator:
    """Evaluates one architecture per call, for one corpus under one split.

    Stateful only in what it keeps for the record: the per-document scores of every
    architecture it has evaluated, so the ranking-versus-threshold question can be
    answered after the campaign rather than requiring another one.
    """

    def __init__(
        self,
        corpus: Corpus,
        split: ProtocolSplit,
        epochs: int = PARTIAL_EPOCHS,
        keep_scores: bool = True,
    ) -> None:
        self.corpus = corpus
        self.split = split
        self.epochs = epochs
        self.keep_scores = keep_scores
        self.scores: dict[str, dict[str, list[float]]] = {}

        self._search_space = original.Sequential_Search_Space(corpus.input_shape)
        self._train = self._partition(
            corpus.train_features,
            corpus.train_documents,
            corpus.train_labels,
            split.train,
            corpus.train_chunks,
        )
        self._validation = self._partition(
            corpus.train_features,
            corpus.train_documents,
            corpus.train_labels,
            split.validation,
            corpus.train_chunks,
        )
        self._test = self._partition(
            corpus.test_features,
            corpus.test_documents,
            corpus.test_labels,
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
        return partition(features, documents, labels, indices, chunks)

    def __call__(self, genotype: Genotype) -> Measurement:
        started = time.time()
        encoding = ",".join(str(gene) for gene in genotype)
        try:
            model = original.Model(
                list(genotype),
                self.corpus.input_shape,
                self._search_space.genel,
                self.corpus.input_shape // 2,
            )
            # Converts every weight to float64. Not cosmetic: the stored features are
            # double and nn.Linear builds float32, so an architecture trained without this
            # raises on its first matrix multiply. ``train_predict`` calls it for the same
            # reason, and skipping it makes every evaluation fail identically — which is
            # how it was found.
            model.weights_init_uniform()
            model.train()
            self._search_space.train_model(
                model,
                train_embeddings=self._train.features,
                train_df=self._train.frame,
                training_cpd=self._train.chunks_per_document,
                n_epochs=self.epochs,
                batch_size=BATCH_SIZE,
                optimizer=OPTIMIZER,
                lr=LEARNING_RATE,
                verbose=False,
            )
            validation = predict_scores(model, self._validation)
            test = predict_scores(model, self._test)
        except Exception as error:  # noqa: BLE001
            # Recorded as a failure with its reason, never as a score of zero. The prior
            # pipeline caught everything and wrote zeros into the results table, which is
            # indistinguishable downstream from an architecture that trained and was bad.
            return Measurement(
                validation_roc_auc=float("nan"),
                validation_balanced_accuracy=float("nan"),
                test_roc_auc=float("nan"),
                test_balanced_accuracy=float("nan"),
                seconds=time.time() - started,
                failed=True,
                detail=f"{type(error).__name__}: {error}",
            )

        if self.keep_scores:
            self.scores[encoding] = {
                "validation": validation.logits,
                "test": test.logits,
            }

        return Measurement(
            validation_roc_auc=validation.roc_auc(),
            validation_balanced_accuracy=validation.balanced_accuracy(),
            test_roc_auc=test.roc_auc(),
            test_balanced_accuracy=test.balanced_accuracy(),
            seconds=time.time() - started,
        )

    def partition_sizes(self) -> dict[str, int]:
        """Recorded with every campaign: a metric's meaning depends on how many pairs."""
        return {
            "train_pairs": self._train.pairs,
            "validation_pairs": self._validation.pairs,
            "test_pairs": self._test.pairs,
            "validation_label_balance": _balance(self._validation.labels),
            "test_label_balance": _balance(self._test.labels),
        }


def _balance(labels: Sequence[int]) -> int:
    """Count of the positive class, so an imbalanced partition is visible in the record."""
    return int(np.sum(np.asarray(labels) == 1))
