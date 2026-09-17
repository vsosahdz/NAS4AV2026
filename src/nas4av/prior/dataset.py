"""Loading a corpus the way the prior implementation assembled it.

Features are read from the stored tensors rather than rebuilt from text, because they are
what the prior campaign actually trained on and rebuilding them would introduce a second
source of disagreement on top of the one being measured. The exception is the punctuation
bag-of-words on the syntactic arm, which was never stored: it is recomputed with the
extracted function, so the assembly is the original one.

The indexing convention is the thing to hold on to. Every quantity here is keyed by
*pair* — one authorship-verification problem — while the feature tensors hold two rows per
pair, the known document followed by the unknown one. So a subset of pairs ``i`` selects
feature rows ``2i`` and ``2i + 1``, and forgetting the factor of two produces a tensor
that loads, trains, and means nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from ..original import pipeline as original
from ..space.cost import input_shape
from .repository import PriorRepositoryMissing, reference_root

#: Corpus to the CSV stems holding its problems. The 2020 files carry an ``s`` suffix for
#: the reduced collection the prior campaign used; the full one is a separate artifact.
CORPUS_FILES: dict[str, tuple[str, str]] = {
    "2013": ("training_data2013En", "testing_data2013En"),
    "2014Essay2": ("training_data2014EnEssay", "testing_data2014EnEssay2"),
    "2014Novel2": ("training_data2014EnNovel", "testing_data2014EnNovel2"),
    "2015": ("training_data2015En", "testing_data2015En"),
    "2020": ("training_data2020s", "testing_data2020s"),
}

#: Feature tensor stems per strategy, with ``{}`` taking the corpus token.
FEATURE_FILES: dict[str, tuple[str, str]] = {
    "sem": (
        "double_training_wembs_nochunks_glove300_year{}",
        "double_test_wembs_nochunks_glove300_year{}",
    ),
    "synBetter": (
        "Double2GramPosTfidfsBetterTrainingNoChunkYear{}",
        "Double2GramPosTfidfsBetterTestNoChunkYear{}",
    ),
    "semBert": (
        "double_training_cembs_bert_500_year{}",
        "double_test_cembs_bert_500_year{}",
    ),
}

#: The separator the prior CSVs use. Backslash, with document text in quoted fields that
#: contain newlines — so a line count is not a row count, and reading these with the
#: default separator silently yields one column.
CSV_SEPARATOR = "\\"


@dataclass
class Corpus:
    """One corpus under one feature strategy, ready for the evaluator."""

    corpus: str
    strategy: str
    train_features: torch.Tensor
    test_features: torch.Tensor
    train_documents: list[str]
    test_documents: list[str]
    train_labels: list[int]
    test_labels: list[int]
    #: Feature rows per document. One for the ``nochunks`` tensors; the BERT arm stores
    #: chunk embeddings instead, at three rows per document, which is why this is derived
    #: rather than assumed.
    train_chunks: int = 1
    test_chunks: int = 1

    @property
    def input_shape(self) -> int:
        return input_shape(self.corpus, self.strategy)

    def __post_init__(self) -> None:
        for name, features, labels in (
            ("train", self.train_features, self.train_labels),
            ("test", self.test_features, self.test_labels),
        ):
            documents = 2 * len(labels)
            rows = features.shape[0]
            if documents == 0 or rows % documents:
                raise ValueError(
                    f"{self.corpus}/{self.strategy} {name}: {rows} feature rows do not divide "
                    f"evenly among {documents} documents. Every document must contribute the "
                    "same number of rows for the pair boundaries to be recoverable."
                )
        self.train_chunks = self.train_features.shape[0] // (2 * len(self.train_labels))
        self.test_chunks = self.test_features.shape[0] // (2 * len(self.test_labels))
        if self.train_features.shape[1] * 2 != self.input_shape:
            raise ValueError(
                f"{self.corpus}/{self.strategy}: assembled width {self.train_features.shape[1]} "
                f"doubles to {self.train_features.shape[1] * 2}, but the declared input shape "
                f"is {self.input_shape}"
            )


def _embeddings_dir() -> Path:
    return reference_root() / "PreprocessedEmbeddings" / "float64"


def _datasets_dir() -> Path:
    return reference_root() / "PreprocessedDatasets"


def _load_tensor(stem: str, device: torch.device) -> torch.Tensor:
    path = _embeddings_dir() / f"{stem}.pt"
    if not path.exists():
        raise PriorRepositoryMissing(f"feature tensor not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def load_corpus(corpus: str, strategy: str, device: torch.device | None = None) -> Corpus:
    """Assemble one corpus exactly as the prior pipeline did.

    For the syntactic strategy that means concatenating the stored PoS-bigram TF-IDF with
    a punctuation bag-of-words computed here from the raw text — the step that makes the
    per-document width 485 rather than 453 on CLEF-PAN 2013, and the one most easily
    missed, since the stored tensor looks complete on its own.
    """
    if corpus not in CORPUS_FILES:
        raise KeyError(f"unknown corpus {corpus!r}; expected one of {sorted(CORPUS_FILES)}")
    if strategy not in FEATURE_FILES:
        raise KeyError(f"unknown strategy {strategy!r}; expected one of {sorted(FEATURE_FILES)}")

    device = device or original.DEVICE
    train_stem, test_stem = CORPUS_FILES[corpus]
    train_frame = pd.read_csv(_datasets_dir() / f"{train_stem}.csv", sep=CSV_SEPARATOR)
    test_frame = pd.read_csv(_datasets_dir() / f"{test_stem}.csv", sep=CSV_SEPARATOR)

    train_documents, train_labels, test_documents, test_labels = original.split_data(
        train_frame, test_frame
    )

    train_stem, test_stem = (name.format(corpus) for name in FEATURE_FILES[strategy])
    train_features = _load_tensor(train_stem, device)
    test_features = _load_tensor(test_stem, device)

    if strategy == "synBetter":
        train_punctuation, test_punctuation = original.get_punctuation_bow(
            original.make_df(train_documents, train_labels),
            original.make_df(test_documents, test_labels),
        )
        train_features = torch.cat([train_features, train_punctuation.to(device)], dim=1)
        test_features = torch.cat([test_features, test_punctuation.to(device)], dim=1)

    return Corpus(
        corpus=corpus,
        strategy=strategy,
        train_features=train_features,
        test_features=test_features,
        train_documents=train_documents,
        test_documents=test_documents,
        train_labels=[int(label) for label in train_labels],
        test_labels=[int(label) for label in test_labels],
    )


@dataclass
class Partition:
    """A subset of pairs, with everything the pipeline needs to consume it."""

    features: torch.Tensor
    frame: pd.DataFrame
    chunks_per_document: list[int]
    labels: list[int]

    @property
    def pairs(self) -> int:
        return len(self.labels)


def partition(
    features: torch.Tensor,
    documents: Sequence[str],
    labels: Sequence[int],
    pair_indices: Sequence[int],
    chunks: int = 1,
) -> Partition:
    """Take the pairs at ``pair_indices``, expanding each into its feature rows.

    ``chunks_per_document`` is what the batching loop reads to find pair boundaries, so it
    has to reflect how many rows each document actually contributes: one for the no-chunk
    tensors, three for the BERT arm's chunk embeddings. Hard-coding one would silently
    mis-slice the chunked arm rather than fail.
    """
    rows: list[int] = []
    subset_documents: list[str] = []
    for index in pair_indices:
        for offset in range(2):
            document = 2 * index + offset
            rows.extend(range(document * chunks, (document + 1) * chunks))
            subset_documents.append(documents[document])
    subset_labels = [int(labels[index]) for index in pair_indices]

    return Partition(
        features=features[rows],
        frame=original.make_df(subset_documents, subset_labels),
        chunks_per_document=[chunks] * len(subset_labels) * 2,
        labels=subset_labels,
    )
