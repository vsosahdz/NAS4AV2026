"""Rebuilding the feature tensors from raw text.

The campaign has always rested on `.pt` files that nobody can regenerate: they were
produced by notebook cells whose save calls are commented out, and the vocabulary that
fixes their column order was never stored. That is survivable while the corpora are fixed
and fatal the moment one is added, because a new corpus has to be vectorised against
*something* and the only honest choice is the same pipeline at the same versions.

So this rebuilds all of them, through the extracted generators rather than a
reimplementation.

The one thing it cannot do is reproduce the original column order. `TfidfVectorizer`
orders its vocabulary alphabetically, so a handful of POS bigrams that differ between
NLTK versions shifts every column after the first change. Measured on CLEF-PAN 2015: the
rebuilt tensor has the stored shape exactly, and a median cosine of 0.29 against it cell
by cell — but 0.9996 once each row's values are sorted. Same features, different
addresses.

Which is why `verify_against_stored` compares sorted rows. A cell-wise check would read
a faithful rebuild as a total failure, and a shape check would pass a rebuild that was
genuinely wrong.
"""

from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .prior.dataset import CSV_SEPARATOR, CORPUS_FILES, FEATURE_FILES
from .prior.repository import reference_root

#: Where rebuilt tensors are written, and where `load_corpus` reads them when
#: NAS4AV_FEATURES points here. Separate from the reference clone on purpose: mixing a
#: rebuilt tensor with a stored one is the failure this module exists to prevent, and
#: writing into the same directory would make it a matter of filename luck.
def rebuilt_root() -> Path:
    return Path(os.environ.get("NAS4AV_FEATURES", "artifacts/features")).resolve()


@dataclass
class RebuildReport:
    """What one rebuild produced, and how it compares to what was stored."""

    corpus: str
    strategy: str
    train_shape: tuple[int, ...]
    test_shape: tuple[int, ...]
    stored_train_shape: tuple[int, ...] | None
    sorted_cosine_median: float | None
    cellwise_cosine_median: float | None
    nonzeros_per_row: float
    stored_nonzeros_per_row: float | None

    @property
    def shape_matches(self) -> bool:
        return self.stored_train_shape is None or self.train_shape == self.stored_train_shape

    @property
    def faithful(self) -> bool:
        """Whether the rebuild reproduces the stored features up to column order."""
        return (
            self.shape_matches
            and self.sorted_cosine_median is not None
            and self.sorted_cosine_median > 0.99
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "strategy": self.strategy,
            "train_shape": list(self.train_shape),
            "test_shape": list(self.test_shape),
            "stored_train_shape": list(self.stored_train_shape)
            if self.stored_train_shape
            else None,
            "shape_matches": self.shape_matches,
            "sorted_cosine_median": round(self.sorted_cosine_median, 6)
            if self.sorted_cosine_median is not None
            else None,
            "cellwise_cosine_median": round(self.cellwise_cosine_median, 6)
            if self.cellwise_cosine_median is not None
            else None,
            "nonzeros_per_row": round(self.nonzeros_per_row, 1),
            "stored_nonzeros_per_row": round(self.stored_nonzeros_per_row, 1)
            if self.stored_nonzeros_per_row is not None
            else None,
            "faithful": self.faithful,
            "note": (
                "cosine is computed on each row's sorted values: the rebuild reproduces "
                "the features but not their column order, because the TF-IDF vocabulary "
                "is ordered alphabetically and a few differing POS bigrams shift it"
            ),
        }


def _frames(corpus: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_stem, test_stem = CORPUS_FILES[corpus]
    directory = reference_root() / "PreprocessedDatasets"
    # Read exactly as `load_corpus` reads them, default engine and all. The Python engine
    # looked like the safer choice and is not: it enforces a 128 KB field limit that a
    # whole document can exceed, and more importantly, a rebuild that parsed its input
    # differently from the pipeline would be a second source of divergence hiding behind
    # the first.
    return (
        pd.read_csv(directory / f"{train_stem}.csv", sep=CSV_SEPARATOR),
        pd.read_csv(directory / f"{test_stem}.csv", sep=CSV_SEPARATOR),
    )


def _documents(corpus: str):
    from .original import pipeline as original

    train_frame, test_frame = _frames(corpus)
    with contextlib.redirect_stdout(io.StringIO()):
        return original.split_data(train_frame, test_frame)


def build_syntactic(corpus: str) -> tuple[torch.Tensor, torch.Tensor]:
    """POS-bigram TF-IDF, with the vocabulary learned on training only.

    The punctuation bag-of-words is *not* concatenated here. `load_corpus` adds it at run
    time from raw text, which is what makes the per-document width `tensor + 32`; building
    it in would double-count it and change every input shape by 32.
    """
    from .original import pipeline as original

    train_documents, train_labels, test_documents, test_labels = _documents(corpus)
    with contextlib.redirect_stdout(io.StringIO()):
        train_pos = original.get_pos_df(original.make_df(train_documents, train_labels))
        test_pos = original.get_pos_df(original.make_df(test_documents, test_labels))
        produced = original.get_tfidf_df((train_pos, test_pos))
    train_tensor, test_tensor = produced[0], produced[1]
    return train_tensor.cpu().double(), test_tensor.cpu().double()


def build_semantic(corpus: str, embeddings: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Averaged GloVe-300 word embeddings, one row per document."""
    from .original import pipeline as original

    train_documents, train_labels, test_documents, test_labels = _documents(corpus)
    with contextlib.redirect_stdout(io.StringIO()):
        train_tensor = original.get_wembs_df(
            original.make_df(train_documents, train_labels), embeddings
        )
        test_tensor = original.get_wembs_df(
            original.make_df(test_documents, test_labels), embeddings
        )
    for tensor in (train_tensor, test_tensor):
        if not torch.is_tensor(tensor):
            raise TypeError(f"expected a tensor from get_wembs_df, got {type(tensor).__name__}")
    return train_tensor.cpu().double(), test_tensor.cpu().double()


def load_glove() -> Any:
    """The pretrained vectors the semantic arm averages.

    Downloaded through gensim, which is how the notebook obtained them
    (`gensim.downloader.load('glove-wiki-gigaword-300')`). About 376 MB on first use and
    cached afterwards.
    """
    import gensim.downloader

    return gensim.downloader.load("glove-wiki-gigaword-300")


def _stored(corpus: str, strategy: str) -> torch.Tensor | None:
    stem = FEATURE_FILES[strategy][0].format(corpus)
    path = reference_root() / "PreprocessedEmbeddings" / "float64" / f"{stem}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False).cpu().double()


def _cosines(left: np.ndarray, right: np.ndarray) -> float:
    numerator = (left * right).sum(axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return float(np.median(numerator / np.clip(denominator, 1e-12, None)))


def verify_against_stored(
    corpus: str, strategy: str, train_tensor: torch.Tensor, test_tensor: torch.Tensor
) -> RebuildReport:
    """Compare a rebuild with the stored tensor, on sorted rows.

    Sorted because the rebuild reproduces the features and not their column order, and a
    cell-wise comparison reads a faithful rebuild as a total failure — measured at 0.29
    against 0.9996 on the same pair of tensors.
    """
    stored = _stored(corpus, strategy)
    rebuilt = train_tensor.numpy()
    if stored is None:
        return RebuildReport(
            corpus=corpus,
            strategy=strategy,
            train_shape=tuple(train_tensor.shape),
            test_shape=tuple(test_tensor.shape),
            stored_train_shape=None,
            sorted_cosine_median=None,
            cellwise_cosine_median=None,
            nonzeros_per_row=float((rebuilt != 0).sum(axis=1).mean()),
            stored_nonzeros_per_row=None,
        )

    reference = stored.numpy()
    comparable = reference.shape == rebuilt.shape
    return RebuildReport(
        corpus=corpus,
        strategy=strategy,
        train_shape=tuple(train_tensor.shape),
        test_shape=tuple(test_tensor.shape),
        stored_train_shape=tuple(stored.shape),
        sorted_cosine_median=_cosines(np.sort(rebuilt, axis=1), np.sort(reference, axis=1))
        if comparable
        else None,
        cellwise_cosine_median=_cosines(rebuilt, reference) if comparable else None,
        nonzeros_per_row=float((rebuilt != 0).sum(axis=1).mean()),
        stored_nonzeros_per_row=float((reference != 0).sum(axis=1).mean()),
    )


def write(corpus: str, strategy: str, train_tensor: torch.Tensor, test_tensor: torch.Tensor) -> Path:
    """Save a rebuild under the names `load_corpus` expects."""
    directory = rebuilt_root()
    directory.mkdir(parents=True, exist_ok=True)
    train_stem, test_stem = (name.format(corpus) for name in FEATURE_FILES[strategy])
    torch.save(train_tensor, directory / f"{train_stem}.pt")
    torch.save(test_tensor, directory / f"{test_stem}.pt")
    return directory


def rebuild(corpus: str, strategy: str, embeddings: Any | None = None) -> RebuildReport:
    """Rebuild one corpus under one strategy, write it, and report the comparison."""
    if strategy == "synBetter":
        train_tensor, test_tensor = build_syntactic(corpus)
    elif strategy == "sem":
        if embeddings is None:
            raise ValueError("the semantic arm needs pretrained vectors; pass load_glove()")
        train_tensor, test_tensor = build_semantic(corpus, embeddings)
    else:
        raise ValueError(
            f"{strategy!r} cannot be rebuilt here: the semBert arm stores chunk embeddings "
            "from a transformer and its generator is not among the extracted cells"
        )
    write(corpus, strategy, train_tensor, test_tensor)
    return verify_against_stored(corpus, strategy, train_tensor, test_tensor)
