"""The strong-input reference: a pretrained encoder over raw text, with nothing fine-tuned.

D5 asks for this as an *upper reference* rather than a peer competitor — the question is
where any of this work sits relative to a model that was trained on far more text than
these corpora contain. The question answering the reviewer's objection about manual design
is the fair-input one, over identical features, and that lives in ``baselines``.

D5 also assumed a fine-tuned transformer. Measuring the corpora first ruled that out, and
not for a reason about compute: the training splits are **10 pairs for 2013 and 70 for
2020**. Fine-tuning a pretrained encoder on seventy examples produces a number whose
variance across seeds exceeds any difference it is meant to establish, which is a worse
outcome than not running it — a reference that cannot be trusted still gets cited as one.

So nothing here is trained. A document becomes the mean of its chunk embeddings, a pair
becomes the cosine between its two documents, and that number is the score. The
consequences are worth being explicit about, because they are what make this reference
honest rather than merely cheap:

**ROC-AUC needs no training at all.** It reads a ranking, and the ranking exists the moment
the embeddings do. So the headline number involves no fitting, no seed, and no corpus-size
floor — it is the same measurement on ten pairs as on ten thousand.

**Balanced accuracy needs a threshold, and a threshold is a fitted parameter.** One scalar,
fitted on the training side and never on test. That is the protocol the whole repair rests
on, and a reference baseline is not exempt from it.

**Every pair overflows the context window**, the shortest corpus by 1.7× and 2020 by 20.5×.
Truncation would make this a first-paragraph baseline that loses for reasons unrelated to
architecture, so documents are chunked and pooled, and the chunk count is recorded per
corpus so a reader can see how much aggregation each number rests on.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .prior.dataset import CSV_SEPARATOR
from .prior.repository import reference_root

#: The encoder, pinned by name and revision. A checkpoint that moves under us would change
#: every number here with nothing in the artifact to show it, so the revision travels into
#: the provenance rather than being left as "the latest".
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "main"

#: Token window per chunk, and the overlap between consecutive chunks. The overlap exists
#: so that a sentence split across a boundary is still seen whole somewhere.
CHUNK_TOKENS = 384
CHUNK_STRIDE = 64

#: Which raw-text file belongs to which corpus. Taken from the preprocessing notebook
#: rather than guessed from the names, because `2014EnEssay` trains against `Essay2` at
#: test time and the pairing is not derivable from the filenames.
RAW_FILES: dict[str, tuple[str, str]] = {
    "2013": ("training_data2013En", "testing_data2013En"),
    "2014Essay2": ("training_data2014EnEssay", "testing_data2014EnEssay2"),
    "2014Novel2": ("training_data2014EnNovel", "testing_data2014EnNovel2"),
    "2015": ("training_data2015En", "testing_data2015En"),
    "2020": ("training_data2020s", "testing_data2020s"),
    "aggregate": ("big_guy_training", "big_guy_test"),
}


class RawTextMissing(RuntimeError):
    """Raised when the raw-text CSVs are not where the reference clone puts them."""


def _raise_csv_field_limit() -> None:
    """Let a single field exceed Python's 128 KB default.

    A whole document sits in one CSV field here, and the aggregate corpus holds documents
    of nearly 16,000 words — comfortably past the limit. The failure is a `ParserError`
    about a field larger than the limit, which reads like a malformed file rather than a
    reader's default, so the five individual corpora load and only the aggregate does not.

    Raised to the platform maximum by stepping down until one is accepted: the limit is a C
    long, so `sys.maxsize` overflows on some builds and the usable ceiling is not knowable
    in advance.
    """
    import csv

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


@dataclass
class RawPairs:
    """One split's raw text, as the CSVs store it."""

    corpus: str
    split: str
    first: list[str]
    second: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.labels)


def load_raw(corpus: str, split: str = "train") -> RawPairs:
    """Read the raw document pairs for one corpus and split.

    Separated by a backslash, which is not a typo in this module: the preprocessing
    notebook wrote these with ``sep='\\\\'`` and a comma-separated read silently returns a
    single column whose name is the whole header.
    """
    if corpus not in RAW_FILES:
        raise RawTextMissing(f"no raw-text mapping for corpus {corpus!r}")
    stem = RAW_FILES[corpus][0 if split == "train" else 1]
    path = Path(reference_root()) / "PreprocessedDatasets" / f"{stem}.csv"
    if not path.exists():
        raise RawTextMissing(f"{path} is absent; the reference clone is incomplete")
    _raise_csv_field_limit()
    frame = pd.read_csv(path, sep=CSV_SEPARATOR, engine="python")
    missing = {"text1", "text2", "label"} - set(frame.columns)
    if missing:
        raise RawTextMissing(f"{path} lacks {sorted(missing)}")
    return RawPairs(
        corpus=corpus,
        split=split,
        first=[str(v) for v in frame["text1"]],
        second=[str(v) for v in frame["text2"]],
        labels=[int(float(v)) for v in frame["label"]],
    )


@dataclass
class Encoder:
    """A frozen pretrained encoder, with its identity recorded.

    Loaded lazily so that importing this module costs nothing and the fair-input baselines
    never pull a transformer they do not use.
    """

    model_name: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    device: str = "cpu"
    _tokenizer: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self._model = AutoModel.from_pretrained(self.model_name, revision=self.revision)
        self._model.eval()
        self._model.to(self.device)
        # Inference only, stated rather than implied. A gradient here would be a bug with no
        # symptom: the numbers would come out the same and the memory would not.
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        torch.set_grad_enabled(False)

    def chunk_count(self, text: str) -> int:
        self.load()
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        step = max(CHUNK_TOKENS - CHUNK_STRIDE, 1)
        return max(1, -(-max(len(ids) - CHUNK_STRIDE, 1) // step))

    def embed(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Mean-pooled embeddings, one row per document, chunked and averaged.

        Two poolings, which is deliberate and worth naming. Within a chunk, tokens are
        mean-pooled with the attention mask so padding contributes nothing. Across chunks, a
        document is the unweighted mean of its chunks — unweighted because a long chunk is
        not more about the author than a short one, and weighting by length would let the
        tail of a document outvote its opening for no reason anybody could defend.
        """
        import torch

        self.load()
        out = np.zeros((len(texts), int(self._model.config.hidden_size)), dtype=np.float64)
        for index, text in enumerate(texts):
            ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
            step = max(CHUNK_TOKENS - CHUNK_STRIDE, 1)
            windows = [ids[start : start + CHUNK_TOKENS] for start in range(0, max(len(ids), 1), step)]
            windows = [w for w in windows if w] or [ids[:CHUNK_TOKENS] or [self._tokenizer.cls_token_id or 0]]
            vectors = []
            for position in range(0, len(windows), batch_size):
                batch = windows[position : position + batch_size]
                encoded = self._tokenizer.pad(
                    {"input_ids": [self._tokenizer.build_inputs_with_special_tokens(w) for w in batch]},
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                hidden = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                vectors.append(pooled.double().cpu().numpy())
            out[index] = np.vstack(vectors).mean(axis=0)
        return out

    def identity(self) -> dict[str, Any]:
        self.load()
        return {
            "model": self.model_name,
            "revision": self.revision,
            "hidden_size": int(self._model.config.hidden_size),
            "max_position_embeddings": int(
                getattr(self._model.config, "max_position_embeddings", -1)
            ),
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_stride": CHUNK_STRIDE,
            "fine_tuned": False,
        }


def cosine_scores(encoder: Encoder, pairs: RawPairs) -> np.ndarray:
    """One score per pair: the cosine between the two documents' embeddings."""
    left = encoder.embed(pairs.first)
    right = encoder.embed(pairs.second)
    numerator = (left * right).sum(axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def fit_threshold(scores: np.ndarray, labels: Sequence[int]) -> float:
    """The one fitted parameter, chosen on the training side only.

    Picked to maximise balanced accuracy over the midpoints between observed scores, which
    is exact rather than a grid: balanced accuracy is piecewise constant in the threshold
    and only changes as the threshold crosses an observed value, so there is nothing
    between the midpoints to search.
    """
    labels_array = np.asarray(labels, dtype=int)
    if labels_array.size == 0 or len(set(labels_array.tolist())) < 2:
        return float(np.median(scores)) if scores.size else 0.0
    ordered = np.unique(scores)
    candidates = (
        (ordered[:-1] + ordered[1:]) / 2.0 if ordered.size > 1 else np.array([ordered[0]])
    )
    best, best_score = float(candidates[0]), -1.0
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        value = _balanced_accuracy(predicted, labels_array)
        if value > best_score:
            best, best_score = float(threshold), value
    return best


def _balanced_accuracy(predicted: np.ndarray, labels: np.ndarray) -> float:
    recalls = []
    for value in (0, 1):
        mask = labels == value
        if mask.any():
            recalls.append(float((predicted[mask] == value).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _roc_auc(scores: np.ndarray, labels: Sequence[int]) -> float | None:
    """Rank-based AUC, which needs no threshold and therefore no fitting."""
    labels_array = np.asarray(labels, dtype=int)
    positives = labels_array == 1
    if not positives.any() or positives.all():
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Mid-ranks for ties, so that a block of identical scores cannot be credited as though
    # it were separated. Ties are common here: identical documents score exactly 1.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            members = inverse == index
            ranks[members] = ranks[members].mean()
    n_pos = int(positives.sum())
    n_neg = int(len(labels_array) - n_pos)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class StrongInputResult:
    """What the reference achieved on one corpus, with its limits attached."""

    corpus: str
    encoder: dict[str, Any]
    train_pairs: int
    test_pairs: int
    threshold: float
    test_roc_auc: float | None
    test_balanced_accuracy: float
    train_balanced_accuracy: float
    chunks_per_document: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "encoder": self.encoder,
            "train_pairs": self.train_pairs,
            "test_pairs": self.test_pairs,
            "threshold": round(self.threshold, 6),
            "test_roc_auc": None if self.test_roc_auc is None else round(self.test_roc_auc, 6),
            "test_balanced_accuracy": round(self.test_balanced_accuracy, 6),
            "train_balanced_accuracy": round(self.train_balanced_accuracy, 6),
            "chunks_per_document": {k: round(v, 2) for k, v in self.chunks_per_document.items()},
            "role": (
                "upper reference, not a peer competitor: this model saw far more text in "
                "pretraining than these corpora contain, and reads raw documents rather "
                "than the feature vectors the searched architectures are restricted to"
            ),
            "nothing_fine_tuned": (
                "no gradient step is taken anywhere. ROC-AUC is therefore free of any "
                "fitting at all; the single fitted parameter is the decision threshold, "
                "chosen on the training split and never on test"
            ),
        }


def evaluate(corpus: str, encoder: Encoder | None = None) -> StrongInputResult:
    """The reference on one corpus, under the repaired protocol."""
    encoder = encoder or Encoder()
    train = load_raw(corpus, "train")
    test = load_raw(corpus, "test")

    train_scores = cosine_scores(encoder, train)
    threshold = fit_threshold(train_scores, train.labels)
    test_scores = cosine_scores(encoder, test)

    chunks = {
        "train": float(
            np.mean([encoder.chunk_count(t) for t in (*train.first, *train.second)])
        ),
        "test": float(np.mean([encoder.chunk_count(t) for t in (*test.first, *test.second)])),
    }
    return StrongInputResult(
        corpus=corpus,
        encoder=encoder.identity(),
        train_pairs=len(train),
        test_pairs=len(test),
        threshold=threshold,
        test_roc_auc=_roc_auc(test_scores, test.labels),
        test_balanced_accuracy=_balanced_accuracy(
            (test_scores >= threshold).astype(int), np.asarray(test.labels, dtype=int)
        ),
        train_balanced_accuracy=_balanced_accuracy(
            (train_scores >= threshold).astype(int), np.asarray(train.labels, dtype=int)
        ),
        chunks_per_document=chunks,
    )


def available() -> bool:
    """Whether the raw text is present, so a caller can skip rather than fail."""
    try:
        return (Path(reference_root()) / "PreprocessedDatasets").is_dir()
    except Exception:
        return False


def offline() -> bool:
    """Whether model downloads are disabled, which makes a first run impossible."""
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
