"""Manually designed architectures, to answer what the search is worth.

Reviewer 1 put it directly: a comparison limited to an SVM "makes it impossible to
determine whether NAS offers any advantage over classic deep model design". That question
has a precise form, and getting the form right matters more than the models.

**Fair input.** These consume the same feature vectors the searched architectures consume
— the same TF-IDF and punctuation concatenation, the same summed embeddings, the same pair
layout — so the only thing that differs is the architecture. This is the comparison that
answers the question.

**Strong input.** A transformer over raw text is a different question: where any of this
sits relative to the state of the art. It is an upper reference, and reporting it as a
peer would confound architecture with representation in the direction that flatters
whichever side one prefers.

Training matches the searched architectures exactly: Adam at 0.001, batches of four pairs,
`BCEWithLogitsLoss`, gradients clipped at one, the same epoch budget. A baseline trained
better than the thing it is compared against is not a baseline.

The hyperparameter budget is declared rather than tuned to taste. The prior work's SVM is
instructive here: it was tuned with `scoring="accuracy"` over 24 random configurations and
then compared on a different metric, and it is retired rather than repaired because a
reproducible weak baseline answers nothing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .evaluator import BATCH_SIZE, LEARNING_RATE, PARTIAL_EPOCHS, Scores
from .prior.dataset import Corpus, Partition, partition
from .protocol import ProtocolSplit, isolated_randomness
from .search.campaign import Measurement

#: Gradient clipping, matching the searched architectures.
CLIP_NORM = 1.0

#: Hyperparameter configurations tried per baseline family. Small and declared: the point
#: is a fair comparison at a stated effort, not the best manual model obtainable.
#:
#: Selection among them uses the validation split, like everything else. A baseline tuned
#: on test would reproduce the defect this whole protocol exists to remove, on the other
#: side of the comparison.
HYPERPARAMETER_BUDGET = 4


class PairMLP(nn.Module):
    """The obvious manual design: a few wide layers over the concatenated pair.

    Included because it is what a practitioner writes first, and because the searched
    space is dominated by Linear operations — thirteen of its twenty-four values — so this
    is the manual design closest to what the search is drawing from.
    """

    def __init__(self, input_shape: int, width: int = 512, depth: int = 3, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_shape
        for _ in range(depth):
            layers += [
                nn.Linear(current, width),
                nn.BatchNorm1d(width, dtype=torch.float64),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            current = width
        layers.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*layers).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairCNN(nn.Module):
    """A 1D convolution over the concatenated pair, then pooling.

    The searched space contains Conv1D operations, so a hand-designed convolutional model
    is the natural manual counterpart. Pooling rather than flattening, which is what the
    search cannot express: its Conv1D is always followed by a Flatten, so the parameter
    count explodes with the input width and `g₂` exists to contain it.
    """

    def __init__(self, input_shape: int, channels: int = 32, kernel: int = 7, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channels, kernel, padding=kernel // 2),
            nn.BatchNorm1d(channels, dtype=torch.float64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(16),
        ).double()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels * 16, 1),
        ).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.unsqueeze(1)))


class PairLSTM(nn.Module):
    """A recurrent model over the pair read as a two-step sequence.

    The known document, then the unknown one. This is the formulation the searched space
    cannot represent at all — it is strictly sequential over layers, with no recurrence —
    so it tests whether the missing operation matters rather than how well the search
    explores what it has.
    """

    def __init__(self, input_shape: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        if input_shape % 2:
            raise ValueError("a pair input must be an even width: one half per document")
        self.per_document = input_shape // 2
        self.lstm = nn.LSTM(self.per_document, hidden, batch_first=True).double()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1)).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.reshape(x.shape[0], 2, self.per_document)
        output, _ = self.lstm(sequence)
        return self.head(output[:, -1, :])


@dataclass(frozen=True)
class BaselineSpec:
    """One manual family and the configurations tried for it."""

    name: str
    build: Callable[..., nn.Module]
    grid: tuple[dict[str, object], ...]

    def configurations(self) -> Sequence[dict[str, object]]:
        return self.grid[:HYPERPARAMETER_BUDGET]


FAIR_INPUT_BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        name="mlp",
        build=PairMLP,
        grid=(
            {"width": 512, "depth": 3, "dropout": 0.3},
            {"width": 256, "depth": 2, "dropout": 0.3},
            {"width": 1024, "depth": 2, "dropout": 0.5},
            {"width": 128, "depth": 4, "dropout": 0.1},
        ),
    ),
    BaselineSpec(
        name="cnn",
        build=PairCNN,
        grid=(
            {"channels": 32, "kernel": 7, "dropout": 0.3},
            {"channels": 64, "kernel": 5, "dropout": 0.3},
            {"channels": 16, "kernel": 11, "dropout": 0.1},
            {"channels": 32, "kernel": 3, "dropout": 0.5},
        ),
    ),
    BaselineSpec(
        name="lstm",
        build=PairLSTM,
        grid=(
            {"hidden": 128, "dropout": 0.3},
            {"hidden": 256, "dropout": 0.3},
            {"hidden": 64, "dropout": 0.1},
            {"hidden": 128, "dropout": 0.5},
        ),
    ),
)


def pair_matrix(part: Partition, input_shape: int) -> torch.Tensor:
    """Fold a partition's feature rows into one row per pair.

    The stored features hold one row per document — or three, on the chunked BERT arm —
    and the searched architectures see a pair as the concatenation of its two documents.
    Chunks are averaged per document before concatenation, which is the only choice that
    leaves the width equal to the searched architectures' input shape.
    """
    rows = part.features
    pairs = part.pairs
    if pairs == 0:
        return rows.new_zeros((0, input_shape))
    per_pair = rows.shape[0] // pairs
    folded = rows.reshape(pairs, 2, per_pair // 2, rows.shape[1]).mean(dim=2)
    return folded.reshape(pairs, -1)


def _train(
    model: nn.Module,
    features: torch.Tensor,
    labels: Sequence[int],
    epochs: int,
    rng: np.random.Generator,
) -> None:
    """The searched architectures' training loop, over pair rows.

    Mirrored deliberately: same optimizer, learning rate, batch size, loss and clipping.
    The sampling uses an injected generator rather than reseeding a global one, which is
    the defect the original loop has and the reason the search there was not exploring.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    target = torch.tensor(list(labels), dtype=torch.float64)
    steps = max(1, len(labels) // BATCH_SIZE)

    model.train()
    for _ in range(epochs):
        for _ in range(steps):
            picked = rng.choice(len(labels), size=min(BATCH_SIZE, len(labels)), replace=False)
            index = torch.as_tensor(np.sort(picked))
            prediction = model(features[index]).reshape(-1)
            loss = loss_fn(prediction, target[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad()


def _score(model: nn.Module, features: torch.Tensor, labels: Sequence[int]) -> Scores:
    model.eval()
    with torch.no_grad():
        logits = model(features).reshape(-1)
    return Scores(logits=[float(value) for value in logits.cpu()], labels=list(labels))


@dataclass
class BaselineResult:
    """One manual family, its chosen configuration, and what it measured."""

    name: str
    configuration: dict[str, object]
    params: int
    measurement: Measurement
    configurations_tried: int
    input_level: str = "fair"
    alternatives: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "input_level": self.input_level,
            "configuration": self.configuration,
            "configurations_tried": self.configurations_tried,
            "params": self.params,
            "validation_roc_auc": self.measurement.validation_roc_auc,
            "validation_balanced_accuracy": self.measurement.validation_balanced_accuracy,
            "test_roc_auc": self.measurement.test_roc_auc,
            "test_balanced_accuracy": self.measurement.test_balanced_accuracy,
            "seconds": self.measurement.seconds,
            "alternatives": self.alternatives,
        }


def evaluate_baseline(
    spec: BaselineSpec,
    corpus: Corpus,
    split: ProtocolSplit,
    test_indices: Sequence[int],
    epochs: int = PARTIAL_EPOCHS,
    seed: int = 20260920,
) -> BaselineResult:
    """Train every configuration, select on validation, read test once.

    The same protocol the searched architectures are held to. Selecting a baseline on test
    would put the leakage back on the other side of the comparison, which is a subtler way
    of making the same mistake.
    """
    train_part = partition(
        corpus.train_features,
        corpus.train_documents,
        corpus.train_labels,
        split.train,
        corpus.train_chunks,
    )
    validation_part = partition(
        corpus.train_features,
        corpus.train_documents,
        corpus.train_labels,
        split.validation,
        corpus.train_chunks,
    )
    test_part = partition(
        corpus.test_features,
        corpus.test_documents,
        corpus.test_labels,
        test_indices,
        corpus.test_chunks,
    )

    shape = corpus.input_shape
    train_x = pair_matrix(train_part, shape)
    validation_x = pair_matrix(validation_part, shape)
    test_x = pair_matrix(test_part, shape)

    started = time.time()
    best: tuple[float, nn.Module, dict[str, object]] | None = None
    alternatives: list[dict[str, object]] = []

    for index, configuration in enumerate(spec.configurations()):
        with isolated_randomness():
            torch.manual_seed(seed + index)
            model = spec.build(shape, **configuration)
            _train(model, train_x, train_part.labels, epochs, np.random.default_rng(seed + index))
            validation = _score(model, validation_x, validation_part.labels)
        value = validation.roc_auc()
        alternatives.append({**configuration, "validation_roc_auc": round(value, 4)})
        if best is None or value > best[0]:
            best = (value, model, dict(configuration))

    assert best is not None
    score, model, configuration = best
    with isolated_randomness():
        validation = _score(model, validation_x, validation_part.labels)
        test = _score(model, test_x, test_part.labels)

    return BaselineResult(
        name=spec.name,
        configuration=configuration,
        params=sum(p.numel() for p in model.parameters()),
        configurations_tried=len(spec.configurations()),
        alternatives=alternatives,
        measurement=Measurement(
            validation_roc_auc=validation.roc_auc(),
            validation_balanced_accuracy=validation.balanced_accuracy(),
            test_roc_auc=test.roc_auc(),
            test_balanced_accuracy=test.balanced_accuracy(),
            seconds=time.time() - started,
        ),
    )
