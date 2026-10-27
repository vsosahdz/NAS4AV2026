"""Building and training a model from the redesigned space.

Written in plain PyTorch rather than through the extracted pipeline, because the pipeline
encodes the first space's assumptions: it concatenates a pair and nothing else, it pads
with identities, and it has no notion of a normalisation layer. Reusing it would constrain
the second space to the shape of the first, which is the thing being tested.

What is *not* changed is everything outside the architecture. Same corpora, same
three-way split, same optimiser, learning rate, batch size and epoch count, same scoring.
A comparison between the two spaces is only about the spaces if nothing else moves, and
every one of those constants is a place where an accidental improvement could be credited
to the redesign.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ..prior.dataset import Partition
from .v2 import COMPARISON_MULTIPLIER, Architecture, Tables, describe


class PairComparison(nn.Module):
    """Combine two document vectors into one, by the operation the genotype names.

    The first space only ever concatenates, which leaves dense layers to discover that a
    comparison was intended. Authorship verification is a question about the relation
    between two texts, so the comparison is the one decision most likely to carry
    structure — which is the whole reason it is searched here.
    """

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.kind == "concat":
            return torch.cat([left, right], dim=-1)
        if self.kind == "absdiff":
            return torch.abs(left - right)
        if self.kind == "product":
            return left * right
        if self.kind == "concat+absdiff":
            return torch.cat([left, right, torch.abs(left - right)], dim=-1)
        if self.kind == "absdiff+product":
            return torch.cat([torch.abs(left - right), left * right], dim=-1)
        raise ValueError(f"unknown comparison {self.kind!r}")


class ResidualDense(nn.Module):
    """A dense block that may add its input back.

    Only when the widths already match. A residual connection across a width change needs
    a projection, and a projection is a layer the genotype did not ask for — adding one
    silently would mean the architecture evaluated is not the architecture encoded.
    """

    def __init__(self, width_in: int, width_out: int, norm: str | None, residual: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(width_in, width_out)
        self.norm: nn.Module | None
        if norm == "layer":
            self.norm = nn.LayerNorm(width_out)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(width_out)
        else:
            self.norm = None
        self.activation = nn.ReLU()
        self.residual = residual and width_in == width_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.norm is not None:
            # BatchNorm over a batch of one has no variance to normalise by and raises in
            # training mode. A single-sample batch is a legitimate tail of any split, so it
            # is skipped rather than allowed to kill an architecture for a reason that has
            # nothing to do with the architecture.
            if not (isinstance(self.norm, nn.BatchNorm1d) and self.training and out.shape[0] < 2):
                out = self.norm(out)
        out = self.activation(out)
        return out + x if self.residual else out


class SelfAttention(nn.Module):
    """Single-token self-attention over the running representation.

    Degenerate as attention goes — one token attends only to itself — and included anyway
    because what it contributes here is the gated projection, not the mixing. It is in the
    alphabet so the measurement can say whether the search finds it useful; asserting in
    advance that it would be is how the first space acquired operations nobody ablated.
    """

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.heads = heads
        self.scale = (width / heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.query(x), self.key(x), self.value(x)
        weights = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * self.scale)
        return self.out(weights * v)


class V2Model(nn.Module):
    """A full architecture from the redesigned space."""

    def __init__(self, architecture: Architecture) -> None:
        super().__init__()
        self.architecture = architecture
        self.compare = PairComparison(architecture.comparison)
        width = architecture.compared_width
        blocks: list[nn.Module] = []
        for layer in architecture.layers:
            if layer.kind == "dense":
                blocks.append(ResidualDense(width, int(layer.width or width), layer.norm,
                                            layer.residual))
                width = int(layer.width or width)
            elif layer.kind == "dropout":
                blocks.append(nn.Dropout(float(layer.p or 0.0)))
            elif layer.kind == "attention":
                blocks.append(SelfAttention(width, int(layer.heads or 1)))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(width, 1)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        """``pairs`` is (batch, 2, per_document_width)."""
        combined = self.compare(pairs[:, 0, :], pairs[:, 1, :])
        return self.head(self.blocks(combined)).reshape(-1)


def pair_tensor(part: Partition) -> torch.Tensor:
    """Rows of (2, per_document_width) from a partition's chunk-level features.

    The stored features are per *chunk*, and a document may have several, so chunks are
    averaged into a document before the pair is formed. That matches what the first
    space's pipeline does inside `compare_eots_ch`, and doing it differently would make
    the two spaces disagree about what their input is.
    """
    features = part.features
    counts = list(part.chunks_per_document)
    documents: list[torch.Tensor] = []
    position = 0
    for count in counts:
        documents.append(features[position : position + count].mean(dim=0))
        position += count
    stacked = torch.stack(documents)
    return stacked.reshape(stacked.shape[0] // 2, 2, stacked.shape[-1])


def build(
    encoding: str | Sequence[int],
    per_document_width: int,
    using: Tables | None = None,
) -> V2Model:
    """Build under the given alphabets, so an ablation variant builds what it encodes."""
    model = V2Model(describe(encoding, per_document_width, using))
    return model.double()


def train_and_score(
    encoding: str | Sequence[int],
    per_document_width: int,
    train: Partition,
    evaluate_on: dict[str, Partition],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    using: Tables | None = None,
) -> dict[str, Any]:
    """Train one architecture and return its logits on each named partition.

    Returns logits rather than scores so the caller decides the metric, which keeps this
    module free of any opinion about ROC-AUC versus balanced accuracy — a distinction this
    project has already been bitten by once.
    """
    torch.manual_seed(seed)
    model = build(encoding, per_document_width, using)

    inputs = pair_tensor(train)
    labels = torch.tensor(train.labels, dtype=torch.float64)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        order = torch.randperm(inputs.shape[0])
        for start in range(0, inputs.shape[0], batch_size):
            index = order[start : start + batch_size]
            optimiser.zero_grad()
            loss = loss_function(model(inputs[index]), labels[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimiser.step()

    model.eval()
    out: dict[str, Any] = {"params": sum(p.numel() for p in model.parameters())}
    with torch.no_grad():
        for name, part in evaluate_on.items():
            out[name] = [float(v) for v in model(pair_tensor(part))]
    return out
