"""A second search space, designed so that a search could in principle beat sampling.

The first space cannot reward a search, and this is measured rather than argued: fitting
the genotype to held-out performance gives R² at or below zero on nine of eleven settings.
An evolutionary algorithm beats random sampling by recombining parents whose good
components are inherited; where structure carries no information about performance there
are no components to inherit, and every search degenerates to sampling.

So the question this space is built to answer is not "is it better" but **"does it have
anything to exploit"** — which `exploitability` measures directly, on a random sample, for
a fraction of a campaign's cost. Three changes, each aimed at a specific reason the first
space is flat.

**The pair comparison becomes a decision.** The first space concatenates the two documents
and leaves dense layers to discover that a comparison was intended. But authorship
verification is a question about the *relation* between two texts, and the operations that
express it — absolute difference, elementwise product — are exactly the inductive bias the
task wants. Making this the first gene means architectures differ in something that should
matter, which is a precondition for structure predicting performance.

**Normalisation and residual connections enter the alphabet.** 23.8% of evaluations in the
prior campaign landed at exactly the no-discrimination point, which is what a network that
failed to train looks like. Whether an architecture trains at all is strongly determined by
normalisation and skip connections, so putting them in the genotype makes a large part of
the performance variance structurally predictable — the honest way to create exploitable
structure rather than hoping for it.

**Depth is explicit and shorter.** The first space is fifteen positions where `Identity`
is one of twenty-four values, so most genotypes are mostly padding and two encodings that
differ in several genes can describe the same network. Eight positions with an explicit
terminator makes depth a decision rather than an accident of how many identities were
drawn.

Nothing here is claimed to perform better. It is claimed to be measurable, and the
measurement is the point.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: Genes: one comparison, then seven block positions.
LENGTH = 8

#: How the two document vectors are combined before anything else. The first space has
#: only `concat`; the rest are the standard pair-encoding operations from the verification
#: literature, and their output widths differ, which is why the gene has to be read before
#: any block can be sized.
COMPARISONS: dict[int, str] = {
    0: "concat",
    1: "absdiff",
    2: "product",
    3: "concat+absdiff",
    4: "absdiff+product",
}

#: Output width of a comparison, as a multiple of the per-document width.
COMPARISON_MULTIPLIER: dict[str, int] = {
    "concat": 2,
    "absdiff": 1,
    "product": 1,
    "concat+absdiff": 3,
    "absdiff+product": 2,
}

#: Block operations. A value is a whole block rather than a bare layer, because "Linear
#: then LayerNorm then ReLU" is the unit a designer actually chooses, and splitting it
#: across three genes makes most of the space describe combinations nobody would write.
BLOCKS: dict[int, dict[str, Any]] = {
    0: {"kind": "stop"},
    **{
        1 + index: {"kind": "dense", "width": 2**exponent, "norm": None, "residual": False}
        for index, exponent in enumerate(range(5, 11))
    },
    **{
        7 + index: {"kind": "dense", "width": 2**exponent, "norm": "layer", "residual": False}
        for index, exponent in enumerate(range(5, 11))
    },
    **{
        13 + index: {"kind": "dense", "width": 2**exponent, "norm": "layer", "residual": True}
        for index, exponent in enumerate(range(6, 9))
    },
    16: {"kind": "dropout", "p": 0.3},
    17: {"kind": "dropout", "p": 0.5},
    18: {"kind": "dense", "width": 256, "norm": "batch", "residual": False},
    19: {"kind": "attention", "heads": 4},
}

ALPHABET = max(len(COMPARISONS), len(BLOCKS))

#: Size of the space, for the record. Smaller than the first space's 24^15 = 5.0e20 on
#: purpose: a space is not better for being larger, and the first one's size was mostly
#: padding and unreachable combinations.
SPACE_SIZE = len(COMPARISONS) * len(BLOCKS) ** (LENGTH - 1)


class MalformedEncoding(ValueError):
    """Raised when an encoding is not a length-8 sequence over the declared alphabets."""


def parse(encoding: str | Sequence[int]) -> tuple[int, ...]:
    genes = (
        tuple(int(t) for t in encoding.split(","))
        if isinstance(encoding, str)
        else tuple(int(g) for g in encoding)
    )
    if len(genes) != LENGTH:
        raise MalformedEncoding(f"expected {LENGTH} genes, got {len(genes)}")
    if genes[0] not in COMPARISONS:
        raise MalformedEncoding(f"gene 0 must be a comparison in {sorted(COMPARISONS)}")
    for gene in genes[1:]:
        if gene not in BLOCKS:
            raise MalformedEncoding(f"block gene {gene} is outside {sorted(BLOCKS)}")
    return genes


def feasible(encoding: str | Sequence[int]) -> bool:
    """Every well-formed encoding is feasible.

    Deliberately so. The first space's constraints exist to bound memory on a 4 GB card —
    `g₁` caps wide convolutions and `g₂` forbids a convolution next to a wide layer — and
    they reject about half of all draws, which makes a uniform sample over the feasible
    region something a rejection sampler has to find rather than something anyone can
    reason about. Here the widths are bounded by the alphabet instead, so feasibility is a
    property of the encoding's shape and nothing has to be rejected.
    """
    try:
        parse(encoding)
    except MalformedEncoding:
        return False
    return True


@dataclass
class Layer:
    """One realised layer, after depth and widths are resolved."""

    kind: str
    width: int | None = None
    norm: str | None = None
    residual: bool = False
    p: float | None = None
    heads: int | None = None


@dataclass
class Architecture:
    """A decoded genotype, with everything derivable from it."""

    encoding: str
    comparison: str
    input_width: int
    compared_width: int
    layers: list[Layer]
    params: int
    final_width: int

    @property
    def depth(self) -> int:
        """Layers actually realised, which is explicit here rather than inferred."""
        return len(self.layers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "comparison": self.comparison,
            "compared_width": self.compared_width,
            "depth": self.depth,
            "params": self.params,
            "final_width": self.final_width,
            "has_norm": any(layer.norm for layer in self.layers),
            "has_residual": any(layer.residual for layer in self.layers),
        }


def describe(encoding: str | Sequence[int], per_document_width: int) -> Architecture:
    """Decode a genotype into its layers and exact parameter count.

    ``per_document_width`` is the width of **one** document's feature vector, not the pair.
    The first space takes the pair width and halves it internally, which is a place that
    has already cost a 2x error once; naming the per-document width here makes the mistake
    harder to make.
    """
    genes = parse(encoding)
    comparison = COMPARISONS[genes[0]]
    width = per_document_width * COMPARISON_MULTIPLIER[comparison]
    compared_width = width

    layers: list[Layer] = []
    params = 0
    for gene in genes[1:]:
        block = BLOCKS[gene]
        kind = block["kind"]
        if kind == "stop":
            break
        if kind == "dense":
            out = int(block["width"])
            # A residual block only connects when the widths already match; otherwise it
            # would need a projection, which is a different block than the genotype names.
            residual = bool(block["residual"]) and out == width
            params += width * out + out
            if block["norm"] == "layer":
                params += 2 * out
            elif block["norm"] == "batch":
                params += 2 * out
            layers.append(Layer("dense", out, block["norm"], residual))
            width = out
        elif kind == "dropout":
            layers.append(Layer("dropout", p=float(block["p"])))
        elif kind == "attention":
            heads = int(block["heads"])
            # Self-attention over the feature vector treated as one token per head group;
            # query, key, value and output projections, each square in the running width.
            params += 4 * (width * width + width)
            layers.append(Layer("attention", width, heads=heads))

    # The decision head, which every architecture carries and which is not searched.
    params += width + 1
    return Architecture(
        encoding=",".join(str(g) for g in genes),
        comparison=comparison,
        input_width=per_document_width,
        compared_width=compared_width,
        layers=layers,
        params=params,
        final_width=width,
    )


def random_encoding(rng: Any) -> tuple[int, ...]:
    """A uniform draw. No rejection, because every well-formed encoding is feasible."""
    return (
        rng.randrange(len(COMPARISONS)),
        *(rng.randrange(len(BLOCKS)) for _ in range(LENGTH - 1)),
    )


def space_summary() -> dict[str, Any]:
    """What the space contains, for the record and for the manuscript's table."""
    return {
        "length": LENGTH,
        "comparisons": list(COMPARISONS.values()),
        "block_alphabet": len(BLOCKS),
        "size": SPACE_SIZE,
        "size_scientific": f"{SPACE_SIZE:.2e}",
        "every_encoding_feasible": True,
        "blocks_with_norm": sum(
            1 for b in BLOCKS.values() if b.get("norm") not in (None, False)
        ),
        "blocks_with_residual": sum(1 for b in BLOCKS.values() if b.get("residual")),
        "rationale": {
            "comparison_is_searched": (
                "authorship verification asks about the relation between two texts, so how "
                "the pair is combined is the decision most likely to matter; the first "
                "space concatenates and never varies it"
            ),
            "norm_and_residual_are_searched": (
                "whether an architecture trains at all is largely set by these, and 23.8% "
                "of the prior campaign sat at the no-discrimination point — the signature "
                "of networks that did not train"
            ),
            "depth_is_explicit": (
                "the first space pads with Identity, so most genotypes are mostly padding "
                "and several encodings describe one network"
            ),
        },
    }


def log10_size() -> float:
    return math.log10(SPACE_SIZE)
