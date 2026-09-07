"""Exact parameter counts, and the input width the first layer actually receives.

Parameter count is the second reported axis alongside predictive performance, and it has
a property performance does not: it is structural rather than learned, so no aspect of
the evaluation protocol can bias it.

It is exact rather than measured. ``Model.build_sequential`` decodes a genotype
deterministically, so the count is closed-form and no forward pass is needed. That is
what makes a front over tens of thousands of architectures affordable on a laptop.

Two transformations sit between the stored feature tensor and the first layer, and
missing either of them makes every parameter count wrong. Both are properties of the
prior implementation and neither is obvious from the feature files alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .genotype import CATEGORIES, CONV_OPERATIONS, LENGTH, categories, linear_width, parse

#: Width of the punctuation bag-of-words concatenated onto the syntactic features.
#: ``get_punctuation_bow`` builds ``[0] * 32``, one slot per character of
#: ``string.punctuation``, indexed by ``string.punctuation.index(char)``. Fixed, so it
#: does not vary by corpus even though the TF-IDF width does.
PUNCTUATION_WIDTH = 32

#: A document *pair* is the concatenation of two document vectors, so the search space is
#: constructed over twice the per-document width:
#:
#:      Sequential_Search_Space(syn_embsz * 2)
#:      Sequential_Search_Space(300 * 2)
#:
#: which is also why ``Model`` receives ``tel = input_shape // 2`` as the per-document
#: embedding length.
PAIR_FACTOR = 2

#: Width of the stored feature tensor, per corpus and feature strategy, read from
#: ``PreprocessedEmbeddings/float64``.
TENSOR_WIDTH: dict[tuple[str, str], int] = {
    ("2013", "synBetter"): 453,
    ("2014Essay2", "synBetter"): 801,
    ("2014Novel2", "synBetter"): 725,
    ("2015", "synBetter"): 653,
    ("2020", "synBetter"): 804,
    ("2013", "sem"): 300,
    ("2014Essay2", "sem"): 300,
    ("2014Novel2", "sem"): 300,
    ("2015", "sem"): 300,
    ("2020", "sem"): 300,
    # BERT features, present for CLEF-PAN 2013 only.
    ("2013", "semBert"): 768,
}

#: Strategies whose features are the TF-IDF tensor concatenated with punctuation counts.
CONCATENATES_PUNCTUATION = frozenset({"synBetter"})


def input_shape(corpus: str, strategy: str) -> int:
    """The width ``Sequential_Search_Space`` is constructed with.

    Cross-check: the prior implementation hardcodes ``syn_embsz = 485`` for CLEF-PAN
    2013, and 453 + 32 = 485. That equality is the only independent confirmation
    available for the punctuation width, so it is also asserted in the tests.
    """
    width = TENSOR_WIDTH[(corpus, strategy)]
    if strategy in CONCATENATES_PUNCTUATION:
        width += PUNCTUATION_WIDTH
    return width * PAIR_FACTOR


@dataclass(frozen=True)
class Architecture:
    """A decoded genotype and the structural quantities the analysis reports over.

    Everything here is derivable from the encoding and the input width, so an
    ``Architecture`` can be built for a genotype that has never been trained. That is
    what lets the parameter axis of a front be complete while the performance axis is
    still being measured.
    """

    encoding: str
    input_shape: int
    params: int
    final_width: int
    max_linear_width: int
    min_linear_width: int
    effective_depth: int
    counts: dict[str, int]

    @property
    def n_linear(self) -> int:
        return self.counts["Linear"]

    @property
    def n_conv1d(self) -> int:
        return self.counts["Conv1D"]

    @property
    def n_dropout(self) -> int:
        return self.counts["Dropout"]

    @property
    def n_activation(self) -> int:
        return self.counts["Activation"]

    @property
    def n_identity(self) -> int:
        return self.counts["Identity"]


def describe(encoding: str | Sequence[int], input_width: int) -> Architecture:
    """Decode a genotype into its exact parameter count and structural features.

    The counting rule follows ``Model.build_sequential`` operation by operation:

        gene 0        Identity()                        no parameters
        gene 1-3      Dropout(p)                        no parameters
        gene 4-6      ReLU / TanH / LeakyReLU           no parameters
        gene 7-19     Linear(curr, out)                 curr * out + out
                      BatchNorm1d(out)                  2 * out
        gene 20-23    Conv1d(1, channels, kernel)       channels * kernel + channels
                      Flatten
                      BatchNorm1d(new)                  2 * new
        tail          Linear(curr, 1)                   curr + 1

    The shape after a Conv1D deserves attention: when the kernel is wider than the
    incoming vector, the prior implementation sets the new width to the channel count
    rather than padding the input up to the kernel size. Padding is the more common
    choice and would give a different width, so the rule is transcribed from the
    implementation rather than assumed.
    """
    genes = parse(encoding)
    params = 0
    curr = input_width
    widths: list[int] = []
    counts = dict.fromkeys(CATEGORIES, 0)
    for name in categories(genes):
        counts[name] += 1

    for gene in genes:
        if gene < 7:
            continue
        if gene <= 19:
            out = linear_width(gene)
            params += curr * out + out  # Linear
            params += 2 * out  # BatchNorm1d, affine
            curr = out
            widths.append(out)
        else:
            kernel, channels = CONV_OPERATIONS[gene]
            params += channels * kernel + channels  # Conv1d(1, channels, kernel)
            curr = channels if kernel > curr else (curr - kernel + 1) * channels
            params += 2 * curr  # BatchNorm1d over the flattened width

    params += curr + 1  # tail Linear(curr, 1), forcing a scalar logit

    return Architecture(
        encoding=encoding if isinstance(encoding, str) else ",".join(str(g) for g in genes),
        input_shape=input_width,
        params=params,
        final_width=curr,
        max_linear_width=max(widths) if widths else 0,
        min_linear_width=min(widths) if widths else 0,
        effective_depth=LENGTH - counts["Identity"],
        counts=counts,
    )


def pareto_front(points: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    """Non-dominated set, minimising parameters and maximising score.

    Ties on parameter count are resolved to the better score before the sweep, so two
    architectures of identical size never both appear.

    A front computed this way is only as informative as the sampling behind it. A search
    guided by performance alone never samples the small-parameter region on purpose, so
    fronts drawn from one are sparse — a handful of points out of thousands of distinct
    parameter values. That sparsity is a property of the sample rather than of the space,
    and is reported alongside any front drawn from such a sample.
    """
    best_at: dict[int, float] = {}
    for size, score in points:
        if size not in best_at or score > best_at[size]:
            best_at[size] = score

    front: list[tuple[int, float]] = []
    ceiling = float("-inf")
    for size in sorted(best_at):
        if best_at[size] > ceiling:
            front.append((size, best_at[size]))
            ceiling = best_at[size]
    return front
