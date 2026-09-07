"""The search space: what an encoding means, and which encodings are legal.

Read from ``Model.int_to_mod`` and ``Model.build_sequential`` in the prior
implementation, which is the authority here: it is the code that produced the results
this work builds on.

The space is every length-15 sequence over an alphabet of 24 operations, so
``24**15 = 5.0e20`` before constraints. Two constraints then cut it roughly in half.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

#: Positions in an encoding.
LENGTH = 15

#: Operations available at each position.
ALPHABET = 24

#: Integer to layer. Verified against ``Model.int_to_mod``:
#:
#:      0        Identity()
#:      1-3      Dropout(0.9), Dropout(0.6), Dropout(0.3)
#:      4-6      ReLU(), TanH(), LeakyReLU()
#:      7-19     Linear(2**0) ... Linear(2**12)
#:      20-23    Conv1D(kernel, channels) over {2,8} x {2,8}
#:
#: Thirteen Linear widths for the thirteen integers 7 through 19, and they are powers of
#: two rather than an arbitrary set — which is easy to mis-transcribe, so ``linear_width``
#: derives them instead of tabulating them.
CONV_OPERATIONS: dict[int, tuple[int, int]] = {
    20: (2, 2),
    21: (2, 8),
    22: (8, 2),
    23: (8, 8),
}

#: Conv1D operations whose channel count is 8. Constrained by g1 because channels drive
#: the parameter count, and the parameter count drove the RAM limit on the 4 GB card the
#: original campaign ran on.
G1_OPERATIONS = frozenset({21, 23})

#: How many of those may appear in one encoding.
G1_LIMIT = 3

#: A Conv1D layer may not sit next to an operation with a value above this. Values 18 and
#: 19 are Linear(2048) and Linear(4096); 20-23 are the Conv1D operations themselves.
G2_NEIGHBOUR_CEILING = 17

CATEGORIES = ("Identity", "Dropout", "Activation", "Linear", "Conv1D")


class MalformedEncoding(ValueError):
    """Raised when an encoding is not a length-15 sequence over [0, 23]."""


def category(gene: int) -> str:
    """Which family an operation belongs to.

    The families are what the structural analysis reports over, and their sizes are the
    base rates any pattern claim has to clear: Linear occupies 13 of the 24 values, so a
    pattern of four consecutive Linear layers is expected rather than discovered.
    """
    if gene == 0:
        return "Identity"
    if 1 <= gene <= 3:
        return "Dropout"
    if 4 <= gene <= 6:
        return "Activation"
    if 7 <= gene <= 19:
        return "Linear"
    if 20 <= gene <= 23:
        return "Conv1D"
    raise MalformedEncoding(f"gene {gene} is outside [0, {ALPHABET - 1}]")


#: Number of alphabet values in each family. The denominator of every base rate.
CATEGORY_SIZES: dict[str, int] = {
    name: sum(1 for gene in range(ALPHABET) if category(gene) == name) for name in CATEGORIES
}


def linear_width(gene: int) -> int:
    """Output width of a Linear operation."""
    if not 7 <= gene <= 19:
        raise MalformedEncoding(f"gene {gene} is not a Linear operation")
    return 2 ** (gene - 7)


def parse(encoding: str | Sequence[int]) -> tuple[int, ...]:
    """Accept either the stored comma-joined string or a sequence of integers."""
    genes = (
        tuple(int(t) for t in encoding.split(","))
        if isinstance(encoding, str)
        else tuple(int(g) for g in encoding)
    )
    if len(genes) != LENGTH:
        raise MalformedEncoding(f"expected {LENGTH} genes, got {len(genes)}")
    for gene in genes:
        if not 0 <= gene < ALPHABET:
            raise MalformedEncoding(f"gene {gene} is outside [0, {ALPHABET - 1}]")
    return genes


def render(genes: Sequence[int]) -> str:
    """The stored form, so a new artifact joins the legacy tables without translation."""
    return ",".join(str(g) for g in genes)


def categories(genes: Sequence[int]) -> tuple[str, ...]:
    return tuple(category(g) for g in genes)


def satisfies_g1(genes: Sequence[int]) -> bool:
    """At most three Conv1D operations with eight channels.

    Counted over *positions*. The distinction is worth stating because the natural
    set-theoretic phrasing — the cardinality of the set of matching gene values — can
    only ever reach two, so a constraint written that way could never bind. Position
    counting is what the prior implementation does and what a limit of three means.
    """
    return sum(1 for gene in genes if gene in G1_OPERATIONS) <= G1_LIMIT


def satisfies_g2(genes: Sequence[int]) -> bool:
    """Every Conv1D layer is surrounded by operations with values in [0, 17].

    Counted over positions, for the same reason as g1.

    The reason for the constraint is parameter growth: a Conv1D fed by a wide Linear, or
    by another Conv1D, multiplies rather than adds, and the prior campaign ran on a
    machine with 16 GB of RAM.
    """
    last = len(genes) - 1
    for i, gene in enumerate(genes):
        if gene not in CONV_OPERATIONS:
            continue
        if i > 0 and genes[i - 1] > G2_NEIGHBOUR_CEILING:
            return False
        if i < last and genes[i + 1] > G2_NEIGHBOUR_CEILING:
            return False
    return True


def feasible(genes: Sequence[int]) -> bool:
    """Both constraints. Infeasible encodings receive the death penalty during search."""
    return satisfies_g1(genes) and satisfies_g2(genes)


def sample_feasible(
    rng: np.random.Generator, count: int, batch: int = 50_000
) -> Iterator[tuple[int, ...]]:
    """Uniform samples of the feasible space, by rejection.

    Uniform over the *feasible* space, not over the alphabet, which is the distribution a
    pattern-frequency null model needs. Sampling the alphabet and keeping the infeasible
    draws would overstate how often Conv1D operations appear next to wide Linear layers —
    exactly the configurations the constraints exclude.

    Rejection is affordable because roughly half of all draws are feasible; the measured
    acceptance rate is about 0.50, so this is not a sampler that needs to be clever.
    """
    produced = 0
    while produced < count:
        draws = rng.integers(0, ALPHABET, size=(batch, LENGTH))
        for row in draws:
            genes = tuple(int(g) for g in row)
            if feasible(genes):
                yield genes
                produced += 1
                if produced >= count:
                    return


def acceptance_rate(rng: np.random.Generator, trials: int = 200_000) -> float:
    """Measured share of uniform alphabet draws that satisfy both constraints.

    Reported rather than derived. The constraints interact — g2 forbids neighbours above
    17, which includes the operations g1 counts — so a closed form would need care and the
    measurement costs a second.
    """
    draws = rng.integers(0, ALPHABET, size=(trials, LENGTH))
    return float(np.mean([feasible(tuple(int(g) for g in row)) for row in draws]))
