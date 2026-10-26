"""What a search strategy needs to know about a space, and nothing more.

The strategies were written against the first space's module-level constants — one
alphabet, one length, one feasibility predicate — which was right while there was one
space and became the obstacle the moment there were two. This is the smallest interface
that lets the same operators run on either.

Two things it deliberately does *not* abstract.

**Nothing about what a genotype means.** A strategy proposes integers and never builds a
model, which is what keeps it impossible for a strategy to select on something the
protocol forbids. Adding a decode method here would invite exactly that.

**Nothing about fitness.** The space says which genotypes exist; the evaluator says how
good they are. Keeping those apart is what lets the exploitability measurement sample a
space without running a search on it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

Genotype = tuple[int, ...]


@dataclass(frozen=True)
class SpaceSpec:
    """The alphabet, the length and the feasibility rule of one search space.

    ``gene_domains`` is per position rather than global, because the second space's first
    gene chooses a pair comparison from five options while the rest choose a block from
    twenty. A single alphabet would make a mutation at position zero propose block values
    that mean nothing there — and since the encoding would still parse as integers, the
    error would surface as architectures that silently differ from what the genotype says.
    """

    name: str
    gene_domains: tuple[tuple[int, ...], ...]
    is_feasible: Callable[[Sequence[int]], bool]
    #: Rejection rate for a uniform draw, measured once and recorded. The first space
    #: rejects about half of all draws through `g₁` and `g₂`; the second rejects none.
    acceptance_rate: float = 1.0

    @property
    def length(self) -> int:
        return len(self.gene_domains)

    def random_gene(self, rng: random.Random, position: int) -> int:
        return rng.choice(self.gene_domains[position])

    def random_genotype(self, rng: random.Random) -> Genotype:
        """A uniform draw from the feasible region, by rejection.

        Rejection rather than repair, in both spaces. A repair operator biases the sample
        toward whatever it repairs into, and the random arm's whole value is that its
        distribution is known.
        """
        while True:
            candidate = tuple(
                self.random_gene(rng, position) for position in range(self.length)
            )
            if self.is_feasible(candidate):
                return candidate

    def size(self) -> int:
        total = 1
        for domain in self.gene_domains:
            total *= len(domain)
        return total

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "length": self.length,
            "gene_domain_sizes": [len(d) for d in self.gene_domains],
            "size_before_constraints": self.size(),
            "acceptance_rate": self.acceptance_rate,
        }


def _v1() -> SpaceSpec:
    from .genotype import ALPHABET, LENGTH, feasible

    return SpaceSpec(
        name="v1",
        gene_domains=tuple(tuple(range(ALPHABET)) for _ in range(LENGTH)),
        is_feasible=feasible,
        acceptance_rate=0.503,
    )


def _v2() -> SpaceSpec:
    from .v2 import BLOCKS, COMPARISONS, LENGTH, feasible

    return SpaceSpec(
        name="v2",
        gene_domains=(
            tuple(sorted(COMPARISONS)),
            *(tuple(sorted(BLOCKS)) for _ in range(LENGTH - 1)),
        ),
        is_feasible=feasible,
        acceptance_rate=1.0,
    )


#: Built lazily so importing this module does not drag in either space's machinery.
_REGISTRY: dict[str, Callable[[], SpaceSpec]] = {"v1": _v1, "v2": _v2}
_CACHE: dict[str, SpaceSpec] = {}


def get(name: str) -> SpaceSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown space {name!r}; known: {sorted(_REGISTRY)}")
    if name not in _CACHE:
        _CACHE[name] = _REGISTRY[name]()
    return _CACHE[name]


def default() -> SpaceSpec:
    """The first space, so that every existing caller keeps its behaviour unchanged."""
    return get("v1")
