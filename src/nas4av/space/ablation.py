"""Which of the redesign's three changes carries the effect.

The second space differs from the first in three ways at once — the pair comparison is
searched, normalisation and residual connections are in the alphabet, and depth is
explicit over eight positions instead of fifteen padded with `Identity`. It measures far
more exploitable than the first, and a change of three things at once tells you nothing
about which one mattered.

That is exactly the defect this project found in the submitted work: the evolutionary
variant departs from a canonical GA in four places, each adopted with a citation and none
ablated. Shipping a redesign with the same problem would be hard to defend.

So each variant removes **one** change and keeps the other two. Reading the results:

- a variant that drops to near zero isolates the change that was carrying the effect
- a variant that stays high says its removed change was not doing the work
- all three staying high would mean the effect comes from their combination, which is a
  real answer and the one that cannot be read off a single number

Everything else is held fixed, including the evaluation protocol, so a difference between
variants is about the variant.
"""

from __future__ import annotations

from typing import Any

from . import v2

#: Blocks with no normalisation and no residual connection — the first space's situation,
#: which offers only width as a choice.
_PLAIN_BLOCKS: dict[int, dict[str, Any]] = {
    0: {"kind": "stop"},
    **{
        1 + index: {"kind": "dense", "width": 2**exponent, "norm": None, "residual": False}
        for index, exponent in enumerate(range(5, 11))
    },
    7: {"kind": "dropout", "p": 0.3},
    8: {"kind": "dropout", "p": 0.5},
    9: {"kind": "attention", "heads": 4},
}


def _domains(comparisons: int, blocks: int, length: int) -> tuple[tuple[int, ...], ...]:
    return (tuple(range(comparisons)), *(tuple(range(blocks)) for _ in range(length - 1)))


#: The variants, each named for the change it **removes**.
VARIANTS: dict[str, dict[str, Any]] = {
    "no-comparison": {
        "removes": "the searched pair comparison",
        "detail": (
            "gene 0 is fixed to concat, as the first space does, so the architecture can no "
            "longer choose how the two documents are related"
        ),
        "comparisons": 1,
        "blocks": v2.BLOCKS,
        "length": v2.LENGTH,
    },
    "no-norm-residual": {
        "removes": "normalisation and residual connections",
        "detail": (
            "the block alphabet keeps width, dropout and attention but loses every "
            "LayerNorm, BatchNorm and skip connection"
        ),
        "comparisons": len(v2.COMPARISONS),
        "blocks": _PLAIN_BLOCKS,
        "length": v2.LENGTH,
    },
    "no-explicit-depth": {
        "removes": "explicit depth",
        "detail": (
            "fifteen positions with no stop gene, padded the way the first space pads, so "
            "depth is again an accident of how many no-ops were drawn"
        ),
        "comparisons": len(v2.COMPARISONS),
        "blocks": {k: val for k, val in v2.BLOCKS.items() if val["kind"] != "stop"}
        | {0: {"kind": "identity"}},
        "length": 16,
    },
}


def summary() -> dict[str, Any]:
    """What each variant removes, for the record and for the manuscript's table."""
    return {
        name: {
            "removes": spec["removes"],
            "detail": spec["detail"],
            "length": spec["length"],
            "comparisons": spec["comparisons"],
            "block_alphabet": len(spec["blocks"]),
        }
        for name, spec in VARIANTS.items()
    }


def register() -> list[str]:
    """Add every variant to the space registry, so the existing machinery can measure them.

    Registered rather than special-cased: the sampler, the featuriser, the strategies and
    the evaluator all read a space through the same interface, so a variant is measured by
    exactly the code that measures the full space. Anything else would make an ablation a
    comparison between two measurement paths.
    """
    from . import spec as spec_module

    names: list[str] = []
    for name, variant in VARIANTS.items():
        spec_module._REGISTRY[name] = _builder(name, variant)
        names.append(name)
    return names


def tables_for(name: str) -> v2.Tables:
    """The alphabets one variant decodes against."""
    variant = VARIANTS[name]
    comparisons = {
        key: value
        for key, value in v2.COMPARISONS.items()
        if key < int(variant["comparisons"])
    }
    return v2.Tables(
        comparisons=comparisons,
        blocks=dict(variant["blocks"]),
        length=int(variant["length"]),
    )


def _builder(name: str, variant: dict[str, Any]):
    from .spec import SpaceSpec

    def build() -> SpaceSpec:
        table = tables_for(name)
        return SpaceSpec(
            name=name,
            gene_domains=(
                tuple(sorted(table.comparisons)),
                *(tuple(sorted(table.blocks)) for _ in range(table.length - 1)),
            ),
            is_feasible=lambda genotype: v2.feasible(genotype, table),
            acceptance_rate=1.0,
        )

    return build


def describe_variant(name: str, encoding: str, per_document_width: int):
    """Decode under a variant's own alphabet, passed explicitly.

    Explicit because the alternative — swapping the module's globals around the call —
    fails silently the moment two decodes overlap, and what it produces is an architecture
    the variant claims it cannot contain.
    """
    return v2.describe(encoding, per_document_width, tables_for(name))
