"""What a discovered model costs to run, as opposed to what it cost to find.

The submitted conclusion says the discovered models "demand significantly fewer
computational resources than large language models" with no number attached, and task
group 7 exists to decide whether that sentence survives contact with one. Three quantities
answer it, and they are not interchangeable:

**Parameters** are structural. They follow from the genotype and the input width alone, so
they can be counted for an architecture that was never trained and no evaluation protocol
can bias them. ``space.cost`` already does this exactly.

**FLOPs** are also structural, and they are the quantity the efficiency claim is really
about: a model can be small and slow. A `Conv1D` over a 4,000-wide input does far more
arithmetic per parameter than a `Linear` of the same size, so parameter count alone ranks
these architectures wrongly.

**Latency** is neither. It is measured, it depends on the machine, and it is the only one
of the three a practitioner feels. It is reported with its platform attached and never
compared across platforms.

One separation matters more than any of the three. The search is a **one-time** cost, paid
once per corpus by whoever runs it; inference is paid per decision, forever, by whoever
deploys the result. The submitted manuscript's cost discussion mixes them, which makes a
20-hour search look like a property of the model it produced.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .space.cost import describe, input_shape
from .space.genotype import CONV_OPERATIONS, category, linear_width, parse

#: Published parameter counts for reference models, for the ratio the conclusion needs.
#: Transcribed from each model's own paper or card, and deliberately spanning three orders
#: of magnitude so the claim is not resting on one flattering comparison.
REFERENCE_MODELS: dict[str, int] = {
    "all-MiniLM-L6-v2": 22_700_000,
    "BERT-base": 110_000_000,
    "RoBERTa-large": 355_000_000,
    "GPT-3": 175_000_000_000,
}


@dataclass
class InferenceCost:
    """What one architecture costs per forward pass."""

    encoding: str
    input_shape: int
    params: int
    flops: int
    effective_depth: int

    @property
    def flops_per_parameter(self) -> float:
        """Arithmetic per stored weight.

        The number that separates small from cheap. A Conv1D reuses each weight across
        every position of its input, so it can be tiny and expensive; a Linear uses each
        weight once. An efficiency claim resting on parameter count alone cannot see the
        difference.
        """
        return self.flops / self.params if self.params else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "input_shape": self.input_shape,
            "params": self.params,
            "flops": self.flops,
            "flops_per_parameter": round(self.flops_per_parameter, 2),
            "effective_depth": self.effective_depth,
        }


def forward_flops(encoding: str | Sequence[int], input_width: int) -> int:
    """Multiply-accumulates in one forward pass, counted as two FLOPs each.

    Closed-form rather than profiled, for the same reason the parameter count is: it has to
    be computable for architectures that were never trained, and a profiler would make the
    number depend on a backend's kernel choices.

    Only the layers that do arithmetic are counted. Dropout is identity at inference,
    Identity is identity everywhere, and activations are one cheap operation per element —
    included, because at fifteen positions over a 4,000-wide tensor they are not nothing,
    but counted as a single FLOP per element rather than as a multiply-accumulate.
    """
    genes = parse(encoding)
    width = input_width
    total = 0
    for gene in genes:
        family = category(gene)
        if family == "Linear":
            out = linear_width(gene)
            total += 2 * width * out
            width = out
        elif family == "Conv1D":
            kernel, channels = CONV_OPERATIONS[gene]
            # The prior pipeline treats the feature vector as a length-`width` signal with
            # one channel, so a convolution slides `kernel` taps over it and emits
            # `channels` maps, which are then flattened back into the running width.
            positions = max(width - kernel + 1, 1)
            total += 2 * positions * kernel * channels
            width = positions * channels
        elif family == "Activation":
            total += width
        # Dropout and Identity cost nothing at inference.
    return total


def cost_of(encoding: str | Sequence[int], input_width: int) -> InferenceCost:
    architecture = describe(encoding, input_width)
    return InferenceCost(
        encoding=architecture.encoding,
        input_shape=input_width,
        params=architecture.params,
        flops=forward_flops(encoding, input_width),
        effective_depth=architecture.effective_depth,
    )


def cost_for_setting(encoding: str | Sequence[int], corpus: str, strategy: str) -> InferenceCost:
    """The same, with the input width looked up rather than passed.

    Worth having separately because the width is where this is easiest to get wrong: it is
    the per-document tensor plus 32 for punctuation on the syntactic strategies, doubled
    because a pair is a concatenation. An off-by-2x here changes every FLOP figure.
    """
    return cost_of(encoding, input_shape(corpus, strategy))


@dataclass
class LatencyMeasurement:
    """Measured wall-clock per forward pass, with the machine it was measured on."""

    encoding: str
    input_shape: int
    batch_size: int
    repeats: int
    median_ms: float
    p90_ms: float
    platform: str
    torch_version: str
    device: str
    threads: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "input_shape": self.input_shape,
            "batch_size": self.batch_size,
            "repeats": self.repeats,
            "median_ms": round(self.median_ms, 4),
            "p90_ms": round(self.p90_ms, 4),
            "per_pair_ms": round(self.median_ms / self.batch_size, 5),
            "platform": self.platform,
            "torch": self.torch_version,
            "device": self.device,
            "threads": self.threads,
            "comparability": (
                "wall-clock is a property of this machine and this thread count, not of "
                "the architecture; compare latencies only within one platform block"
            ),
        }


def measure_latency(
    encoding: str | Sequence[int],
    input_width: int,
    batch_size: int = 1,
    repeats: int = 50,
    warmup: int = 5,
) -> LatencyMeasurement:
    """Time a forward pass, reporting the median rather than the mean.

    The median because the distribution has a tail and no floor: a scheduler interruption
    can only make a measurement longer, never shorter, so the mean drifts upward with
    whatever else the machine was doing. The p90 travels alongside it so that a long tail
    is visible rather than hidden by the median.

    Warm-up iterations are discarded. The first pass through a fresh model pays for lazy
    allocation and cache misses that no later pass repeats, and including it would
    overstate a small model's cost far more than a large one's.
    """
    import contextlib
    import io

    import torch

    from .original import pipeline as original

    torch.set_grad_enabled(False)
    threads = torch.get_num_threads()
    genes = list(parse(encoding))

    # Built through the extracted pipeline rather than reconstructed here, so that what is
    # timed is the model the campaign actually evaluates. A reimplementation would be
    # timing a different object and attributing the number to this one. Both constructors
    # print a summary, which is captured for the same reason the evaluator captures it.
    with contextlib.redirect_stdout(io.StringIO()):
        space = original.Sequential_Search_Space(input_width)
        model = original.Model(genes, input_width, space.genel, input_width // 2)
        # Converts every weight to float64. The stored features are double and nn.Linear
        # builds float32, so without this the first matrix multiply raises — and a latency
        # measured in float32 would not be the cost of the model that runs.
        model.weights_init_uniform()
    model.eval()

    # The shapes the pipeline expects, which are not the obvious ones. A row of
    # ``embeddings`` is a *chunk*, ``cpd`` says how many consecutive rows belong to each
    # document, and the model pairs documents two at a time into one row of width
    # ``input_width``. So a batch of B pairs with one chunk per document is 2B rows of
    # half the width — passing (B, input_width) instead silently measures a different
    # computation, or raises, depending on the first layer.
    per_document = input_width // 2
    rows = 2 * batch_size
    embeddings = torch.randn(rows, per_document, dtype=torch.float64)
    cpd = [1] * rows

    for _ in range(warmup):
        model(cpd, embeddings=embeddings)

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(cpd, embeddings=embeddings)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()

    return LatencyMeasurement(
        encoding=",".join(str(g) for g in parse(encoding)),
        input_shape=input_width,
        batch_size=batch_size,
        repeats=repeats,
        median_ms=statistics.median(samples),
        p90_ms=samples[min(int(0.9 * len(samples)), len(samples) - 1)],
        platform=platform.platform(),
        torch_version=torch.__version__,
        device="cpu",
        threads=threads,
    )


def against_references(params: int) -> dict[str, Any]:
    """The efficiency claim as ratios rather than as an adjective.

    The submitted conclusion asserts the discovered models need far fewer resources than
    large language models and attaches no number. These are the numbers, across three
    orders of magnitude, so the sentence is checked against the weakest comparison it
    implies rather than the most flattering one.
    """
    ratios = {
        name: round(size / params, 1) if params else None for name, size in REFERENCE_MODELS.items()
    }
    smallest = min(REFERENCE_MODELS.values())
    return {
        "discovered_params": params,
        "references": REFERENCE_MODELS,
        "ratios": ratios,
        "weakest_comparison": {
            "model": min(REFERENCE_MODELS, key=lambda k: REFERENCE_MODELS[k]),
            "ratio": round(smallest / params, 1) if params else None,
        },
        "caveat": (
            "a parameter ratio is not a capability ratio: the reference models solve a "
            "far broader class of problem and were not trained for this task, so this "
            "bounds the resource claim and says nothing about what is lost"
        ),
    }


def separate_costs(
    search_hours: float, evaluations: int, inference_ms: float, decisions: int = 1_000_000
) -> dict[str, Any]:
    """One-time search against per-decision inference, which the manuscript conflates.

    The crossover is the point of the comparison. A search costing twenty hours is paid
    once by whoever runs it; a model costing a millisecond per decision is paid by whoever
    deploys it, on every decision, forever. Reporting them as one figure makes a search
    cost look like a property of the model it produced.
    """
    inference_hours = inference_ms * decisions / 1000.0 / 3600.0
    return {
        "one_time_search": {
            "hours": round(search_hours, 2),
            "evaluations": evaluations,
            "seconds_per_evaluation": round(search_hours * 3600 / evaluations, 3)
            if evaluations
            else None,
            "paid_by": "whoever runs the search, once per corpus",
        },
        "per_decision_inference": {
            "milliseconds": round(inference_ms, 4),
            "hours_per_million_decisions": round(inference_hours, 3),
            "paid_by": "whoever deploys the model, on every decision",
        },
        "decisions_to_equal_the_search": int(search_hours * 3600 * 1000 / inference_ms)
        if inference_ms
        else None,
    }


def front_with_flops(
    encodings: Iterable[str], corpus: str, strategy: str
) -> list[dict[str, Any]]:
    """Costs for a set of architectures on one setting, cheapest first."""
    width = input_shape(corpus, strategy)
    return sorted(
        (cost_of(encoding, width).as_dict() for encoding in encodings),
        key=lambda row: row["flops"],
    )
