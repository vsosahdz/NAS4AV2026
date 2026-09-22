"""Planning and running the campaign, one resumable unit at a time.

A full comparison is tens of hours of compute, so the unit of work is one run rather than
one campaign, and each unit writes its own artifact the moment it finishes. An
interruption costs at most the run in flight, and resuming is a matter of skipping what is
already on disk rather than remembering where things were.

The identity of a unit is what it measures — corpus, feature strategy, search strategy,
novelty filter, run index — and not when it was launched. Two artifacts can therefore
never disagree about which configuration produced them, and re-running a sweep after an
interruption cannot silently produce a second copy of a result under a different name.

Per-architecture scores are written beside each run rather than inside it. They are much
larger than the summary, they are only needed for the ranking-versus-threshold question,
and keeping them separate means a reader who wants the results table does not have to
parse megabytes of logits to get it.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .artifacts import artifacts_root, write_json
from .evaluator import PipelineEvaluator
from .prior.dataset import load_corpus
from .protocol import run_seeds, seed_everything, three_way_split
from .search.campaign import SearchRun, run_search
from .search.strategy import (
    CanonicalGenetic,
    EvolutionaryVariant,
    NoveltyFilter,
    SearchStrategy,
    UniformRandom,
)

#: Every (corpus, feature strategy) the prior campaign produced, including the BERT arm it
#: did not report.
SETTINGS: tuple[tuple[str, str], ...] = (
    ("2013", "sem"),
    ("2013", "synBetter"),
    ("2013", "semBert"),
    ("2014Essay2", "sem"),
    ("2014Essay2", "synBetter"),
    ("2014Novel2", "sem"),
    ("2014Novel2", "synBetter"),
    ("2015", "sem"),
    ("2015", "synBetter"),
    ("2020", "sem"),
    ("2020", "synBetter"),
)

#: Search strategies, by the name that appears in an artifact.
STRATEGIES: dict[str, type[SearchStrategy]] = {
    "evolutionary-variant": EvolutionaryVariant,
    "canonical-ga": CanonicalGenetic,
    "uniform-random": UniformRandom,
}

#: Population and budget, held at the prior campaign's values so the two are comparable.
#: 24 individuals over 50 generations is 1,200 evaluations per run.
POPULATION_SIZE = 24
BUDGET = 1_200
RUNS = 5

#: Base seed for the whole sweep. Every run seed is derived from it, so the campaign is
#: reproducible from one number and two runs can never accidentally share a stream.
BASE_SEED = 20260920


@dataclass(frozen=True)
class WorkUnit:
    """One run: what it measures, and the seed that makes it reproducible."""

    corpus: str
    features: str
    strategy: str
    run: int
    seed: int
    novelty: bool = True
    budget: int = BUDGET
    population_size: int = POPULATION_SIZE

    @property
    def identifier(self) -> str:
        filter_tag = "novelty" if self.novelty else "nofilter"
        return f"{self.corpus}_{self.features}_{self.strategy}_{filter_tag}_run{self.run}"

    @property
    def artifact_path(self) -> str:
        return f"sweep/{self.identifier}.json"

    @property
    def scores_path(self) -> str:
        return f"sweep/scores/{self.identifier}.json"

    def build_strategy(self) -> SearchStrategy:
        """Construct the strategy with its own generator, seeded from the unit."""
        strategy = STRATEGIES[self.strategy](
            population_size=self.population_size, rng=random.Random(self.seed)
        )
        return NoveltyFilter(strategy) if self.novelty else strategy  # type: ignore[return-value]


def plan(
    settings: Sequence[tuple[str, str]] = SETTINGS,
    strategies: Sequence[str] = ("evolutionary-variant",),
    runs: int = RUNS,
    novelty: bool = True,
    base_seed: int = BASE_SEED,
    budget: int = BUDGET,
) -> list[WorkUnit]:
    """Enumerate the sweep without running anything.

    Separated from execution so the cost of a configuration can be read before it is
    paid, and so a plan can be split across machines by taking slices of the same list.
    """
    seeds = run_seeds(base_seed, runs)
    units: list[WorkUnit] = []
    for corpus, features in settings:
        for strategy in strategies:
            if strategy not in STRATEGIES:
                raise KeyError(f"unknown strategy {strategy!r}; expected {sorted(STRATEGIES)}")
            for index in range(runs):
                units.append(
                    WorkUnit(
                        corpus=corpus,
                        features=features,
                        strategy=strategy,
                        run=index + 1,
                        seed=seeds[index],
                        novelty=novelty,
                        budget=budget,
                    )
                )
    return units


def completed(units: Sequence[WorkUnit]) -> set[str]:
    """Which units already have a readable artifact.

    A truncated file — an interruption mid-write — is treated as incomplete rather than
    skipped, because the alternative is a sweep that silently carries a partial result
    forward into its conclusions.
    """
    done: set[str] = set()
    for unit in units:
        path = artifacts_root() / unit.artifact_path
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("run", {}).get("evaluations"):
            done.add(unit.identifier)
    return done


def run_unit(unit: WorkUnit, keep_scores: bool = True) -> SearchRun:
    """Execute one unit and write its artifacts."""
    seed_everything(unit.seed)
    corpus = load_corpus(unit.corpus, unit.features)
    split = three_way_split(unit.corpus, corpus.train_labels, corpus.test_labels, seed=unit.seed)
    evaluator = PipelineEvaluator(corpus, split, keep_scores=keep_scores)

    started = time.time()
    run = run_search(
        strategy=unit.build_strategy(),
        evaluator=evaluator,
        split=split,
        input_shape=corpus.input_shape,
        budget=unit.budget,
        strategy_seed=unit.seed,
    )
    elapsed = time.time() - started

    write_json(
        unit.artifact_path,
        {
            "unit": {
                "corpus": unit.corpus,
                "features": unit.features,
                "strategy": unit.strategy,
                "run": unit.run,
                "seed": unit.seed,
                "novelty_filter": unit.novelty,
                "budget": unit.budget,
                "population_size": unit.population_size,
            },
            "partitions": evaluator.partition_sizes(),
            "supports_clean_selection": split.supports_clean_selection,
            "wall_clock_seconds": round(elapsed, 3),
            "run": run.as_dict(),
        },
    )
    if keep_scores and evaluator.scores:
        write_json(unit.scores_path, {"unit": unit.identifier, "scores": evaluator.scores})
    return run


def run_sweep(
    units: Sequence[WorkUnit],
    resume: bool = True,
    keep_scores: bool = True,
) -> Iterator[tuple[WorkUnit, SearchRun | None]]:
    """Run a plan, yielding each unit as it finishes.

    A generator rather than a function that returns at the end, because at tens of hours
    the caller needs progress it can print and a point at which it can stop.
    """
    done = completed(units) if resume else set()
    for unit in units:
        if unit.identifier in done:
            yield unit, None
            continue
        yield unit, run_unit(unit, keep_scores=keep_scores)


def estimate(
    units: Sequence[WorkUnit], seconds_per_evaluation: dict[str, float]
) -> dict[str, object]:
    """Projected cost of a plan, from measured per-setting rates.

    Takes the rates rather than measuring them, so a plan can be costed on a machine that
    is not the one that will run it — which is the situation whenever the question is
    whether something belongs on a cluster.
    """
    total = 0.0
    unknown: list[str] = []
    for unit in units:
        key = f"{unit.corpus}/{unit.features}"
        rate = seconds_per_evaluation.get(key)
        if rate is None:
            unknown.append(key)
            continue
        total += rate * unit.budget
    return {
        "units": len(units),
        "evaluations": sum(unit.budget for unit in units),
        "projected_hours": round(total / 3600, 2),
        "settings_without_a_measured_rate": sorted(set(unknown)),
    }


def artifact_directory() -> Path:
    return artifacts_root() / "sweep"
