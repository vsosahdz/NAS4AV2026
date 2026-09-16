"""The driver, tested against a fake evaluator.

No torch, no corpus. That is the point of injecting the evaluator: the properties that
make a budget-matched comparison valid are properties of the driver, and they should be
checkable in milliseconds rather than after a training run.
"""

from __future__ import annotations

import random

import pytest

from nas4av.protocol import three_way_split
from nas4av.search.campaign import Measurement, SearchRun, run_search
from nas4av.search.strategy import (
    CanonicalGenetic,
    EvolutionaryVariant,
    Genotype,
    NoveltyFilter,
    UniformRandom,
)

INPUT_SHAPE = 600


def a_split(corpus: str = "2015"):
    return three_way_split(corpus, [0] * 50 + [1] * 50, [0] * 20 + [1] * 20, seed=1)


class FakeEvaluator:
    """Scores an architecture from its genotype, deterministically and instantly.

    The test score is deliberately anti-correlated with the validation score, so any
    accidental selection on the test side would show up as the search getting *worse* on
    validation rather than as a subtle bias.
    """

    def __init__(self, fail_every: int = 0) -> None:
        self.calls = 0
        self.fail_every = fail_every
        self.seen: list[Genotype] = []

    def __call__(self, genotype: Genotype) -> Measurement:
        self.calls += 1
        self.seen.append(genotype)
        if self.fail_every and self.calls % self.fail_every == 0:
            return Measurement(0.0, 0.0, 0.0, 0.0, 0.0, failed=True, detail="synthetic failure")
        value = sum(genotype) / (len(genotype) * 23)
        return Measurement(
            validation_roc_auc=value,
            validation_balanced_accuracy=value * 0.9,
            test_roc_auc=1.0 - value,
            test_balanced_accuracy=1.0 - value,
            seconds=0.001,
        )


def test_budget_is_spent_exactly():
    """Every strategy gets the same number of evaluations, whatever its population size."""
    for size in (4, 7, 10):
        evaluator = FakeEvaluator()
        strategy = UniformRandom(population_size=size, rng=random.Random(1))
        run = run_search(strategy, evaluator, a_split(), INPUT_SHAPE, budget=30, strategy_seed=1)
        assert run.evaluations == 30
        assert evaluator.calls == 30


def test_a_generation_that_would_overrun_is_truncated():
    """An overrun is precisely how a budget-matched comparison stops being one."""
    evaluator = FakeEvaluator()
    strategy = UniformRandom(population_size=8, rng=random.Random(1))
    run = run_search(strategy, evaluator, a_split(), INPUT_SHAPE, budget=20, strategy_seed=1)
    assert run.evaluations == 20
    assert len([r for r in run.records if r.generation == 2]) == 4


def test_budget_below_one_generation_is_refused():
    with pytest.raises(ValueError):
        run_search(
            UniformRandom(population_size=10, rng=random.Random(1)),
            FakeEvaluator(),
            a_split(),
            INPUT_SHAPE,
            budget=5,
            strategy_seed=1,
        )


@pytest.mark.parametrize("metric", ["test_roc_auc", "test_balanced_accuracy", "params"])
def test_a_test_side_fitness_is_refused(metric):
    """The defect that sank the prior submission, refused structurally.

    A caller cannot opt into optimising the test split, however convenient it would be.
    """
    with pytest.raises(ValueError, match="fitness_metric"):
        run_search(
            UniformRandom(population_size=4, rng=random.Random(1)),
            FakeEvaluator(),
            a_split(),
            INPUT_SHAPE,
            budget=8,
            strategy_seed=1,
            fitness_metric=metric,
        )


def test_the_search_optimises_validation_and_not_test():
    """With the fake's anti-correlated scores, selecting on test would invert the trend."""
    evaluator = FakeEvaluator()
    strategy = EvolutionaryVariant(population_size=8, rng=random.Random(4), elite_count=2)
    run = run_search(strategy, evaluator, a_split(), INPUT_SHAPE, budget=80, strategy_seed=4)

    first = [r.validation_roc_auc for r in run.records if r.generation == 0]
    last = max(r.generation for r in run.records)
    final = [r.validation_roc_auc for r in run.records if r.generation == last]
    assert max(final) >= max(first)

    best = run.best()
    assert best is not None
    assert best.validation_roc_auc == max(r.validation_roc_auc for r in run.records)


def test_best_ignores_failed_evaluations():
    evaluator = FakeEvaluator(fail_every=3)
    strategy = UniformRandom(population_size=6, rng=random.Random(2))
    run = run_search(strategy, evaluator, a_split(), INPUT_SHAPE, budget=24, strategy_seed=2)
    assert run.failures > 0
    best = run.best()
    assert best is not None
    assert not best.failed


def test_a_fully_failing_generation_halts_without_fabricating_a_row():
    """A placeholder row would inflate the count the budget comparison rests on."""

    class AlwaysFails:
        def __call__(self, genotype: Genotype) -> Measurement:
            return Measurement(0.0, 0.0, 0.0, 0.0, 0.0, failed=True, detail="always")

    strategy = UniformRandom(population_size=5, rng=random.Random(1))
    run = run_search(strategy, AlwaysFails(), a_split(), INPUT_SHAPE, budget=50, strategy_seed=1)
    assert run.halted
    assert run.evaluations == 5
    assert all(record.genotype for record in run.records)


def test_parameter_counts_are_recorded_for_every_evaluation():
    """The size axis has to be complete even where the performance axis is not."""
    evaluator = FakeEvaluator(fail_every=4)
    strategy = UniformRandom(population_size=6, rng=random.Random(3))
    run = run_search(strategy, evaluator, a_split(), INPUT_SHAPE, budget=18, strategy_seed=3)
    assert all(record.params > 0 for record in run.records)


def test_both_metrics_are_recorded():
    evaluator = FakeEvaluator()
    strategy = UniformRandom(population_size=4, rng=random.Random(1))
    run = run_search(strategy, evaluator, a_split(), INPUT_SHAPE, budget=8, strategy_seed=1)
    for record in run.records:
        assert record.validation_roc_auc != record.validation_balanced_accuracy
        assert record.test_roc_auc >= 0.0


def test_the_run_carries_the_split_summary_and_the_strategy_name():
    evaluator = FakeEvaluator()
    strategy = NoveltyFilter(UniformRandom(population_size=5, rng=random.Random(1)))
    run = run_search(strategy, evaluator, a_split("2020"), INPUT_SHAPE, budget=25, strategy_seed=7)
    payload = run.as_dict()
    assert payload["corpus"] == "2020"
    assert payload["strategy_seed"] == 7
    assert "novelty" in str(payload["strategy"])
    assert payload["split"]["validation_pairs"] > 0
    assert payload["input_shape"] == INPUT_SHAPE


def test_novelty_statistics_travel_into_the_record():
    """How hard the filter worked is a result, so it must reach the artifact."""
    evaluator = FakeEvaluator()
    strategy = NoveltyFilter(UniformRandom(population_size=6, rng=random.Random(9)))
    run = run_search(strategy, evaluator, a_split(), INPUT_SHAPE, budget=36, strategy_seed=9)
    assert run.novelty_replacements == strategy.replacements
    assert run.novelty_exhausted == strategy.exhausted


def test_the_test_split_is_never_released_by_the_search():
    """The search must finish without the gate having been opened."""
    split = a_split()
    evaluator = FakeEvaluator()
    strategy = CanonicalGenetic(population_size=6, rng=random.Random(1))
    run_search(strategy, evaluator, split, INPUT_SHAPE, budget=30, strategy_seed=1)
    assert not split.test_was_released


def test_runs_are_reproducible_from_the_strategy_generator():
    def one():
        return run_search(
            EvolutionaryVariant(population_size=6, rng=random.Random(123)),
            FakeEvaluator(),
            a_split(),
            INPUT_SHAPE,
            budget=30,
            strategy_seed=123,
        )

    first, second = one(), one()
    assert [r.genotype for r in first.records] == [r.genotype for r in second.records]


def test_different_seeds_explore_differently():
    """Runs that share their architectures are not replicates, which is the prior defect."""

    def one(seed: int) -> set[str]:
        run = run_search(
            EvolutionaryVariant(population_size=8, rng=random.Random(seed)),
            FakeEvaluator(),
            a_split(),
            INPUT_SHAPE,
            budget=64,
            strategy_seed=seed,
        )
        return {record.genotype for record in run.records}

    first, second = one(1), one(2)
    overlap = len(first & second) / min(len(first), len(second))
    assert overlap < 0.2


def test_on_generation_callback_sees_progress():
    seen: list[int] = []
    run_search(
        UniformRandom(population_size=5, rng=random.Random(1)),
        FakeEvaluator(),
        a_split(),
        INPUT_SHAPE,
        budget=20,
        strategy_seed=1,
        on_generation=lambda generation, run: seen.append(run.evaluations),
    )
    assert seen == [5, 10, 15, 20]


def test_search_run_with_no_usable_records_has_no_best():
    run = SearchRun(
        strategy="x",
        corpus="2015",
        strategy_seed=1,
        split={},
        budget=1,
        fitness_metric="validation_roc_auc",
    )
    assert run.best() is None
