"""Search strategies: budget, feasibility, and the properties that make them comparable.

These run without torch and without the prior campaign's data, because a strategy is a
function from fitnesses to proposals — everything about training is injected. That is what
makes the ablations the reviewers asked for cheap to run.
"""

from __future__ import annotations

import random

import pytest

from nas4av.search.strategy import (
    CanonicalGenetic,
    EvolutionaryVariant,
    NoveltyFilter,
    Scored,
    UniformRandom,
)
from nas4av.space.genotype import ALPHABET, LENGTH, feasible

STRATEGIES = (UniformRandom, CanonicalGenetic, EvolutionaryVariant)


def build(cls, size=8, seed=1):
    return cls(population_size=size, rng=random.Random(seed))


def score(genotypes, rng=None):
    rng = rng or random.Random(0)
    return [Scored(g, rng.random()) for g in genotypes]


@pytest.mark.parametrize("cls", STRATEGIES, ids=lambda c: c.__name__)
def test_every_proposal_is_feasible(cls):
    """Infeasible children must never reach the evaluator.

    An infeasible architecture costs a full evaluation to discover and returns nothing,
    so a strategy that emits them is quietly spending the budget it is being compared on.
    """
    strategy = build(cls)
    population = strategy.initial_population()
    for _ in range(12):
        assert all(feasible(g) for g in population)
        population = strategy.next_generation(score(population))
    assert all(feasible(g) for g in population)


@pytest.mark.parametrize("cls", STRATEGIES, ids=lambda c: c.__name__)
def test_population_size_is_exact(cls):
    """The budget is enforced by count, so a short generation is a cheated comparison."""
    strategy = build(cls, size=11)
    assert len(strategy.initial_population()) == 11
    for _ in range(5):
        population = strategy.next_generation(score(strategy.initial_population()))
        assert len(population) == 11


@pytest.mark.parametrize("cls", STRATEGIES, ids=lambda c: c.__name__)
def test_every_proposal_has_the_right_shape(cls):
    strategy = build(cls)
    for genotype in strategy.next_generation(score(strategy.initial_population())):
        assert len(genotype) == LENGTH
        assert all(0 <= gene < ALPHABET for gene in genotype)


@pytest.mark.parametrize("cls", STRATEGIES, ids=lambda c: c.__name__)
def test_strategies_are_reproducible_from_their_generator(cls):
    first = build(cls, seed=99).initial_population()
    second = build(cls, seed=99).initial_population()
    assert first == second


@pytest.mark.parametrize("cls", STRATEGIES, ids=lambda c: c.__name__)
def test_a_strategy_does_not_draw_from_the_global_generator(cls):
    """The defect this whole design exists to prevent.

    If a strategy reached for ``random`` directly, then training — which reseeds the
    global generator on every mini-batch — would steer the search. Advancing the global
    generator between two identically-seeded strategies must change nothing.
    """
    strategy_a = build(cls, seed=5)
    random.seed(20)
    first = strategy_a.initial_population()

    strategy_b = build(cls, seed=5)
    random.seed(999)
    [random.random() for _ in range(1000)]
    second = strategy_b.initial_population()

    assert first == second


def test_rejects_a_population_too_small_to_recombine():
    with pytest.raises(ValueError):
        build(UniformRandom, size=1)


# ── what distinguishes the two genetic strategies ──────────────────────────────


def test_the_variant_preserves_its_elite():
    """Elitism is the modification most likely to matter, so it is pinned directly."""
    strategy = EvolutionaryVariant(population_size=8, rng=random.Random(3), elite_count=2)
    population = strategy.initial_population()
    scored = [Scored(g, float(i)) for i, g in enumerate(population)]
    best = sorted(scored, key=lambda s: s.fitness, reverse=True)[:2]

    children = strategy.next_generation(scored)
    for elite in best:
        assert elite.genotype in children


def test_the_canonical_algorithm_does_not_preserve_an_elite():
    """Which is the point of having it: the difference has to be attributable."""
    strategy = CanonicalGenetic(population_size=8, rng=random.Random(3))
    population = strategy.initial_population()
    scored = [Scored(g, float(i)) for i, g in enumerate(population)]
    champion = max(scored, key=lambda s: s.fitness).genotype

    survived = sum(champion in strategy.next_generation(scored) for _ in range(20))
    assert survived < 20


def test_elite_count_must_leave_room_for_children():
    with pytest.raises(ValueError):
        EvolutionaryVariant(population_size=4, rng=random.Random(1), elite_count=4)


def test_uniform_random_ignores_fitness():
    """The control must be blind, or it is not a control."""
    strategy = build(UniformRandom, seed=17)
    population = strategy.initial_population()
    high = strategy.next_generation([Scored(g, 1.0) for g in population])

    strategy = build(UniformRandom, seed=17)
    population = strategy.initial_population()
    low = strategy.next_generation([Scored(g, 0.0) for g in population])

    assert high == low


def test_roulette_survives_an_all_ties_population():
    """Not a rare edge case: much of this space evaluates to the same value.

    A fitness-proportionate selector that divides by a zero total would crash on exactly
    the populations this search produces most often.
    """
    strategy = CanonicalGenetic(population_size=6, rng=random.Random(8))
    population = strategy.initial_population()
    children = strategy.next_generation([Scored(g, 0.0) for g in population])
    assert len(children) == 6
    assert all(feasible(g) for g in children)


def test_binomial_crossover_can_take_genes_from_both_parents():
    strategy = EvolutionaryVariant(population_size=4, rng=random.Random(2))
    a = tuple([0] * LENGTH)
    b = tuple([4] * LENGTH)
    mixed = [strategy._binomial_crossover(a, b) for _ in range(40)]
    assert any(0 in child and 4 in child for child in mixed)


# ── the novelty filter ─────────────────────────────────────────────────────────


def test_novelty_filter_never_repeats_an_architecture():
    wrapped = NoveltyFilter(build(UniformRandom, size=6, seed=4))
    seen: list[tuple[int, ...]] = []
    population = wrapped.initial_population()
    for _ in range(10):
        seen.extend(population)
        population = wrapped.next_generation(score(population))
    seen.extend(population)
    assert len(seen) == len(set(seen))


def test_novelty_filter_reports_how_often_it_intervened():
    """Its intervention count is a result, not an implementation detail.

    In the prior search this filter was the only source of diversity, so how hard it had
    to work is what separates a strategy that explores from one that is being carried.
    """
    strategy = build(UniformRandom, size=4, seed=6)
    wrapped = NoveltyFilter(strategy)
    assert wrapped.replacements == 0

    population = wrapped.initial_population()
    # Feed the same population back so every proposal collides with the seen set.
    stuck = NoveltyFilter(build(UniformRandom, size=4, seed=6))
    stuck.seen = set(population)
    stuck.next_generation(score(population))
    assert stuck.replacements > 0


def test_novelty_filter_keeps_the_population_size_when_it_cannot_freshen():
    """A shrunken generation would make two strategies incomparable at one budget."""
    strategy = build(UniformRandom, size=5, seed=2)
    wrapped = NoveltyFilter(strategy, attempts=1)
    population = wrapped.initial_population()
    wrapped.seen = set(population)
    assert len(wrapped.next_generation(score(population))) == 5


def test_novelty_filter_reports_its_name_through_the_wrapper():
    wrapped = NoveltyFilter(build(EvolutionaryVariant))
    assert "novelty" in wrapped.name
    assert "evolutionary-variant" in wrapped.name


@pytest.mark.parametrize("cls", STRATEGIES, ids=lambda c: c.__name__)
def test_names_are_recorded_and_distinct(cls):
    assert build(cls).name
    assert len({build(c).name for c in STRATEGIES}) == len(STRATEGIES)
