"""Search strategies, budget-matched and interchangeable.

Three of the reviewer's questions reduce to one requirement: a comparison between the
evolutionary variant, the canonical genetic algorithm, and uniform random sampling, each
given the same number of evaluations. That comparison is only meaningful if the budget is
enforced by the driver rather than trusted to each implementation, so strategies here
propose populations and never count their own spending.

Two design choices carry more weight than they look.

**Each strategy owns its generator.** Not Python's global one. The prior arrangement had
training reseed the global generator on every mini-batch, which left the operators
restarting from a fixed state after every evaluation; an injected ``random.Random`` cannot
be reached by anything downstream. ``protocol.isolated_randomness`` guards the same
boundary from the other side, and both are kept.

**The novelty filter is declared, not assumed.** The prior search rejected any child it
had already evaluated and re-drew until it found a new one. That filter was not an
optimisation — with the operators restarting from a fixed state it was the only source of
diversity in the run. Here it is a wrapper any strategy can be given or refused, so its
contribution to a result is visible instead of built in.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from ..space.genotype import ALPHABET, LENGTH, feasible

Genotype = tuple[int, ...]


@dataclass(frozen=True)
class Scored:
    """One evaluated architecture, as a strategy sees it.

    Deliberately thin. A strategy is given a fitness and nothing else — not the test
    score, not the parameter count — so that a strategy cannot select on something the
    protocol forbids even by accident.
    """

    genotype: Genotype
    fitness: float


class SearchStrategy(ABC):
    """Proposes populations; never evaluates, never counts its own budget."""

    #: Individuals per generation. The driver spends exactly this many evaluations per
    #: call, which is what makes two strategies comparable at a stated budget.
    population_size: int

    def __init__(self, population_size: int, rng: random.Random) -> None:
        if population_size < 2:
            raise ValueError("a population needs at least two individuals to recombine")
        self.population_size = population_size
        self._rng = rng

    @property
    @abstractmethod
    def name(self) -> str:
        """Recorded with every result, so an artifact says what produced it."""

    def random_genotype(self) -> Genotype:
        """A uniform draw from the feasible space, by rejection.

        Rejection rather than repair. A repair operator would bias the sample toward
        whatever it repairs into, and the whole point of the random arm is that its
        distribution is known.
        """
        while True:
            candidate = tuple(self._rng.randrange(ALPHABET) for _ in range(LENGTH))
            if feasible(candidate):
                return candidate

    def initial_population(self) -> list[Genotype]:
        """Uniform over the feasible space for every strategy.

        Shared on purpose: if two strategies started from different distributions, a
        difference between them at the end could not be attributed to the search.
        """
        return [self.random_genotype() for _ in range(self.population_size)]

    @abstractmethod
    def next_generation(self, scored: Sequence[Scored]) -> list[Genotype]:
        """Propose the next population from the last one's fitnesses."""


class UniformRandom(SearchStrategy):
    """The control the NAS literature treats as mandatory.

    Ignores fitness entirely. Any margin an informed strategy shows over this one at the
    same budget is what the search bought; without it, a reported result is compatible
    with the space simply containing good architectures.
    """

    @property
    def name(self) -> str:
        return "uniform-random"

    def next_generation(self, scored: Sequence[Scored]) -> list[Genotype]:
        return [self.random_genotype() for _ in range(self.population_size)]


class CanonicalGenetic(SearchStrategy):
    """The textbook algorithm the variant is measured against.

    Fitness-proportionate selection, single-point crossover, per-gene uniform mutation,
    and no elitism — the four places the variant departs from it. Keeping them together in
    one class is what lets the comparison attribute a difference to the modifications
    rather than to an accumulation of unrelated choices.
    """

    def __init__(
        self,
        population_size: int,
        rng: random.Random,
        crossover_probability: float = 0.9,
        mutation_probability: float = 1 / LENGTH,
    ) -> None:
        super().__init__(population_size, rng)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability

    @property
    def name(self) -> str:
        return "canonical-ga"

    def _roulette(self, scored: Sequence[Scored]) -> Genotype:
        """Fitness-proportionate selection.

        Falls back to a uniform pick when the fitnesses are all equal, which is not a rare
        edge case here: close to a quarter of this space evaluates to the metric's
        no-discrimination point, so whole populations of ties do occur.
        """
        total = sum(item.fitness for item in scored)
        if total <= 0:
            return self._rng.choice(scored).genotype
        threshold = self._rng.uniform(0, total)
        running = 0.0
        for item in scored:
            running += item.fitness
            if running >= threshold:
                return item.genotype
        return scored[-1].genotype

    def _single_point_crossover(self, a: Genotype, b: Genotype) -> tuple[Genotype, Genotype]:
        if self._rng.random() >= self.crossover_probability:
            return a, b
        cut = self._rng.randrange(1, LENGTH)
        return a[:cut] + b[cut:], b[:cut] + a[cut:]

    def _mutate(self, genotype: Genotype) -> Genotype:
        return tuple(
            self._rng.randrange(ALPHABET) if self._rng.random() < self.mutation_probability else g
            for g in genotype
        )

    def next_generation(self, scored: Sequence[Scored]) -> list[Genotype]:
        children: list[Genotype] = []
        while len(children) < self.population_size:
            first, second = self._single_point_crossover(
                self._roulette(scored), self._roulette(scored)
            )
            for child in (first, second):
                if len(children) < self.population_size:
                    children.append(self._repair_or_redraw(self._mutate(child)))
        return children

    def _repair_or_redraw(self, genotype: Genotype) -> Genotype:
        """The death penalty: an infeasible child is replaced by a fresh draw.

        Named for what it is. The prior implementation called this a death penalty while
        implementing replacement, and the distinction matters for how the budget is
        spent — a penalised individual costs an evaluation, a replaced one does not.
        """
        return genotype if feasible(genotype) else self.random_genotype()


class EvolutionaryVariant(SearchStrategy):
    """The strategy the prior work used, with its four modifications made explicit.

    Elitism, deterministic binary tournament, binomial crossover borrowed from
    differential evolution, and uniform integer mutation. Each is a departure from
    :class:`CanonicalGenetic` and each was adopted with a citation and never ablated;
    holding both classes to the same interface is what makes ablating them cheap.
    """

    def __init__(
        self,
        population_size: int,
        rng: random.Random,
        crossover_probability: float = 0.9,
        binomial_probability: float = 0.7,
        mutation_probability: float = 1 / LENGTH,
        elite_count: int = 1,
    ) -> None:
        super().__init__(population_size, rng)
        if not 0 <= elite_count < population_size:
            raise ValueError("elite_count must leave room for at least one child")
        self.crossover_probability = crossover_probability
        self.binomial_probability = binomial_probability
        self.mutation_probability = mutation_probability
        self.elite_count = elite_count

    @property
    def name(self) -> str:
        return f"evolutionary-variant(elite={self.elite_count})"

    def _tournament(self, scored: Sequence[Scored]) -> Genotype:
        """Deterministic binary tournament: sample two, keep the better."""
        first, second = self._rng.sample(list(scored), 2)
        return first.genotype if first.fitness >= second.fitness else second.genotype

    def _binomial_crossover(self, a: Genotype, b: Genotype) -> Genotype:
        """Per-gene exchange, with one position forced.

        Forcing a position is what makes it a crossover rather than a coin flip that
        sometimes returns a parent unchanged — without it, a low exchange probability
        produces clones and the operator quietly stops doing anything.
        """
        if self._rng.random() >= self.crossover_probability:
            return a
        forced = self._rng.randrange(LENGTH)
        return tuple(
            b[i] if (i == forced or self._rng.random() < self.binomial_probability) else a[i]
            for i in range(LENGTH)
        )

    def _mutate(self, genotype: Genotype) -> Genotype:
        return tuple(
            self._rng.randrange(ALPHABET) if self._rng.random() < self.mutation_probability else g
            for g in genotype
        )

    def next_generation(self, scored: Sequence[Scored]) -> list[Genotype]:
        ranked = sorted(scored, key=lambda item: item.fitness, reverse=True)
        children: list[Genotype] = [item.genotype for item in ranked[: self.elite_count]]
        while len(children) < self.population_size:
            child = self._binomial_crossover(self._tournament(scored), self._tournament(scored))
            child = self._mutate(child)
            children.append(child if feasible(child) else self.random_genotype())
        return children


class NoveltyFilter:
    """Wraps a strategy so it never proposes an architecture already evaluated.

    Separated from the strategies because of what it turned out to be doing. In the prior
    search the operators restarted from a fixed random state after every evaluation, so
    this filter — re-drawing until a child was new — was the only thing producing
    diversity across a run. A component with that much influence should be switchable and
    reported, not implicit in every strategy.

    ``replacements`` counts how often the filter had to intervene. A high count means the
    strategy is proposing mostly-seen candidates and the filter is doing the exploring.
    """

    def __init__(self, strategy: SearchStrategy, attempts: int = 50) -> None:
        self.strategy = strategy
        self.attempts = attempts
        self.seen: set[Genotype] = set()
        self.replacements = 0
        self.exhausted = 0

    @property
    def name(self) -> str:
        return f"{self.strategy.name}+novelty"

    @property
    def population_size(self) -> int:
        return self.strategy.population_size

    def _freshen(self, proposed: list[Genotype]) -> list[Genotype]:
        out: list[Genotype] = []
        for genotype in proposed:
            candidate = genotype
            for _ in range(self.attempts):
                if candidate not in self.seen:
                    break
                candidate = self.strategy.random_genotype()
                self.replacements += 1
            else:
                # Budget is spent either way, so a duplicate is recorded rather than
                # dropped: silently shrinking a generation would make two strategies
                # incomparable at the same nominal budget.
                self.exhausted += 1
            self.seen.add(candidate)
            out.append(candidate)
        return out

    def initial_population(self) -> list[Genotype]:
        return self._freshen(self.strategy.initial_population())

    def next_generation(self, scored: Sequence[Scored]) -> list[Genotype]:
        return self._freshen(self.strategy.next_generation(scored))
