"""The driver: spends a fixed budget on a strategy and records what it bought.

The budget is counted here rather than inside any strategy, because the comparison the
reviewers asked for — the variant against the canonical algorithm against uniform random
— is only meaningful if none of the three can quietly spend more than the others.

Every evaluation is wrapped in ``isolated_randomness``. Without it the extracted training
loop reseeds the global generator the strategies would otherwise be exposed to; the
strategies also hold their own generators, and keeping both guards is deliberate, since
either alone is a single point of failure for a defect that is invisible in the output.

What a strategy sees and what is recorded are different on purpose. A strategy receives
only the validation fitness. The test score travels straight into the record without
passing through the search, which is the structural form of the protocol rather than a
promise to be careful.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

from ..protocol import ProtocolSplit, isolated_randomness
from ..space.cost import describe
from .strategy import Genotype, Scored, SearchStrategy


@dataclass(frozen=True)
class Measurement:
    """What one evaluation returns.

    Both metrics are recorded for every architecture. ROC-AUC is what the literature
    reports and what makes a comparison against published results like-for-like; balanced
    accuracy is what the prior campaign measured under the name AUC, and it is kept so the
    two campaigns remain comparable to each other.
    """

    validation_roc_auc: float
    validation_balanced_accuracy: float
    test_roc_auc: float
    test_balanced_accuracy: float
    seconds: float
    failed: bool = False
    detail: str = ""


class Evaluator(Protocol):
    """Trains one architecture and measures it. Injected, so the driver stays testable."""

    def __call__(self, genotype: Genotype) -> Measurement: ...


@dataclass
class EvaluationRecord:
    """One row of the campaign's output."""

    generation: int
    genotype: str
    params: int
    validation_roc_auc: float
    validation_balanced_accuracy: float
    test_roc_auc: float
    test_balanced_accuracy: float
    seconds: float
    failed: bool
    detail: str


@dataclass
class SearchRun:
    """Everything one run produced, and enough context to know what it means."""

    strategy: str
    corpus: str
    strategy_seed: int
    split: dict[str, object]
    budget: int
    fitness_metric: str
    input_shape: int = 0
    records: list[EvaluationRecord] = field(default_factory=list)
    novelty_replacements: int = 0
    novelty_exhausted: int = 0
    #: Set when the run stopped before spending its budget, with the reason. An empty
    #: string means the budget was spent as planned.
    halted: str = ""

    @property
    def evaluations(self) -> int:
        return len(self.records)

    @property
    def failures(self) -> int:
        return sum(1 for record in self.records if record.failed)

    def best(self) -> EvaluationRecord | None:
        """Best by the fitness the search actually optimised, never by the test score."""
        usable = [record for record in self.records if not record.failed]
        if not usable:
            return None
        return max(usable, key=lambda record: getattr(record, self.fitness_metric))

    def as_dict(self) -> dict[str, object]:
        best = self.best()
        return {
            "strategy": self.strategy,
            "corpus": self.corpus,
            "strategy_seed": self.strategy_seed,
            "split": self.split,
            "budget": self.budget,
            "input_shape": self.input_shape,
            "fitness_metric": self.fitness_metric,
            "halted": self.halted,
            "evaluations": self.evaluations,
            "failures": self.failures,
            "novelty_replacements": self.novelty_replacements,
            "novelty_exhausted": self.novelty_exhausted,
            "best": asdict(best) if best else None,
            "records": [asdict(record) for record in self.records],
        }


#: Which measurement the search optimises. Validation only — the test fields exist on the
#: record so a reader can see them afterwards, not so a strategy can select on them.
FITNESS_METRICS = ("validation_roc_auc", "validation_balanced_accuracy")


def run_search(
    strategy: SearchStrategy,
    evaluator: Evaluator,
    split: ProtocolSplit,
    input_shape: int,
    budget: int,
    strategy_seed: int,
    fitness_metric: str = "validation_roc_auc",
    on_generation: Callable[[int, SearchRun], None] | None = None,
) -> SearchRun:
    """Spend ``budget`` evaluations and return everything they produced.

    The budget is a count of evaluations, not of generations, so strategies with different
    population sizes remain comparable. A generation that would overrun is truncated
    rather than allowed to finish, since an overrun is exactly the way a budget-matched
    comparison stops being one.
    """
    if fitness_metric not in FITNESS_METRICS:
        raise ValueError(
            f"fitness_metric must be one of {FITNESS_METRICS}; a test-side metric would "
            "put the test split back inside the objective, which is the defect this "
            "protocol exists to remove"
        )
    if budget < strategy.population_size:
        raise ValueError("the budget must cover at least one full generation")

    run = SearchRun(
        strategy=getattr(strategy, "name", type(strategy).__name__),
        corpus=split.corpus,
        strategy_seed=strategy_seed,
        split=split.as_dict(),
        budget=budget,
        input_shape=input_shape,
        fitness_metric=fitness_metric,
    )

    population = strategy.initial_population()
    generation = 0
    while True:
        remaining = budget - run.evaluations
        if remaining <= 0:
            break
        scored = _evaluate(population[:remaining], evaluator, run, generation, fitness_metric)
        if on_generation is not None:
            on_generation(generation, run)
        if run.evaluations >= budget:
            break
        if not scored:
            # Every architecture in the generation failed to build, so there is nothing to
            # breed from. Recorded on the run rather than as a fabricated evaluation: a
            # placeholder row would inflate the count the budget comparison rests on.
            run.halted = f"generation {generation} produced no usable individual"
            break
        population = strategy.next_generation(scored)
        generation += 1

    for attribute in ("replacements", "exhausted"):
        value = getattr(strategy, attribute, None)
        if value is not None:
            setattr(run, f"novelty_{attribute}", value)
    return run


def _evaluate(
    population: Sequence[Genotype],
    evaluator: Evaluator,
    run: SearchRun,
    generation: int,
    fitness_metric: str,
) -> list[Scored]:
    scored: list[Scored] = []
    for genotype in population:
        with isolated_randomness():
            measurement = evaluator(genotype)
        architecture = describe(genotype, run.input_shape)
        record = EvaluationRecord(
            generation=generation,
            genotype=architecture.encoding,
            params=architecture.params,
            validation_roc_auc=measurement.validation_roc_auc,
            validation_balanced_accuracy=measurement.validation_balanced_accuracy,
            test_roc_auc=measurement.test_roc_auc,
            test_balanced_accuracy=measurement.test_balanced_accuracy,
            seconds=measurement.seconds,
            failed=measurement.failed,
            detail=measurement.detail,
        )
        run.records.append(record)
        if not measurement.failed:
            scored.append(Scored(genotype, getattr(record, fitness_metric)))
    return scored
