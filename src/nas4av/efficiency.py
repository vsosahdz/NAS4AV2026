"""The performance-versus-size front, built from campaign artifacts.

Parameter count is the second reported axis, and it has a property performance does not:
it is structural rather than learned, so no aspect of the evaluation protocol can bias it.
That is what lets this axis be complete while the other is still being measured — a front
can be drawn for architectures that failed to train, with their size known and their
performance absent, which is a different thing from a front that quietly omits them.

Two tolerances, fixed before any front is computed, because a tolerance chosen after
seeing the reductions is a knob rather than a criterion.

**Per corpus, from the data.** The width of the bootstrap interval on that corpus. A
performance difference smaller than the variation resampling the evaluation set would
produce is one nobody can act on, so an architecture within it is not worse in any sense a
reader can use.

**For the abstract, from outside.** The smallest architecture that still exceeds the
corpus's published CLEF-PAN baseline. An external reference with no tolerance to argue
about, and the form a result takes when it has to survive a sentence.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import artifacts_root
from .space.cost import pareto_front

#: Published CLEF-PAN baselines, transcribed from the overview papers of each edition as
#: the submitted manuscript's Table 7 reports them. External reference points, not
#: measurements of this work, and 2013's baseline is a random guess so it is excluded —
#: beating chance is not a claim worth making.
CLEF_BASELINES: dict[str, float] = {
    "2014Essay2": 0.5430,
    "2014Novel2": 0.4530,
    "2015": 0.6390,
    "2020": 0.7470,
}


@dataclass
class Point:
    """One architecture on the size axis, with whatever performance it has."""

    encoding: str
    params: int
    score: float | None
    strategy: str
    run: int
    failed: bool = False


@dataclass
class Front:
    """A non-dominated set, with the sampling behind it stated."""

    setting: str
    metric: str
    points: list[tuple[int, float]]
    population: int
    distinct_param_values: int
    failed: int
    tolerance: float
    tolerance_origin: str
    knees: list[dict[str, Any]] = field(default_factory=list)
    baseline: dict[str, Any] | None = None

    @property
    def best(self) -> tuple[int, float] | None:
        return self.points[-1] if self.points else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "setting": self.setting,
            "metric": self.metric,
            "population": self.population,
            "distinct_param_values": self.distinct_param_values,
            "failed_evaluations": self.failed,
            "front_size": len(self.points),
            "tolerance": round(self.tolerance, 5),
            "tolerance_origin": self.tolerance_origin,
            "front": [{"params": size, "score": round(value, 4)} for size, value in self.points],
            "knees": self.knees,
            "baseline": self.baseline,
            "sampling_note": (
                "a search guided by performance alone never samples the small-parameter "
                "region on purpose, so a front drawn from one is sparse by construction; "
                "front_size against distinct_param_values is the measure of that"
            ),
        }


def load_points(
    directory: Path | None = None,
    setting: str | None = None,
    metric: str = "test_roc_auc",
) -> Iterator[Point]:
    """Every evaluated architecture from the campaign artifacts.

    Reads the test-side score because a front is a *reported* object: the search selected
    on validation and never saw this. Drawing the front on validation would describe what
    the search optimised rather than what it found.
    """
    root = directory or (artifacts_root() / "sweep")
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        unit = payload["unit"]
        key = f"{unit['corpus']}/{unit['features']}"
        if setting is not None and key != setting:
            continue
        for record in payload["run"]["records"]:
            yield Point(
                encoding=record["genotype"],
                params=record["params"],
                score=None if record["failed"] else record[metric],
                strategy=unit["strategy"],
                run=unit["run"],
                failed=record["failed"],
            )


def build_front(
    points: Iterable[Point],
    setting: str,
    tolerance: float,
    tolerance_origin: str,
    metric: str = "test_roc_auc",
    baseline: float | None = None,
) -> Front:
    """The non-dominated set, with the knees a reader actually wants.

    A failed evaluation contributes its size and no performance. Dropping it entirely
    would be the smaller lie: the population a front was drawn from is part of what the
    front means, and a reader cannot tell a sparse front from a filtered one.
    """
    collected = list(points)
    usable = [point for point in collected if point.score is not None and not point.failed]
    front = pareto_front([(point.params, float(point.score)) for point in usable])  # type: ignore[arg-type]

    knees: list[dict[str, Any]] = []
    if front:
        best_params, best_score = front[-1]
        within = [(size, value) for size, value in front if value >= best_score - tolerance]
        if within:
            size, value = min(within, key=lambda pair: pair[0])
            knees.append(
                {
                    "within_tolerance": round(tolerance, 5),
                    "params": size,
                    "score": round(value, 4),
                    "score_cost": round(best_score - value, 4),
                    "reduction_vs_best": round(best_params / size, 1) if size else None,
                    "best_params": best_params,
                    "best_score": round(best_score, 4),
                }
            )

    baseline_record = None
    if baseline is not None:
        reaching = [point for point in usable if float(point.score) >= baseline]  # type: ignore[arg-type]
        if reaching:
            smallest = min(reaching, key=lambda point: point.params)
            baseline_record = {
                "baseline": baseline,
                "reached": True,
                "smallest_params": smallest.params,
                "its_score": round(float(smallest.score), 4),  # type: ignore[arg-type]
                "share_of_population_reaching": round(len(reaching) / len(usable), 4),
            }
        else:
            baseline_record = {"baseline": baseline, "reached": False}

    return Front(
        setting=setting,
        metric=metric,
        points=front,
        population=len(collected),
        distinct_param_values=len({point.params for point in collected}),
        failed=sum(1 for point in collected if point.failed),
        tolerance=tolerance,
        tolerance_origin=tolerance_origin,
        knees=knees,
        baseline=baseline_record,
    )


def compare_on_both_axes(
    front: Front,
    baselines: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Where the manual baselines sit relative to the front.

    The comparison the two-axis claim rests on. A baseline that the front dominates on
    both axes is beaten; one that dominates the front is a result against this work; and
    one that neither dominates nor is dominated is the interesting case, which a
    single-axis table would have hidden.
    """
    rows = []
    for baseline in baselines:
        size = int(baseline["params"])
        score = float(baseline["score"])
        dominated_by_front = any(
            point_size <= size
            and point_score >= score
            and (point_size, point_score) != (size, score)
            for point_size, point_score in front.points
        )
        dominates_front = all(
            size <= point_size and score >= point_score for point_size, point_score in front.points
        ) and bool(front.points)
        rows.append(
            {
                "name": baseline["name"],
                "params": size,
                "score": round(score, 4),
                "dominated_by_front": dominated_by_front,
                "dominates_whole_front": dominates_front,
                "verdict": (
                    "beaten by the front"
                    if dominated_by_front
                    else "dominates the front"
                    if dominates_front
                    else "neither dominates"
                ),
            }
        )
    return {
        "setting": front.setting,
        "metric": front.metric,
        "front_size": len(front.points),
        "baselines": rows,
    }


def size_against_pretrained(params: int, reference: str, reference_params: int) -> dict[str, Any]:
    """The efficiency claim, as a ratio rather than as an adjective.

    The submitted conclusion states the discovered models "demand significantly fewer
    computational resources than large language models" with no number attached. This is
    the number, and if it is smaller than the sentence implies then the sentence changes.
    """
    return {
        "discovered_params": params,
        "reference": reference,
        "reference_params": reference_params,
        "ratio": round(reference_params / params, 1) if params else None,
    }
