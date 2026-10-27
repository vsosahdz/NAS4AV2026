"""Does the search buy anything? The comparison the whole campaign exists to answer.

The submitted manuscript reports what its evolutionary variant found and never asks what a
simpler method would have found at the same cost. Three reviewer questions reduce to that
one, and the NAS literature treats a random-search control as mandatory for exactly this
reason: a reported result is otherwise compatible with the search space simply containing
good architectures.

So three strategies at an identical budget of 1,200 evaluations, five runs each, over
eleven settings. The comparison is built on three rules that are not negotiable once the
numbers exist, because each of them is a place where a favourable answer could be
manufactured after the fact.

**Selection is on validation, reporting is on test.** The per-run number is the test score
of whichever architecture had the best *validation* score. Ranking strategies by their best
test score would reward whichever one searched hardest against the reporting split, which
is the defect this entire rebuild exists to remove.

**Runs are aggregated by median, not by maximum.** A maximum over five runs measures how
lucky a strategy got, and it is biased upward by run count rather than by quality. The
median says what one run of this strategy typically delivers, which is what somebody
choosing a method actually needs.

**Both blocking structures are reported.** Eleven settings give usable power and are not
independent; five corpora are independent and sit exactly at the conventional floor.
Reporting whichever one produced the preferred answer would be the same selection defect in
a new place.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import artifacts_root
from .statistics import both_structures, bootstrap_interval, equivalence, rope_from_intervals

#: Space names that may appear as a leading token in a unit identifier. Read from the
#: registry rather than hardcoded, so adding a third space cannot leave this behind.
def _space_prefixes() -> frozenset[str]:
    from .space.spec import _REGISTRY

    return frozenset(_REGISTRY) - {"v1"}


_SPACE_PREFIXES = _space_prefixes()

#: The strategies the campaign ran, with the names the artifacts carry.
STRATEGIES = ("evolutionary-variant", "canonical-ga", "uniform-random")

#: The control the comparison is framed against. Uniform random is the question "did the
#: search help", and making it the control shrinks the correction family from every pair to
#: every-strategy-against-random, which costs less power for the question that matters.
CONTROL = "uniform-random"


@dataclass
class RunOutcome:
    """One run of one strategy on one setting, reduced to the number that is reportable."""

    setting: str
    strategy: str
    run: int
    validation: float
    test: float
    params: int
    encoding: str
    evaluations: int
    failures: int


@dataclass
class StrategyOnSetting:
    """What a strategy delivered on one setting, across its runs."""

    setting: str
    strategy: str
    runs: list[RunOutcome] = field(default_factory=list)

    @property
    def median_test(self) -> float:
        return statistics.median(run.test for run in self.runs)

    @property
    def median_validation(self) -> float:
        return statistics.median(run.validation for run in self.runs)

    @property
    def spread(self) -> float:
        """Run-to-run spread, which is not a standard error and is never called one.

        These runs share an initialisation distribution and differ only in seed, so the
        spread describes how much the search wanders — not the precision of an estimate of
        anything. D14 is the finding that made this distinction necessary.
        """
        return statistics.pstdev([run.test for run in self.runs]) if len(self.runs) > 1 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "setting": self.setting,
            "strategy": self.strategy,
            "runs": len(self.runs),
            "median_test": round(self.median_test, 6),
            "median_validation": round(self.median_validation, 6),
            "run_to_run_spread": round(self.spread, 6),
            "best_run_test": round(max(run.test for run in self.runs), 6),
            "median_params": int(statistics.median(run.params for run in self.runs)),
            "failures": sum(run.failures for run in self.runs),
            "spread_note": (
                "run-to-run spread under a shared initialisation distribution, not a "
                "standard error and not a replication interval"
            ),
        }


def load_outcomes(
    directory: Path | None = None, metric: str = "test_roc_auc"
) -> list[RunOutcome]:
    """Every run's reportable outcome, selected on validation.

    The selection happens here rather than in the artifact so that it is visible and
    changeable: an artifact records every evaluation, and which one counts is a decision
    about the protocol, not a property of the data.
    """
    root = directory or (artifacts_root() / "sweep")
    validation_metric = metric.replace("test_", "validation_")
    outcomes: list[RunOutcome] = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        unit = payload["unit"]
        records = [r for r in payload["run"]["records"] if not r["failed"]]
        if not records:
            continue
        chosen = max(records, key=lambda r: r[validation_metric])
        outcomes.append(
            RunOutcome(
                setting=f"{unit['corpus']}/{unit['features']}",
                strategy=unit["strategy"],
                run=int(unit["run"]),
                validation=float(chosen[validation_metric]),
                test=float(chosen[metric]),
                params=int(chosen["params"]),
                encoding=str(chosen["genotype"]),
                evaluations=int(payload["run"]["evaluations"]),
                failures=int(payload["run"]["failures"]),
            )
        )
    return outcomes


def group(outcomes: Iterable[RunOutcome]) -> dict[tuple[str, str], StrategyOnSetting]:
    grouped: dict[tuple[str, str], StrategyOnSetting] = {}
    for outcome in outcomes:
        key = (outcome.setting, outcome.strategy)
        if key not in grouped:
            grouped[key] = StrategyOnSetting(outcome.setting, outcome.strategy)
        grouped[key].runs.append(outcome)
    return grouped


def strategy_table(outcomes: Iterable[RunOutcome]) -> dict[str, dict[str, float]]:
    """Treatment to {setting: median test score}, as the Friedman test wants it."""
    table: dict[str, dict[str, float]] = {}
    for (setting, strategy), entry in group(outcomes).items():
        table.setdefault(strategy, {})[setting] = entry.median_test
    return table


def budget_is_matched(outcomes: Iterable[RunOutcome]) -> dict[str, Any]:
    """Whether the strategies really did spend the same number of evaluations.

    Checked rather than assumed. A budget enforced by the driver is the premise the whole
    comparison rests on, and a strategy that quietly spent more would beat the others for a
    reason that has nothing to do with search quality. The novelty filter makes this a live
    risk: it redraws duplicates, and an implementation that dropped them instead would
    shrink a generation without saying so.
    """
    spent: dict[str, set[int]] = {}
    for outcome in outcomes:
        spent.setdefault(outcome.strategy, set()).add(outcome.evaluations)
    return {
        "evaluations_per_run": {k: sorted(v) for k, v in sorted(spent.items())},
        "matched": len({value for values in spent.values() for value in values}) == 1,
    }


def compare(
    directory: Path | None = None, metric: str = "test_roc_auc", rope: float | None = None
) -> dict[str, Any]:
    """The whole comparison, as one report.

    Returns rather than prints, so the same object serves the CLI, the tests, and whatever
    builds the manuscript's tables — three consumers that must not disagree about what the
    numbers were.
    """
    outcomes = load_outcomes(directory, metric=metric)
    if not outcomes:
        return {"error": "no artifacts found", "metric": metric}

    grouped = group(outcomes)
    table = strategy_table(outcomes)
    budget = budget_is_matched(outcomes)

    present = sorted(table)
    missing_strategies = [s for s in STRATEGIES if s not in table]
    comparison: dict[str, Any]
    if len(present) < 2:
        comparison = {"error": f"only {present} present; a comparison needs at least two"}
    else:
        control = CONTROL if CONTROL in table else None
        comparison = both_structures(table, metric=metric, control=control)

    # The equivalence region comes from the data rather than from a convention, and it is
    # derived from *document* resampling rather than from run spread. Those measure
    # different things: run spread says how much the search wanders, while what limits this
    # study is which documents happened to land in the evaluation set. A region built from
    # run spread would widen whenever a strategy got unlucky, which is backwards.
    region, region_origin = _region(directory, rope, metric)

    per_setting = {}
    for (setting, strategy), entry in sorted(grouped.items()):
        per_setting.setdefault(setting, {})[strategy] = entry.as_dict()

    return {
        "metric": metric,
        "selection": "best validation score per run; the test score of that architecture is reported",
        "aggregation": "median across runs, never maximum",
        "strategies_present": present,
        "strategies_missing": missing_strategies,
        "units": len(outcomes),
        "budget": budget,
        "rope": {"region": round(region, 6), "origin": region_origin},
        "equivalence": _equivalence_against_control(table, region, region_origin),
        "comparison": comparison,
        "per_setting": per_setting,
    }


def _region(
    directory: Path | None, override: float | None, metric: str
) -> tuple[float, str]:
    """The equivalence region, from document resampling on the stored per-architecture scores.

    Falls back to zero rather than to a convention when the scores are absent. A region of
    zero makes every equivalence claim fail, which is the right failure: an equivalence
    result rests entirely on where the region came from, and inventing one would make the
    weakest number in the report the least visible.
    """
    if override is not None:
        return override, "supplied by the caller"

    root = (directory or (artifacts_root() / "sweep")) / "scores"
    if not root.is_dir():
        return 0.0, "unavailable: no per-architecture scores were stored"

    # The stored scores are logits only. The labels are not stored beside them because they
    # are a property of the corpus rather than of an evaluation — the same thirty test
    # pairs for every architecture ever run on 2013 — so duplicating them 1,200 times per
    # unit would be storing one fact 1,200 times and inviting it to disagree with itself.
    from .prior.dataset import load_corpus

    labels_for: dict[str, list[int]] = {}

    def test_labels(corpus: str, features: str) -> list[int] | None:
        if corpus not in labels_for:
            try:
                labels_for[corpus] = list(load_corpus(corpus, features).test_labels)
            except Exception:  # noqa: BLE001 — absent corpora give no region, not a crash
                labels_for[corpus] = []
        return labels_for[corpus] or None

    # One interval per setting, not per architecture. The region describes how much the
    # evaluation set's composition moves a score, which is a property of the corpus; taking
    # all 1,200 architectures of a unit would weight settings by how many were run on them
    # and measure the spread between architectures instead.
    seen: set[str] = set()
    intervals = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # A scores file records its unit as the identifier string rather than as the
        # structured record the artifact carries, so corpus and features are read off the
        # name. Underscore-separated and safe to split: the strategies that contain a
        # separator use a hyphen.
        #
        # A unit from a space other than the first carries that space as a leading token,
        # so it is stripped before the corpus is read. Without this, `v2_2013_sem_...`
        # parses as corpus "v2" and features "2013", no corpus loads, and the region comes
        # back as "unavailable" — which looks like missing scores rather than a parse bug.
        identifier = payload.get("unit")
        parts = identifier.split("_") if isinstance(identifier, str) else []
        if parts and parts[0] in _SPACE_PREFIXES:
            parts = parts[1:]
        if len(parts) < 2:
            continue
        corpus, features = parts[0], parts[1]
        setting = f"{corpus}/{features}"
        if setting in seen:
            continue
        labels = test_labels(corpus, features)
        if labels is None:
            continue
        for record in list(payload.get("scores", {}).values())[:1]:
            scores = record.get("test")
            if not scores or len(scores) != len(labels) or len(set(labels)) < 2:
                continue
            seen.add(setting)
            intervals.append(
                bootstrap_interval(
                    scores,
                    labels,
                    metric="roc_auc" if metric.endswith("roc_auc") else "balanced_accuracy",
                    resamples=400,
                )
            )
    if not intervals:
        return 0.0, "unavailable: stored scores carried no usable two-class split"
    return (
        rope_from_intervals(intervals),
        f"median half-width of bootstrap intervals over {len(intervals)} settings' test documents",
    )


def _equivalence_against_control(
    table: dict[str, dict[str, float]], region: float, region_origin: str
) -> dict[str, Any]:
    """Where a strategy is not merely 'not significantly different' but equivalent.

    The distinction a null result cannot make on its own. Failing to reject says nothing
    about whether a difference exists; an equivalence test against a region fixed in
    advance says the difference is too small to matter, which is a claim with content —
    and for a search method that costs more than random, "indistinguishable from random"
    is the finding, not the absence of one.
    """
    if CONTROL not in table:
        return {"error": f"control {CONTROL} absent"}
    if region <= 0:
        return {"error": "no region could be derived, so no equivalence claim is made"}
    out: dict[str, Any] = {}
    for strategy in sorted(table):
        if strategy == CONTROL:
            continue
        out[strategy] = equivalence(
            table, strategy, CONTROL, rope=region, rope_origin=region_origin
        ).as_dict()
    return out
