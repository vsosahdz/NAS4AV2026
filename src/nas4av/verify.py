"""Re-run a finished unit and compare it against what it produced the first time.

A claim of reproducibility that cannot be executed is a claim about intentions. This runs
the comparison: take a stored artifact, re-run the unit it describes from the seed it
recorded, and report whether the two agree evaluation by evaluation.

**What agreement means here is narrower than it sounds, and the narrowing is the point.**
A unit is reproducible on the platform that produced it. Across platforms it is not
expected to be: the campaign already contains artifacts from two machines whose torch
versions differ by a minor release, and float64 reduction order is not guaranteed across
builds. So the comparison reports the provenance of both sides, and a mismatch between
different platforms is a measurement of that gap rather than a failure.

The reference platform is declared rather than assumed. A reader who reproduces on
something else gets a number for how far their platform sits from it, which is more useful
than a pass that was only ever going to pass on one machine.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import artifacts_root, provenance
from .sweep import WorkUnit, run_unit

#: Fields compared evaluation by evaluation. The genotype and the parameter count are
#: structural and must match exactly; the measurements are compared with a tolerance,
#: because a reduction-order difference is not a different result.
EXACT_FIELDS = ("generation", "genotype", "params")
MEASURED_FIELDS = (
    "validation_roc_auc",
    "validation_balanced_accuracy",
    "test_roc_auc",
    "test_balanced_accuracy",
)

#: Tolerance on a measured field. Tighter than any difference this work reports, and loose
#: enough that a float64 reduction-order difference does not register as a disagreement.
TOLERANCE = 1e-9

#: Fields of the provenance that make two artifacts comparable. A difference in any of
#: them means a mismatch is expected rather than alarming, and it is reported as such.
PLATFORM_FIELDS = ("platform", "torch_device")
PACKAGE_FIELDS = ("torch", "numpy", "scipy")


@dataclass
class Divergence:
    """One evaluation where the two runs disagree."""

    index: int
    field: str
    stored: Any
    replayed: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation": self.index,
            "field": self.field,
            "stored": self.stored,
            "replayed": self.replayed,
        }


@dataclass
class VerificationResult:
    """Whether a unit reproduced, and what would explain it if not."""

    unit: str
    evaluations: int
    structural_divergences: list[Divergence] = field(default_factory=list)
    measured_divergences: list[Divergence] = field(default_factory=list)
    same_platform: bool = True
    stored_environment: dict[str, Any] = field(default_factory=dict)
    replayed_environment: dict[str, Any] = field(default_factory=dict)

    @property
    def reproduced(self) -> bool:
        return not self.structural_divergences and not self.measured_divergences

    @property
    def verdict(self) -> str:
        """Three outcomes, because two would hide the one that matters.

        A mismatch on the same platform is a defect. The same mismatch across platforms is
        a measurement of the gap between them, and calling both "failed" would make the
        first invisible among the second.
        """
        if self.reproduced:
            return "reproduced"
        if self.structural_divergences:
            return "structural mismatch"
        return (
            "numeric mismatch, same platform"
            if self.same_platform
            else ("numeric mismatch, different platform")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "verdict": self.verdict,
            "reproduced": self.reproduced,
            "evaluations_compared": self.evaluations,
            "same_platform": self.same_platform,
            "tolerance": TOLERANCE,
            "structural_divergences": [d.as_dict() for d in self.structural_divergences[:10]],
            "measured_divergences": [d.as_dict() for d in self.measured_divergences[:10]],
            "structural_divergence_count": len(self.structural_divergences),
            "measured_divergence_count": len(self.measured_divergences),
            "stored_environment": self.stored_environment,
            "replayed_environment": self.replayed_environment,
        }


def _environment(record: dict[str, Any]) -> dict[str, Any]:
    packages = record.get("packages", {})
    return {
        **{name: record.get(name) for name in PLATFORM_FIELDS},
        **{name: packages.get(name) for name in PACKAGE_FIELDS},
        "python": record.get("python"),
    }


def unit_from_artifact(payload: dict[str, Any]) -> WorkUnit:
    """Rebuild the unit an artifact describes, so the replay is of the same thing.

    Taken from the artifact rather than from a plan: a plan can drift as defaults change,
    and the point of a replay is to re-run what was run, not what would be run today.
    """
    stored = payload["unit"]
    return WorkUnit(
        corpus=stored["corpus"],
        features=stored["features"],
        strategy=stored["strategy"],
        run=stored["run"],
        seed=stored["seed"],
        novelty=stored["novelty_filter"],
        budget=stored["budget"],
        population_size=stored["population_size"],
    )


def verify_unit(identifier: str, keep_scores: bool = False) -> VerificationResult:
    """Replay one stored unit and compare, without overwriting what it is compared to.

    The replay writes to a scratch artifact root so a failed verification cannot replace
    the artifact that failed it — which would make the discrepancy unreproducible, in a
    module about reproducibility.
    """
    stored_path = artifacts_root() / "sweep" / f"{identifier}.json"
    if not stored_path.exists():
        raise FileNotFoundError(f"no stored artifact at {stored_path}")
    payload = json.loads(stored_path.read_text(encoding="utf-8"))

    unit = unit_from_artifact(payload)
    stored_records = payload["run"]["records"]

    scratch = artifacts_root() / "verification" / identifier
    scratch.mkdir(parents=True, exist_ok=True)
    import os

    previous = os.environ.get("NAS4AV_ARTIFACTS")
    os.environ["NAS4AV_ARTIFACTS"] = str(scratch)
    try:
        replayed_run = run_unit(unit, keep_scores=keep_scores)
    finally:
        if previous is None:
            os.environ.pop("NAS4AV_ARTIFACTS", None)
        else:
            os.environ["NAS4AV_ARTIFACTS"] = previous

    replayed_records = [record.__dict__ for record in replayed_run.records]
    stored_environment = _environment(payload["provenance"])
    replayed_environment = _environment(provenance())

    result = VerificationResult(
        unit=identifier,
        evaluations=min(len(stored_records), len(replayed_records)),
        same_platform=stored_environment == replayed_environment,
        stored_environment=stored_environment,
        replayed_environment=replayed_environment,
    )

    if len(stored_records) != len(replayed_records):
        result.structural_divergences.append(
            Divergence(-1, "evaluation_count", len(stored_records), len(replayed_records))
        )

    for index, (stored, replayed) in enumerate(zip(stored_records, replayed_records, strict=False)):
        for name in EXACT_FIELDS:
            if stored.get(name) != replayed.get(name):
                result.structural_divergences.append(
                    Divergence(index, name, stored.get(name), replayed.get(name))
                )
        for name in MEASURED_FIELDS:
            left, right = stored.get(name), replayed.get(name)
            if _differs(left, right):
                result.measured_divergences.append(Divergence(index, name, left, right))

    return result


def _differs(left: Any, right: Any) -> bool:
    """NaN equals NaN here: a failed evaluation that fails again has reproduced."""
    if left is None or right is None:
        return left is not right
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return False
        return not math.isclose(left, right, rel_tol=0.0, abs_tol=TOLERANCE)
    return bool(left != right)


def verify_many(identifiers: Sequence[str], keep_scores: bool = False) -> dict[str, Any]:
    """Verify several units and summarise. Used by the replication tier."""
    results = [verify_unit(name, keep_scores=keep_scores) for name in identifiers]
    return {
        "units": len(results),
        "reproduced": sum(1 for r in results if r.reproduced),
        "verdicts": {r.unit: r.verdict for r in results},
        "details": [r.as_dict() for r in results],
    }


def stored_units() -> list[str]:
    """Identifiers of every unit with a stored artifact."""
    directory = artifacts_root() / "sweep"
    return sorted(path.stem for path in directory.glob("*.json")) if directory.is_dir() else []


def reference_platform(sample: int = 1) -> dict[str, Any]:
    """What the stored artifacts say they were produced on.

    A replication package has to name the platform its numbers came from, and the
    artifacts already carry it — so it is read rather than written down separately, where
    it could drift.
    """
    environments: dict[str, int] = {}
    directory = artifacts_root() / "sweep"
    for path in sorted(directory.glob("*.json"))[: sample or None]:
        record = _environment(json.loads(path.read_text(encoding="utf-8"))["provenance"])
        environments[json.dumps(record, sort_keys=True)] = (
            environments.get(json.dumps(record, sort_keys=True), 0) + 1
        )
    return {
        "distinct_environments": len(environments),
        "environments": [
            {"count": count, **json.loads(key)} for key, count in environments.items()
        ],
        "homogeneous": len(environments) <= 1,
    }


def artifact_path(identifier: str) -> Path:
    return artifacts_root() / "sweep" / f"{identifier}.json"
