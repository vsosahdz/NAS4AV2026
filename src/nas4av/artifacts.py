"""Writing artifacts, so that no reported number is ever typed by hand.

Every stage writes its outputs here with a provenance header. The convention is not
generic hygiene; it guards a specific failure mode. A results table can be arithmetically
correct and still unverifiable, if the code that produced it reads inputs that are no
longer in the tree — and that combination looks fine until somebody tries to re-run it.

A provenance header is what makes the difference between a number and a measurement.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

#: Packages whose versions are recorded with every artifact. A number produced by a
#: different resolution of these is a different number.
TRACKED_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "torch")

#: Commit of github.com/luisferro2/NAS_4_AV that the prior results were read from.
#: Recorded with every artifact that touches them, because "the prior results" is not
#: a citable object and this is.
LEGACY_UPSTREAM_SHA = "65ee41046f8c3417e4159f8f9bc96ca1a7ddf84b"


def artifacts_root() -> Path:
    """Where artifacts land. Overridable so tests never write into the real tree."""
    return Path(os.environ.get("NAS4AV_ARTIFACTS", "./artifacts")).resolve()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except Exception:
        return None


def provenance() -> dict[str, Any]:
    """What is needed to know whether two artifacts are comparable.

    ``git_dirty`` matters: an artifact produced from uncommitted work cannot be
    reproduced from the recorded revision, and recording the fact is better than
    discovering it later.

    ``torch_device`` matters more here than it would elsewhere. The prior results were
    produced on a CUDA device; the models use float64 BatchNorm, which Apple MPS does not
    support, so a local reproduction is a CPU reproduction. Two artifacts that differ
    only in this field are not expected to agree bit-for-bit, and the field is how a
    reader knows which comparison they are looking at.
    """
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"

    status = _git("status", "--porcelain")
    return {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_device": _torch_device(),
        "determinism_environment": {
            var: os.environ.get(var)
            for var in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "seed": os.environ.get("NAS4AV_SEED"),
        "legacy_upstream_sha": LEGACY_UPSTREAM_SHA,
        "command": " ".join(sys.argv),
    }


def _torch_device() -> str | None:
    """The device a training artifact was produced on, or None if torch is absent.

    Reported rather than inferred at read time, because the artifact outlives the
    machine.
    """
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    return "cpu"


def write_json(relative_path: str, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact with its provenance header."""
    target = artifacts_root() / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {"provenance": provenance(), **payload}
    target.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def write_table(relative_path: str, rows: list[dict[str, Any]], columns: list[str]) -> Path:
    """Write a CSV artifact, with provenance alongside it as a sidecar.

    CSV rather than JSON for the tables meant to be consumed by others: the
    per-architecture metadata table should open in anything, including a spreadsheet,
    without a parser.
    """
    target = artifacts_root() / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(relative_path + ".provenance.json", {"rows": len(rows), "columns": columns})
    return target
