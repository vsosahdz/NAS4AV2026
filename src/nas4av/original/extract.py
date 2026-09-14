"""Regenerate ``pipeline.py`` from the prior implementation's notebook.

The extraction is a tool rather than a one-off paste, for one reason: if the transcription
is checkable, then a disagreement between our results and the prior ones is a finding
about the work. If it is a paste, a disagreement is equally likely to be a typo, and no
amount of care makes that distinguishable after the fact.

So the notebook is the source of truth, the cell list below is the manifest, and the only
edits applied are the device substitutions enumerated in ``DEVICE_SUBSTITUTIONS`` — each
of which is asserted to fire, so a silently-missed one cannot ship.

    python -m nas4av.original.extract --check     # verify pipeline.py is current
    python -m nas4av.original.extract --write     # regenerate it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..prior.repository import reference_root

#: Notebook code cells to extract, in the order they must appear. Indices are into the
#: code cells only, skipping markdown.
#:
#: Deliberately not everything. The graph search space, the ensemble machinery, the
#: preprocessing and the SVM baseline are excluded: the features are already stored as
#: tensors, and the parts that are being replaced should not be carried along.
CELLS: tuple[tuple[int, str], ...] = (
    (4, "debug"),
    # Label construction. Extracted rather than reimplemented because the pairing is
    # load-bearing and easy to get subtly wrong: each CSV row yields two documents and
    # one label, so the document frame is twice the length of the label list, and
    # ``make_df`` pads the tail with -1 values that are never read.
    (5, "split_data"),
    (9, "make_df"),
    (26, "RDropout"),
    (28, "Model"),
    (30, "Search_Space"),
    (32, "Sequential_Search_Space"),
    (44, "TrainingType"),
    (45, "train_predict"),
    (46, "train_predict_evaluate"),
)

#: Every device edit, applied in order. The prior implementation hardcodes CUDA in eleven
#: places; a machine without an NVIDIA device cannot run a single evaluation until they
#: are parameterised.
#:
#: Each entry is (pattern, replacement, expected_occurrences). The count is asserted, so
#: a notebook revision that adds or removes a hardcoded device fails the extraction
#: instead of silently producing code that runs on the wrong one.
DEVICE_SUBSTITUTIONS: tuple[tuple[str, str, int], ...] = (
    ("""torch.device("cuda" if torch.cuda.is_available() else "cpu")""", "DEVICE", 1),
    ("""device='cuda'""", "device=DEVICE", 1),
    (""".to(torch.device('cuda'))""", ".to(DEVICE)", 6),
    (""".to('cuda')""", ".to(DEVICE)", 3),
)

HEADER = '''"""Extracted from the prior implementation's notebook. Do not edit by hand.

Regenerate with ``python -m nas4av.original.extract --write``; the cell manifest and the
device substitutions live in ``extract.py`` next to this file.

This module is kept as close to its source as running outside Jupyter allows, which is
why it is exempt from linting and type checking. The point is not that the code is good —
several things in it are the subject of the work — but that a reproduction mismatch
should be a finding about the prior results rather than a transcription error of ours.

Two properties are worth knowing before calling anything here.

``Search_Space.metrics`` reports balanced accuracy under the name AUC. Predictions are
binarised at ``logit > 0`` by ``logit2int``, a confusion matrix is built from the 0/1
labels, and ``calc_auc`` averages the two class-wise recalls. No continuous score reaches
the metric.

``train_predict_evaluate`` swallows every exception and records the architecture with
zeros for all measurements. A row of zeros is therefore a failure, not a measurement —
though in the prior campaign's 66,000 rows, none occurred.
"""

from __future__ import annotations

import random
import time
from copy import deepcopy
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import KFold


class _NotExtracted:
    """Stands in for classes the manifest deliberately leaves behind.

    ``train_predict`` and ``train_predict_evaluate`` dispatch on the search space with an
    ``isinstance`` chain that includes the graph variants. Leaving those names undefined
    would be worse than it looks: ``train_predict_evaluate`` wraps everything in a bare
    ``except``, so a ``NameError`` would be caught and recorded as an architecture scoring
    zero on every measurement — a failure indistinguishable, in the output table, from a
    real evaluation.

    Binding them here keeps the dispatch well-defined and always false.
    """


Graph_Search_Space = _NotExtracted
Model_FG = _NotExtracted

#: Where tensors and modules live. The prior implementation hardcoded CUDA; this is the
#: only behavioural change the extraction makes.
#:
#: float64 BatchNorm is used throughout, which Apple MPS does not implement, so MPS is
#: deliberately not offered here — it would fail at the first BatchNorm rather than fall
#: back, and a silent CPU fallback would misreport what produced a number.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''


def notebook_path() -> Path:
    return reference_root() / "NAS_AV.ipynb"


def code_cells() -> list[str]:
    with notebook_path().open() as handle:
        notebook = json.load(handle)
    return [
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]


def build() -> str:
    """Extract, substitute, then prepend the header.

    The substitutions are applied to the extracted body alone. Running them over the
    header too would rewrite the very line that defines ``DEVICE``, and the occurrence
    counts below would be counting our own text.
    """
    cells = code_cells()
    parts = []
    for index, name in CELLS:
        if index >= len(cells):
            raise SystemExit(f"notebook has {len(cells)} code cells; {name} expects {index}")
        source = cells[index].strip("\n")
        rule = "─" * max(0, 66 - len(name))
        parts.append(f"\n\n# ── {name} {rule}\n# code cell {index}\n\n{source}\n")

    body = "".join(parts)
    for pattern, replacement, expected in DEVICE_SUBSTITUTIONS:
        found = body.count(pattern)
        if found != expected:
            raise SystemExit(
                f"device substitution {pattern!r} matched {found} times, expected {expected}. "
                "The notebook changed; update DEVICE_SUBSTITUTIONS deliberately rather than "
                "loosening the check."
            )
        body = body.replace(pattern, replacement)

    leftovers = [line for line in body.splitlines() if "cuda" in line.lower()]
    if leftovers:
        raise SystemExit(
            "a hardcoded device survived the substitutions:\n  " + "\n  ".join(leftovers)
        )
    return HEADER + body


def target() -> Path:
    return Path(__file__).resolve().parent / "pipeline.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate pipeline.py")
    group.add_argument("--check", action="store_true", help="fail if pipeline.py is stale")
    args = parser.parse_args(argv)

    generated = build()
    path = target()

    if args.write:
        path.write_text(generated, encoding="utf-8")
        print(f"wrote {path} ({generated.count(chr(10))} lines)")
        return 0

    if not path.exists():
        print(f"{path} does not exist; run --write", file=sys.stderr)
        return 1
    if path.read_text(encoding="utf-8") != generated:
        print(f"{path} is stale; run --write", file=sys.stderr)
        return 1
    print(f"{path} is current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
