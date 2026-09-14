"""The extracted pipeline must stay a faithful transcription.

``pipeline.py`` is generated from the prior implementation's notebook by ``extract.py``.
If it is edited by hand, or if the notebook changes underneath it, the comparison between
our results and the prior ones stops meaning anything — a disagreement becomes as likely
to be our typo as their finding.

So the generated file is checked rather than trusted, and the device substitutions carry
expected occurrence counts that the extractor asserts.
"""

from __future__ import annotations

import pytest

from nas4av.original import extract
from nas4av.prior.repository import PriorRepositoryMissing

pytestmark = pytest.mark.legacy


@pytest.fixture(scope="module")
def _require_notebook():
    try:
        if not extract.notebook_path().exists():
            pytest.skip("the prior implementation's notebook is not present")
    except PriorRepositoryMissing as exc:
        pytest.skip(str(exc).splitlines()[0])


def test_generated_pipeline_is_current(_require_notebook):
    """Regenerating must be a no-op. If this fails, run ``--write`` and read the diff."""
    assert extract.build() == extract.target().read_text(encoding="utf-8")


def test_no_hardcoded_device_survives(_require_notebook):
    """The substitutions are the only behavioural change the extraction makes.

    A missed one is not a cosmetic problem: on a machine without an NVIDIA device it
    raises at the first tensor, and on a machine with one it would silently pin work to
    a device the caller did not choose.
    """
    generated = extract.build()
    body = generated.split("DEVICE = torch.device", 1)[1]
    assert "cuda" not in body.split("\n", 1)[1].lower()


def test_every_substitution_is_exercised(_require_notebook):
    """Each entry in the table must fire exactly as many times as it claims.

    An entry that matches nothing is dead weight that hides a real pattern; one that
    matches more than expected means the notebook changed shape. Both should fail loudly
    rather than be absorbed.
    """
    cells = extract.code_cells()
    body = "".join(cells[index] for index, _ in extract.CELLS)
    for pattern, _, expected in extract.DEVICE_SUBSTITUTIONS:
        assert body.count(pattern) == expected, f"{pattern!r} no longer matches {expected}"


def test_manifest_covers_what_the_harness_imports(_require_notebook):
    """The names the rest of the package reaches for must survive extraction."""
    from nas4av.original import pipeline

    for name in (
        "Model",
        "Search_Space",
        "Sequential_Search_Space",
        "TrainingType",
        "train_predict",
        "train_predict_evaluate",
        "split_data",
        "make_df",
    ):
        assert hasattr(pipeline, name), f"{name} is missing from the extraction"
