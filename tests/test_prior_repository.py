"""The loader, checked against what the prior campaign actually contains.

Marked ``legacy`` because they need the pinned clone, which is obtained rather than
vendored and so is absent by default. They are worth having anyway: every figure this
project derives from the prior work passes through this module, and the counts below are
what the rest of the analysis assumes.
"""

from __future__ import annotations

import pytest

from nas4av.prior import repository as repo

pytestmark = pytest.mark.legacy


@pytest.fixture(scope="module")
def _require_clone():
    try:
        repo.results_root()
    except repo.PriorRepositoryMissing as exc:
        pytest.skip(str(exc).splitlines()[0])


def test_fitness_inversion_is_exact_on_a_known_row(_require_clone):
    """The first row of 2013/sem run 1, taken from the published table.

    A fitness of 0.6921875 with a test score of 0.65625 must give a training score of
    0.8 exactly. If the fitness weights were ever misread, this is where it shows.
    """
    assert repo.recover_train_auc(0.6921875, 0.65625) == pytest.approx(0.8, abs=1e-12)


def test_every_recovered_training_auc_is_a_probability(_require_clone):
    """66,000 inversions, no failures.

    An inversion that produced a value outside [0, 1] would mean ``combined accuracy``
    is not the fitness, and every recovered training score would rest on a misreading.
    """
    outside = 0
    total = 0
    for evaluation in repo.search_evaluations():
        total += 1
        if not -1e-9 <= evaluation.train_auc <= 1 + 1e-9:
            outside += 1
    assert total == 66_000
    assert outside == 0


def test_the_auc_column_is_balanced_accuracy_not_roc_auc(_require_clone):
    """The signature of a thresholded metric, visible in the data alone.

    ``calc_auc`` averages the two class-wise recalls of a confusion matrix built from
    predictions already binarised by ``logit2int``. That is balanced accuracy at a single
    operating point, not an area under a ranking curve, and it is what every column named
    AUC in the prior tables holds.

    The data confirm it without reading the code. Balanced accuracy equals plain accuracy
    exactly when the label distribution is balanced, and differs otherwise. So the column
    agrees with accuracy in **100.00%** of rows on the corpora whose labels are balanced
    and in under 2% on the two that are not — a pattern a real ROC-AUC over continuous
    scores would essentially never produce.
    """
    balanced_on_test = {"2014Essay2", "2014Novel2", "2015"}
    agree: dict[str, int] = {}
    total: dict[str, int] = {}
    for evaluation in repo.search_evaluations():
        key = evaluation.setting
        total[key] = total.get(key, 0) + 1
        if abs(evaluation.test_auc - evaluation.test_accuracy) < 1e-9:
            agree[key] = agree.get(key, 0) + 1

    for setting, count in total.items():
        corpus = setting.split("/")[0]
        share = agree.get(setting, 0) / count
        if corpus in balanced_on_test:
            assert share == 1.0, f"{setting}: expected exact equality, got {share:.4f}"
        else:
            assert share < 0.05, f"{setting}: expected disagreement, got {share:.4f}"


def test_the_recovered_signal_is_identical_to_accuracy_where_labels_are_balanced(
    _require_clone,
):
    """Which limits what the recovery buys.

    It is exact, and on nine of the eleven settings it returns a number the tables
    already carried: only CLEF-PAN 2020 has an imbalanced training split. So a selection
    signal built from it is training accuracy in all but one setting, and saying
    otherwise would overstate it.
    """
    identical: dict[str, bool] = {}
    for evaluation in repo.search_evaluations():
        key = evaluation.setting
        same = abs(evaluation.train_auc - evaluation.train_accuracy) < 1e-9
        identical[key] = identical.get(key, True) and same

    always_identical = [key for key, value in identical.items() if value]
    assert len(always_identical) == 9
    assert all(not key.startswith("2020/") for key in always_identical)


def test_three_feature_strategies_are_present(_require_clone):
    """Two embedding strategies and one syntactic one, with BERT on 2013 alone."""
    found = repo.settings()
    assert ("2013", "semBert") in found
    assert len(found) == 11
    assert sum(1 for _, strategy in found if strategy == "semBert") == 1


def test_fully_trained_counts_fall_short_of_a_full_top_24_selection(_require_clone):
    """A top-24 from five runs would give 120 per setting; every setting holds fewer."""
    per_setting: dict[str, int] = {}
    for evaluation in repo.fully_trained():
        per_setting[evaluation.setting] = per_setting.get(evaluation.setting, 0) + 1
    assert sum(per_setting.values()) == 1_010
    assert len(per_setting) == 10  # semBert has no fully trained counterpart
    assert all(count < 120 for count in per_setting.values())


def test_ensembles_are_pre_sorted_by_test_auc(_require_clone):
    """Which is what makes taking the first rows of one of these a test-split selection."""
    for corpus in repo.CORPORA:
        rows = repo.ensembles(corpus)
        assert len(rows) == 252
        aucs = [row["test auc"] for row in rows]
        assert aucs == sorted(aucs, reverse=True)


def test_ensembles_offer_no_recoverable_training_auc(_require_clone):
    """No fitness column, so the ensemble stage has only training accuracy to select on.

    A property of the stored tables rather than a choice, and the one stage where the
    recovery cannot help.
    """
    columns = set(repo.ensembles("2013")[0])
    assert "combined accuracy" not in columns
    assert {"training accuracy", "test auc"} <= columns


def test_convergence_traces_are_monotone_and_end_at_the_run_best(_require_clone):
    traces = repo.convergence("2013", "sem")
    assert len(traces) == 5
    for trace in traces.values():
        fitness = [value for value, _ in trace]
        assert len(fitness) == 50
        assert fitness == sorted(fitness)


def test_runs_are_not_independent_replicates(_require_clone):
    """Two independent samples of 1,200 from a space of 24**15 would share nothing.

    They share more than half, so the runs cannot have been seeded independently — which
    is why dispersion across them is not a standard error.
    """
    by_run: dict[int, set[str]] = {}
    for evaluation in repo.search_evaluations(corpus="2013", strategy="sem"):
        by_run.setdefault(evaluation.run, set()).add(evaluation.encoding)
    assert len(by_run) == 5
    first, second = by_run[1], by_run[2]
    overlap = len(first & second) / min(len(first), len(second))
    assert overlap > 0.4


def test_the_evaluator_is_deterministic_given_the_encoding(_require_clone):
    """Shared encodings carry identical fitness, so the evaluator has no measurement noise.

    Which means any disagreement between this estimate and a fully trained result is
    systematic rather than noise.
    """
    seen: dict[str, float] = {}
    disagreements = 0
    for evaluation in repo.search_evaluations(corpus="2020", strategy="sem"):
        previous = seen.get(evaluation.encoding)
        if previous is not None and abs(previous - evaluation.fitness) > 1e-12:
            disagreements += 1
        seen[evaluation.encoding] = evaluation.fitness
    assert disagreements == 0
