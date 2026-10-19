"""The selection pipeline, checked against the table it is meant to replace.

Task 3.2 is the load-bearing test here and it is deliberately circular-looking: the
reimplementation must reproduce the *submitted* Table 7 exactly when its signal is set to
test. That is the only way to know the stages were transcribed faithfully before anything
is concluded from running them on a different signal. A repair that quietly also changed
the pipeline would measure the repair plus the change and attribute all of it to the
repair.

The published values are transcribed from the manuscript, not computed, so they are
something to check against rather than a second copy of the same derivation.
"""

from __future__ import annotations

import pytest

from nas4av.prior.repository import PriorRepositoryMissing
from nas4av.selection import (
    REPORTED,
    TABLE7_TOP,
    UnknownSignal,
    ensemble_signal,
    evaluation_signal,
    optimism,
    rank_ensembles,
    report,
)

#: Table 7's `Avg.` rows, all twenty-five values, transcribed from the submitted
#: manuscript. Four decimals because that is what the manuscript prints, so a
#: reproduction can only ever be checked to within half of the last digit.
PUBLISHED: dict[str, dict[str, float]] = {
    "2013": {
        "test accuracy": 0.8000,
        "test precision": 0.7452,
        "test recall": 0.8810,
        "test fscore": 0.8031,
        "test auc": 0.8051,
    },
    "2014Essay2": {
        "test accuracy": 0.7433,
        "test precision": 0.7074,
        "test recall": 0.8300,
        "test fscore": 0.7638,
        "test auc": 0.7433,
    },
    "2014Novel2": {
        "test accuracy": 0.7733,
        "test precision": 0.7580,
        "test recall": 0.8067,
        "test fscore": 0.7804,
        "test auc": 0.7733,
    },
    "2015": {
        "test accuracy": 0.6927,
        "test precision": 0.6927,
        "test recall": 0.6960,
        "test fscore": 0.6934,
        "test auc": 0.6927,
    },
    "2020": {
        "test accuracy": 0.9000,
        "test precision": 0.8154,
        "test recall": 0.9722,
        "test fscore": 0.8859,
        "test auc": 0.9120,
    },
}


def available(corpus: str) -> bool:
    try:
        report(corpus)
    except (PriorRepositoryMissing, FileNotFoundError):
        return False
    return True


needs_repository = pytest.mark.skipif(
    not available("2013"), reason="the reference clone is not present"
)


class TestSignals:
    def test_an_unknown_signal_is_refused_rather_than_defaulted(self) -> None:
        """A silent fallback would make the optimism gap compare two different things."""
        from nas4av.prior.repository import Evaluation

        evaluation = Evaluation(
            encoding="0" + ",0" * 14,
            corpus="2013",
            strategy="sem",
            run=1,
            test_auc=0.6,
            train_auc=0.7,
            fitness=0.625,
            train_accuracy=0.8,
            test_accuracy=0.6,
            seconds=1.0,
        )
        assert evaluation_signal(evaluation, "test") == 0.6
        assert evaluation_signal(evaluation, "train") == 0.7
        with pytest.raises(UnknownSignal):
            evaluation_signal(evaluation, "validation")

    def test_the_ensemble_stage_refuses_a_signal_it_cannot_compute(self) -> None:
        """`ensemblesDF` has no fitness column, so recovered training AUC does not exist."""
        row = {"models": "0,1,2,3,4", "test auc": 0.7, "training accuracy": 0.9}
        assert ensemble_signal(row, "test") == 0.7
        assert ensemble_signal(row, "train") == 0.9
        with pytest.raises(UnknownSignal):
            ensemble_signal(row, "fitness")

    def test_ranking_is_by_the_named_signal_not_by_stored_order(self) -> None:
        """The stored tables arrive pre-sorted by test score; a slice is a test selection."""
        rows = [
            {"models": "a", "test auc": 0.9, "training accuracy": 0.1},
            {"models": "b", "test auc": 0.8, "training accuracy": 0.9},
            {"models": "c", "test auc": 0.7, "training accuracy": 0.5},
        ]
        assert [r["models"] for r in rank_ensembles(rows, "test", k=2)] == ["a", "b"]
        assert [r["models"] for r in rank_ensembles(rows, "train", k=2)] == ["b", "c"]


@needs_repository
class TestTable7:
    @pytest.mark.parametrize("corpus", sorted(PUBLISHED))
    def test_the_reimplementation_reproduces_the_submitted_table(self, corpus: str) -> None:
        """Task 3.2. Four decimals, which is the precision the manuscript prints."""
        produced = report(corpus, signal="test")
        for column in REPORTED:
            assert produced.average[column] == pytest.approx(
                PUBLISHED[corpus][column], abs=5e-5
            ), f"{corpus} {column}"

    def test_it_averages_three_ensembles(self) -> None:
        produced = report("2013", signal="test")
        assert len(produced.top_ensembles) == TABLE7_TOP
        assert produced.pool == 252

    def test_the_spread_is_labelled_as_what_it_is(self) -> None:
        """Three selected values, not replicates; the output has to say so."""
        payload = report("2013", signal="test").as_dict()
        assert "not a dispersion over replicates" in payload["spread_note"]
        assert "standard error" in payload["spread_note"]


@needs_repository
class TestOptimism:
    def test_selecting_without_test_does_not_beat_selecting_on_it(self) -> None:
        """The direction the gap must have, if it measures what it claims to.

        Ranking on the reported split cannot do worse on that split than ranking on
        anything else. A negative gap would mean the test-ranked arm lost on its own
        criterion, which would be a bug in the pipeline rather than a finding.
        """
        measured = optimism("2013")
        assert measured["gap"]["test auc"] >= 0

    def test_the_ensemble_stage_records_its_degradation(self) -> None:
        measured = optimism("2013")
        clean = measured["selected_without_test"]
        assert clean["degraded_at_ensemble_stage"] is True
        assert clean["ensemble_signal_used"] == "training_accuracy"

    def test_the_result_is_declared_a_lower_bound(self) -> None:
        """Stages 2-4 had already selected on test, so this is not the whole optimism."""
        assert "lower bound" in optimism("2013")["bound"]
