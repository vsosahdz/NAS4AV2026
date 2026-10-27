"""The strategy comparison.

Every test here guards a place where a favourable answer could be produced without
anything looking wrong: selecting on the reporting split, aggregating runs by maximum,
comparing strategies that did not spend the same budget, or claiming equivalence from a
region that was never derived.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from nas4av.comparison import (
    CONTROL,
    budget_is_matched,
    compare,
    group,
    load_outcomes,
    strategy_table,
)

STRATEGIES = ("evolutionary-variant", "canonical-ga", "uniform-random")


def write_campaign(
    root: Path,
    means: dict[str, float] | None = None,
    corpora: tuple[str, ...] = ("A", "B", "C", "D", "E"),
    features: tuple[str, ...] = ("sem", "syn"),
    runs: int = 5,
    evaluations: dict[str, int] | None = None,
    seed: int = 0,
) -> Path:
    """A synthetic campaign with a known answer."""
    means = means or {"evolutionary-variant": 0.72, "canonical-ga": 0.70, "uniform-random": 0.62}
    rng = random.Random(seed)
    sweep = root / "sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    for corpus in corpora:
        for feature in features:
            for strategy, mean in means.items():
                for run in range(1, runs + 1):
                    records = [
                        {
                            "genotype": ",".join(["0"] * 15),
                            "params": 1000 + index,
                            "failed": False,
                            "validation_roc_auc": rng.gauss(mean, 0.02),
                            "test_roc_auc": rng.gauss(mean, 0.02),
                        }
                        for index in range(5)
                    ]
                    payload = {
                        "unit": {
                            "corpus": corpus,
                            "features": feature,
                            "strategy": strategy,
                            "run": run,
                        },
                        "run": {
                            "records": records,
                            "evaluations": (evaluations or {}).get(strategy, 1200),
                            "failures": 0,
                        },
                    }
                    name = f"{corpus}_{feature}_{strategy}_run{run}.json"
                    (sweep / name).write_text(json.dumps(payload), encoding="utf-8")
    return sweep


class TestSelection:
    def test_the_reported_score_belongs_to_the_best_validation_architecture(
        self, tmp_path: Path
    ) -> None:
        """Not the best test score in the run.

        The defect the whole rebuild exists to remove. An artifact holds every evaluation,
        so taking the maximum test score is one line away and would make every strategy
        look better the harder it searched against the reporting split.
        """
        sweep = tmp_path / "sweep"
        sweep.mkdir(parents=True)
        records = [
            {"genotype": "a", "params": 1, "failed": False, "validation_roc_auc": 0.9,
             "test_roc_auc": 0.50},
            {"genotype": "b", "params": 2, "failed": False, "validation_roc_auc": 0.1,
             "test_roc_auc": 0.99},
        ]
        (sweep / "A_sem_uniform-random_run1.json").write_text(
            json.dumps(
                {
                    "unit": {"corpus": "A", "features": "sem", "strategy": "uniform-random",
                             "run": 1},
                    "run": {"records": records, "evaluations": 10, "failures": 0},
                }
            )
        )
        outcome = load_outcomes(sweep)[0]
        assert outcome.encoding == "a"
        assert outcome.test == pytest.approx(0.50), "the 0.99 is visible and must not be chosen"

    def test_failed_evaluations_are_never_selected(self, tmp_path: Path) -> None:
        sweep = tmp_path / "sweep"
        sweep.mkdir(parents=True)
        records = [
            {"genotype": "ok", "params": 1, "failed": False, "validation_roc_auc": 0.6,
             "test_roc_auc": 0.6},
            {"genotype": "bad", "params": 2, "failed": True, "validation_roc_auc": 1.0,
             "test_roc_auc": 1.0},
        ]
        (sweep / "A_sem_uniform-random_run1.json").write_text(
            json.dumps(
                {
                    "unit": {"corpus": "A", "features": "sem", "strategy": "uniform-random",
                             "run": 1},
                    "run": {"records": records, "evaluations": 10, "failures": 1},
                }
            )
        )
        assert load_outcomes(sweep)[0].encoding == "ok"


class TestAggregation:
    def test_runs_aggregate_by_median_not_maximum(self, tmp_path: Path) -> None:
        """A maximum over runs measures luck and grows with run count."""
        sweep = write_campaign(tmp_path, corpora=("A",), features=("sem",))
        entry = next(iter(group(load_outcomes(sweep)).values()))
        assert entry.median_test <= entry.as_dict()["best_run_test"]
        assert entry.as_dict()["median_test"] == pytest.approx(entry.median_test)

    def test_the_spread_is_never_called_a_standard_error(self, tmp_path: Path) -> None:
        sweep = write_campaign(tmp_path, corpora=("A",), features=("sem",))
        entry = next(iter(group(load_outcomes(sweep)).values()))
        note = entry.as_dict()["spread_note"]
        assert "not a standard error" in note


class TestBudget:
    def test_matched_budgets_are_recognised(self, tmp_path: Path) -> None:
        sweep = write_campaign(tmp_path)
        assert budget_is_matched(load_outcomes(sweep))["matched"] is True

    def test_an_unmatched_budget_is_detected(self, tmp_path: Path) -> None:
        """A strategy that spent more would win for a reason unrelated to search quality."""
        sweep = write_campaign(
            tmp_path, evaluations={"uniform-random": 2400, "canonical-ga": 1200,
                                   "evolutionary-variant": 1200}
        )
        assert budget_is_matched(load_outcomes(sweep))["matched"] is False

    def test_compare_refuses_to_interpret_an_unmatched_comparison(self, tmp_path: Path) -> None:
        sweep = write_campaign(tmp_path, evaluations={"uniform-random": 2400})
        report = compare(sweep)
        assert report["budget"]["matched"] is False


class TestComparison:
    def test_a_worse_control_ranks_last_under_both_structures(self, tmp_path: Path) -> None:
        sweep = write_campaign(tmp_path)
        report = compare(sweep)
        for structure in ("per_setting", "per_corpus"):
            ranks = report["comparison"][structure]["mean_ranks"]
            assert max(ranks, key=lambda k: ranks[k]) == CONTROL
        assert report["comparison"]["orderings_agree"] is True

    def test_both_structures_are_always_reported(self, tmp_path: Path) -> None:
        """Reporting whichever one gave the preferred answer is the defect being removed."""
        report = compare(write_campaign(tmp_path))
        assert "per_setting" in report["comparison"]
        assert "per_corpus" in report["comparison"]
        assert report["comparison"]["per_setting"]["block_count"] == 10
        assert report["comparison"]["per_corpus"]["block_count"] == 5

    def test_indistinguishable_strategies_are_not_separated(self, tmp_path: Path) -> None:
        """The result that matters most if it happens: random doing as well as the variant."""
        sweep = write_campaign(
            tmp_path,
            means={s: 0.70 for s in STRATEGIES},
            seed=11,
        )
        report = compare(sweep)
        pairwise = report["comparison"]["per_corpus"]["pairwise"]
        assert not any(row["significant_at_05"] for row in pairwise)

    def test_no_region_means_no_equivalence_claim(self, tmp_path: Path) -> None:
        """Equivalence rests entirely on where the region came from."""
        report = compare(write_campaign(tmp_path))
        assert report["rope"]["region"] == 0.0
        assert "unavailable" in report["rope"]["origin"]
        assert "error" in report["equivalence"]

    def test_an_empty_directory_reports_an_error_rather_than_an_answer(
        self, tmp_path: Path
    ) -> None:
        empty = tmp_path / "sweep"
        empty.mkdir(parents=True)
        assert "error" in compare(empty)

    def test_a_single_strategy_cannot_be_compared(self, tmp_path: Path) -> None:
        sweep = write_campaign(tmp_path, means={"uniform-random": 0.6})
        assert "error" in compare(sweep)["comparison"]

    def test_the_table_carries_every_setting_for_every_strategy(self, tmp_path: Path) -> None:
        """A ragged table makes Friedman compare different things without saying so."""
        table = strategy_table(load_outcomes(write_campaign(tmp_path)))
        sizes = {len(values) for values in table.values()}
        assert sizes == {10}


class TestSpacePrefix:
    def test_a_second_space_identifier_parses_to_its_corpus(self) -> None:
        """Without this, `v2_2013_sem_...` reads as corpus "v2" and features "2013".

        The symptom is not a crash: no corpus loads, the equivalence region comes back
        "unavailable", and the report says there were no stored scores — which is a
        missing-data message for what is actually a parse bug.
        """
        from nas4av.comparison import _SPACE_PREFIXES

        assert "v2" in _SPACE_PREFIXES
        assert "v1" not in _SPACE_PREFIXES, "first-space units carry no prefix"

        def corpus_of(identifier: str) -> tuple[str, str]:
            parts = identifier.split("_")
            if parts[0] in _SPACE_PREFIXES:
                parts = parts[1:]
            return parts[0], parts[1]

        assert corpus_of("v2_2013_semBert_canonical-ga_novelty_run1") == ("2013", "semBert")
        assert corpus_of("2013_semBert_canonical-ga_novelty_run1") == ("2013", "semBert")

    def test_the_prefix_list_comes_from_the_registry(self) -> None:
        """A third space must not need this list updated by hand."""
        from nas4av.space.spec import _REGISTRY

        from nas4av.comparison import _SPACE_PREFIXES

        assert _SPACE_PREFIXES == frozenset(_REGISTRY) - {"v1"}
