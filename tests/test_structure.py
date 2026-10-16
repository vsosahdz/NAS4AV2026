"""The structural analysis, and the properties the rebuild exists to guarantee.

These test what the encoding and the attribution promise, not scikit-learn's arithmetic:
that no operation is placed on a scale with an operation from another family, that a
position's importance survives being spread across eleven correlated columns, that the
"absent" sentinel cannot be confused with a real parameter value, and that a base rate is
computed for the question a reader is actually asking.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4av.space.genotype import ALPHABET, CATEGORY_SIZES, LENGTH, category
from nas4av.structure import (
    ABSENT,
    DROPOUT_RATES,
    consecutive_run,
    encode,
    encode_many,
    family_groups,
    feature_names,
    grouped_permutation_importance,
    null_probability_of_run,
    position_groups,
    split_frequency,
)


def every_operation_at(position: int) -> list[str]:
    """One encoding per alphabet value, all differing only at ``position``."""
    return [
        ",".join(str(gene if index == position else 0) for index in range(LENGTH))
        for gene in range(ALPHABET)
    ]


class TestEncoding:
    def test_one_position_contributes_one_family_indicator(self) -> None:
        """The defect the rebuild exists to fix: families must not share an axis."""
        for encoding in every_operation_at(3):
            row = encode(encoding).reshape(LENGTH, -1)
            indicators = row[3, : len(CATEGORY_SIZES)]
            assert indicators.sum() == 1.0, "exactly one family is active"
            assert set(np.unique(indicators)) <= {0.0, 1.0}

    def test_absent_is_not_a_real_parameter_value(self) -> None:
        """``Linear(1)`` has log2-width 0, so zero cannot mean "not a Linear layer"."""
        linear_one = encode("7" + ",0" * (LENGTH - 1)).reshape(LENGTH, -1)
        identity = encode("0" * 1 + ",0" * (LENGTH - 1)).reshape(LENGTH, -1)
        width_column = len(CATEGORY_SIZES) + 3
        assert linear_one[0, width_column] == 0.0
        assert identity[0, width_column] == ABSENT
        assert ABSENT != 0.0

    def test_linear_widths_stay_ordered_within_the_family(self) -> None:
        """Ordinality is absent between families and real inside one; keep the real part."""
        width_column = len(CATEGORY_SIZES) + 3
        widths = [
            encode(f"{gene}" + ",0" * (LENGTH - 1)).reshape(LENGTH, -1)[0, width_column]
            for gene in range(7, 20)
        ]
        assert widths == sorted(widths)
        assert widths == [float(k) for k in range(13)]

    def test_dropout_rates_are_the_declared_ones(self) -> None:
        rate_column = len(CATEGORY_SIZES) + 4
        for gene, rate in DROPOUT_RATES.items():
            row = encode(f"{gene}" + ",0" * (LENGTH - 1)).reshape(LENGTH, -1)
            assert row[0, rate_column] == pytest.approx(rate)

    def test_distinct_operations_never_collide(self) -> None:
        """Twenty-four operations must produce twenty-four distinct feature rows."""
        rows = {encode(e).tobytes() for e in every_operation_at(7)}
        assert len(rows) == ALPHABET

    def test_feature_names_match_the_vector(self) -> None:
        assert len(feature_names()) == encode("0" + ",0" * (LENGTH - 1)).size


class TestGroups:
    def test_positions_partition_every_column(self) -> None:
        groups = position_groups()
        columns = [c for cols in groups.values() for c in cols]
        assert sorted(columns) == list(range(len(feature_names())))
        assert len(groups) == LENGTH

    def test_every_family_group_is_non_empty(self) -> None:
        for name, columns in family_groups().items():
            assert columns, f"{name} contributes no columns"


class TestImportance:
    def fitted(self, seed: int = 0):
        """A tree fitted on a target that depends only on position 4's family."""
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(seed)
        genes = rng.integers(0, ALPHABET, size=(600, LENGTH))
        encodings = [",".join(str(g) for g in row) for row in genes]
        target = np.array(
            [1.0 if category(int(row[4])) == "Linear" else 0.0 for row in genes]
        ) + rng.normal(0, 0.01, size=600)
        features = encode_many(encodings)
        model = RandomForestRegressor(n_estimators=24, random_state=seed, n_jobs=1)
        model.fit(features, target)
        return model, features, target

    def test_grouped_importance_finds_the_position_that_matters(self) -> None:
        """The whole reason importance is grouped.

        The signal sits at one position, which spans eleven correlated columns. Permuting
        them one at a time lets the model read the signal off the survivors, so the
        position that drives the target would report as unimportant.
        """
        model, features, target = self.fitted()
        ranked = grouped_permutation_importance(
            model, features, target, position_groups(), repeats=5
        )
        assert ranked[0].group == "position_04"
        assert ranked[0].mean > 10 * max(item.mean for item in ranked[1:])

    def test_importance_is_sorted_and_carries_its_spread(self) -> None:
        model, features, target = self.fitted()
        ranked = grouped_permutation_importance(
            model, features, target, family_groups(), repeats=5
        )
        assert [item.mean for item in ranked] == sorted(
            (item.mean for item in ranked), reverse=True
        )
        assert all(item.std >= 0 for item in ranked)
        assert all(item.columns > 0 for item in ranked)

    def test_split_frequency_reports_both_readings(self) -> None:
        """A count and a gain answer different questions; the submitted figures conflate them."""
        model, features, target = self.fitted()
        report = split_frequency(model, position_groups())
        assert report["position_04"]["gain_share"] == max(
            row["gain_share"] for row in report.values()
        )
        # Shares are reported rounded to five places, so fifteen of them sum to within
        # fifteen half-ulps of one. Asserting exact unity would be asserting against the
        # rounding rather than against the normalisation.
        tolerance = LENGTH * 0.5e-5
        assert sum(row["split_share"] for row in report.values()) == pytest.approx(
            1.0, abs=tolerance
        )
        assert sum(row["gain_share"] for row in report.values()) == pytest.approx(
            1.0, abs=tolerance
        )


class TestBaseRates:
    @pytest.mark.parametrize(
        ("family", "length"), [("Linear", 4), ("Linear", 2), ("Dropout", 2), ("Conv1D", 3)]
    )
    def test_recurrence_agrees_with_simulation(self, family: str, length: int) -> None:
        rng = np.random.default_rng(7)
        draws = rng.integers(0, ALPHABET, size=(40_000, LENGTH))
        empirical = float(np.mean([consecutive_run(row, family, length) for row in draws]))
        assert null_probability_of_run(family, length) == pytest.approx(empirical, abs=0.01)

    def test_anywhere_is_not_per_window(self) -> None:
        """The arithmetic Section IV-D's reasoning implies understates the base rate.

        ``p**4`` is the chance a *given* window of four is all-Linear. A reader comparing
        it against an observed frequency means the chance of seeing the pattern anywhere,
        and with twelve windows available those differ by more than five times.
        """
        per_window = (CATEGORY_SIZES["Linear"] / ALPHABET) ** 4
        anywhere = null_probability_of_run("Linear", 4)
        assert per_window == pytest.approx(0.0861, abs=1e-4)
        assert anywhere > 5 * per_window

    def test_a_run_longer_than_the_encoding_is_impossible(self) -> None:
        assert null_probability_of_run("Linear", LENGTH + 1) == 0.0

    def test_consecutive_run_needs_adjacency(self) -> None:
        alternating = [7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7]
        assert not consecutive_run(alternating, "Linear", 2)
        assert consecutive_run([7, 7] + [0] * 13, "Linear", 2)
