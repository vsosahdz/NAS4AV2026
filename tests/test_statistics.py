"""The statistical protocol.

These test the guarantees, not the arithmetic of scipy: that a ragged comparison is
refused rather than silently computed over whatever overlaps, that a correction is applied
and is monotone, that an underpowered comparison says so, and that overlapping intervals
are never reported as equivalence.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4av.statistics import (
    MINIMUM_BLOCKS,
    ComparisonStructureError,
    bootstrap_interval,
    describe_difference,
    friedman_with_holm,
    holm,
)


def separable(n: int = 60, seed: int = 1) -> tuple[list[float], list[int]]:
    rng = np.random.default_rng(seed)
    labels = [0] * (n // 2) + [1] * (n // 2)
    scores = [float(rng.normal(-1 if label == 0 else 1)) for label in labels]
    return scores, labels


# ── intervals ──────────────────────────────────────────────────────────────────


def test_an_interval_brackets_its_estimate():
    scores, labels = separable()
    interval = bootstrap_interval(scores, labels, seed=7)
    assert interval.low <= interval.estimate <= interval.high
    assert interval.width > 0


def test_a_small_evaluation_set_produces_a_wide_interval():
    """Thirty pairs cannot resolve the differences the submitted manuscript claims."""
    wide = bootstrap_interval(*separable(n=30, seed=2), seed=7)
    narrow = bootstrap_interval(*separable(n=600, seed=2), seed=7)
    assert wide.width > narrow.width
    assert wide.width > 0.1


def test_an_interval_reports_whether_it_covers_chance():
    """A result whose interval contains 0.5 has not distinguished itself from chance."""
    rng = np.random.default_rng(3)
    labels = [0, 1] * 30
    noise = [float(rng.normal()) for _ in labels]
    assert bootstrap_interval(noise, labels, seed=7).covers(0.5)
    assert not bootstrap_interval(*separable(n=400, seed=4), seed=7).covers(0.5)


def test_the_resample_is_stratified_by_class():
    """An unstratified resample of a small set produces single-class draws.

    Both metrics are undefined there, so an unstratified interval would be computed from
    whichever resamples happened to survive — a different quantity under the same name.
    Stratification makes every resample usable, which shows up as no NaN in the interval.
    """
    interval = bootstrap_interval(*separable(n=20, seed=5), resamples=2000, seed=7)
    assert not np.isnan(interval.low)
    assert not np.isnan(interval.high)


def test_an_interval_is_reproducible_from_its_seed():
    scores, labels = separable()
    first = bootstrap_interval(scores, labels, seed=11, resamples=500)
    second = bootstrap_interval(scores, labels, seed=11, resamples=500)
    assert (first.low, first.high) == (second.low, second.high)


def test_a_single_class_evaluation_set_yields_no_interval():
    interval = bootstrap_interval([0.1, 0.5, 0.9], [1, 1, 1], seed=7)
    assert np.isnan(interval.low)
    assert interval.resamples == 0


def test_balanced_accuracy_is_available_as_the_legacy_metric():
    """Kept so the new campaign stays comparable with the prior one."""
    scores, labels = separable()
    interval = bootstrap_interval(scores, labels, metric="balanced_accuracy", seed=7)
    assert 0.0 <= interval.estimate <= 1.0


def test_an_unknown_metric_is_refused():
    with pytest.raises(ValueError, match="metric"):
        bootstrap_interval([0.1], [1], metric="f1")


def test_mismatched_scores_and_labels_are_refused():
    with pytest.raises(ValueError, match="same evaluation units"):
        bootstrap_interval([0.1, 0.2], [1])


# ── the correction ─────────────────────────────────────────────────────────────


def test_holm_is_monotone_and_never_exceeds_one():
    adjusted = holm([0.001, 0.01, 0.04, 0.2, 0.9])
    assert adjusted == sorted(adjusted)
    assert max(adjusted) <= 1.0


def test_holm_is_less_conservative_than_bonferroni():
    """Which is why it is the one used: same error rate, more power."""
    raw = [0.01, 0.02, 0.03]
    assert holm(raw)[-1] < len(raw) * raw[-1]


def test_holm_preserves_input_order():
    raw = [0.5, 0.001, 0.2]
    adjusted = holm(raw)
    assert adjusted[1] == min(adjusted)


# ── comparisons ────────────────────────────────────────────────────────────────


def table(**treatments: dict[str, float]) -> dict[str, dict[str, float]]:
    return dict(treatments)


def test_a_ragged_comparison_is_refused():
    """A Friedman test over ragged blocks silently compares different things."""
    with pytest.raises(ComparisonStructureError, match="ragged"):
        friedman_with_holm(
            table(
                a={"c1": 0.7, "c2": 0.6, "c3": 0.5},
                b={"c1": 0.6, "c2": 0.5},
            ),
            metric="roc_auc",
        )


def test_treatments_with_no_shared_block_are_refused():
    with pytest.raises(ComparisonStructureError, match="share no block"):
        friedman_with_holm(table(a={"c1": 0.7}, b={"c2": 0.6}), metric="roc_auc")


def test_a_single_treatment_is_refused():
    with pytest.raises(ComparisonStructureError, match="at least two"):
        friedman_with_holm(table(a={"c1": 0.7, "c2": 0.6}), metric="roc_auc")


def test_fewer_than_five_blocks_is_flagged_underpowered():
    """Five corpora is exactly the floor, so whether settings or corpora are the blocks
    decides whether any omnibus test in this study is usable."""
    blocks = {f"c{i}": 0.5 + 0.01 * i for i in range(3)}
    comparison = friedman_with_holm(
        table(
            a=blocks,
            b={k: v - 0.02 for k, v in blocks.items()},
            c={k: v - 0.04 for k, v in blocks.items()},
        ),
        metric="roc_auc",
    )
    assert len(comparison.blocks) < MINIMUM_BLOCKS
    assert comparison.underpowered


def test_a_clear_winner_takes_the_best_mean_rank():
    blocks = [f"c{i}" for i in range(6)]
    comparison = friedman_with_holm(
        table(
            strong={b: 0.80 + 0.001 * i for i, b in enumerate(blocks)},
            middle={b: 0.70 + 0.001 * i for i, b in enumerate(blocks)},
            weak={b: 0.60 + 0.001 * i for i, b in enumerate(blocks)},
        ),
        metric="roc_auc",
    )
    assert comparison.mean_ranks["strong"] < comparison.mean_ranks["weak"]
    assert comparison.omnibus_p < 0.05
    assert not comparison.underpowered


def test_every_pairwise_result_carries_a_corrected_p():
    blocks = [f"c{i}" for i in range(6)]
    comparison = friedman_with_holm(
        table(
            a={b: 0.8 for b in blocks},
            b={b: 0.7 for b in blocks},
            c={b: 0.6 for b in blocks},
        ),
        metric="roc_auc",
    )
    assert len(comparison.pairwise) == 3
    for record in comparison.pairwise:
        assert "p_holm" in record
        assert record["p_holm"] >= record["p_raw"]


def test_a_control_shrinks_the_comparison_family():
    """Comparing against one control costs less power than all pairs."""
    blocks = [f"c{i}" for i in range(6)]
    data = table(
        random={b: 0.60 for b in blocks},
        variant={b: 0.70 for b in blocks},
        canonical={b: 0.65 for b in blocks},
    )
    all_pairs = friedman_with_holm(data, metric="roc_auc")
    against = friedman_with_holm(data, metric="roc_auc", control="random")
    assert len(all_pairs.pairwise) == 3
    assert len(against.pairwise) == 2
    assert against.control == "random"


def test_an_unknown_control_is_refused():
    blocks = {f"c{i}": 0.5 for i in range(6)}
    with pytest.raises(ComparisonStructureError, match="control"):
        friedman_with_holm(table(a=blocks, b=blocks), metric="roc_auc", control="nope")


def test_identical_treatments_give_a_p_of_one_rather_than_an_error():
    """A coarse metric produces ties, and Wilcoxon refuses an all-zero difference."""
    blocks = {f"c{i}": 0.5 for i in range(6)}
    comparison = friedman_with_holm(
        table(a=dict(blocks), b=dict(blocks), c=dict(blocks)), metric="roc_auc"
    )
    assert comparison.pairwise[0]["p_raw"] == 1.0


def test_two_treatments_skip_the_omnibus():
    """An omnibus over two is the paired test again, reported twice under two names."""
    blocks = [f"c{i}" for i in range(6)]
    comparison = friedman_with_holm(
        table(a={b: 0.8 for b in blocks}, b={b: 0.6 for b in blocks}), metric="roc_auc"
    )
    assert comparison.omnibus_p is None
    assert len(comparison.pairwise) == 1
    assert comparison.as_dict()["omnibus"]["test"] is None


# ── what overlapping intervals do and do not show ──────────────────────────────


def test_separated_intervals_indicate_a_difference():
    strong = bootstrap_interval(*separable(n=400, seed=1), seed=7)
    weak = bootstrap_interval([0.0] * 200 + [0.0] * 200, [0] * 200 + [1] * 200, seed=7)
    result = describe_difference(strong, weak)
    assert result["intervals_separate"]


def test_overlapping_intervals_are_never_reported_as_equivalence():
    """The inference a reader is most likely to make unprompted, refused in the output."""
    first = bootstrap_interval(*separable(n=40, seed=1), seed=7)
    second = bootstrap_interval(*separable(n=40, seed=2), seed=7)
    result = describe_difference(first, second)
    assert not result["intervals_separate"]
    assert "do not indicate" in result["note"]
