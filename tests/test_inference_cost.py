"""What a model costs to run.

The efficiency claim is the second axis of the reframed contribution, so these guard the
properties that claim rests on: that FLOPs are not a restatement of the parameter count,
that the three quantities are kept distinct, and that a one-time search cost is never
reported as a per-decision one.
"""

from __future__ import annotations

import pytest

from nas4av.inference_cost import (
    REFERENCE_MODELS,
    against_references,
    cost_for_setting,
    cost_of,
    forward_flops,
    separate_costs,
)
from nas4av.space.genotype import LENGTH

WIDTH = 600


def only(gene: int, length: int = LENGTH) -> str:
    """An encoding of one operation followed by identities."""
    return ",".join([str(gene)] + ["0"] * (length - 1))


class TestFlops:
    def test_identity_and_dropout_cost_nothing_at_inference(self) -> None:
        """Dropout is identity once eval() is called; counting it would inflate every model."""
        assert forward_flops(only(0), WIDTH) == 0
        for dropout in (1, 2, 3):
            assert forward_flops(only(dropout), WIDTH) == 0

    def test_a_linear_layer_is_two_flops_per_weight(self) -> None:
        """One multiply and one add per weight, each weight used exactly once."""
        # gene 10 is Linear(2**3) = Linear(8)
        assert forward_flops(only(10), WIDTH) == 2 * WIDTH * 8

    def test_an_activation_costs_one_per_element_not_two(self) -> None:
        for activation in (4, 5, 6):
            assert forward_flops(only(activation), WIDTH) == WIDTH

    def test_flops_are_not_a_restatement_of_the_parameter_count(self) -> None:
        """The whole reason this module exists beside `space.cost`.

        A Conv1D reuses each weight across every position of its input, so it can be tiny
        and expensive; a Linear uses each weight once and sits at 2.0 by construction. If
        both families gave the same ratio, parameter count would already answer the
        efficiency question and FLOPs would be decoration.
        """
        linear = cost_of(only(19), WIDTH)
        wide_conv = cost_of(only(23), WIDTH)
        assert linear.flops_per_parameter == pytest.approx(2.0, abs=0.1)
        assert wide_conv.flops_per_parameter > 2 * linear.flops_per_parameter

    def test_a_smaller_model_can_cost_more_arithmetic_per_weight(self) -> None:
        """"Small" and "cheap" are different claims, and the manuscript conflates them."""
        big_linear = cost_of(only(19), WIDTH)
        small_conv = cost_of(only(23), WIDTH)
        assert small_conv.params < big_linear.params
        assert small_conv.flops_per_parameter > big_linear.flops_per_parameter

    def test_flops_grow_with_the_input_width(self) -> None:
        narrow = forward_flops(only(10), 100)
        wide = forward_flops(only(10), 1000)
        assert wide == 10 * narrow

    def test_the_input_width_is_looked_up_per_setting(self) -> None:
        """An off-by-2x in the width changes every FLOP figure, so it is not passed by hand."""
        encoding = only(10)
        for corpus, strategy in (("2013", "sem"), ("2015", "synBetter")):
            measured = cost_for_setting(encoding, corpus, strategy)
            assert measured.input_shape % 2 == 0, "a pair is a concatenation of two documents"
            assert measured.flops == forward_flops(encoding, measured.input_shape)


class TestReferenceRatios:
    def test_the_weakest_comparison_is_the_one_reported(self) -> None:
        """The conclusion says "far fewer resources" with no number.

        A claim like that has to survive its least flattering comparison, not its most, so
        the smallest reference is singled out rather than left for a reader to find.
        """
        report = against_references(2_000_000)
        assert report["weakest_comparison"]["model"] == min(
            REFERENCE_MODELS, key=lambda k: REFERENCE_MODELS[k]
        )
        assert report["weakest_comparison"]["ratio"] <= min(
            r for r in report["ratios"].values() if r is not None
        )

    def test_it_states_that_a_size_ratio_is_not_a_capability_ratio(self) -> None:
        assert "capability" in against_references(1_000_000)["caveat"]

    def test_a_zero_parameter_model_does_not_divide_by_zero(self) -> None:
        report = against_references(0)
        assert all(value is None for value in report["ratios"].values())


class TestCostSeparation:
    def test_search_and_inference_are_reported_separately(self) -> None:
        """Mixing them makes a 20-hour search look like a property of the model."""
        report = separate_costs(search_hours=20.37, evaluations=66_000, inference_ms=0.33)
        assert "once per corpus" in report["one_time_search"]["paid_by"]
        assert "every decision" in report["per_decision_inference"]["paid_by"]

    def test_the_crossover_is_reported_in_decisions(self) -> None:
        """How much deployment it takes for inference to matter as much as the search."""
        report = separate_costs(search_hours=1.0, evaluations=1000, inference_ms=1.0)
        # One hour of search against one millisecond per decision: 3.6 million decisions.
        assert report["decisions_to_equal_the_search"] == 3_600_000

    def test_zero_latency_does_not_divide_by_zero(self) -> None:
        report = separate_costs(search_hours=1.0, evaluations=10, inference_ms=0.0)
        assert report["decisions_to_equal_the_search"] is None

    def test_seconds_per_evaluation_is_derived_not_assumed(self) -> None:
        report = separate_costs(search_hours=2.0, evaluations=7200, inference_ms=1.0)
        assert report["one_time_search"]["seconds_per_evaluation"] == pytest.approx(1.0)
