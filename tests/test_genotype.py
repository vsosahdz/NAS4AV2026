"""The encoding and the two constraints.

Two of these tests pin the *position-counting* semantics of the constraints. A
set-theoretic phrasing of the same rule — the cardinality of the set of matching gene
values — can only ever reach two, so a limit of three written that way would never bind.
The tests exist so that a future reader cannot quietly reintroduce that reading.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4av.space.genotype import (
    ALPHABET,
    CATEGORY_SIZES,
    LENGTH,
    MalformedEncoding,
    acceptance_rate,
    categories,
    category,
    feasible,
    linear_width,
    parse,
    render,
    sample_feasible,
    satisfies_g1,
    satisfies_g2,
)


def test_category_sizes_are_the_base_rates_the_null_model_uses():
    """Linear occupies 13 of 24 values, which is why four consecutive Linear layers are
    expected rather than discovered.

    Any claim that a layer pattern characterises good architectures has to clear these
    rates first, so the arithmetic behind them is worth pinning.
    """
    assert CATEGORY_SIZES == {
        "Identity": 1,
        "Dropout": 3,
        "Activation": 3,
        "Linear": 13,
        "Conv1D": 4,
    }
    assert sum(CATEGORY_SIZES.values()) == ALPHABET
    assert CATEGORY_SIZES["Linear"] / ALPHABET == pytest.approx(0.5417, abs=1e-4)


def test_linear_widths_are_powers_of_two_from_one_to_4096():
    """Powers of two from 2**0 to 2**12, derived rather than tabulated."""
    assert [linear_width(g) for g in range(7, 20)] == [2**k for k in range(13)]
    assert linear_width(7) == 1
    assert linear_width(19) == 4096


def test_category_rejects_values_outside_the_alphabet():
    with pytest.raises(MalformedEncoding):
        category(ALPHABET)
    with pytest.raises(MalformedEncoding):
        category(-1)


def test_parse_accepts_the_stored_string_form():
    stored = "16,10,16,9,2,17,19,12,11,8,23,10,17,5,11"
    assert parse(stored) == (16, 10, 16, 9, 2, 17, 19, 12, 11, 8, 23, 10, 17, 5, 11)
    assert render(parse(stored)) == stored


def test_parse_rejects_the_wrong_length():
    with pytest.raises(MalformedEncoding):
        parse([0] * (LENGTH - 1))


def test_g1_counts_positions_not_values():
    """Four eight-channel Conv1D layers violate g1, however few distinct values they use.

    Under a set-of-values reading this encoding would be feasible, because ``{21}`` has
    one element. That is the reading the constraint must not have.
    """
    genes = [21, 0, 21, 0, 21, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0]
    assert sum(1 for g in genes if g in (21, 23)) == 4
    assert not satisfies_g1(genes)
    assert satisfies_g1([21, 0, 21, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_g1_counts_both_eight_channel_operations_together():
    """21 and 23 share one budget of three, not one each."""
    assert not satisfies_g1([21, 0, 21, 0, 23, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0])
    assert satisfies_g1([21, 0, 23, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_g1_ignores_two_channel_conv_operations():
    """20 and 22 have two channels and are not counted; only their neighbours matter."""
    assert satisfies_g1([20, 0, 22, 0, 20, 0, 22, 0, 20, 0, 22, 0, 20, 0, 22])


def test_g2_forbids_a_wide_linear_beside_a_conv():
    """Values above 17 are Linear(2048), Linear(4096) and the Conv1D operations."""
    assert not satisfies_g2([18, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert not satisfies_g2([0, 20, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert satisfies_g2([17, 20, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_g2_forbids_adjacent_conv_operations():
    assert not satisfies_g2([0, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_g2_ignores_missing_neighbours_at_the_ends():
    """A Conv1D at position 1 has no predecessor, and at position 15 no successor.

    An implementation that indexed ``genes[-1]`` for the missing neighbour would silently
    wrap to the last gene and reject legal encodings.
    """
    assert satisfies_g2([20, 0] + [0] * 13)
    assert satisfies_g2([0] * 13 + [0, 20])
    assert not satisfies_g2([20, 18] + [0] * 13)
    assert not satisfies_g2([0] * 13 + [18, 20])


def test_feasible_requires_both_constraints():
    both_ok = [17, 20, 17, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert feasible(both_ok)
    assert not feasible([21, 0, 21, 0, 21, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0])  # g1
    assert not feasible([18, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # g2


def test_sampler_only_yields_feasible_encodings():
    rng = np.random.default_rng(20260904)
    drawn = list(sample_feasible(rng, 500))
    assert len(drawn) == 500
    assert all(feasible(genes) for genes in drawn)
    assert all(len(genes) == LENGTH for genes in drawn)


def test_sampler_is_reproducible_from_its_seed():
    """Same seed, same sample. A null model is a reported number and has to behave as one."""
    first = list(sample_feasible(np.random.default_rng(7), 50))
    second = list(sample_feasible(np.random.default_rng(7), 50))
    assert first == second


def test_acceptance_rate_is_about_one_half():
    """Measured, and reported rather than derived; the constraints interact.

    The value matters because it is the difference between sampling the alphabet and
    sampling the feasible space, and a pattern null model needs the latter.
    """
    rate = acceptance_rate(np.random.default_rng(20260904), trials=20_000)
    assert 0.45 < rate < 0.55


def test_categories_maps_position_by_position():
    genes = [0, 1, 4, 7, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert categories(genes)[:5] == ("Identity", "Dropout", "Activation", "Linear", "Conv1D")
