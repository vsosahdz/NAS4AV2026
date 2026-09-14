"""The split and the randomness isolation.

The isolation tests matter more than they look. The defect they guard against is not
hypothetical: the prior training loop reseeded Python's global generator on every batch,
the evolutionary operators drew from that same generator, and the result was five search
runs sharing most of their architectures. A test that merely checks the context manager
returns cleanly would not have caught it.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from nas4av.protocol import (
    MINIMUM_VALIDATION_PAIRS,
    ProtocolSplit,
    SplitLeakError,
    isolated_randomness,
    run_seeds,
    seed_everything,
    stratified_holdout,
    three_way_split,
)


def test_holdout_preserves_the_label_balance():
    """Balanced accuracy is undefined in one term if a split holds a single class."""
    labels = [0] * 50 + [1] * 50
    _, held = stratified_holdout(labels, 0.25, seed=1)
    held_labels = [labels[index] for index in held]
    assert set(held_labels) == {0, 1}
    assert abs(held_labels.count(0) - held_labels.count(1)) <= 1


def test_holdout_partitions_without_overlap_or_loss():
    labels = [0] * 30 + [1] * 30
    kept, held = stratified_holdout(labels, 0.3, seed=7)
    assert set(kept) | set(held) == set(range(len(labels)))
    assert not set(kept) & set(held)


def test_holdout_never_consumes_a_class_entirely():
    """A fraction near one must still leave something to train on."""
    labels = [0] * 4 + [1] * 4
    kept, held = stratified_holdout(labels, 0.95, seed=3)
    assert kept, "the training side was emptied"
    assert {labels[i] for i in kept} == {0, 1}


def test_holdout_is_reproducible_from_its_seed():
    labels = [0, 1] * 40
    assert stratified_holdout(labels, 0.25, seed=11) == stratified_holdout(labels, 0.25, seed=11)
    assert stratified_holdout(labels, 0.25, seed=11) != stratified_holdout(labels, 0.25, seed=12)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_holdout_rejects_degenerate_fractions(fraction):
    with pytest.raises(ValueError):
        stratified_holdout([0, 1, 0, 1], fraction, seed=1)


def test_a_corpus_too_small_to_select_on_says_so():
    """CLEF-PAN 2013 has ten training pairs in total, so no split of it clears the floor.

    Reported rather than worked around: a validation split of two or three pairs would
    produce a selection signal with fewer distinct values than the population has
    candidates.
    """
    split = three_way_split("2013", [0] * 5 + [1] * 5, [0] * 15 + [1] * 15, seed=1)
    assert len(split.train) + len(split.validation) == 10
    assert not split.supports_clean_selection


def test_a_larger_corpus_supports_clean_selection():
    split = three_way_split("2015", [0] * 50 + [1] * 50, [0] * 500 + [1] * 500, seed=1)
    assert len(split.validation) >= MINIMUM_VALIDATION_PAIRS
    assert split.supports_clean_selection


def test_the_test_split_is_not_handed_out_silently():
    split = three_way_split("2015", [0] * 50 + [1] * 50, [0] * 20 + [1] * 20, seed=1)
    assert not split.test_was_released
    with pytest.raises(SplitLeakError):
        split.release_test("   ")
    assert not split.test_was_released

    released = split.release_test("final reporting, once")
    assert len(released) == 40
    assert split.test_was_released
    assert split.as_dict()["test_released"] is True


def test_split_summary_records_what_a_reader_needs():
    split = three_way_split("2020", [0] * 35 + [1] * 35, [0] * 15 + [1] * 15, seed=9)
    summary = split.as_dict()
    assert summary["corpus"] == "2020"
    assert summary["seed"] == 9
    assert summary["train_pairs"] + summary["validation_pairs"] == 70
    assert summary["test_pairs"] == 30


# ── randomness isolation ───────────────────────────────────────────────────────


def test_isolation_restores_the_python_generator():
    """The concrete defect: a callee reseeding the generator its caller draws from."""
    random.seed(1234)
    expected = [random.random() for _ in range(5)]

    random.seed(1234)
    with isolated_randomness():
        random.seed(20)  # what the prior training loop does on every batch
        [random.random() for _ in range(50)]
    assert [random.random() for _ in range(5)] == expected


def test_isolation_restores_numpy():
    """``RDropout`` seeds numpy's legacy global generator from the layer width."""
    np.random.seed(99)
    expected = np.random.rand(4).tolist()

    np.random.seed(99)
    with isolated_randomness():
        np.random.seed(7)
        np.random.rand(100)
    assert np.random.rand(4).tolist() == expected


def test_isolation_restores_torch():
    torch = pytest.importorskip("torch")
    torch.manual_seed(5)
    expected = torch.rand(3).tolist()

    torch.manual_seed(5)
    with isolated_randomness():
        torch.manual_seed(1000)
        torch.rand(100)
    assert torch.rand(3).tolist() == expected


def test_isolation_holds_when_the_body_raises():
    """A failed evaluation must not leave the search drawing from a foreign state."""
    random.seed(77)
    expected = [random.random() for _ in range(3)]

    random.seed(77)
    with pytest.raises(RuntimeError), isolated_randomness():
        random.seed(20)
        raise RuntimeError("an architecture failed to build")
    assert [random.random() for _ in range(3)] == expected


def test_two_isolated_evaluations_do_not_converge_on_one_state():
    """The property the prior arrangement lacked.

    Without isolation, two evaluations that each reseed to a constant leave the caller at
    the same place both times, so the next draw is identical — which is how five runs
    came to share most of their architectures.
    """

    def evaluate_without_isolation():
        random.seed(20)

    def evaluate_with_isolation():
        with isolated_randomness():
            random.seed(20)

    random.seed(1)
    evaluate_without_isolation()
    first_unisolated = [random.randint(0, 23) for _ in range(15)]
    evaluate_without_isolation()
    second_unisolated = [random.randint(0, 23) for _ in range(15)]
    assert first_unisolated == second_unisolated, "precondition: the defect reproduces"

    random.seed(1)
    evaluate_with_isolation()
    first = [random.randint(0, 23) for _ in range(15)]
    evaluate_with_isolation()
    second = [random.randint(0, 23) for _ in range(15)]
    assert first != second


# ── seeding ────────────────────────────────────────────────────────────────────


def test_run_seeds_are_distinct_and_reproducible():
    seeds = run_seeds(2026, runs=5)
    assert len(set(seeds)) == 5
    assert run_seeds(2026, runs=5) == seeds
    assert run_seeds(2027, runs=5) != seeds


def test_run_seeds_are_not_consecutive():
    """Consecutive integers give correlated opening states in some generators."""
    seeds = sorted(run_seeds(7, runs=5))
    gaps = [b - a for a, b in itertools.pairwise(seeds)]
    assert min(gaps) > 1


def test_seed_everything_makes_a_run_reproducible():
    seed_everything(42)
    first = (random.random(), float(np.random.rand()))
    seed_everything(42)
    assert (random.random(), float(np.random.rand())) == first


def test_protocol_split_is_constructible_directly():
    """Used by the harness when the partition comes from a manifest rather than a draw."""
    split = ProtocolSplit(corpus="2015", train=(0, 1, 2), validation=(3, 4), seed=1)
    assert split.as_dict()["train_pairs"] == 3
    assert not split.supports_clean_selection
