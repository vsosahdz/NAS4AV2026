"""Parameter counting, checked against PyTorch rather than against itself.

The closed form in ``space.cost.describe`` is what makes a front over tens of thousands
of architectures affordable, and it is worth exactly as much as its agreement with the
module the decoder actually builds. So the central test here transcribes
``Model.build_sequential`` into a real ``nn.Sequential`` and asks PyTorch to count.

If those two ever disagree, the closed form is wrong and every parameter figure derived
from it is wrong with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4av.space.cost import (
    PAIR_FACTOR,
    PUNCTUATION_WIDTH,
    TENSOR_WIDTH,
    describe,
    input_shape,
    pareto_front,
)
from nas4av.space.genotype import CONV_OPERATIONS, LENGTH, linear_width, sample_feasible

torch = pytest.importorskip("torch")
nn = torch.nn


def build_sequential(genes, input_width):
    """Transcribed from ``Model.build_sequential`` in the prior implementation.

    Kept deliberately close to its source, including the Conv1D width rule, because this
    function is the reference the closed form is checked against.
    """
    modules: list[nn.Module] = []
    curr = input_width
    for gene in genes:
        if gene < 7:
            if gene == 0:
                modules.append(nn.Identity())
            elif gene <= 3:
                modules.append(nn.Dropout(0.9))
            else:
                modules.append(nn.ReLU())
        elif gene <= 19:
            out = linear_width(gene)
            modules.append(nn.Linear(curr, out))
            curr = out
            modules.append(nn.BatchNorm1d(curr, dtype=torch.float64))
        else:
            kernel, channels = CONV_OPERATIONS[gene]
            modules.append(nn.Conv1d(1, channels, kernel))
            modules.append(nn.Flatten(1, -1))
            curr = channels if kernel > curr else (curr - kernel + 1) * channels
            modules.append(nn.BatchNorm1d(curr, dtype=torch.float64))
    modules.append(nn.Linear(curr, 1))
    return nn.Sequential(*modules)


def torch_parameter_count(genes, input_width):
    model = build_sequential(genes, input_width)
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize(
    "genes",
    [
        pytest.param([0] * LENGTH, id="all-identity"),
        pytest.param([4] * LENGTH, id="all-activation"),
        pytest.param([1, 2, 3] * 5, id="all-dropout"),
        pytest.param([7] * LENGTH, id="narrowest-linear"),
        pytest.param([19] + [7] * 14, id="widest-linear-first"),
        pytest.param([20, 0, 21, 0, 22, 0, 23, 0, 7, 0, 8, 0, 9, 0, 10], id="conv-spaced"),
        pytest.param([16, 10, 16, 9, 2, 17, 19, 12, 11, 8, 23, 10, 17, 5, 11], id="legacy-row"),
    ],
)
def test_closed_form_matches_torch(genes):
    """The arithmetic and the module must agree exactly, not approximately."""
    width = 600
    assert describe(genes, width).params == torch_parameter_count(genes, width)


def test_closed_form_matches_torch_over_random_feasible_sample():
    """Agreement has to hold across the space, not only on hand-picked cases.

    Seeded, because a test that fails on one machine in twenty is a test nobody trusts.
    """
    rng = np.random.default_rng(20260904)
    for genes in sample_feasible(rng, 60):
        for width in (600, 970, 1536):
            assert describe(genes, width).params == torch_parameter_count(genes, width)


def test_input_shape_reproduces_the_hardcoded_upstream_value():
    """``syn_embsz = 485`` for CLEF-PAN 2013, and 453 + 32 = 485.

    This is the only independent check available on the punctuation width, so it earns
    its own test: the prior implementation states the per-document syntactic width for
    one corpus, and the rule has to land on it.
    """
    assert TENSOR_WIDTH[("2013", "synBetter")] + PUNCTUATION_WIDTH == 485
    assert input_shape("2013", "synBetter") == 485 * PAIR_FACTOR


def test_embedding_strategies_do_not_gain_punctuation():
    """Only the syntactic arm concatenates punctuation counts.

    ``train_embeddings_sem = training_wembs`` upstream, with the concatenating
    alternative present but commented out. Adding 32 here would inflate every semantic
    parameter count.
    """
    assert input_shape("2015", "sem") == 300 * PAIR_FACTOR
    assert input_shape("2013", "semBert") == 768 * PAIR_FACTOR


def test_identity_only_architecture_is_a_single_linear():
    """Fifteen Identity operations leave the tail Linear as the only parameters."""
    arch = describe([0] * LENGTH, 600)
    assert arch.params == 600 + 1
    assert arch.effective_depth == 0
    assert arch.n_identity == LENGTH


def test_effective_depth_counts_non_identity_positions():
    arch = describe([0, 0, 7, 4, 0, 8, 1, 9, 0, 10, 5, 11, 0, 12, 6], 600)
    assert arch.n_identity == 5
    assert arch.effective_depth == LENGTH - 5


def test_pareto_front_keeps_the_better_score_at_equal_size():
    front = pareto_front([(100, 0.60), (100, 0.70), (200, 0.65)])
    assert front == [(100, 0.70)]


def test_pareto_front_is_monotone_in_both_axes():
    points = [(10, 0.50), (20, 0.55), (30, 0.52), (40, 0.60), (50, 0.58)]
    front = pareto_front(points)
    sizes = [size for size, _ in front]
    scores = [score for _, score in front]
    assert sizes == sorted(sizes)
    assert scores == sorted(scores)
    assert (30, 0.52) not in front  # dominated by (20, 0.55)
