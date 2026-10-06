"""Manual baselines: the shape of the comparison, more than the models.

The properties worth guarding are the ones that make the comparison mean something —
same input, same training budget, selection on validation — because a baseline that is
unfair in either direction answers the reviewer's question wrongly rather than not at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4av.baselines import (
    FAIR_INPUT_BASELINES,
    HYPERPARAMETER_BUDGET,
    PairCNN,
    PairLSTM,
    PairMLP,
    evaluate_baseline,
    pair_matrix,
)
from nas4av.prior.dataset import Partition
from nas4av.protocol import three_way_split

torch = pytest.importorskip("torch")


def fake_partition(pairs: int, width: int, chunks: int = 1) -> Partition:
    rows = torch.arange(pairs * 2 * chunks * width, dtype=torch.float64)
    return Partition(
        features=rows.reshape(pairs * 2 * chunks, width),
        frame=None,  # type: ignore[arg-type]
        chunks_per_document=[chunks] * pairs * 2,
        labels=[index % 2 for index in range(pairs)],
    )


# ── the pair folding, which every baseline depends on ──────────────────────────


def test_pair_matrix_yields_one_row_per_pair_at_the_searched_width():
    """A baseline must see exactly what a searched architecture sees."""
    part = fake_partition(pairs=5, width=300)
    folded = pair_matrix(part, input_shape=600)
    assert folded.shape == (5, 600)


def test_pair_matrix_concatenates_the_two_documents_in_order():
    part = fake_partition(pairs=2, width=3)
    folded = pair_matrix(part, input_shape=6)
    # pair 0 is document rows 0 and 1
    assert folded[0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert folded[1].tolist() == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]


def test_pair_matrix_averages_chunks_before_concatenating():
    """The chunked BERT arm holds three rows per document.

    Averaging is the only fold that leaves the width equal to the searched architectures'
    input shape, which is what makes the comparison fair rather than merely adjacent.
    """
    part = fake_partition(pairs=1, width=2, chunks=3)
    folded = pair_matrix(part, input_shape=4)
    assert folded.shape == (1, 4)
    # document 0 is rows 0,1,2 -> means (2, 3); document 1 is rows 3,4,5 -> means (8, 9)
    assert folded[0].tolist() == [2.0, 3.0, 8.0, 9.0]


def test_pair_matrix_handles_an_empty_partition():
    part = fake_partition(pairs=0, width=300)
    assert pair_matrix(part, input_shape=600).shape == (0, 600)


# ── the models ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("build", "kwargs"),
    [
        (PairMLP, {"width": 32, "depth": 2}),
        (PairCNN, {"channels": 8, "kernel": 5}),
        (PairLSTM, {"hidden": 16}),
    ],
    ids=["mlp", "cnn", "lstm"],
)
def test_every_baseline_maps_a_pair_batch_to_one_logit(build, kwargs):
    model = build(600, **kwargs)
    batch = torch.randn(4, 600, dtype=torch.float64)
    assert model(batch).reshape(-1).shape == (4,)


@pytest.mark.parametrize(
    ("build", "kwargs"),
    [
        (PairMLP, {"width": 32, "depth": 2}),
        (PairCNN, {"channels": 8, "kernel": 5}),
        (PairLSTM, {"hidden": 16}),
    ],
    ids=["mlp", "cnn", "lstm"],
)
def test_every_baseline_is_float64(build, kwargs):
    """The stored features are double, so a float32 model fails on its first multiply.

    The searched architectures hit exactly this, and it is why their weights are
    converted before training rather than left at the layer default.
    """
    model = build(600, **kwargs)
    assert all(p.dtype == torch.float64 for p in model.parameters())


def test_the_recurrent_baseline_refuses_an_odd_width():
    """It reads the input as two documents, so an odd width has no such reading."""
    with pytest.raises(ValueError, match="even width"):
        PairLSTM(601)


def test_the_recurrent_baseline_reads_the_pair_as_two_steps():
    """The formulation the searched space cannot express, which is why it is included."""
    model = PairLSTM(600, hidden=16)
    assert model.per_document == 300
    assert model.lstm.input_size == 300


# ── the comparison's shape ─────────────────────────────────────────────────────


def test_every_family_declares_a_bounded_budget():
    """Tuning effort is disclosed, so a weak baseline cannot be mistaken for evidence."""
    for spec in FAIR_INPUT_BASELINES:
        assert 0 < len(spec.configurations()) <= HYPERPARAMETER_BUDGET
        assert len({tuple(sorted(c.items())) for c in spec.configurations()}) == len(
            spec.configurations()
        )


def test_the_families_are_distinct_and_named():
    names = [spec.name for spec in FAIR_INPUT_BASELINES]
    assert sorted(names) == ["cnn", "lstm", "mlp"]


def test_selection_happens_on_validation_and_records_what_it_rejected(monkeypatch):
    """A baseline tuned on test would put the leakage on the other side of the comparison.

    Uses a tiny synthetic corpus so the assertion is about the protocol rather than the
    numbers.
    """
    from nas4av.baselines import BaselineSpec
    from nas4av.prior.dataset import Corpus
    from nas4av.space import cost

    width, pairs, test_pairs = 8, 12, 6
    # Registered rather than patched after the fact: Corpus checks its assembled width
    # against the declared input shape while it is being constructed.
    monkeypatch.setitem(cost.TENSOR_WIDTH, ("synthetic", "sem"), width)
    corpus = Corpus(
        corpus="synthetic",
        strategy="sem",
        train_features=torch.randn(pairs * 2, width, dtype=torch.float64),
        test_features=torch.randn(test_pairs * 2, width, dtype=torch.float64),
        train_documents=[f"d{i}" for i in range(pairs * 2)],
        test_documents=[f"t{i}" for i in range(test_pairs * 2)],
        train_labels=[i % 2 for i in range(pairs)],
        test_labels=[i % 2 for i in range(test_pairs)],
    )
    split = three_way_split("synthetic", corpus.train_labels, corpus.test_labels, seed=1)
    spec = BaselineSpec(
        name="mlp",
        build=PairMLP,
        grid=({"width": 8, "depth": 1}, {"width": 16, "depth": 1}),
    )
    result = evaluate_baseline(
        spec, corpus, split, test_indices=range(test_pairs), epochs=1, seed=5
    )

    assert result.configurations_tried == 2
    assert len(result.alternatives) == 2
    assert all("validation_roc_auc" in entry for entry in result.alternatives)
    chosen = max(result.alternatives, key=lambda entry: entry["validation_roc_auc"])
    assert result.configuration["width"] == chosen["width"]
    assert result.params > 0
    assert result.input_level == "fair"


def test_training_uses_an_injected_generator_not_a_global_one():
    """The defect the original loop has: a callee steering the stream its caller reads."""
    import random

    from nas4av.baselines import _train

    features = torch.randn(8, 16, dtype=torch.float64)
    labels = [0, 1] * 4

    random.seed(1)
    model = PairMLP(16, width=8, depth=1)
    torch.manual_seed(0)
    _train(model, features, labels, epochs=1, rng=np.random.default_rng(7))
    after = random.random()

    random.seed(1)
    assert after == random.random()
