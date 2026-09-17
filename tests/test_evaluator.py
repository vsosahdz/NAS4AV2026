"""Corpus assembly and the evaluator, against the real stored features.

Marked ``legacy`` because they need the pinned clone. The partition arithmetic is the
part worth guarding: every quantity is keyed by pair while the feature tensors hold two
rows per document and, on the BERT arm, three rows per document. A mis-slice there
produces tensors that load, train, and mean nothing.
"""

from __future__ import annotations

import pytest

from nas4av.evaluator import PipelineEvaluator, Scores
from nas4av.prior.dataset import Corpus, load_corpus, partition
from nas4av.prior.repository import PriorRepositoryMissing
from nas4av.protocol import three_way_split

pytestmark = pytest.mark.legacy


@pytest.fixture(scope="module")
def _require_clone():
    try:
        from nas4av.prior.repository import results_root

        results_root()
    except PriorRepositoryMissing as exc:
        pytest.skip(str(exc).splitlines()[0])


# ── partition arithmetic, which needs no corpus ────────────────────────────────


def test_partition_selects_two_rows_per_pair():
    import torch

    features = torch.arange(20).reshape(10, 2).double()
    documents = [f"d{i}" for i in range(10)]
    labels = [0, 1, 0, 1, 0]
    part = partition(features, documents, labels, [1, 3])
    assert part.features.tolist() == [[4, 5], [6, 7], [12, 13], [14, 15]]
    assert part.chunks_per_document == [1, 1, 1, 1]
    assert part.labels == [1, 1]


def test_partition_follows_the_chunk_count():
    """The BERT arm stores three rows per document; hard-coding one would mis-slice it."""
    import torch

    features = torch.arange(36).reshape(18, 2).double()
    documents = [f"d{i}" for i in range(6)]
    labels = [0, 1, 0]
    part = partition(features, documents, labels, [1], chunks=3)
    # pair 1 is documents 2 and 3, so rows 6..11
    assert part.features.tolist() == [[12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]]
    assert part.chunks_per_document == [3, 3]


def test_a_corpus_with_ragged_rows_is_refused():
    import torch

    with pytest.raises(ValueError, match="divide"):
        Corpus(
            corpus="2015",
            strategy="sem",
            train_features=torch.zeros(7, 300).double(),
            test_features=torch.zeros(4, 300).double(),
            train_documents=["a", "b", "c", "d"],
            test_documents=["e", "f", "g", "h"],
            train_labels=[0, 1],
            test_labels=[0, 1],
        )


# ── assembly against the stored tensors ────────────────────────────────────────


@pytest.mark.parametrize(
    ("corpus", "strategy", "expected_width"),
    [
        ("2013", "sem", 600),
        ("2013", "synBetter", 970),
        ("2013", "semBert", 1536),
        ("2015", "synBetter", 1370),
        ("2020", "sem", 600),
    ],
)
def test_assembled_width_matches_the_declared_input_shape(
    _require_clone, corpus, strategy, expected_width
):
    """The syntactic arm only reaches its width after punctuation is concatenated."""
    loaded = load_corpus(corpus, strategy)
    assert loaded.input_shape == expected_width
    assert loaded.train_features.shape[1] * 2 == expected_width


def test_the_bert_arm_is_chunked_and_the_others_are_not(_require_clone):
    assert load_corpus("2013", "semBert").train_chunks == 3
    assert load_corpus("2013", "sem").train_chunks == 1
    assert load_corpus("2013", "synBetter").train_chunks == 1


def test_pair_counts_reproduce_the_published_data_description(_require_clone):
    """Train and test problem counts, per corpus."""
    expected = {
        "2013": (10, 30),
        "2014Essay2": (200, 200),
        "2014Novel2": (100, 200),
        "2015": (100, 500),
        "2020": (70, 30),
    }
    for corpus, (train_pairs, test_pairs) in expected.items():
        loaded = load_corpus(corpus, "sem")
        assert len(loaded.train_labels) == train_pairs
        assert len(loaded.test_labels) == test_pairs


# ── the evaluator's protocol guarantees ────────────────────────────────────────


@pytest.mark.slow
def test_an_evaluation_produces_both_metrics_and_keeps_its_scores(_require_clone):
    loaded = load_corpus("2020", "sem")
    split = three_way_split("2020", loaded.train_labels, loaded.test_labels, seed=3)
    evaluator = PipelineEvaluator(loaded, split)

    genotype = (10, 0, 4, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    measurement = evaluator(genotype)

    assert not measurement.failed, measurement.detail
    assert 0.0 <= measurement.validation_roc_auc <= 1.0
    assert 0.0 <= measurement.test_roc_auc <= 1.0
    assert measurement.seconds > 0

    encoding = ",".join(str(gene) for gene in genotype)
    assert encoding in evaluator.scores
    kept = evaluator.scores[encoding]
    assert len(kept["validation"]) == split.validation.__len__()
    assert len(kept["test"]) == len(loaded.test_labels)


@pytest.mark.slow
def test_a_broken_architecture_is_recorded_as_failed_not_as_zero(_require_clone):
    """The prior pipeline wrote zeros, which downstream cannot tell from a bad model."""
    loaded = load_corpus("2020", "sem")
    split = three_way_split("2020", loaded.train_labels, loaded.test_labels, seed=3)
    evaluator = PipelineEvaluator(loaded, split)

    # A width of one followed by a kernel-8 convolution: the shape arithmetic collapses.
    measurement = evaluator((7, 23, 23, 23, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    if measurement.failed:
        assert measurement.detail
        assert measurement.validation_roc_auc != measurement.validation_roc_auc  # NaN


def test_scores_return_the_no_discrimination_value_on_a_single_class():
    """Small validation partitions can hold one class, and that must not raise."""
    single = Scores(logits=[0.4, -0.2, 1.1], labels=[1, 1, 1])
    assert single.roc_auc() == 0.5
    assert single.balanced_accuracy() == 0.5


def test_scores_separate_a_ranking_metric_from_a_thresholded_one():
    """The distinction the prior work collapsed.

    These logits rank perfectly but all fall on one side of zero, so ROC-AUC is 1.0 while
    balanced accuracy is 0.5. A campaign that records only the second cannot tell a model
    that ranks well from one that does not.
    """
    scores = Scores(logits=[1.0, 2.0, 3.0, 4.0], labels=[0, 0, 1, 1])
    assert scores.roc_auc() == 1.0
    assert scores.balanced_accuracy() == 0.5
