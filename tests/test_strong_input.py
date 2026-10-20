"""The strong-input reference.

Most of what can go wrong here is silent. A threshold fitted on the wrong split, an AUC
that mishandles ties, a document read through a window that drops 95% of it — each
produces a plausible number rather than an error, and a reference baseline that is quietly
wrong is worse than none, because it still gets cited as one.

The encoder tests need a model download, so they skip rather than fail where there is no
network. The arithmetic tests do not, and they are where the guarantees live.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4av.strong_input import (
    CHUNK_STRIDE,
    CHUNK_TOKENS,
    RAW_FILES,
    Encoder,
    RawTextMissing,
    _balanced_accuracy,
    _roc_auc,
    available,
    fit_threshold,
    load_raw,
    offline,
)

needs_raw = pytest.mark.skipif(not available(), reason="the reference clone is not present")


def encoder_usable() -> bool:
    if offline():
        return False
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


needs_encoder = pytest.mark.skipif(
    not (available() and encoder_usable()), reason="no encoder or no raw text"
)


class TestRanking:
    def test_a_perfect_ranking_is_one_and_its_inverse_is_zero(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        assert _roc_auc(scores, [0, 0, 1, 1]) == pytest.approx(1.0)
        assert _roc_auc(-scores, [0, 0, 1, 1]) == pytest.approx(0.0)

    def test_total_ties_are_half_not_one(self) -> None:
        """Ties are common here: identical documents score exactly 1.

        Without mid-ranks a block of equal scores gets credited by whatever order the sort
        happened to produce, which turns a model that cannot separate anything into one
        that separates everything.
        """
        assert _roc_auc(np.ones(4), [0, 0, 1, 1]) == pytest.approx(0.5)

    def test_partial_ties_sit_between(self) -> None:
        scores = np.array([0.5, 0.5, 0.5, 0.9])
        value = _roc_auc(scores, [0, 0, 1, 1])
        assert value is not None and 0.5 < value < 1.0

    def test_one_class_has_no_auc_rather_than_a_default(self) -> None:
        """A corpus with no negatives cannot produce a ranking score, and must say so."""
        assert _roc_auc(np.array([0.1, 0.2]), [1, 1]) is None
        assert _roc_auc(np.array([0.1, 0.2]), [0, 0]) is None


class TestThreshold:
    def test_it_separates_a_separable_set(self) -> None:
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        threshold = fit_threshold(scores, [0, 0, 1, 1])
        assert 0.2 < threshold < 0.8
        predicted = (scores >= threshold).astype(int)
        assert _balanced_accuracy(predicted, np.array([0, 0, 1, 1])) == pytest.approx(1.0)

    def test_it_balances_rather_than_maximising_raw_accuracy(self) -> None:
        """On a skewed set, raw accuracy is maximised by predicting the majority class.

        Balanced accuracy is the metric the rest of this work reports, and a threshold
        fitted to a different objective would make the reference incomparable with
        everything beside it.
        """
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.95])
        labels = [0, 0, 0, 0, 0, 1]
        threshold = fit_threshold(scores, labels)
        predicted = (scores >= threshold).astype(int)
        assert predicted[-1] == 1, "the single positive must be recovered"
        assert _balanced_accuracy(predicted, np.array(labels)) == pytest.approx(1.0)

    def test_a_single_class_does_not_crash(self) -> None:
        assert isinstance(fit_threshold(np.array([0.3, 0.4]), [1, 1]), float)


class TestBalancedAccuracy:
    def test_predicting_one_class_scores_half(self) -> None:
        labels = np.array([0, 0, 1, 1])
        assert _balanced_accuracy(np.ones(4, dtype=int), labels) == pytest.approx(0.5)
        assert _balanced_accuracy(np.zeros(4, dtype=int), labels) == pytest.approx(0.5)

    def test_it_is_not_raw_accuracy_on_a_skewed_set(self) -> None:
        labels = np.array([0] * 9 + [1])
        predicted = np.zeros(10, dtype=int)
        assert float((predicted == labels).mean()) == pytest.approx(0.9)
        assert _balanced_accuracy(predicted, labels) == pytest.approx(0.5)


@needs_raw
class TestRawText:
    @pytest.mark.parametrize("corpus", sorted(RAW_FILES))
    def test_every_declared_corpus_loads_both_splits(self, corpus: str) -> None:
        for split in ("train", "test"):
            pairs = load_raw(corpus, split)
            assert len(pairs) > 0
            assert len(pairs.first) == len(pairs.second) == len(pairs.labels)
            assert set(pairs.labels) <= {0, 1}

    def test_the_aggregate_corpus_loads_despite_its_field_size(self) -> None:
        """A whole document sits in one CSV field, past Python's 128 KB default.

        This is the only corpus that trips it, so a reader who tests on the individual
        corpora sees nothing wrong.
        """
        assert len(load_raw("aggregate", "train")) == 480

    def test_an_unmapped_corpus_is_refused(self) -> None:
        with pytest.raises(RawTextMissing):
            load_raw("not-a-corpus")

    def test_the_text_is_not_a_single_mangled_column(self) -> None:
        """A comma-separated read of these files silently returns one column.

        It does not raise, so the symptom is an encoder embedding the header.
        """
        pairs = load_raw("2015", "train")
        assert all(len(t.split()) > 20 for t in pairs.first[:5])


@needs_encoder
class TestEncoder:
    def test_a_document_longer_than_the_window_is_chunked_not_truncated(self) -> None:
        """2020 overflows a 512-token window by 20x; truncation would read 5% of it."""
        encoder = Encoder()
        long_text = " ".join(["word"] * (CHUNK_TOKENS * 4))
        assert encoder.chunk_count(long_text) > 3

    def test_a_short_document_is_one_chunk(self) -> None:
        assert Encoder().chunk_count("a short sentence.") == 1

    def test_embedding_a_document_gives_one_vector(self) -> None:
        encoder = Encoder()
        vectors = encoder.embed(["one short text", " ".join(["word"] * (CHUNK_TOKENS * 3))])
        assert vectors.shape[0] == 2
        assert vectors.shape[1] == encoder.identity()["hidden_size"]
        assert np.isfinite(vectors).all()

    def test_identical_documents_are_maximally_similar(self) -> None:
        encoder = Encoder()
        vectors = encoder.embed(["the same words twice over", "the same words twice over"])
        cosine = float(
            vectors[0] @ vectors[1] / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
        )
        assert cosine == pytest.approx(1.0, abs=1e-6)

    def test_the_encoder_declares_that_nothing_was_fine_tuned(self) -> None:
        identity = Encoder().identity()
        assert identity["fine_tuned"] is False
        assert identity["chunk_tokens"] == CHUNK_TOKENS
        assert identity["chunk_stride"] == CHUNK_STRIDE
        assert identity["revision"]
