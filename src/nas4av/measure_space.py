"""Measure a space's exploitability on a real corpus, before committing to a campaign.

Draws a uniform sample from a space, trains each architecture under the same protocol the
campaign uses, and fits the genotype to held-out performance. The result says whether a
search could beat sampling on that space — the precondition the first space fails on nine
of eleven settings, discovered after 165 units had been spent finding out.

Everything outside the architecture is held fixed between spaces: same corpora, same
three-way split, same optimiser, learning rate, batch size and epochs. A difference
between two spaces means something only if nothing else moved.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .evaluator import BATCH_SIZE, LEARNING_RATE, PARTIAL_EPOCHS
from .exploitability import (
    Exploitability,
    featurise_v1,
    featurise_v2,
    measure,
    sample_v1,
    sample_v2,
)
from .prior.dataset import load_corpus, partition
from .protocol import three_way_split

#: Which sampler and featuriser belong to each space. The first two are named explicitly
#: because the first space's sampler has to reject against `g1` and `g2`; everything else
#: is resolved through the registry, so an ablation variant is measured by exactly the code
#: that measures the full space rather than by a parallel path.
SPACES = {
    "v1": (sample_v1, featurise_v1),
    "v2": (sample_v2, featurise_v2),
}


def _resolve(space: str):
    if space in SPACES:
        return SPACES[space]
    from .exploitability import featurise_spec, sample_spec
    from .space.ablation import VARIANTS, register

    if space in VARIANTS:
        register()
    from .space.spec import _REGISTRY

    if space not in _REGISTRY:
        raise KeyError(f"unknown space {space!r}")
    return (lambda count, seed: sample_spec(space, count, seed)), featurise_spec(space)


def _roc_auc(logits: Sequence[float], labels: Sequence[int]) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(list(labels), list(logits)))


def evaluate_sample(
    space: str,
    corpus_name: str,
    features: str,
    count: int = 300,
    seed: int = 20260920,
    epochs: int = PARTIAL_EPOCHS,
    progress: bool = False,
) -> Exploitability:
    """Draw, train and fit — the whole measurement for one space on one setting."""
    sampler, featuriser = _resolve(space)
    encodings = sampler(count, seed)

    corpus = load_corpus(corpus_name, features)
    split = three_way_split(
        corpus_name, corpus.train_labels, corpus.test_labels, seed=seed
    )
    train = partition(
        corpus.train_features, corpus.train_documents, corpus.train_labels,
        split.train, corpus.train_chunks,
    )
    validation = partition(
        corpus.train_features, corpus.train_documents, corpus.train_labels,
        split.validation, corpus.train_chunks,
    )
    test = partition(
        corpus.test_features, corpus.test_documents, corpus.test_labels,
        split.release_test("exploitability is measured on held-out performance"),
        corpus.test_chunks,
    )

    scores: list[float | None] = []
    started = time.perf_counter()
    for index, encoding in enumerate(encodings):
        scores.append(_evaluate_one(space, encoding, corpus, train, {"test": test}, epochs, seed))
        if progress and (index + 1) % 25 == 0:
            rate = (time.perf_counter() - started) / (index + 1)
            print(
                f"    {index + 1}/{len(encodings)}  {rate:.2f}s each  "
                f"{rate * (len(encodings) - index - 1) / 60:.1f} min left",
                flush=True,
            )

    return measure(
        encodings,
        scores,
        featuriser,
        space=space,
        setting=f"{corpus_name}/{features}",
        seed=seed,
    )


def _evaluate_one(
    space: str,
    encoding: str,
    corpus: Any,
    train: Any,
    evaluate_on: dict[str, Any],
    epochs: int,
    seed: int,
) -> float | None:
    """One architecture's held-out score, or None if it failed to train.

    A failure returns None rather than a low score. The two are different facts: a low
    score says the architecture trained and was bad, while a failure says nothing was
    measured — and feeding failures in as zeros would let the fit predict "this one breaks"
    and report that as exploitable structure about quality.
    """
    try:
        if space != "v1":
            from .space.ablation import VARIANTS, tables_for
            from .space.v2_model import train_and_score

            # An ablation variant builds against its own alphabet, passed explicitly. The
            # full space passes None and gets its own.
            using = tables_for(space) if space in VARIANTS else None
            per_document = corpus.input_shape // 2
            result = train_and_score(
                encoding, per_document, train, evaluate_on,
                epochs=epochs, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, seed=seed, using=using,
            )
            return _roc_auc(result["test"], evaluate_on["test"].labels)

        import contextlib
        import io

        import torch

        from .original import pipeline as original

        with contextlib.redirect_stdout(io.StringIO()):
            space_object = original.Sequential_Search_Space(corpus.input_shape)
            model = original.Model(
                [int(g) for g in encoding.split(",")],
                corpus.input_shape, space_object.genel, corpus.input_shape // 2,
            )
            model.weights_init_uniform()
            model.train()
            space_object.train_model(
                model, train_embeddings=train.features, train_df=train.frame,
                training_cpd=train.chunks_per_document, n_epochs=epochs,
                batch_size=BATCH_SIZE, optimizer="Adam", lr=LEARNING_RATE, verbose=False,
            )
            model.eval()
            with torch.no_grad():
                part = evaluate_on["test"]
                logits = model(part.chunks_per_document, embeddings=part.features)
        return _roc_auc([float(v) for v in logits.reshape(-1)], evaluate_on["test"].labels)
    except Exception:  # noqa: BLE001 — a failure is a recorded outcome, not a crash
        return None


def failure_rate(space: str, corpus_name: str, features: str, count: int = 50,
                 seed: int = 1) -> dict[str, Any]:
    """How often a uniform draw fails to train at all.

    Worth knowing before a sample of hundreds: a space where most draws fail produces an
    exploitability measurement over whatever survived, which is a different space.
    """
    result = evaluate_sample(space, corpus_name, features, count=count, seed=seed)
    return {
        "space": space,
        "setting": f"{corpus_name}/{features}",
        "sampled": result.sampled,
        "failed": result.failed,
        "rate": round(result.failed / result.sampled, 4) if result.sampled else None,
    }
