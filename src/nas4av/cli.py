"""Command-line entry points.

    python -m nas4av.cli plan --strategies evolutionary-variant canonical-ga uniform-random
    python -m nas4av.cli rate --settings 2015/sem 2020/sem
    python -m nas4av.cli sweep --strategies evolutionary-variant --runs 5
    python -m nas4av.cli status

``plan`` and ``status`` read nothing heavier than the artifact directory. ``rate`` and
``sweep`` need the prior campaign's feature tensors and a torch install, because they
train.

The separation is the point: a sweep is tens of hours, so what it will cost should be
answerable before it starts, and what it has already done should be answerable while it
runs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .artifacts import write_json
from .sweep import (
    BASE_SEED,
    BUDGET,
    RUNS,
    SETTINGS,
    STRATEGIES,
    WorkUnit,
    artifact_directory,
    completed,
    estimate,
    plan,
    run_sweep,
)


def _parse_settings(values: Sequence[str] | None) -> list[tuple[str, str]]:
    if not values:
        return list(SETTINGS)
    parsed: list[tuple[str, str]] = []
    for value in values:
        if "/" not in value:
            raise SystemExit(f"a setting looks like corpus/features, got {value!r}")
        corpus, features = value.split("/", 1)
        if (corpus, features) not in SETTINGS:
            raise SystemExit(f"unknown setting {value!r}; expected one of {_setting_names()}")
        parsed.append((corpus, features))
    return parsed


def _setting_names() -> list[str]:
    return [f"{corpus}/{features}" for corpus, features in SETTINGS]


def _plan_from_args(args: argparse.Namespace) -> list[WorkUnit]:
    return plan(
        settings=_parse_settings(args.settings),
        strategies=args.strategies,
        runs=args.runs,
        novelty=not args.no_novelty,
        base_seed=args.base_seed,
        budget=args.budget,
    )


def cmd_plan(args: argparse.Namespace) -> int:
    units = _plan_from_args(args)
    done = completed(units)
    print(f"  {len(units)} units, {sum(u.budget for u in units):,} evaluations")
    print(f"  {len(done)} already complete\n")
    for unit in units:
        mark = "done" if unit.identifier in done else "    "
        print(f"    [{mark}] {unit.identifier:<56} seed {unit.seed}")
    if args.rates:
        rates = dict(pair.split("=") for pair in args.rates)
        projection = estimate(units, {key: float(value) for key, value in rates.items()})
        print(f"\n  projected: {projection['projected_hours']} h")
        if projection["settings_without_a_measured_rate"]:
            print(f"  no measured rate for: {projection['settings_without_a_measured_rate']}")
    return 0


def cmd_rate(args: argparse.Namespace) -> int:
    """Measure seconds per evaluation, per setting, so a plan can be costed."""
    import random
    import time

    from .evaluator import PipelineEvaluator
    from .prior.dataset import load_corpus
    from .protocol import three_way_split
    from .search.campaign import run_search
    from .search.strategy import UniformRandom

    rates: dict[str, object] = {}
    for corpus_name, features in _parse_settings(args.settings):
        corpus = load_corpus(corpus_name, features)
        split = three_way_split(corpus_name, corpus.train_labels, corpus.test_labels, seed=11)
        evaluator = PipelineEvaluator(corpus, split, keep_scores=False)
        started = time.time()
        run = run_search(
            strategy=UniformRandom(population_size=args.samples, rng=random.Random(3)),
            evaluator=evaluator,
            split=split,
            input_shape=corpus.input_shape,
            budget=args.samples,
            strategy_seed=3,
        )
        seconds = (time.time() - started) / max(run.evaluations, 1)
        key = f"{corpus_name}/{features}"
        rates[key] = {
            "seconds_per_evaluation": round(seconds, 4),
            "partitions": evaluator.partition_sizes(),
            "supports_clean_selection": split.supports_clean_selection,
            "failures": run.failures,
        }
        print(
            f"  {key:<24}{seconds:>7.2f} s/eval   clean selection: {split.supports_clean_selection}"
        )
    path = write_json("sweep/rates.json", {"samples_per_setting": args.samples, "rates": rates})
    print(f"\n  written: {path}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    units = _plan_from_args(args)
    print(f"  {len(units)} units, {sum(u.budget for u in units):,} evaluations")
    print(f"  artifacts: {artifact_directory()}\n")
    ran = skipped = 0
    for unit, run in run_sweep(units, resume=not args.no_resume, keep_scores=not args.no_scores):
        if run is None:
            skipped += 1
            print(f"    [skip] {unit.identifier}")
            continue
        ran += 1
        best = run.best()
        summary = (
            f"best val {getattr(best, run.fitness_metric):.4f}" if best else "no usable result"
        )
        halted = f"  HALTED: {run.halted}" if run.halted else ""
        print(
            f"    [run ] {unit.identifier:<56} {run.evaluations:>5} evals, "
            f"{run.failures:>4} failed, {summary}{halted}"
        )
    print(f"\n  {ran} run, {skipped} skipped")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    units = _plan_from_args(args)
    done = completed(units)
    directory = artifact_directory()
    print(f"  artifacts: {directory}")
    print(f"  {len(done)} of {len(units)} units complete")
    remaining = [unit for unit in units if unit.identifier not in done]
    if remaining:
        print(f"  {sum(u.budget for u in remaining):,} evaluations outstanding")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nas4av", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(target: argparse.ArgumentParser) -> None:
        target.add_argument("--settings", nargs="*", help="corpus/features, default all")
        target.add_argument(
            "--strategies",
            nargs="*",
            default=["evolutionary-variant"],
            choices=sorted(STRATEGIES),
        )
        target.add_argument("--runs", type=int, default=RUNS)
        target.add_argument("--budget", type=int, default=BUDGET)
        target.add_argument("--base-seed", type=int, default=BASE_SEED)
        target.add_argument(
            "--no-novelty",
            action="store_true",
            help="disable the novelty filter, to measure what it contributes",
        )

    planner = sub.add_parser("plan", help="enumerate the sweep without running it")
    shared(planner)
    planner.add_argument("--rates", nargs="*", help="corpus/features=seconds, to project the cost")
    planner.set_defaults(func=cmd_plan)

    rate = sub.add_parser("rate", help="measure seconds per evaluation, per setting")
    rate.add_argument("--settings", nargs="*", help="corpus/features, default all")
    rate.add_argument("--samples", type=int, default=6)
    rate.set_defaults(func=cmd_rate)

    sweeper = sub.add_parser("sweep", help="run the plan, resuming by default")
    shared(sweeper)
    sweeper.add_argument("--no-resume", action="store_true", help="re-run completed units")
    sweeper.add_argument("--no-scores", action="store_true", help="do not keep per-document scores")
    sweeper.set_defaults(func=cmd_sweep)

    status = sub.add_parser("status", help="what the artifact directory already holds")
    shared(status)
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
