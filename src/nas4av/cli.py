"""Command-line entry points.

    python -m nas4av.cli plan --strategies evolutionary-variant canonical-ga uniform-random
    python -m nas4av.cli rate --settings 2015/sem 2020/sem
    python -m nas4av.cli sweep --strategies evolutionary-variant --runs 5
    python -m nas4av.cli status
    python -m nas4av.cli verify --units <identifier> ...
    python -m nas4av.cli platform

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
    from .sweep import interrupted

    units = _plan_from_args(args)
    done = completed(units)
    directory = artifact_directory()
    print(f"  artifacts: {directory}")
    print(f"  {len(done)} of {len(units)} units complete")
    remaining = [unit for unit in units if unit.identifier not in done]
    if remaining:
        print(f"  {sum(u.budget for u in remaining):,} evaluations outstanding")
    partial = interrupted()
    if partial:
        print(f"\n  {len(partial)} unit(s) stopped partway and will resume from a checkpoint:")
        for name in partial[:10]:
            print(f"    {name}")
        print("  A long list here means runs are being killed, not that resuming is working.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-run stored units and report whether they reproduce."""
    from .verify import stored_units, verify_many

    identifiers = args.units or stored_units()[: args.limit]
    if not identifiers:
        print("  no stored units to verify")
        return 1
    print(f"  verifying {len(identifiers)} unit(s)\n")
    report = verify_many(identifiers, keep_scores=False)
    for name, verdict in report["verdicts"].items():
        mark = "ok  " if verdict == "reproduced" else "DIFF"
        print(f"    [{mark}] {name:<56} {verdict}")
    print(f"\n  {report['reproduced']} of {report['units']} reproduced")
    path = write_json("verification/report.json", report)
    print(f"  written: {path}")
    return 0 if report["reproduced"] == report["units"] else 1


def cmd_platform(args: argparse.Namespace) -> int:
    """What the stored artifacts say they were produced on."""
    from .verify import reference_platform

    report = reference_platform(sample=args.sample)
    for entry in report["environments"]:
        print(
            f"  {entry['count']:>4} units   {entry.get('platform')}   "
            f"torch {entry.get('torch')}   numpy {entry.get('numpy')}   {entry.get('torch_device')}"
        )
    if not report["homogeneous"]:
        print("\n  The campaign is NOT homogeneous. Results from different platforms are")
        print("  not expected to agree bit-for-bit; regenerate or declare the mix.")
    path = write_json("verification/platform.json", report)
    print(f"\n  written: {path}")
    return 0


def cmd_structure(args: argparse.Namespace) -> int:
    """What the genotype says about performance, attributed to nameable features.

    The source is an argument rather than an assumption, and the choice is a real one.
    The legacy database is 66,000 evaluations selected on test and scored with balanced
    accuracy under the name AUC, so it describes a searched population rather than a
    measured one; the clean campaign is smaller and sound. Running the same analysis over
    both is what separates "this is what the rebuild changes" from "this is what the new
    data says", and those are different claims.
    """
    from .structure import (
        consecutive_run,
        encode_many,
        enrichment,
        family_groups,
        grouped_permutation_importance,
        null_probability_of_run,
        position_groups,
        split_frequency,
    )

    population = _structure_population(args)
    if len(population) < args.minimum:
        print(f"  {len(population)} evaluations is below the floor of {args.minimum}")
        return 1

    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    encodings = [encoding for encoding, _ in population]
    scores = np.array([score for _, score in population], dtype=np.float64)
    features = encode_many(encodings)

    # Importance is read on data the model did not fit. On the training split a forest can
    # memorise 66,000 rows, and permuting a column then measures how much of its own fit it
    # loses rather than how much the feature predicts.
    train_x, test_x, train_y, test_y = train_test_split(
        features, scores, test_size=0.3, random_state=args.seed
    )
    model = RandomForestRegressor(
        n_estimators=args.trees, random_state=args.seed, n_jobs=-1, min_samples_leaf=5
    )
    model.fit(train_x, train_y)
    fit = {"train_r2": float(model.score(train_x, train_y)), "test_r2": float(model.score(test_x, test_y))}
    print(f"  {len(population)} evaluations   R² train {fit['train_r2']:.3f}   test {fit['test_r2']:.3f}")
    if fit["test_r2"] <= 0:
        print("  The model does not predict out of sample, so no importance read off it means")
        print("  anything. Reported below for the record, and not as evidence.")

    by_family = grouped_permutation_importance(model, test_x, test_y, family_groups(), repeats=args.repeats, seed=args.seed)
    by_position = grouped_permutation_importance(model, test_x, test_y, position_groups(), repeats=args.repeats, seed=args.seed)
    print("\n  importance by operation family")
    for item in by_family:
        print(f"    {item.group:<12} {item.mean:+.5f} ± {item.std:.5f}")
    print("\n  importance by position (top five)")
    for item in by_position[:5]:
        print(f"    {item.group:<12} {item.mean:+.5f} ± {item.std:.5f}")

    patterns = []
    for family, length in (("Linear", 4), ("Linear", 3), ("Linear", 2), ("Dropout", 2), ("Conv1D", 2)):
        measured = enrichment(
            population,
            f"{length} consecutive {family}",
            lambda genes, f=family, n=length: consecutive_run(genes, f, n),
            null=null_probability_of_run(family, length),
            top_fraction=args.top_fraction,
        )
        patterns.append(measured)
    print(f"\n  pattern enrichment, top {args.top_fraction:.0%} against the sampled population")
    print(f"    {'pattern':<24} {'in top':>8} {'in pop':>8} {'null':>8} {'enrich':>7}")
    for item in patterns:
        print(
            f"    {item.pattern:<24} {item.top_frequency:>8.3f} {item.population_frequency:>8.3f} "
            f"{item.null_probability:>8.3f} {item.ratio:>7.2f}"
        )
    survivors = [item.pattern for item in patterns if item.ratio >= args.enrichment_floor]
    print(f"\n  clearing an enrichment of {args.enrichment_floor}: {', '.join(survivors) or 'none'}")

    report = {
        "source": args.source,
        "setting": args.setting,
        "population": len(population),
        "model": {"trees": args.trees, "seed": args.seed, **fit},
        "importance_by_family": [item.as_dict() for item in by_family],
        "importance_by_position": [item.as_dict() for item in by_position],
        "split_frequency_by_family": split_frequency(model, family_groups()),
        "split_frequency_by_position": split_frequency(model, position_groups()),
        "enrichment": [item.as_dict() for item in patterns],
        "enrichment_floor": args.enrichment_floor,
        "patterns_clearing_the_floor": survivors,
        "note": (
            "enrichment is against the sampled population, which for the legacy source is "
            "EA-guided and therefore already biased toward whatever the search favoured; "
            "the null is stated alongside and is the unconstrained-space probability of the "
            "pattern appearing anywhere in fifteen positions, not in a given window"
        ),
    }
    stem = f"{args.source}_{(args.setting or 'all').replace('/', '_')}"
    path = write_json(f"structure/{stem}.json", report)
    print(f"  written: {path}")
    return 0


def _structure_population(args: argparse.Namespace) -> list[tuple[str, float]]:
    """(encoding, score) pairs from whichever database was asked for."""
    if args.source == "legacy":
        from .prior.repository import search_evaluations

        corpus, _, strategy = (args.setting or "/").partition("/")
        rows = search_evaluations(corpus or None, strategy or None)
        # The legacy column named `test auc` is balanced accuracy, and it is also what the
        # search selected on. Both defects are recorded elsewhere; this analysis is
        # descriptive of that population and says so rather than repairing it silently.
        return [(row.encoding, row.test_auc) for row in rows]

    from .efficiency import load_points

    return [
        (point.encoding, float(point.score))
        for point in load_points(setting=args.setting, metric=args.metric)
        if point.score is not None and not point.failed
    ]


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

    verify = sub.add_parser("verify", help="re-run stored units and compare")
    verify.add_argument("--units", nargs="*", help="identifiers; default the first --limit")
    verify.add_argument("--limit", type=int, default=3)
    verify.set_defaults(func=cmd_verify)

    platform = sub.add_parser("platform", help="what the stored artifacts were produced on")
    platform.add_argument("--sample", type=int, default=0, help="0 reads every artifact")
    platform.set_defaults(func=cmd_platform)

    structure = sub.add_parser(
        "structure", help="what the genotype says about performance, by family and position"
    )
    structure.add_argument(
        "--source",
        choices=("legacy", "campaign"),
        default="campaign",
        help="legacy: the 66,000 prior evaluations; campaign: the clean sweep artifacts",
    )
    structure.add_argument("--setting", help="corpus/features, default pooled over all")
    structure.add_argument(
        "--metric",
        default="test_roc_auc",
        help="which campaign score to attribute; ignored for the legacy source",
    )
    structure.add_argument("--trees", type=int, default=300)
    structure.add_argument("--repeats", type=int, default=20)
    structure.add_argument("--seed", type=int, default=BASE_SEED)
    structure.add_argument(
        "--top-fraction",
        type=float,
        default=0.05,
        help="what counts as a top performer when measuring enrichment",
    )
    structure.add_argument(
        "--enrichment-floor",
        type=float,
        default=2.7,
        help="enrichment a pattern must clear to survive as a recommendation (D6)",
    )
    structure.add_argument(
        "--minimum",
        type=int,
        default=200,
        help="refuse to fit below this many evaluations",
    )
    structure.set_defaults(func=cmd_structure)

    status = sub.add_parser("status", help="what the artifact directory already holds")
    shared(status)
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
