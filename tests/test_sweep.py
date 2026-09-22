"""Planning, resumption, and the CLI that drives them.

Resumption is the part that earns its tests. A campaign is tens of hours, so it will be
interrupted, and the failure modes of a careless resume are both silent: re-running work
that is already done wastes the budget, and skipping work that only half-finished carries
a partial result into the conclusions.
"""

from __future__ import annotations

import json

import pytest

from nas4av import cli
from nas4av.artifacts import artifacts_root
from nas4av.sweep import (
    BUDGET,
    SETTINGS,
    STRATEGIES,
    WorkUnit,
    completed,
    estimate,
    plan,
    run_sweep,
)


@pytest.fixture(autouse=True)
def _isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("NAS4AV_ARTIFACTS", str(tmp_path))
    return tmp_path


def write_artifact(unit: WorkUnit, evaluations: int = 10, truncated: bool = False) -> None:
    path = artifacts_root() / unit.artifact_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"unit": {"corpus": unit.corpus}, "run": {"evaluations": evaluations}}
    text = json.dumps(payload)
    path.write_text(text[: len(text) // 2] if truncated else text, encoding="utf-8")


# ── planning ───────────────────────────────────────────────────────────────────


def test_plan_covers_every_setting_strategy_and_run():
    units = plan(strategies=["evolutionary-variant", "uniform-random"], runs=3)
    assert len(units) == len(SETTINGS) * 2 * 3
    assert len({unit.identifier for unit in units}) == len(units)


def test_runs_of_one_setting_get_distinct_seeds():
    """Runs sharing a seed would repeat each other, which is the prior defect."""
    units = [u for u in plan(runs=5) if u.corpus == "2015" and u.features == "sem"]
    assert len({unit.seed for unit in units}) == 5


def test_the_same_run_index_shares_a_seed_across_settings():
    """So a difference between two settings is not confounded with a different draw."""
    units = plan(runs=3)
    by_run: dict[int, set[int]] = {}
    for unit in units:
        by_run.setdefault(unit.run, set()).add(unit.seed)
    assert all(len(seeds) == 1 for seeds in by_run.values())


def test_a_plan_is_reproducible_from_its_base_seed():
    assert [u.seed for u in plan(base_seed=7)] == [u.seed for u in plan(base_seed=7)]
    assert [u.seed for u in plan(base_seed=7)] != [u.seed for u in plan(base_seed=8)]


def test_an_unknown_strategy_is_refused():
    with pytest.raises(KeyError):
        plan(strategies=["bayesian"])


def test_the_novelty_flag_changes_the_identifier():
    """Two configurations must never write to the same artifact."""
    with_filter = plan(runs=1, novelty=True)[0]
    without = plan(runs=1, novelty=False)[0]
    assert with_filter.identifier != without.identifier
    assert "nofilter" in without.identifier


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_registered_strategy_builds(name):
    unit = WorkUnit(corpus="2015", features="sem", strategy=name, run=1, seed=5)
    strategy = unit.build_strategy()
    assert strategy.population_size == unit.population_size
    assert len(strategy.initial_population()) == unit.population_size


def test_building_without_the_novelty_filter_returns_the_bare_strategy():
    unit = WorkUnit(
        corpus="2015", features="sem", strategy="uniform-random", run=1, seed=5, novelty=False
    )
    assert "novelty" not in unit.build_strategy().name


# ── resumption ─────────────────────────────────────────────────────────────────


def test_nothing_is_complete_in_an_empty_directory():
    assert completed(plan(runs=2)) == set()


def test_a_written_unit_is_recognised():
    units = plan(runs=2)
    write_artifact(units[0])
    assert completed(units) == {units[0].identifier}


def test_a_truncated_artifact_is_not_treated_as_complete():
    """An interruption mid-write must cost a re-run, not a partial result in the table."""
    units = plan(runs=2)
    write_artifact(units[0], truncated=True)
    assert completed(units) == set()


def test_an_artifact_with_no_evaluations_is_not_complete():
    units = plan(runs=2)
    write_artifact(units[0], evaluations=0)
    assert completed(units) == set()


def test_resume_skips_only_what_is_done(monkeypatch):
    """``run_unit`` is replaced, because the real one trains for hours.

    An earlier version of this test called the real driver and quietly launched four
    1,200-evaluation campaigns, which is the same mistake in miniature that resumption
    exists to prevent.
    """
    executed: list[str] = []

    def fake_run_unit(unit, keep_scores=True):
        executed.append(unit.identifier)
        return None

    monkeypatch.setattr("nas4av.sweep.run_unit", fake_run_unit)

    units = plan(runs=2)[:4]
    write_artifact(units[0])
    write_artifact(units[2])

    outcomes = list(run_sweep(units, resume=True, keep_scores=False))
    skipped = {
        unit.identifier for unit, run in outcomes if run is None and unit.identifier not in executed
    }
    assert skipped == {units[0].identifier, units[2].identifier}
    assert executed == [units[1].identifier, units[3].identifier]


def test_resume_disabled_re_runs_everything(monkeypatch):
    executed: list[str] = []
    monkeypatch.setattr(
        "nas4av.sweep.run_unit",
        lambda unit, keep_scores=True: executed.append(unit.identifier),
    )
    units = plan(runs=2)[:3]
    for unit in units:
        write_artifact(unit)

    list(run_sweep(units, resume=False, keep_scores=False))
    assert executed == [unit.identifier for unit in units]


# ── cost projection ────────────────────────────────────────────────────────────


def test_estimate_uses_the_measured_rate_per_setting():
    units = plan(settings=[("2015", "sem")], runs=2, budget=100)
    projection = estimate(units, {"2015/sem": 2.0})
    assert projection["evaluations"] == 200
    assert projection["projected_hours"] == pytest.approx(200 * 2.0 / 3600, abs=0.01)
    assert projection["settings_without_a_measured_rate"] == []


def test_estimate_names_the_settings_it_could_not_cost():
    """A projection that silently omitted a setting would understate the campaign."""
    units = plan(settings=[("2015", "sem"), ("2020", "sem")], runs=1)
    projection = estimate(units, {"2015/sem": 1.0})
    assert projection["settings_without_a_measured_rate"] == ["2020/sem"]


# ── the command line ───────────────────────────────────────────────────────────


def test_plan_command_lists_units(capsys):
    assert cli.main(["plan", "--settings", "2015/sem", "--runs", "2"]) == 0
    out = capsys.readouterr().out
    assert "2 units" in out
    assert f"{2 * BUDGET:,} evaluations" in out


def test_plan_command_projects_a_cost_when_given_rates(capsys):
    assert cli.main(["plan", "--settings", "2015/sem", "--runs", "1", "--rates", "2015/sem=2"]) == 0
    assert "projected:" in capsys.readouterr().out


def test_status_command_counts_what_is_on_disk(capsys):
    units = plan(settings=[("2015", "sem")], runs=2)
    write_artifact(units[0])
    assert cli.main(["status", "--settings", "2015/sem", "--runs", "2"]) == 0
    assert "1 of 2 units complete" in capsys.readouterr().out


def test_an_unknown_setting_is_refused():
    with pytest.raises(SystemExit):
        cli.main(["plan", "--settings", "1999/sem"])


def test_a_malformed_setting_is_refused():
    with pytest.raises(SystemExit):
        cli.main(["plan", "--settings", "2015"])


def test_the_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])
