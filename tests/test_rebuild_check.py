"""The before/after check on rebuilt features.

The risk this guards is a comfortable one: declaring that rebuilding changed nothing
because the test was too weak to notice. So the tests build campaigns where a conclusion
*does* flip and check it is reported, alongside ones where only the scores move.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nas4av.rebuild_check import CONTROL, REBUILT_SETTINGS, SettingShift, compare_rebuild


def write_units(root: Path, scores: dict[str, dict[str, float]]) -> Path:
    """A minimal campaign: one run per (setting, strategy) at a fixed score."""
    sweep = root / "sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    for setting, strategies in scores.items():
        corpus, features = setting.split("/")
        for strategy, value in strategies.items():
            payload = {
                "unit": {
                    "corpus": corpus,
                    "features": features,
                    "strategy": strategy,
                    "run": 1,
                },
                "run": {
                    "records": [
                        {
                            "genotype": "0",
                            "params": 1,
                            "failed": False,
                            "validation_roc_auc": value,
                            "test_roc_auc": value,
                        }
                    ],
                    "evaluations": 10,
                    "failures": 0,
                },
            }
            name = f"{corpus}_{features}_{strategy}_run1.json"
            (sweep / name).write_text(json.dumps(payload), encoding="utf-8")
    return sweep


class TestShift:
    def test_a_held_verdict_is_recognised(self) -> None:
        shift = SettingShift(
            setting="x/y",
            before={"evolutionary-variant": 0.7, CONTROL: 0.6},
            after={"evolutionary-variant": 0.68, CONTROL: 0.61},
        )
        assert shift.search_beat_random_before and shift.search_beat_random_after
        assert shift.verdict_held
        assert shift.largest_shift == pytest.approx(0.02)

    def test_a_flipped_verdict_is_reported(self) -> None:
        """The failure that matters: the rebuild overturning a claim."""
        shift = SettingShift(
            setting="x/y",
            before={"evolutionary-variant": 0.7, CONTROL: 0.6},
            after={"evolutionary-variant": 0.55, CONTROL: 0.62},
        )
        assert shift.search_beat_random_before
        assert not shift.search_beat_random_after
        assert not shift.verdict_held

    def test_ordering_is_tracked_separately_from_the_verdict(self) -> None:
        """Two searches can swap places while both still beat random."""
        shift = SettingShift(
            setting="x/y",
            before={"evolutionary-variant": 0.70, "canonical-ga": 0.68, CONTROL: 0.5},
            after={"evolutionary-variant": 0.67, "canonical-ga": 0.69, CONTROL: 0.5},
        )
        assert shift.verdict_held
        assert not shift.ordering_held


class TestCompare:
    def scores(self, control: float) -> dict[str, dict[str, float]]:
        return {
            setting: {
                "evolutionary-variant": 0.70,
                "canonical-ga": 0.68,
                CONTROL: control,
            }
            for setting in REBUILT_SETTINGS
        }

    def test_scores_moving_without_conclusions_moving(self, tmp_path: Path) -> None:
        before = write_units(tmp_path / "before", self.scores(0.50))
        after = write_units(tmp_path / "after", self.scores(0.52))
        report = compare_rebuild(before, after)
        assert report["settings_compared"] == len(REBUILT_SETTINGS)
        assert report["verdict_held"] == len(REBUILT_SETTINGS)
        assert report["verdict_changed"] == []
        assert "changes scores and not conclusions" in report["conclusion"]

    def test_a_conclusion_that_flips_is_not_called_cosmetic(self, tmp_path: Path) -> None:
        before = write_units(tmp_path / "before", self.scores(0.50))
        flipped = self.scores(0.90)  # random now beats both searches everywhere
        after = write_units(tmp_path / "after", flipped)
        report = compare_rebuild(before, after)
        assert report["verdict_held"] == 0
        assert sorted(report["verdict_changed"]) == sorted(REBUILT_SETTINGS)
        assert "cannot be treated as cosmetic" in report["conclusion"]

    def test_a_missing_setting_is_reported_rather_than_skipped(self, tmp_path: Path) -> None:
        """A setting absent from the re-run must not be silently counted as holding."""
        before = write_units(tmp_path / "before", self.scores(0.50))
        partial = {k: v for k, v in self.scores(0.52).items() if k != REBUILT_SETTINGS[0]}
        after = write_units(tmp_path / "after", partial)
        report = compare_rebuild(before, after)
        assert REBUILT_SETTINGS[0] in report["settings_missing"]
        assert report["settings_compared"] == len(REBUILT_SETTINGS) - 1

    def test_it_states_that_scores_are_expected_to_differ(self, tmp_path: Path) -> None:
        before = write_units(tmp_path / "before", self.scores(0.50))
        after = write_units(tmp_path / "after", self.scores(0.52))
        assert "necessarily differ" in compare_rebuild(before, after)["what_is_compared"]
