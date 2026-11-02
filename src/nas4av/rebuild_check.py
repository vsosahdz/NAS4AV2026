"""Did rebuilding the features change any conclusion?

Regenerating the feature tensors from raw text changes the input to every experiment on
the four settings whose features actually moved. A cosine of 0.9996 on sorted rows says
the conclusions should survive, and "should" is not a measurement — so the affected units
were re-run and this compares the two.

What it compares is deliberately narrow. Per-unit scores *will* differ, because the
features differ; a test that demanded they match would be testing the wrong thing and
would fail by construction. What has to hold is the part the manuscript rests on:

- the strategies' **ordering** on each setting
- whether a search still beats uniform random where it did before
- the **magnitude** of the shift, so a reader can see it is small rather than be told so

A conclusion that survives this is one that did not depend on which NLTK version tagged
the corpus, which is the only reason to trust it in the first place.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Settings whose features changed under the rebuild. The rest reproduce to a cosine above
#: 0.9999 and their original artifacts stand.
REBUILT_SETTINGS = (
    "2013/synBetter",
    "2014Essay2/synBetter",
    "2014Novel2/synBetter",
    "2015/synBetter",
)

CONTROL = "uniform-random"


@dataclass
class SettingShift:
    """How one setting's result moved when its features were rebuilt."""

    setting: str
    before: dict[str, float]
    after: dict[str, float]

    @property
    def ordering_before(self) -> list[str]:
        return sorted(self.before, key=lambda k: self.before[k], reverse=True)

    @property
    def ordering_after(self) -> list[str]:
        return sorted(self.after, key=lambda k: self.after[k], reverse=True)

    @property
    def ordering_held(self) -> bool:
        return self.ordering_before == self.ordering_after

    @property
    def search_beat_random_before(self) -> bool:
        return max(v for k, v in self.before.items() if k != CONTROL) > self.before[CONTROL]

    @property
    def search_beat_random_after(self) -> bool:
        return max(v for k, v in self.after.items() if k != CONTROL) > self.after[CONTROL]

    @property
    def verdict_held(self) -> bool:
        """The claim that matters: did a search beat random, and does it still?"""
        return self.search_beat_random_before == self.search_beat_random_after

    @property
    def largest_shift(self) -> float:
        return max(abs(self.after[k] - self.before[k]) for k in self.before if k in self.after)

    def as_dict(self) -> dict[str, Any]:
        return {
            "setting": self.setting,
            "before": {k: round(v, 4) for k, v in self.before.items()},
            "after": {k: round(v, 4) for k, v in self.after.items()},
            "ordering_before": self.ordering_before,
            "ordering_after": self.ordering_after,
            "ordering_held": self.ordering_held,
            "search_beat_random_before": self.search_beat_random_before,
            "search_beat_random_after": self.search_beat_random_after,
            "verdict_held": self.verdict_held,
            "largest_shift": round(self.largest_shift, 4),
        }


def _medians(directory: Path, metric: str = "test_roc_auc") -> dict[str, dict[str, float]]:
    """Median test score per (setting, strategy), selected on validation."""
    from .comparison import group, load_outcomes

    out: dict[str, dict[str, float]] = {}
    for (setting, strategy), entry in group(load_outcomes(directory, metric=metric)).items():
        out.setdefault(setting, {})[strategy] = entry.median_test
    return out


def compare_rebuild(
    original: Path, rebuilt: Path, settings: Iterable[str] = REBUILT_SETTINGS
) -> dict[str, Any]:
    """Compare the affected settings before and after the feature rebuild."""
    before, after = _medians(original), _medians(rebuilt)
    shifts: list[SettingShift] = []
    missing: list[str] = []
    for setting in settings:
        if setting not in before or setting not in after:
            missing.append(setting)
            continue
        shared = set(before[setting]) & set(after[setting])
        if not shared:
            missing.append(setting)
            continue
        shifts.append(
            SettingShift(
                setting=setting,
                before={k: before[setting][k] for k in shared},
                after={k: after[setting][k] for k in shared},
            )
        )

    held = [s for s in shifts if s.verdict_held]
    return {
        "settings_compared": len(shifts),
        "settings_missing": missing,
        "verdict_held": len(held),
        "verdict_changed": [s.setting for s in shifts if not s.verdict_held],
        "ordering_held": sum(1 for s in shifts if s.ordering_held),
        "median_largest_shift": round(
            statistics.median([s.largest_shift for s in shifts]), 4
        )
        if shifts
        else None,
        "max_shift": round(max((s.largest_shift for s in shifts), default=0.0), 4),
        "conclusion": (
            "the rebuild changes scores and not conclusions"
            if shifts and len(held) == len(shifts)
            else "the rebuild changes at least one conclusion; it cannot be treated as cosmetic"
        ),
        "what_is_compared": (
            "per-unit scores necessarily differ because the features differ; what is "
            "checked is whether a search still beats uniform random where it did before, "
            "and how far the numbers moved"
        ),
        "settings": [s.as_dict() for s in shifts],
    }


def write_report(payload: dict[str, Any], name: str = "rebuild/comparison.json") -> str:
    from .artifacts import write_json

    return str(write_json(name, json.loads(json.dumps(payload))))
