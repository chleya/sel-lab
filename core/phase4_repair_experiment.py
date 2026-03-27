# -*- coding: utf-8 -*-
"""
Targeted repair sweep for Phase 4 evolution failure modes.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase4_failure_analysis import analyze_phase4_failure
from core.runtime import resolve_canonical_results_path, save_json


def run_phase4_repair_sweep() -> Dict:
    settings = [
        {"name": "baseline", "threshold": 0.30, "age_decay": 0.05, "initial_tension": 0.5},
        {"name": "lower_threshold", "threshold": 0.10, "age_decay": 0.05, "initial_tension": 0.5},
        {"name": "slower_decay", "threshold": 0.30, "age_decay": 0.005, "initial_tension": 0.5},
        {"name": "both_adjusted", "threshold": 0.10, "age_decay": 0.005, "initial_tension": 0.5},
        {"name": "primed_evolution", "threshold": 0.10, "age_decay": 0.005, "initial_tension": 1.0},
    ]

    results: List[Dict] = []
    for setting in settings:
        payload = analyze_phase4_failure(
            epochs=15,
            seed=42,
            evolution_threshold=setting["threshold"],
            age_decay=setting["age_decay"],
            initial_tension=setting["initial_tension"],
        )
        results.append({"name": setting["name"], **payload["config"], **payload["findings"]})

    best = min(results, key=lambda row: row["final_accuracy_gap"])
    summary = {"results": results, "best": best}
    target = resolve_canonical_results_path("phase4_repair_sweep.json")
    save_json(summary, target)
    print(f"Saved repair sweep: {target}")
    return summary


def main():
    summary = run_phase4_repair_sweep()
    print("\nPhase 4 Repair Sweep")
    for row in summary["results"]:
        print(
            f"  {row['name']}: gap={row['final_accuracy_gap']:+.1%}, "
            f"expanded={row['triggered_expansion']}, "
            f"mean_tension={row['mean_tension']:.3f}, "
            f"lr_end={row['final_effective_lr']:.5f}"
        )
    best = summary["best"]
    print(
        f"\nBest setting: {best['name']} "
        f"(gap={best['final_accuracy_gap']:+.1%}, expanded={best['triggered_expansion']})"
    )


if __name__ == "__main__":
    main()
