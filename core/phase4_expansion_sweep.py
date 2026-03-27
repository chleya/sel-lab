# -*- coding: utf-8 -*-
"""
Sweep Phase 4 expansion parameters after fixing age decay.
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


def run_phase4_expansion_sweep() -> Dict:
    thresholds = [0.30, 0.15, 0.10, 0.08, 0.05]
    initial_tensions = [0.5, 0.75, 1.0]
    results: List[Dict] = []

    for threshold in thresholds:
        for initial_tension in initial_tensions:
            payload = analyze_phase4_failure(
                epochs=15,
                seed=42,
                evolution_threshold=threshold,
                age_decay=0.005,
                initial_tension=initial_tension,
            )
            findings = payload["findings"]
            results.append(
                {
                    "threshold": threshold,
                    "initial_tension": initial_tension,
                    "triggered_expansion": findings["triggered_expansion"],
                    "expansion_epochs": findings["expansion_epochs"],
                    "final_accuracy_gap": findings["final_accuracy_gap"],
                    "mean_accuracy_gap": findings["mean_accuracy_gap"],
                    "mean_tension": findings["mean_tension"],
                    "max_tension": findings["max_tension"],
                    "dominant_failure_mode": findings["dominant_failure_mode"],
                }
            )

    best_gap = min(results, key=lambda row: row["final_accuracy_gap"])
    best_expanding = min(
        [row for row in results if row["triggered_expansion"]],
        key=lambda row: row["final_accuracy_gap"],
        default=None,
    )
    summary = {"results": results, "best_gap": best_gap, "best_expanding": best_expanding}
    target = resolve_canonical_results_path("phase4_expansion_sweep.json")
    save_json(summary, target)
    print(f"Saved expansion sweep: {target}")
    return summary


def main():
    summary = run_phase4_expansion_sweep()
    print("\nPhase 4 Expansion Sweep")
    for row in summary["results"]:
        print(
            f"  thr={row['threshold']:.2f}, initT={row['initial_tension']:.2f}: "
            f"gap={row['final_accuracy_gap']:+.1%}, "
            f"expanded={row['triggered_expansion']}, "
            f"mode={row['dominant_failure_mode']}"
        )

    best_gap = summary["best_gap"]
    print(
        f"\nBest overall: thr={best_gap['threshold']:.2f}, "
        f"initT={best_gap['initial_tension']:.2f}, gap={best_gap['final_accuracy_gap']:+.1%}"
    )
    if summary["best_expanding"] is not None:
        best_expanding = summary["best_expanding"]
        print(
            f"Best expanding setup: thr={best_expanding['threshold']:.2f}, "
            f"initT={best_expanding['initial_tension']:.2f}, "
            f"gap={best_expanding['final_accuracy_gap']:+.1%}"
        )
    else:
        print("Best expanding setup: none triggered expansion")


if __name__ == "__main__":
    main()
