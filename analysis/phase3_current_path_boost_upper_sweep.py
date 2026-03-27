# -*- coding: utf-8 -*-
"""
Upper-band sweep for current-path learning-rate boost on the broader digits benchmark family.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase3_ablation import AblationConfig, run_phase3_ablations
from core.runtime import resolve_exploratory_results_path, save_json


DIGITS_SUITES = ("digits_pairs", "digits_pairs_noisy", "digits_pairs_permuted")
BOOST_VALUES = (2.00, 2.25, 2.50)
TARGET_JSON = resolve_exploratory_results_path("phase3_current_path_boost_upper_sweep.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_CURRENT_PATH_BOOST_UPPER_SWEEP.md"


def build_config(task_suite: str, boost: float) -> AblationConfig:
    return replace(
        AblationConfig(),
        input_size=64,
        hidden_size=32,
        task_suite=task_suite,
        specialist_merge_scale=0.02,
        specialist_distill_strength=0.50,
        current_path_lr_boost=boost,
        learning_rate=0.02,
        epochs_per_task=35,
    )


def run_sweep() -> Dict:
    rows: List[Dict] = []
    for boost in BOOST_VALUES:
        suite_rows: List[Dict] = []
        for task_suite in DIGITS_SUITES:
            print("\n" + "=" * 60)
            print(f"Phase 3 Current Path Boost Upper Sweep: boost={boost:.2f}, suite={task_suite}")
            print("=" * 60)
            result = run_phase3_ablations(
                config=build_config(task_suite, boost),
                result_filename=f"phase3_current_path_boost_upper_{str(boost).replace('.', '_')}_{task_suite}.json",
                policies=(
                    "adapt_only",
                    "task_specialist_clone_limited_merge",
                    "task_specialist_clone_current_path_boost_merge",
                ),
            )
            suite_rows.append(
                {
                    "task_suite": task_suite,
                    "adapt_only": result["policies"]["adapt_only"]["summary"],
                    "task_specialist_clone_limited_merge": result["policies"]["task_specialist_clone_limited_merge"]["summary"],
                    "task_specialist_clone_current_path_boost_merge": result["policies"]["task_specialist_clone_current_path_boost_merge"]["summary"],
                }
            )

        rows.append(
            {
                "current_path_lr_boost": boost,
                "mean_avg_accuracy_delta_vs_adapt_only": float(
                    sum(
                        row["task_specialist_clone_current_path_boost_merge"]["avg_accuracy_delta_vs_adapt_only"]
                        for row in suite_rows
                    )
                    / len(suite_rows)
                ),
                "mean_current_accuracy_delta_vs_adapt_only": float(
                    sum(
                        row["task_specialist_clone_current_path_boost_merge"]["current_accuracy_delta_vs_adapt_only"]
                        for row in suite_rows
                    )
                    / len(suite_rows)
                ),
                "mean_forgetting_delta_vs_adapt_only": float(
                    sum(
                        row["task_specialist_clone_current_path_boost_merge"]["forgetting_delta_vs_adapt_only"]
                        for row in suite_rows
                    )
                    / len(suite_rows)
                ),
                "mean_avg_accuracy_gain_vs_fixed": float(
                    sum(
                        row["task_specialist_clone_current_path_boost_merge"]["avg_accuracy_gain_vs_fixed"]
                        for row in suite_rows
                    )
                    / len(suite_rows)
                ),
                "suite_rows": suite_rows,
            }
        )

    best = max(rows, key=lambda row: row["mean_avg_accuracy_delta_vs_adapt_only"])
    return {
        "task_suites": list(DIGITS_SUITES),
        "boost_values": list(BOOST_VALUES),
        "best": best,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    best = payload["best"]
    lines = [
        "# Phase 3 Current Path Boost Upper Sweep",
        "",
        f"- Digits suites: `{', '.join(payload['task_suites'])}`",
        f"- Best current-path LR boost in upper band: `{best['current_path_lr_boost']:.2f}`",
        f"- Best mean avg delta vs adapt_only: {best['mean_avg_accuracy_delta_vs_adapt_only']:+.1%}",
        f"- Best mean current delta vs adapt_only: {best['mean_current_accuracy_delta_vs_adapt_only']:+.1%}",
        f"- Best mean forgetting delta vs adapt_only: {best['mean_forgetting_delta_vs_adapt_only']:+.1%}",
        f"- Best mean gain vs fixed: {best['mean_avg_accuracy_gain_vs_fixed']:+.1%}",
        "",
        "## Aggregate Sweep",
        "",
        "| Boost | Mean Avg Delta | Mean Current Delta | Mean Forgetting Delta | Mean Gain vs Fixed |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['current_path_lr_boost']:.2f} | {row['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} | "
            f"{row['mean_current_accuracy_delta_vs_adapt_only']:+.1%} | {row['mean_forgetting_delta_vs_adapt_only']:+.1%} | "
            f"{row['mean_avg_accuracy_gain_vs_fixed']:+.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = run_sweep()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved current-path boost upper sweep JSON: {TARGET_JSON}")
    print(f"Saved current-path boost upper sweep markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
