# -*- coding: utf-8 -*-
"""
Evaluate a dual-path current/memory split on the broader digits benchmark family.
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
from core.runtime import resolve_canonical_results_path, save_json


DIGITS_SUITES = ("digits_pairs", "digits_pairs_noisy", "digits_pairs_permuted")
POLICIES = (
    "task_specialist_clone_limited_merge",
    "task_specialist_clone_current_path_boost_merge",
    "task_specialist_clone_dual_path_boost_merge",
)
TARGET_JSON = resolve_canonical_results_path("phase3_dual_path_boost_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_DUAL_PATH_BOOST_BENCHMARK.md"


def build_config(task_suite: str) -> AblationConfig:
    return replace(
        AblationConfig(),
        input_size=64,
        hidden_size=32,
        task_suite=task_suite,
        specialist_merge_scale=0.02,
        specialist_distill_strength=0.50,
        current_path_lr_boost=2.50,
        specialist_retention_strength=0.12,
        learning_rate=0.02,
        epochs_per_task=35,
    )


def run_benchmark() -> Dict:
    rows: List[Dict] = []
    win_counts = {name: 0 for name in POLICIES}
    for task_suite in DIGITS_SUITES:
        print("\n" + "=" * 60)
        print(f"Phase 3 Dual Path Boost Benchmark: suite={task_suite}")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(task_suite),
            result_filename=f"phase3_dual_path_boost_{task_suite}_results.json",
            policies=("adapt_only",) + POLICIES,
        )
        summaries = {name: result["policies"][name]["summary"] for name in ("adapt_only",) + POLICIES}
        best_name = max(POLICIES, key=lambda name: summaries[name]["avg_accuracy_delta_vs_adapt_only"])
        win_counts[best_name] += 1
        rows.append(
            {
                "task_suite": task_suite,
                "best_policy": best_name,
                "adapt_only": summaries["adapt_only"],
                **{name: summaries[name] for name in POLICIES},
            }
        )

    aggregate = {}
    for policy_name in POLICIES:
        aggregate[policy_name] = {
            "mean_avg_accuracy_delta_vs_adapt_only": float(
                sum(row[policy_name]["avg_accuracy_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_current_accuracy_delta_vs_adapt_only": float(
                sum(row[policy_name]["current_accuracy_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_forgetting_delta_vs_adapt_only": float(
                sum(row[policy_name]["forgetting_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_avg_accuracy_gain_vs_fixed": float(
                sum(row[policy_name]["avg_accuracy_gain_vs_fixed"] for row in rows) / len(rows)
            ),
            "suite_wins": win_counts[policy_name],
        }
    overall_best = max(POLICIES, key=lambda name: aggregate[name]["mean_avg_accuracy_delta_vs_adapt_only"])
    return {
        "task_suites": list(DIGITS_SUITES),
        "current_path_lr_boost": 2.50,
        "specialist_retention_strength": 0.12,
        "policies": list(POLICIES),
        "overall_best_policy": overall_best,
        "aggregate": aggregate,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Dual Path Boost Benchmark",
        "",
        f"- Digits suites: `{', '.join(payload['task_suites'])}`",
        f"- Shared tuned configuration: current-path boost `{payload['current_path_lr_boost']:.2f}`, retention strength `{payload['specialist_retention_strength']:.2f}`",
        f"- Overall best policy by mean avg delta vs adapt_only: `{payload['overall_best_policy']}`",
        "",
        "## Aggregate Comparison",
        "",
    ]
    for policy_name in payload["policies"]:
        aggregate = payload["aggregate"][policy_name]
        lines.append(
            f"- `{policy_name}`: mean avg delta {aggregate['mean_avg_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"mean current delta {aggregate['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"mean forgetting delta {aggregate['mean_forgetting_delta_vs_adapt_only']:+.1%}, "
            f"mean gain vs fixed {aggregate['mean_avg_accuracy_gain_vs_fixed']:+.1%}, "
            f"suite wins {aggregate['suite_wins']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = run_benchmark()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved dual-path boost benchmark JSON: {TARGET_JSON}")
    print(f"Saved dual-path boost benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
