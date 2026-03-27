# -*- coding: utf-8 -*-
"""
Test an explicit dual-mode update rule against plasticity and interference baselines.
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


TASK_SUITES = ("feature_shift_noisy", "digits_pairs_noisy", "digits_pairs_permuted")
POLICIES = (
    "adapt_only",
    "task_specialist_clone_limited_merge",
    "task_specialist_clone_current_path_boost_merge",
    "task_specialist_clone_dual_mode_update_merge",
)
TARGET_JSON = resolve_canonical_results_path("phase3_dual_mode_update_diagnostic.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_DUAL_MODE_UPDATE_DIAGNOSTIC.md"


def build_config(task_suite: str) -> AblationConfig:
    base = replace(
        AblationConfig(),
        task_suite=task_suite,
        learning_rate=0.02,
        epochs_per_task=25,
        runs=3,
        mode_switch_conflict_threshold=0.35,
        mode_switch_min_current_boost=1.10,
        specialist_retention_strength=0.12,
    )
    if task_suite.startswith("digits_pairs"):
        return replace(
            base,
            input_size=64,
            hidden_size=32,
            specialist_merge_scale=0.02,
            specialist_distill_strength=0.50,
            current_path_lr_boost=2.50,
        )
    return replace(
        base,
        specialist_merge_scale=0.05,
        current_path_lr_boost=2.00,
    )


def run_benchmark() -> Dict:
    rows: List[Dict] = []
    win_counts = {name: 0 for name in POLICIES if name != "adapt_only"}
    for task_suite in TASK_SUITES:
        print("\n" + "=" * 60)
        print(f"Phase 3 Dual-Mode Update Diagnostic: suite={task_suite}")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(task_suite),
            result_filename=f"phase3_dual_mode_update_{task_suite}.json",
            policies=POLICIES,
        )
        summaries = {name: result["policies"][name]["summary"] for name in POLICIES}
        candidates = tuple(name for name in POLICIES if name != "adapt_only")
        best_policy = max(candidates, key=lambda name: summaries[name]["avg_accuracy_delta_vs_adapt_only"])
        win_counts[best_policy] += 1
        rows.append({"task_suite": task_suite, "best_policy": best_policy, **summaries})

    aggregate = {}
    for policy_name in POLICIES:
        aggregate[policy_name] = {
            "mean_avg_accuracy_delta_vs_adapt_only": float(
                sum(row[policy_name]["avg_accuracy_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_current_accuracy_delta_vs_adapt_only": float(
                sum(row[policy_name]["current_accuracy_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_prior_accuracy_delta_vs_adapt_only": float(
                sum(row[policy_name]["prior_accuracy_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_forgetting_delta_vs_adapt_only": float(
                sum(row[policy_name]["forgetting_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_avg_accuracy_gain_vs_fixed": float(
                sum(row[policy_name]["avg_accuracy_gain_vs_fixed"] for row in rows) / len(rows)
            ),
            "suite_wins": win_counts.get(policy_name, 0),
        }
    overall_best_policy = max(
        (name for name in POLICIES if name != "adapt_only"),
        key=lambda name: aggregate[name]["mean_avg_accuracy_delta_vs_adapt_only"],
    )
    config = build_config(TASK_SUITES[0])
    return {
        "task_suites": list(TASK_SUITES),
        "policies": list(POLICIES),
        "overall_best_policy": overall_best_policy,
        "aggregate": aggregate,
        "rows": rows,
        "config": {
            "mode_switch_conflict_threshold": config.mode_switch_conflict_threshold,
            "mode_switch_min_current_boost": config.mode_switch_min_current_boost,
            "specialist_retention_strength": config.specialist_retention_strength,
        },
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Dual-Mode Update Diagnostic",
        "",
        "Conflict-aware dual-mode update: downshift current-path boost and switch archived specialists to retention-only when conflict rises.",
        "",
        f"- Task suites: `{', '.join(payload['task_suites'])}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
        f"- Config: threshold `{payload['config']['mode_switch_conflict_threshold']:.2f}`, min boost `{payload['config']['mode_switch_min_current_boost']:.2f}`, retention `{payload['config']['specialist_retention_strength']:.2f}`",
        "",
        "## Aggregate",
        "",
    ]
    for policy_name in payload["policies"]:
        row = payload["aggregate"][policy_name]
        lines.append(
            f"- `{policy_name}`: avg delta {row['mean_avg_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"current delta {row['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"prior delta {row['mean_prior_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"forgetting delta {row['mean_forgetting_delta_vs_adapt_only']:+.1%}, "
            f"gain vs fixed {row['mean_avg_accuracy_gain_vs_fixed']:+.1%}, "
            f"suite wins {row['suite_wins']}"
        )
    lines.extend(["", "## Per Suite", ""])
    for row in payload["rows"]:
        lines.append(f"### {row['task_suite']}")
        lines.append("")
        lines.append(f"- Best policy: `{row['best_policy']}`")
        for policy_name in payload["policies"]:
            summary = row[policy_name]
            lines.append(
                f"- `{policy_name}`: avg delta {summary['avg_accuracy_delta_vs_adapt_only']:+.1%}, "
                f"current delta {summary['current_accuracy_delta_vs_adapt_only']:+.1%}, "
                f"prior delta {summary['prior_accuracy_delta_vs_adapt_only']:+.1%}, "
                f"forgetting delta {summary['forgetting_delta_vs_adapt_only']:+.1%}, "
                f"gain vs fixed {summary['avg_accuracy_gain_vs_fixed']:+.1%}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = run_benchmark()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved dual-mode update diagnostic JSON: {TARGET_JSON}")
    print(f"Saved dual-mode update diagnostic markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
