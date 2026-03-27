# -*- coding: utf-8 -*-
"""
Estimate the upside of high-level regime switching on the formal Phase 3 stress map.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import load_json_result, resolve_canonical_results_path, save_json


TARGET_JSON = resolve_canonical_results_path("phase3_regime_switch_oracle.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_REGIME_SWITCH_ORACLE.md"

FAMILY_RESULT_FILE = "phase3_diagnostic_benchmark_family.json"

REGIME_POLICY_BY_BENCHMARK = {
    "plasticity_stress": "task_specialist_clone_current_path_boost_merge",
    "interference_stress": "task_specialist_clone_limited_merge",
    "interference_stress_strong": "task_specialist_clone_limited_merge",
    "retrieval_ambiguity_stress": "task_specialist_clone_current_path_boost_merge",
}

BASELINE_POLICIES = (
    "task_specialist_clone_current_path_boost_merge",
    "task_specialist_clone_limited_merge",
    "task_specialist_clone_mode_switch_boost_merge",
    "task_specialist_clone_dual_mode_update_merge",
)


def build_payload() -> Dict:
    family = load_json_result(FAMILY_RESULT_FILE)
    rows = family["rows"]

    aggregate: Dict[str, Dict[str, float]] = {}
    comparison_policies = list(BASELINE_POLICIES) + ["regime_switch_oracle"]
    for policy_name in comparison_policies:
        policy_rows: List[Dict] = []
        for row in rows:
            if policy_name == "regime_switch_oracle":
                chosen = REGIME_POLICY_BY_BENCHMARK[row["benchmark_name"]]
                summary = row["summaries"][chosen]
                policy_rows.append({"benchmark_name": row["benchmark_name"], "policy_name": chosen, "summary": summary})
            elif policy_name in row["summaries"]:
                policy_rows.append({"benchmark_name": row["benchmark_name"], "policy_name": policy_name, "summary": row["summaries"][policy_name]})

        aggregate[policy_name] = {
            "benchmarks_present": len(policy_rows),
            "mean_avg_accuracy_delta_vs_adapt_only": float(
                sum(item["summary"]["avg_accuracy_delta_vs_adapt_only"] for item in policy_rows) / len(policy_rows)
            ),
            "mean_current_accuracy_delta_vs_adapt_only": float(
                sum(item["summary"]["current_accuracy_delta_vs_adapt_only"] for item in policy_rows) / len(policy_rows)
            ),
            "mean_prior_accuracy_delta_vs_adapt_only": float(
                sum(item["summary"]["prior_accuracy_delta_vs_adapt_only"] for item in policy_rows) / len(policy_rows)
            ),
            "mean_forgetting_delta_vs_adapt_only": float(
                sum(item["summary"]["forgetting_delta_vs_adapt_only"] for item in policy_rows) / len(policy_rows)
            ),
            "mean_avg_accuracy_gain_vs_fixed": float(
                sum(item["summary"]["avg_accuracy_gain_vs_fixed"] for item in policy_rows) / len(policy_rows)
            ),
        }

    benchmark_choices = []
    for row in rows:
        chosen_policy = REGIME_POLICY_BY_BENCHMARK[row["benchmark_name"]]
        benchmark_choices.append(
            {
                "benchmark_name": row["benchmark_name"],
                "task_suite": row["task_suite"],
                "chosen_policy": chosen_policy,
                "summary": row["summaries"][chosen_policy],
                "best_policy": row["best_policy"],
            }
        )

    return {
        "source_family": family["family_name"],
        "comparison_policies": comparison_policies,
        "regime_policy_by_benchmark": REGIME_POLICY_BY_BENCHMARK,
        "aggregate": aggregate,
        "benchmark_choices": benchmark_choices,
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Regime Switch Oracle",
        "",
        "Upper-bound check for a high-level regime switch that selects one policy per stress benchmark.",
        "",
        f"- Source family: `{payload['source_family']}`",
        "",
        "## Aggregate",
        "",
    ]
    for policy_name in payload["comparison_policies"]:
        row = payload["aggregate"][policy_name]
        lines.append(
            f"- `{policy_name}`: mean avg delta {row['mean_avg_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"current delta {row['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"prior delta {row['mean_prior_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"forgetting delta {row['mean_forgetting_delta_vs_adapt_only']:+.1%}, "
            f"gain vs fixed {row['mean_avg_accuracy_gain_vs_fixed']:+.1%}"
        )
    lines.extend(["", "## Per Benchmark Choice", ""])
    for row in payload["benchmark_choices"]:
        summary = row["summary"]
        lines.extend(
            [
                f"### {row['benchmark_name']}",
                "",
                f"- Task suite: `{row['task_suite']}`",
                f"- Oracle choice: `{row['chosen_policy']}`",
                f"- Benchmark winner: `{row['best_policy']}`",
                f"- Chosen result: avg delta {summary['avg_accuracy_delta_vs_adapt_only']:+.1%}, current delta {summary['current_accuracy_delta_vs_adapt_only']:+.1%}, prior delta {summary['prior_accuracy_delta_vs_adapt_only']:+.1%}, forgetting delta {summary['forgetting_delta_vs_adapt_only']:+.1%}, gain vs fixed {summary['avg_accuracy_gain_vs_fixed']:+.1%}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    payload = build_payload()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved regime-switch oracle JSON: {TARGET_JSON}")
    print(f"Saved regime-switch oracle markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
