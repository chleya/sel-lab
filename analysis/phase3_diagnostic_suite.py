# -*- coding: utf-8 -*-
"""
Run a compact Phase 3 diagnostic suite that separates plasticity, interference, and retrieval stress.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase3_benchmark_family import build_stress_benchmark_config, build_summary_row
from core.phase3_ablation import run_phase3_ablations
from core.phase3_registry import PHASE3_DIAGNOSTIC_BENCHMARKS, resolve_phase3_diagnostic_benchmark
from core.runtime import resolve_canonical_results_path, save_json


TARGET_JSON = resolve_canonical_results_path("phase3_diagnostic_suite.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_DIAGNOSTIC_SUITE.md"
DIAGNOSTIC_ORDER = (
    "plasticity_stress",
    "interference_stress",
    "retrieval_ambiguity_stress",
)


def run_diagnostics() -> Dict:
    rows: List[Dict] = []
    family_win_counts: Dict[str, int] = {}

    for diagnostic_name in DIAGNOSTIC_ORDER:
        spec = resolve_phase3_diagnostic_benchmark(diagnostic_name)
        task_suite = str(spec["task_suite"])
        policies = tuple(spec["policies"])
        print("\n" + "=" * 60)
        print(f"Phase 3 Diagnostic Suite: {diagnostic_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_stress_benchmark_config(task_suite, epochs_per_task=20),
            result_filename=f"phase3_diagnostic_{diagnostic_name}_{task_suite}.json",
            policies=policies,
        )
        summaries = build_summary_row(result, policies)
        candidate_policies = tuple(policy for policy in policies if policy != "adapt_only")
        best_policy = max(
            candidate_policies,
            key=lambda name: summaries[name]["avg_accuracy_delta_vs_adapt_only"],
        )
        best_family = result["policy_metadata"][best_policy]["family"]
        family_win_counts[best_family] = family_win_counts.get(best_family, 0) + 1
        rows.append(
            {
                "diagnostic_name": diagnostic_name,
                "task_suite": task_suite,
                "benchmark_intent": spec["benchmark_intent"],
                "best_policy": best_policy,
                "best_family": best_family,
                "policies": summaries,
            }
        )

    return {
        "diagnostic_order": list(DIAGNOSTIC_ORDER),
        "diagnostic_specs": {
            name: resolve_phase3_diagnostic_benchmark(name) for name in DIAGNOSTIC_ORDER
        },
        "family_win_counts": family_win_counts,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Diagnostic Suite",
        "",
        "Compact diagnostic suite for separating plasticity, interference, and retrieval pressure.",
        "",
        "## Family Wins",
        "",
    ]
    for family_name, wins in sorted(payload["family_win_counts"].items()):
        lines.append(f"- `{family_name}`: {wins}")
    lines.extend(["", "## Diagnostics", ""])
    for row in payload["rows"]:
        lines.extend(
            [
                f"### {row['diagnostic_name']}",
                "",
                f"- Task suite: `{row['task_suite']}`",
                f"- Intent: {row['benchmark_intent']}",
                f"- Best policy: `{row['best_policy']}`",
                f"- Best family: `{row['best_family']}`",
                "",
            ]
        )
        for policy_name, summary in row["policies"].items():
            lines.append(
                f"- `{policy_name}`: avg delta vs adapt_only {summary['avg_accuracy_delta_vs_adapt_only']:+.1%}, "
                f"current delta {summary['current_accuracy_delta_vs_adapt_only']:+.1%}, "
                f"prior delta {summary['prior_accuracy_delta_vs_adapt_only']:+.1%}, "
                f"forgetting delta {summary['forgetting_delta_vs_adapt_only']:+.1%}, "
                f"gain vs fixed {summary['avg_accuracy_gain_vs_fixed']:+.1%}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = run_diagnostics()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved diagnostic suite JSON: {TARGET_JSON}")
    print(f"Saved diagnostic suite markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
