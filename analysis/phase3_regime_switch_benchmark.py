# -*- coding: utf-8 -*-
"""
Benchmark an explicit regime-switch policy on the formal Phase 3 stress map.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase3_benchmark_family import run_policy_benchmark_family
from core.runtime import resolve_canonical_results_path, save_json


FAMILY_NAME = "expanded_stress_map"
TARGET_JSON = resolve_canonical_results_path("phase3_regime_switch_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_REGIME_SWITCH_BENCHMARK.md"

def run_benchmark() -> Dict:
    return run_policy_benchmark_family(
        family_name=FAMILY_NAME,
        policies=(
            "adapt_only",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_current_path_boost_merge",
            "task_specialist_clone_regime_switch_merge",
        ),
        result_prefix="phase3_regime_switch",
        title="Phase 3 Regime Switch Benchmark",
    )


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Regime Switch Benchmark",
        "",
        "Explicit regime-selection policy benchmarked on the formal stress-map family.",
        "",
        f"- Family: `{payload['family_name']}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
        "",
        "## Aggregate",
        "",
    ]
    for policy_name, row in payload["aggregate"].items():
        lines.append(
            f"- `{policy_name}`: mean avg delta {row['mean_avg_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"current delta {row['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"prior delta {row['mean_prior_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"forgetting delta {row['mean_forgetting_delta_vs_adapt_only']:+.1%}, "
            f"gain vs fixed {row['mean_avg_accuracy_gain_vs_fixed']:+.1%}, "
            f"wins {row['benchmark_wins']}"
        )
    lines.extend(["", "## Per Benchmark", ""])
    for row in payload["rows"]:
        lines.extend(
            [
                f"### {row['benchmark_name']}",
                "",
                f"- Task suite: `{row['task_suite']}`",
                f"- Intent: {row['benchmark_intent']}",
                f"- Best policy: `{row['best_policy']}`",
                "",
            ]
        )
        for policy_name, summary in row["summaries"].items():
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
    print(f"Saved regime-switch benchmark JSON: {TARGET_JSON}")
    print(f"Saved regime-switch benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
