# -*- coding: utf-8 -*-
"""
Formal Phase 3 diagnostic benchmark family for screening mechanisms across stress types.
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
TARGET_JSON = resolve_canonical_results_path("phase3_diagnostic_benchmark_family.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_DIAGNOSTIC_BENCHMARK_FAMILY.md"

def _build_row(
    benchmark_name: str,
    task_suite: str,
    spec: Dict[str, object],
    summaries: Dict[str, Dict[str, float]],
    best_policy: str,
    result: Dict,
) -> Dict:
    return {
        "benchmark_name": benchmark_name,
        "task_suite": task_suite,
        "benchmark_intent": spec["benchmark_intent"],
        "policies": list(spec["policies"]),
        "best_policy": best_policy,
        "best_family": result["policy_metadata"][best_policy]["family"],
        "summaries": summaries,
    }


def run_family() -> Dict:
    payload = run_policy_benchmark_family(
        family_name=FAMILY_NAME,
        policies=None,
        result_prefix="phase3_diag_family",
        title="Phase 3 Diagnostic Benchmark Family",
        config_overrides={
            "mode_switch_conflict_threshold": 0.35,
            "mode_switch_min_current_boost": 1.10,
            "specialist_retention_strength": 0.12,
        },
        row_builder=_build_row,
    )
    family_win_counts: Dict[str, int] = {}
    for row in payload["rows"]:
        best_family = row["best_family"]
        family_win_counts[best_family] = family_win_counts.get(best_family, 0) + 1
    payload["family_win_counts"] = family_win_counts
    return payload


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Diagnostic Benchmark Family",
        "",
        "Formal stress-map benchmark family for screening Phase 3 mechanisms across plasticity, interference, and retrieval pressure.",
        "",
        f"- Family: `{payload['family_name']}`",
        f"- Benchmarks: `{', '.join(payload['benchmark_names'])}`",
        "",
        "## Policy Wins",
        "",
    ]
    for policy_name, wins in sorted(payload["policy_win_counts"].items()):
        lines.append(f"- `{policy_name}`: {wins}")
    lines.extend(["", "## Family Wins", ""])
    for family_name, wins in sorted(payload["family_win_counts"].items()):
        lines.append(f"- `{family_name}`: {wins}")
    lines.extend(["", "## Aggregate", ""])
    for policy_name, row in sorted(payload["aggregate"].items()):
        lines.append(
            f"- `{policy_name}`: mean avg delta {row['mean_avg_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"current delta {row['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"prior delta {row['mean_prior_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"forgetting delta {row['mean_forgetting_delta_vs_adapt_only']:+.1%}, "
            f"gain vs fixed {row['mean_avg_accuracy_gain_vs_fixed']:+.1%}, "
            f"wins {row['benchmark_wins']}/{row['benchmarks_present']}"
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
                f"- Best family: `{row['best_family']}`",
                "",
            ]
        )
        for policy_name in row["policies"]:
            summary = row["summaries"][policy_name]
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
    payload = run_family()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved diagnostic benchmark family JSON: {TARGET_JSON}")
    print(f"Saved diagnostic benchmark family markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
