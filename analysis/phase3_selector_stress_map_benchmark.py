# -*- coding: utf-8 -*-
"""
Formal selector stress-map benchmark that combines the main stress family with selector-specific gates.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase3_benchmark_family import build_stress_benchmark_config, run_policy_benchmark_family
from core.runtime import resolve_canonical_results_path, save_json


FAMILY_NAME = "selector_full_map"
TARGET_JSON = resolve_canonical_results_path("phase3_selector_stress_map_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_SELECTOR_STRESS_MAP_BENCHMARK.md"


def run_benchmark() -> Dict:
    payload = run_policy_benchmark_family(
        family_name=FAMILY_NAME,
        policies=(
            "adapt_only",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_current_path_boost_merge",
            "task_specialist_clone_regime_switch_merge",
            "task_specialist_clone_diagnostic_selector_merge",
            "task_specialist_clone_fingerprint_selector_merge",
            "task_specialist_clone_signature_selector_merge",
            "task_specialist_clone_fitted_selector_merge",
        ),
        result_prefix="phase3_selector_stress_map",
        title="Phase 3 Selector Stress Map Benchmark",
        config_overrides={
            "selector_conflict_threshold": 0.30,
            "selector_confidence_threshold": 0.22,
            "selector_input_nonnegative_threshold": 0.85,
            "selector_input_abs_mean_threshold": 0.80,
            "selector_input_zero_ratio_threshold": 0.55,
            "selector_conflict_peak_scale": 1.10,
        },
    )
    selector_config = build_stress_benchmark_config(
        "feature_shift",
        selector_conflict_threshold=0.30,
        selector_confidence_threshold=0.22,
        selector_input_nonnegative_threshold=0.85,
        selector_input_abs_mean_threshold=0.80,
        selector_input_zero_ratio_threshold=0.55,
        selector_conflict_peak_scale=1.10,
    )
    payload["selector_config"] = {
        "selector_conflict_threshold": selector_config.selector_conflict_threshold,
        "selector_confidence_threshold": selector_config.selector_confidence_threshold,
        "selector_input_nonnegative_threshold": selector_config.selector_input_nonnegative_threshold,
        "selector_input_abs_mean_threshold": selector_config.selector_input_abs_mean_threshold,
        "selector_input_zero_ratio_threshold": selector_config.selector_input_zero_ratio_threshold,
        "selector_conflict_peak_scale": selector_config.selector_conflict_peak_scale,
    }
    return payload


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Selector Stress Map Benchmark",
        "",
        "Formal selector stress-map benchmark combining the main stress family with nonnegative and adversarial selector gates.",
        "",
        f"- Family: `{payload['family_name']}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
        (
            f"- Selector config: conflict `{payload['selector_config']['selector_conflict_threshold']:.2f}`, "
            f"confidence `{payload['selector_config']['selector_confidence_threshold']:.2f}`, "
            f"nonnegative ratio `{payload['selector_config']['selector_input_nonnegative_threshold']:.2f}`, "
            f"abs-mean `{payload['selector_config']['selector_input_abs_mean_threshold']:.2f}`, "
            f"zero ratio `{payload['selector_config']['selector_input_zero_ratio_threshold']:.2f}`, "
            f"conflict peak scale `{payload['selector_config']['selector_conflict_peak_scale']:.2f}`"
        ),
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
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = run_benchmark()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved selector stress-map benchmark JSON: {TARGET_JSON}")
    print(f"Saved selector stress-map benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
