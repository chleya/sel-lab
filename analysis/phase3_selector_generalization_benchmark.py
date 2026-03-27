# -*- coding: utf-8 -*-
"""
Stress-test selector generalization on an interference benchmark with digits-like nonnegative inputs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase3_ablation import AblationConfig, run_phase3_ablations
from core.runtime import resolve_canonical_results_path, save_json


TASK_SUITE = "feature_shift_nonnegative"
TARGET_JSON = resolve_canonical_results_path("phase3_selector_generalization_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_SELECTOR_GENERALIZATION_BENCHMARK.md"


def build_config() -> AblationConfig:
    return replace(
        AblationConfig(),
        task_suite=TASK_SUITE,
        learning_rate=0.02,
        epochs_per_task=25,
        runs=3,
        specialist_merge_scale=0.05,
        specialist_distill_strength=0.35,
        current_path_lr_boost=2.00,
        selector_conflict_threshold=0.30,
        selector_confidence_threshold=0.22,
        selector_input_nonnegative_threshold=0.85,
        selector_input_abs_mean_threshold=0.80,
    )


def run_benchmark() -> Dict:
    policies = (
        "adapt_only",
        "task_specialist_clone_limited_merge",
        "task_specialist_clone_current_path_boost_merge",
        "task_specialist_clone_regime_switch_merge",
        "task_specialist_clone_diagnostic_selector_merge",
        "task_specialist_clone_fingerprint_selector_merge",
        "task_specialist_clone_signature_selector_merge",
    )
    result = run_phase3_ablations(
        config=build_config(),
        result_filename="phase3_selector_generalization_feature_shift_nonnegative.json",
        policies=policies,
    )
    summaries = {name: result["policies"][name]["summary"] for name in policies}
    best_policy = max(
        (name for name in policies if name != "adapt_only"),
        key=lambda name: summaries[name]["avg_accuracy_delta_vs_adapt_only"],
    )
    return {
        "task_suite": TASK_SUITE,
        "best_policy": best_policy,
        "policies": summaries,
        "selector_config": {
            "selector_conflict_threshold": build_config().selector_conflict_threshold,
            "selector_confidence_threshold": build_config().selector_confidence_threshold,
            "selector_input_nonnegative_threshold": build_config().selector_input_nonnegative_threshold,
            "selector_input_abs_mean_threshold": build_config().selector_input_abs_mean_threshold,
        },
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Selector Generalization Benchmark",
        "",
        "Generalization check on an interference-style benchmark with nonnegative, digits-like inputs.",
        "",
        f"- Task suite: `{payload['task_suite']}`",
        f"- Best policy: `{payload['best_policy']}`",
        f"- Scalar selector thresholds: conflict `{payload['selector_config']['selector_conflict_threshold']:.2f}`, confidence `{payload['selector_config']['selector_confidence_threshold']:.2f}`",
        f"- Fingerprint thresholds: nonnegative ratio `{payload['selector_config']['selector_input_nonnegative_threshold']:.2f}`, abs-mean `{payload['selector_config']['selector_input_abs_mean_threshold']:.2f}`",
        "",
        "## Policies",
        "",
    ]
    for policy_name, summary in payload["policies"].items():
        lines.append(
            f"- `{policy_name}`: avg delta {summary['avg_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"current delta {summary['current_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"prior delta {summary['prior_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"forgetting delta {summary['forgetting_delta_vs_adapt_only']:+.1%}, "
            f"gain vs fixed {summary['avg_accuracy_gain_vs_fixed']:+.1%}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = run_benchmark()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved selector generalization benchmark JSON: {TARGET_JSON}")
    print(f"Saved selector generalization benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
