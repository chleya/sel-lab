# -*- coding: utf-8 -*-
"""
Adversarial selector benchmark on an interference task family with digits-like nonnegative embedded inputs.
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


TASK_SUITE = "feature_shift_embedded"
TARGET_JSON = resolve_canonical_results_path("phase3_selector_adversarial_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_SELECTOR_ADVERSARIAL_BENCHMARK.md"


def build_config() -> AblationConfig:
    return replace(
        AblationConfig(),
        task_suite=TASK_SUITE,
        input_size=64,
        hidden_size=32,
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


def run_benchmark(selector_weights: Dict[str, float] | None = None) -> Dict:
    selector_weights = selector_weights or {}
    config = replace(
        build_config(),
        selector_fit_conflict_weight=selector_weights.get("conflict_score", 0.0),
        selector_fit_conflict_peak_weight=selector_weights.get("conflict_peak", 0.0),
        selector_fit_confidence_weight=selector_weights.get("confidence", 0.0),
        selector_fit_abs_mean_weight=selector_weights.get("input_abs_mean", 0.0),
        selector_fit_nonnegative_weight=selector_weights.get("input_nonnegative_ratio", 0.0),
        selector_fit_zero_ratio_weight=selector_weights.get("input_zero_ratio", 0.0),
        selector_fit_bias=selector_weights.get("bias", 0.0),
    )
    policies = (
        "adapt_only",
        "task_specialist_clone_limited_merge",
        "task_specialist_clone_current_path_boost_merge",
        "task_specialist_clone_regime_switch_merge",
        "task_specialist_clone_diagnostic_selector_merge",
        "task_specialist_clone_fingerprint_selector_merge",
        "task_specialist_clone_signature_selector_merge",
        "task_specialist_clone_fitted_selector_merge",
    )
    result = run_phase3_ablations(
        config=config,
        result_filename="phase3_selector_adversarial_feature_shift_embedded.json",
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
            "selector_conflict_threshold": config.selector_conflict_threshold,
            "selector_confidence_threshold": config.selector_confidence_threshold,
            "selector_input_nonnegative_threshold": config.selector_input_nonnegative_threshold,
            "selector_input_abs_mean_threshold": config.selector_input_abs_mean_threshold,
        },
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Selector Adversarial Benchmark",
        "",
        "Adversarial selector benchmark on an interference task encoded as digits-like nonnegative inputs.",
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
    print(f"Saved selector adversarial benchmark JSON: {TARGET_JSON}")
    print(f"Saved selector adversarial benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
