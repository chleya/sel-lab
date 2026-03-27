# -*- coding: utf-8 -*-
"""
Fit a linear selector over task-signature features and benchmark it against existing selector baselines.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase3_ablation import AblationConfig, run_phase3_ablations
from core.phase3_common import create_task
from core.phase3_model import ReusePolicyNetwork
from core.phase3_registry import resolve_phase3_diagnostic_benchmark, resolve_phase3_diagnostic_family
from core.runtime import resolve_canonical_results_path, save_json
from analysis.phase3_benchmark_family import aggregate_policy_summaries, build_stress_benchmark_config


FAMILY_NAME = "expanded_stress_map"
TARGET_JSON = resolve_canonical_results_path("phase3_fitted_selector_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_FITTED_SELECTOR_BENCHMARK.md"
PROBE_POLICY = "task_specialist_clone_current_path_boost_merge"


def build_config(task_suite: str, selector_weights: Dict[str, float] | None = None) -> AblationConfig:
    selector_weights = selector_weights or {}
    base = build_stress_benchmark_config(
        task_suite,
        selector_conflict_threshold=0.30,
        selector_confidence_threshold=0.22,
        selector_input_nonnegative_threshold=0.85,
        selector_input_abs_mean_threshold=0.80,
        selector_input_zero_ratio_threshold=0.55,
        selector_conflict_peak_scale=1.10,
        selector_fit_conflict_weight=selector_weights.get("conflict_score", 0.0),
        selector_fit_conflict_peak_weight=selector_weights.get("conflict_peak", 0.0),
        selector_fit_confidence_weight=selector_weights.get("confidence", 0.0),
        selector_fit_abs_mean_weight=selector_weights.get("input_abs_mean", 0.0),
        selector_fit_nonnegative_weight=selector_weights.get("input_nonnegative_ratio", 0.0),
        selector_fit_zero_ratio_weight=selector_weights.get("input_zero_ratio", 0.0),
        selector_fit_bias=selector_weights.get("bias", 0.0),
    )
    return base


def regime_label_for_task_suite(task_suite: str) -> float:
    if task_suite.startswith("digits_pairs"):
        return -1.0
    return 1.0


def collect_signature_examples(task_suite: str, runs: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    config = build_config(task_suite)
    feature_rows: List[np.ndarray] = []
    labels: List[float] = []
    for run in range(runs):
        seed = run * 100 + 42
        learner = ReusePolicyNetwork(config, policy=PROBE_POLICY, seed=seed)
        for task_id in range(config.num_tasks):
            X, y = create_task(task_id, task_suite=config.task_suite)
            for epoch in range(config.epochs_per_task):
                for i in range(len(X)):
                    learner.learn(X[i], y[i], lr=config.learning_rate)
                if (epoch + 1) % config.evolve_interval == 0:
                    learner.evolve()
            feature_rows.append(learner._task_signature_feature_vector().copy())
            labels.append(regime_label_for_task_suite(task_suite))
            learner.reset_after_task()
    return np.vstack(feature_rows), np.array(labels, dtype=float)


def fit_selector_weights() -> Dict[str, float]:
    benchmark_names = resolve_phase3_diagnostic_family(FAMILY_NAME)
    feature_blocks = []
    label_blocks = []
    for benchmark_name in benchmark_names:
        task_suite = str(resolve_phase3_diagnostic_benchmark(benchmark_name)["task_suite"])
        features, labels = collect_signature_examples(task_suite)
        feature_blocks.append(features)
        label_blocks.append(labels)
    X = np.vstack(feature_blocks)
    y = np.concatenate(label_blocks)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std <= 1e-8, 1.0, X_std)
    X_norm = (X - X_mean) / X_std
    X_aug = np.concatenate([X_norm, np.ones((len(X_norm), 1))], axis=1)
    solution = np.linalg.pinv(X_aug) @ y
    scaled_weights = solution[:-1] / X_std
    bias = float(solution[-1] - np.dot(scaled_weights, X_mean))
    return {
        "conflict_score": float(scaled_weights[0]),
        "conflict_peak": float(scaled_weights[1]),
        "confidence": float(scaled_weights[2]),
        "input_abs_mean": float(scaled_weights[3]),
        "input_nonnegative_ratio": float(scaled_weights[4]),
        "input_zero_ratio": float(scaled_weights[5]),
        "bias": bias,
    }


def run_benchmark() -> Dict:
    selector_weights = fit_selector_weights()
    benchmark_names = resolve_phase3_diagnostic_family(FAMILY_NAME)
    rows: List[Dict] = []
    policy_win_counts: Dict[str, int] = {}
    policies = (
        "adapt_only",
        "task_specialist_clone_limited_merge",
        "task_specialist_clone_current_path_boost_merge",
        "task_specialist_clone_regime_switch_merge",
        "task_specialist_clone_fingerprint_selector_merge",
        "task_specialist_clone_signature_selector_merge",
        "task_specialist_clone_fitted_selector_merge",
    )
    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        print("\n" + "=" * 60)
        print(f"Phase 3 Fitted Selector Benchmark: {benchmark_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(task_suite, selector_weights=selector_weights),
            result_filename=f"phase3_fitted_selector_{benchmark_name}_{task_suite}.json",
            policies=policies,
        )
        summaries = {name: result["policies"][name]["summary"] for name in policies}
        candidates = tuple(name for name in policies if name != "adapt_only")
        best_policy = max(candidates, key=lambda name: summaries[name]["avg_accuracy_delta_vs_adapt_only"])
        policy_win_counts[best_policy] = policy_win_counts.get(best_policy, 0) + 1
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "task_suite": task_suite,
                "benchmark_intent": spec["benchmark_intent"],
                "best_policy": best_policy,
                "summaries": summaries,
            }
        )

    aggregate = aggregate_policy_summaries(rows, policies)
    for policy_name, row in aggregate.items():
        row["benchmark_wins"] = policy_win_counts.get(policy_name, 0)

    overall_best_policy = max(
        (name for name in aggregate if name != "adapt_only"),
        key=lambda name: aggregate[name]["mean_avg_accuracy_delta_vs_adapt_only"],
    )
    return {
        "family_name": FAMILY_NAME,
        "probe_policy": PROBE_POLICY,
        "selector_weights": selector_weights,
        "overall_best_policy": overall_best_policy,
        "policy_win_counts": policy_win_counts,
        "aggregate": aggregate,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    weight_parts = ", ".join(
        f"{name}={value:+.3f}" for name, value in payload["selector_weights"].items()
    )
    lines = [
        "# Phase 3 Fitted Selector Benchmark",
        "",
        "Linear fitted selector over task-signature features benchmarked against existing selector baselines.",
        "",
        f"- Family: `{payload['family_name']}`",
        f"- Probe policy for feature collection: `{payload['probe_policy']}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
        f"- Fitted weights: {weight_parts}",
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
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = run_benchmark()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved fitted selector benchmark JSON: {TARGET_JSON}")
    print(f"Saved fitted selector benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
