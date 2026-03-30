# -*- coding: utf-8 -*-
"""
Fit an outcome-supervised selector over richer online regime features and benchmark it on the full selector stress map.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase3_benchmark_family import aggregate_policy_summaries, build_stress_benchmark_config
from core.phase3_ablation import run_phase3_ablations
from core.phase3_common import create_task
from core.phase3_model import ReusePolicyNetwork, TASK_REGIME_FEATURE_NAMES
from core.phase3_registry import resolve_phase3_diagnostic_benchmark, resolve_phase3_diagnostic_family
from core.runtime import resolve_canonical_results_path, save_json


TRAIN_FAMILY_NAME = "expanded_stress_map"
EVAL_FAMILY_NAME = "selector_full_map"
TARGET_JSON = resolve_canonical_results_path("phase3_outcome_selector_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_OUTCOME_SELECTOR_BENCHMARK.md"
PROBE_POLICY = "task_specialist_clone_current_path_boost_merge"
COMPETING_POLICIES = (
    "task_specialist_clone_limited_merge",
    "task_specialist_clone_current_path_boost_merge",
)


def build_config(task_suite: str, selector_payload: Dict[str, object] | None = None):
    selector_payload = selector_payload or {}
    return build_stress_benchmark_config(
        task_suite,
        selector_outcome_fit_weights=tuple(
            selector_payload.get("weights", (0.0,) * len(TASK_REGIME_FEATURE_NAMES))
        ),
        selector_outcome_fit_bias=float(selector_payload.get("bias", 0.0)),
    )


def collect_regime_examples(task_suite: str, runs: int = 3) -> np.ndarray:
    config = build_config(task_suite)
    feature_rows: List[np.ndarray] = []
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
            feature_rows.append(learner._task_regime_feature_vector().copy())
            learner.reset_after_task()
    return np.vstack(feature_rows)


def benchmark_margin_for_task_suite(task_suite: str) -> float:
    result = run_phase3_ablations(
        config=build_stress_benchmark_config(task_suite),
        result_filename=f"phase3_outcome_selector_supervision_{task_suite}.json",
        policies=("adapt_only",) + COMPETING_POLICIES,
    )
    summaries = {name: result["policies"][name]["summary"] for name in COMPETING_POLICIES}
    limited = summaries["task_specialist_clone_limited_merge"]
    boost = summaries["task_specialist_clone_current_path_boost_merge"]
    margin = (
        (limited["avg_accuracy_gain_vs_fixed"] - boost["avg_accuracy_gain_vs_fixed"])
        + 0.5 * (limited["prior_accuracy_delta_vs_adapt_only"] - boost["prior_accuracy_delta_vs_adapt_only"])
        + 0.25 * (limited["forgetting_delta_vs_adapt_only"] - boost["forgetting_delta_vs_adapt_only"])
        - 0.25 * (limited["current_accuracy_delta_vs_adapt_only"] - boost["current_accuracy_delta_vs_adapt_only"])
    )
    return float(margin)


def fit_selector_payload() -> Dict[str, object]:
    benchmark_names = resolve_phase3_diagnostic_family(TRAIN_FAMILY_NAME)
    feature_blocks = []
    target_blocks = []
    supervision_rows = []
    for benchmark_name in benchmark_names:
        task_suite = str(resolve_phase3_diagnostic_benchmark(benchmark_name)["task_suite"])
        features = collect_regime_examples(task_suite)
        target = benchmark_margin_for_task_suite(task_suite)
        feature_blocks.append(features)
        target_blocks.append(np.full(len(features), target, dtype=float))
        supervision_rows.append(
            {
                "benchmark_name": benchmark_name,
                "task_suite": task_suite,
                "target_margin": target,
            }
        )
    X = np.vstack(feature_blocks)
    y = np.concatenate(target_blocks)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std <= 1e-8, 1.0, X_std)
    X_norm = (X - X_mean) / X_std
    X_aug = np.concatenate([X_norm, np.ones((len(X_norm), 1))], axis=1)
    ridge = 1e-3
    solution = np.linalg.solve(
        (X_aug.T @ X_aug) + ridge * np.eye(X_aug.shape[1]),
        X_aug.T @ y,
    )
    scaled_weights = solution[:-1] / X_std
    bias = float(solution[-1] - np.dot(scaled_weights, X_mean))
    return {
        "feature_names": list(TASK_REGIME_FEATURE_NAMES),
        "weights": tuple(float(value) for value in scaled_weights),
        "bias": bias,
        "supervision_rows": supervision_rows,
    }


def run_benchmark() -> Dict:
    selector_payload = fit_selector_payload()
    benchmark_names = resolve_phase3_diagnostic_family(EVAL_FAMILY_NAME)
    rows: List[Dict] = []
    policy_win_counts: Dict[str, int] = {}
    policies = (
        "adapt_only",
        "task_specialist_clone_limited_merge",
        "task_specialist_clone_current_path_boost_merge",
        "task_specialist_clone_regime_switch_merge",
        "task_specialist_clone_fingerprint_selector_merge",
        "task_specialist_clone_signature_selector_merge",
        "task_specialist_clone_dynamics_selector_merge",
        "task_specialist_clone_outcome_selector_merge",
    )
    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        print("\n" + "=" * 60)
        print(f"Phase 3 Outcome Selector Benchmark: {benchmark_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(task_suite, selector_payload=selector_payload),
            result_filename=f"phase3_outcome_selector_{benchmark_name}_{task_suite}.json",
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
                "policies": list(policies),
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
        "train_family_name": TRAIN_FAMILY_NAME,
        "eval_family_name": EVAL_FAMILY_NAME,
        "probe_policy": PROBE_POLICY,
        "selector_payload": selector_payload,
        "overall_best_policy": overall_best_policy,
        "policy_win_counts": policy_win_counts,
        "aggregate": aggregate,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    weight_parts = ", ".join(
        f"{name}={value:+.3f}"
        for name, value in zip(payload["selector_payload"]["feature_names"], payload["selector_payload"]["weights"])
    )
    lines = [
        "# Phase 3 Outcome Selector Benchmark",
        "",
        "Outcome-supervised selector over richer online regime features benchmarked on the full selector stress map.",
        "",
        f"- Train family: `{payload['train_family_name']}`",
        f"- Eval family: `{payload['eval_family_name']}`",
        f"- Probe policy for feature collection: `{payload['probe_policy']}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
        f"- Fitted weights: {weight_parts}, bias={payload['selector_payload']['bias']:+.3f}",
        "",
        "## Outcome Supervision Targets",
        "",
    ]
    for row in payload["selector_payload"]["supervision_rows"]:
        lines.append(
            f"- `{row['benchmark_name']}` / `{row['task_suite']}`: target margin {row['target_margin']:+.3f}"
        )
    lines.extend(["", "## Aggregate", ""])
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
    print(f"Saved outcome selector benchmark JSON: {TARGET_JSON}")
    print(f"Saved outcome selector benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
