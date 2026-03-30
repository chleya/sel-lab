# -*- coding: utf-8 -*-
"""
Fit a structured pairwise-ranking selector over regime features and benchmark it on the full selector stress map.
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


TRAIN_FAMILY_NAME = "selector_full_map"
EVAL_FAMILY_NAME = "selector_full_map"
TARGET_JSON = resolve_canonical_results_path("phase3_ranking_selector_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_RANKING_SELECTOR_BENCHMARK.md"
PROBE_POLICY = "task_specialist_clone_current_path_boost_merge"
STRUCTURED_SUITE_TARGETS = {
    "plasticity_stress": -3.0,
    "interference_stress": 1.5,
    "interference_stress_strong": 2.0,
    "retrieval_ambiguity_stress": -3.5,
    "mixed_regime_stress": 0.8,
    "selector_generalization_gate": -0.8,
    "selector_adversarial_gate": 2.8,
    "selector_sparse_adversarial_gate": -1.8,
}


def build_config(task_suite: str, selector_payload: Dict[str, object] | None = None):
    selector_payload = selector_payload or {}
    return build_stress_benchmark_config(
        task_suite,
        selector_ranking_fit_weights=tuple(
            selector_payload.get("weights", (0.0,) * len(TASK_REGIME_FEATURE_NAMES))
        ),
        selector_ranking_fit_bias=float(selector_payload.get("bias", 0.0)),
    )


def collect_regime_examples(task_suite: str, runs: int = 1) -> np.ndarray:
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


def fit_pairwise_ranking_weights(
    feature_rows: np.ndarray,
    target_scores: np.ndarray,
) -> np.ndarray:
    weights = np.zeros(feature_rows.shape[1], dtype=float)
    pairs: List[Tuple[int, int, float, float]] = []
    for idx in range(len(feature_rows)):
        for jdx in range(idx + 1, len(feature_rows)):
            delta = target_scores[idx] - target_scores[jdx]
            if abs(delta) <= 1e-8:
                continue
            direction = 1.0 if delta > 0.0 else -1.0
            pairs.append((idx, jdx, direction, abs(delta)))
    for _step in range(4000):
        grad = np.zeros_like(weights)
        for idx, jdx, direction, pair_weight in pairs:
            diff = feature_rows[idx] - feature_rows[jdx]
            margin = direction * float(np.dot(weights, diff))
            if margin > 50.0:
                sigmoid = 0.0
            elif margin < -50.0:
                sigmoid = 1.0
            else:
                sigmoid = 1.0 / (1.0 + np.exp(margin))
            grad += (-pair_weight * direction * sigmoid) * diff
        grad = grad / max(1, len(pairs)) + 1e-3 * weights
        weights -= 0.2 * grad
    return weights


def fit_selector_payload() -> Dict[str, object]:
    benchmark_names = resolve_phase3_diagnostic_family(TRAIN_FAMILY_NAME)
    feature_rows = []
    target_rows = []
    suite_rows = []
    for benchmark_name in benchmark_names:
        task_suite = str(resolve_phase3_diagnostic_benchmark(benchmark_name)["task_suite"])
        suite_features = collect_regime_examples(task_suite, runs=1)
        suite_centroid = np.mean(suite_features, axis=0)
        target_value = float(STRUCTURED_SUITE_TARGETS[benchmark_name])
        feature_rows.append(suite_centroid)
        target_rows.append(target_value)
        suite_rows.append(
            {
                "benchmark_name": benchmark_name,
                "task_suite": task_suite,
                "target_preference": target_value,
                "feature_centroid": [float(value) for value in suite_centroid],
            }
        )
    X = np.vstack(feature_rows)
    y = np.array(target_rows, dtype=float)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std <= 1e-8, 1.0, X_std)
    X_norm = (X - X_mean) / X_std
    normalized_weights = fit_pairwise_ranking_weights(X_norm, y)
    scaled_weights = normalized_weights / X_std
    bias = float(-np.dot(scaled_weights, X_mean))
    return {
        "feature_names": list(TASK_REGIME_FEATURE_NAMES),
        "weights": tuple(float(value) for value in scaled_weights),
        "bias": bias,
        "suite_rows": suite_rows,
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
        "task_specialist_clone_outcome_selector_merge",
        "task_specialist_clone_ranking_selector_merge",
    )
    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        print("\n" + "=" * 60)
        print(f"Phase 3 Ranking Selector Benchmark: {benchmark_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(task_suite, selector_payload=selector_payload),
            result_filename=f"phase3_ranking_selector_{benchmark_name}_{task_suite}.json",
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
        "# Phase 3 Ranking Selector Benchmark",
        "",
        "Structured pairwise-ranking selector over richer online regime features benchmarked on the full selector stress map.",
        "",
        f"- Train family: `{payload['train_family_name']}`",
        f"- Eval family: `{payload['eval_family_name']}`",
        f"- Probe policy for feature collection: `{payload['probe_policy']}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
        f"- Ranking weights: {weight_parts}, bias={payload['selector_payload']['bias']:+.3f}",
        "",
        "## Structured Preference Targets",
        "",
    ]
    for row in payload["selector_payload"]["suite_rows"]:
        lines.append(
            f"- `{row['benchmark_name']}` / `{row['task_suite']}`: target preference {row['target_preference']:+.2f}"
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
    print(f"Saved ranking selector benchmark JSON: {TARGET_JSON}")
    print(f"Saved ranking selector benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
