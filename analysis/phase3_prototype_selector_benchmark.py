# -*- coding: utf-8 -*-
"""
Prototype selector benchmark: fit a nearest-exemplar regime selector on the formal family and evaluate it on the full selector stress map.
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
from core.phase3_model import ReusePolicyNetwork
from core.phase3_registry import resolve_phase3_diagnostic_benchmark, resolve_phase3_diagnostic_family
from core.runtime import resolve_canonical_results_path, save_json


TRAIN_FAMILY_NAME = "expanded_stress_map"
EVAL_FAMILY_NAME = "selector_full_map"
TARGET_JSON = resolve_canonical_results_path("phase3_prototype_selector_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_PROTOTYPE_SELECTOR_BENCHMARK.md"
PROBE_POLICY = "task_specialist_clone_current_path_boost_merge"


def build_config(task_suite: str, selector_payload: Dict[str, object] | None = None):
    selector_payload = selector_payload or {}
    return build_stress_benchmark_config(
        task_suite,
        selector_conflict_threshold=0.30,
        selector_confidence_threshold=0.22,
        selector_input_nonnegative_threshold=0.85,
        selector_input_abs_mean_threshold=0.80,
        selector_input_zero_ratio_threshold=0.55,
        selector_conflict_peak_scale=1.10,
        selector_feature_mean=tuple(selector_payload.get("feature_mean", (0.0,) * 6)),
        selector_feature_std=tuple(selector_payload.get("feature_std", (1.0,) * 6)),
        selector_prototype_vectors=tuple(selector_payload.get("prototype_vectors", ())),
        selector_prototype_labels=tuple(selector_payload.get("prototype_labels", ())),
        selector_prototype_top_k=int(selector_payload.get("top_k", 3)),
        selector_prototype_bandwidth=float(selector_payload.get("bandwidth", 1.0)),
    )


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


def fit_selector_prototypes() -> Dict[str, object]:
    benchmark_names = resolve_phase3_diagnostic_family(TRAIN_FAMILY_NAME)
    feature_blocks = []
    label_blocks = []
    for benchmark_name in benchmark_names:
        task_suite = str(resolve_phase3_diagnostic_benchmark(benchmark_name)["task_suite"])
        features, labels = collect_signature_examples(task_suite)
        feature_blocks.append(features)
        label_blocks.append(labels)
    X = np.vstack(feature_blocks)
    y = np.concatenate(label_blocks)
    feature_mean = np.mean(X, axis=0)
    feature_std = np.std(X, axis=0)
    feature_std = np.where(feature_std <= 1e-8, 1.0, feature_std)
    X_norm = (X - feature_mean) / feature_std
    return {
        "feature_mean": tuple(float(value) for value in feature_mean),
        "feature_std": tuple(float(value) for value in feature_std),
        "prototype_vectors": tuple(tuple(float(value) for value in row) for row in X_norm),
        "prototype_labels": tuple(float(value) for value in y),
        "top_k": 3,
        "bandwidth": 1.0,
    }


def run_benchmark() -> Dict:
    selector_payload = fit_selector_prototypes()
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
        "task_specialist_clone_fitted_selector_merge",
        "task_specialist_clone_prototype_selector_merge",
    )
    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        print("\n" + "=" * 60)
        print(f"Phase 3 Prototype Selector Benchmark: {benchmark_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(task_suite, selector_payload=selector_payload),
            result_filename=f"phase3_prototype_selector_{benchmark_name}_{task_suite}.json",
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
    lines = [
        "# Phase 3 Prototype Selector Benchmark",
        "",
        "Nearest-exemplar selector fit on the formal stress family and evaluated on the full selector stress map.",
        "",
        f"- Train family: `{payload['train_family_name']}`",
        f"- Eval family: `{payload['eval_family_name']}`",
        f"- Probe policy for feature collection: `{payload['probe_policy']}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
        f"- Prototype count: {len(payload['selector_payload']['prototype_vectors'])}",
        f"- Top-k vote: {payload['selector_payload']['top_k']}",
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
    print(f"Saved prototype selector benchmark JSON: {TARGET_JSON}")
    print(f"Saved prototype selector benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
