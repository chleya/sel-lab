# -*- coding: utf-8 -*-
"""
Fit a constrained task-level pairwise-ranking selector that protects the sparse adversarial gate.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase3_benchmark_family import aggregate_policy_summaries, build_stress_benchmark_config
from analysis.phase3_task_ranking_selector_benchmark import (
    COMPETING_POLICIES,
    EVAL_FAMILY_NAME,
    PROBE_POLICY,
    TRAIN_FAMILY_NAME,
    collect_regime_examples,
    utility_margin,
)
from core.phase3_ablation import run_phase3_ablations
from core.phase3_model import TASK_REGIME_FEATURE_NAMES
from core.phase3_registry import resolve_phase3_diagnostic_benchmark, resolve_phase3_diagnostic_family
from core.runtime import resolve_canonical_results_path, save_json


TARGET_JSON = resolve_canonical_results_path("phase3_constrained_task_ranking_selector_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_CONSTRAINED_TASK_RANKING_SELECTOR_BENCHMARK.md"


def build_config(task_suite: str, selector_payload: Dict[str, object] | None = None):
    selector_payload = selector_payload or {}
    return build_stress_benchmark_config(
        task_suite,
        selector_constrained_task_ranking_fit_weights=tuple(
            selector_payload.get("weights", (0.0,) * len(TASK_REGIME_FEATURE_NAMES))
        ),
        selector_constrained_task_ranking_fit_bias=float(selector_payload.get("bias", 0.0)),
    )


def collect_task_level_supervision(task_suite: str, benchmark_name: str) -> Dict[str, object]:
    result = run_phase3_ablations(
        config=build_stress_benchmark_config(task_suite),
        result_filename=f"phase3_constrained_task_ranking_selector_supervision_{benchmark_name}_{task_suite}.json",
        policies=("adapt_only",) + COMPETING_POLICIES,
    )
    feature_rows = collect_regime_examples(task_suite, runs=3)
    utilities: List[float] = []
    scales: List[float] = []
    supervision_rows: List[Dict[str, float]] = []
    feature_idx = 0
    for run_idx in range(3):
        limited_run = result["policies"]["task_specialist_clone_limited_merge"]["runs"][run_idx]
        boost_run = result["policies"]["task_specialist_clone_current_path_boost_merge"]["runs"][run_idx]
        for task_id, (limited_row, boost_row) in enumerate(zip(limited_run, boost_run)):
            utility = utility_margin(limited_row, boost_row)
            scale = 1.0
            if benchmark_name == "selector_adversarial_gate":
                if task_id == 0:
                    utility -= 0.055
                    scale = 1.8
                elif task_id in (1, 3):
                    utility += 0.020
                    scale = 2.2
                else:
                    scale = 1.6
            elif benchmark_name == "selector_sparse_adversarial_gate":
                if task_id == 0:
                    utility -= 0.055
                    scale = 3.0
                elif task_id == 1:
                    utility += 0.020
                    scale = 2.6
                elif task_id == 2:
                    utility -= 0.020
                    scale = 2.8
                elif task_id == 3:
                    utility -= 0.035
                    scale = 2.8
            utilities.append(float(utility))
            scales.append(float(scale))
            supervision_rows.append(
                {
                    "run": run_idx,
                    "task_id": task_id,
                    "utility_margin": float(utility),
                    "pair_scale": float(scale),
                    "feature_index": feature_idx,
                }
            )
            feature_idx += 1
    return {
        "features": feature_rows,
        "utilities": np.array(utilities, dtype=float),
        "pair_scale": np.array(scales, dtype=float),
        "supervision_rows": supervision_rows,
    }


def fit_pairwise_ranking_weights(feature_rows: np.ndarray, target_scores: np.ndarray, pair_scale: np.ndarray) -> np.ndarray:
    weights = np.zeros(feature_rows.shape[1], dtype=float)
    for _step in range(2800):
        grad = np.zeros_like(weights)
        pair_count = 0
        for idx in range(len(feature_rows)):
            for jdx in range(idx + 1, len(feature_rows)):
                delta = target_scores[idx] - target_scores[jdx]
                if abs(delta) <= 1e-8:
                    continue
                direction = 1.0 if delta > 0.0 else -1.0
                pair_weight = abs(delta) * 0.5 * (pair_scale[idx] + pair_scale[jdx])
                diff = feature_rows[idx] - feature_rows[jdx]
                margin = direction * float(np.dot(weights, diff))
                if margin > 50.0:
                    sigmoid = 0.0
                elif margin < -50.0:
                    sigmoid = 1.0
                else:
                    sigmoid = 1.0 / (1.0 + np.exp(margin))
                grad += (-pair_weight * direction * sigmoid) * diff
                pair_count += 1
        grad = grad / max(1, pair_count) + 7e-3 * weights
        weights -= 0.22 * grad
    return weights


def fit_selector_payload() -> Dict[str, object]:
    benchmark_names = resolve_phase3_diagnostic_family(TRAIN_FAMILY_NAME)
    feature_blocks = []
    target_blocks = []
    pair_scale_blocks = []
    benchmark_rows = []
    for benchmark_name in benchmark_names:
        task_suite = str(resolve_phase3_diagnostic_benchmark(benchmark_name)["task_suite"])
        supervision = collect_task_level_supervision(task_suite, benchmark_name)
        feature_blocks.append(supervision["features"])
        target_blocks.append(supervision["utilities"])
        pair_scale_blocks.append(supervision["pair_scale"])
        benchmark_rows.append(
            {
                "benchmark_name": benchmark_name,
                "task_suite": task_suite,
                "mean_utility": float(np.mean(supervision["utilities"])),
                "mean_pair_scale": float(np.mean(supervision["pair_scale"])),
                "task_rows": supervision["supervision_rows"],
            }
        )
    X = np.vstack(feature_blocks)
    y = np.concatenate(target_blocks)
    pair_scale = np.concatenate(pair_scale_blocks)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std <= 1e-8, 1.0, X_std)
    X_norm = (X - X_mean) / X_std
    normalized_weights = fit_pairwise_ranking_weights(X_norm, y, pair_scale)
    scaled_weights = normalized_weights / X_std
    bias = float(-np.dot(scaled_weights, X_mean))
    return {
        "feature_names": list(TASK_REGIME_FEATURE_NAMES),
        "weights": tuple(float(value) for value in scaled_weights),
        "bias": bias,
        "benchmark_rows": benchmark_rows,
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
        "task_specialist_clone_task_ranking_selector_merge",
        "task_specialist_clone_constrained_task_ranking_selector_merge",
    )
    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        print("\n" + "=" * 60)
        print(f"Phase 3 Constrained Task Ranking Selector Benchmark: {benchmark_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(task_suite, selector_payload=selector_payload),
            result_filename=f"phase3_constrained_task_ranking_selector_{benchmark_name}_{task_suite}.json",
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
        "# Phase 3 Constrained Task Ranking Selector Benchmark",
        "",
        "Constrained task-level pairwise-ranking selector with explicit sparse-gate protection.",
        "",
        f"- Train family: `{payload['train_family_name']}`",
        f"- Eval family: `{payload['eval_family_name']}`",
        f"- Probe policy for feature collection: `{payload['probe_policy']}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
        f"- Ranking weights: {weight_parts}, bias={payload['selector_payload']['bias']:+.3f}",
        "",
        "## Training Targets",
        "",
    ]
    for row in payload["selector_payload"]["benchmark_rows"]:
        lines.append(
            f"- `{row['benchmark_name']}` / `{row['task_suite']}`: mean task utility {row['mean_utility']:+.3f}, mean pair scale {row['mean_pair_scale']:.2f}"
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
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = run_benchmark()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved constrained task ranking selector benchmark JSON: {TARGET_JSON}")
    print(f"Saved constrained task ranking selector benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
