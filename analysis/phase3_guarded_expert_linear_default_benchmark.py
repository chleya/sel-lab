# -*- coding: utf-8 -*-
"""
Preserve sparse and embedded guarded-expert branches, but learn only a linear default-region fallback.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase3_benchmark_family import aggregate_policy_summaries
from analysis.phase3_guarded_expert_selector_benchmark import (
    EVAL_FAMILY_NAME,
    TRAIN_FAMILY_NAME,
)
from analysis.phase3_hierarchical_selector_benchmark import (
    build_config as build_linear_reference_config,
    load_baseline_payloads,
    selector_region_from_feature_row,
)
from analysis.phase3_task_ranking_selector_benchmark import (
    collect_task_level_supervision,
    fit_pairwise_ranking_weights,
)
from core.phase3_ablation import run_phase3_ablations
from core.phase3_model import TASK_REGIME_FEATURE_NAMES
from core.phase3_registry import resolve_phase3_diagnostic_benchmark, resolve_phase3_diagnostic_family
from core.runtime import resolve_canonical_results_path, save_json


TARGET_JSON = resolve_canonical_results_path("phase3_guarded_expert_linear_default_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_GUARDED_EXPERT_LINEAR_DEFAULT_BENCHMARK.md"


def build_config(
    task_suite: str,
    selector_payload: Dict[str, object] | None = None,
    baseline_payloads: Dict[str, object] | None = None,
):
    selector_payload = selector_payload or {}
    baseline_payloads = baseline_payloads or {}
    cfg = build_linear_reference_config(task_suite, {}, baseline_payloads)
    cfg.selector_guarded_expert_linear_default_fit_weights = tuple(
        selector_payload.get("weights", (0.0,) * len(TASK_REGIME_FEATURE_NAMES))
    )
    cfg.selector_guarded_expert_linear_default_fit_bias = float(selector_payload.get("bias", 0.0))
    return cfg


def fit_default_head(features: np.ndarray, utilities: np.ndarray, pair_scale: np.ndarray) -> Dict[str, object]:
    if len(features) == 0:
        return {
            "weights": tuple(0.0 for _ in TASK_REGIME_FEATURE_NAMES),
            "bias": 0.0,
            "sample_count": 0,
            "mean_utility": 0.0,
        }
    X_mean = np.mean(features, axis=0)
    X_std = np.std(features, axis=0)
    X_std = np.where(X_std <= 1e-8, 1.0, X_std)
    X_norm = (features - X_mean) / X_std
    normalized_weights = fit_pairwise_ranking_weights(X_norm, utilities, pair_scale)
    scaled_weights = normalized_weights / X_std
    bias = float(-np.dot(scaled_weights, X_mean))
    return {
        "weights": tuple(float(value) for value in scaled_weights),
        "bias": bias,
        "sample_count": int(len(features)),
        "mean_utility": float(np.mean(utilities)),
    }


def fit_selector_payload() -> Dict[str, object]:
    benchmark_names = resolve_phase3_diagnostic_family(TRAIN_FAMILY_NAME)
    feature_rows = []
    utility_rows = []
    pair_scale_rows = []
    benchmark_rows = []
    for benchmark_name in benchmark_names:
        task_suite = str(resolve_phase3_diagnostic_benchmark(benchmark_name)["task_suite"])
        supervision = collect_task_level_supervision(task_suite, benchmark_name)
        features = supervision["features"]
        utilities = supervision["utilities"]
        region_counts = {"default": 0, "embedded": 0, "sparse_embedded": 0}
        task_rows = []
        for idx, (feature_row, utility) in enumerate(zip(features, utilities)):
            region_name = selector_region_from_feature_row(feature_row)
            region_counts[region_name] += 1
            task_row = {
                "feature_index": idx,
                "region": region_name,
                "utility_margin": float(utility),
            }
            if region_name == "default":
                scale = 1.0
                feature_rows.append(feature_row)
                utility_rows.append(float(utility))
                pair_scale_rows.append(scale)
                task_row["pair_scale"] = scale
            task_rows.append(task_row)
        benchmark_rows.append(
            {
                "benchmark_name": benchmark_name,
                "task_suite": task_suite,
                "region_counts": region_counts,
                "mean_utility": float(np.mean(utilities)),
                "task_rows": task_rows,
            }
        )
    payload = fit_default_head(
        np.vstack(feature_rows),
        np.array(utility_rows, dtype=float),
        np.array(pair_scale_rows, dtype=float),
    )
    payload["benchmark_rows"] = benchmark_rows
    payload["feature_names"] = list(TASK_REGIME_FEATURE_NAMES)
    return payload


def run_benchmark() -> Dict:
    selector_payload = fit_selector_payload()
    baseline_payloads = load_baseline_payloads()
    benchmark_names = resolve_phase3_diagnostic_family(EVAL_FAMILY_NAME)
    rows: List[Dict] = []
    policy_win_counts: Dict[str, int] = {}
    policies = (
        "adapt_only",
        "task_specialist_clone_signature_selector_merge",
        "task_specialist_clone_task_ranking_selector_merge",
        "task_specialist_clone_two_stage_selector_merge",
        "task_specialist_clone_guarded_expert_selector_merge",
        "task_specialist_clone_guarded_expert_linear_default_merge",
        "task_specialist_clone_guarded_expert_boost_default_merge",
        "task_specialist_clone_guarded_expert_merge_default_merge",
    )
    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        print("\n" + "=" * 60)
        print(f"Phase 3 Guarded Expert Linear Default Benchmark: {benchmark_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(
                task_suite,
                selector_payload=selector_payload,
                baseline_payloads=baseline_payloads,
            ),
            result_filename=f"phase3_guarded_expert_linear_default_{benchmark_name}_{task_suite}.json",
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
        "baseline_payload_sources": baseline_payloads["sources"],
        "selector_payload": selector_payload,
        "overall_best_policy": overall_best_policy,
        "policy_win_counts": policy_win_counts,
        "aggregate": aggregate,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Guarded Expert Linear Default Benchmark",
        "",
        "Checks whether the guarded-expert default-region fallback can be simplified from quadratic to linear without losing the selector frontier.",
        "",
        f"- Train family: `{payload['train_family_name']}`",
        f"- Eval family: `{payload['eval_family_name']}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
        f"- Default-head samples: `{payload['selector_payload']['sample_count']}`",
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
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = run_benchmark()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved guarded expert linear default benchmark JSON: {TARGET_JSON}")
    print(f"Saved guarded expert linear default benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
