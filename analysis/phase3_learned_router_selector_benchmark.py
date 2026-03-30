# -*- coding: utf-8 -*-
"""
Fit a learned three-way router over regime features and benchmark it on the full selector stress map.
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


TARGET_JSON = resolve_canonical_results_path("phase3_learned_router_selector_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_LEARNED_ROUTER_SELECTOR_BENCHMARK.md"
TASK_RANKING_SOURCE_JSON = resolve_canonical_results_path("phase3_task_ranking_selector_benchmark.json")
CLASS_NAMES = ("plasticity_boost", "interference_control", "sparse_plasticity_boost")


def load_task_ranking_payload() -> Dict[str, object]:
    import json

    payload = json.loads(Path(TASK_RANKING_SOURCE_JSON).read_text(encoding="utf-8"))
    selector_payload = payload["selector_payload"]
    return {
        "weights": tuple(selector_payload["weights"]),
        "bias": float(selector_payload["bias"]),
        "source_benchmark": str(TASK_RANKING_SOURCE_JSON),
    }


def build_config(
    task_suite: str,
    selector_payload: Dict[str, object] | None = None,
    *,
    task_ranking_payload: Dict[str, object] | None = None,
):
    selector_payload = selector_payload or {}
    task_ranking_payload = task_ranking_payload or {}
    return build_stress_benchmark_config(
        task_suite,
        selector_task_ranking_fit_weights=tuple(
            task_ranking_payload.get("weights", (0.0,) * len(TASK_REGIME_FEATURE_NAMES))
        ),
        selector_task_ranking_fit_bias=float(task_ranking_payload.get("bias", 0.0)),
        selector_router_fit_weights=tuple(
            selector_payload.get("weights", (0.0,) * (3 * len(TASK_REGIME_FEATURE_NAMES)))
        ),
        selector_router_fit_bias=tuple(selector_payload.get("bias", (0.0, 0.0, 0.0))),
    )


def route_label(benchmark_name: str, utility: float) -> int:
    if benchmark_name == "selector_sparse_adversarial_gate" and utility <= -0.004:
        return 2
    if utility >= 0.004:
        return 1
    return 0


def example_weight(benchmark_name: str, label: int, utility: float) -> float:
    weight = 1.0 + 12.0 * abs(float(utility))
    if benchmark_name == "selector_adversarial_gate":
        weight *= 2.2
        if label == 1:
            weight *= 1.2
    elif benchmark_name == "selector_sparse_adversarial_gate":
        weight *= 2.6
        if label == 2:
            weight *= 1.35
    return float(weight)


def collect_router_supervision(task_suite: str, benchmark_name: str) -> Dict[str, object]:
    result = run_phase3_ablations(
        config=build_stress_benchmark_config(task_suite),
        result_filename=f"phase3_learned_router_selector_supervision_{benchmark_name}_{task_suite}.json",
        policies=("adapt_only",) + COMPETING_POLICIES,
    )
    feature_rows = collect_regime_examples(task_suite, runs=3)
    labels: List[int] = []
    weights: List[float] = []
    supervision_rows: List[Dict[str, float | int | str]] = []
    feature_idx = 0
    for run_idx in range(3):
        limited_run = result["policies"]["task_specialist_clone_limited_merge"]["runs"][run_idx]
        boost_run = result["policies"]["task_specialist_clone_current_path_boost_merge"]["runs"][run_idx]
        for task_id, (limited_row, boost_row) in enumerate(zip(limited_run, boost_run)):
            utility = utility_margin(limited_row, boost_row)
            label = route_label(benchmark_name, utility)
            sample_weight = example_weight(benchmark_name, label, utility)
            labels.append(label)
            weights.append(sample_weight)
            supervision_rows.append(
                {
                    "run": run_idx,
                    "task_id": task_id,
                    "utility_margin": float(utility),
                    "route_label": CLASS_NAMES[label],
                    "sample_weight": float(sample_weight),
                    "feature_index": feature_idx,
                }
            )
            feature_idx += 1
    return {
        "features": feature_rows,
        "labels": np.array(labels, dtype=int),
        "sample_weight": np.array(weights, dtype=float),
        "supervision_rows": supervision_rows,
    }


def fit_router_weights(feature_rows: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    class_count = len(CLASS_NAMES)
    weights = np.zeros((class_count, feature_rows.shape[1]), dtype=float)
    bias = np.zeros(class_count, dtype=float)
    targets = np.eye(class_count, dtype=float)[labels]
    for _step in range(3200):
        logits = feature_rows @ weights.T + bias[None, :]
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= np.sum(probs, axis=1, keepdims=True)
        diff = (probs - targets) * sample_weight[:, None]
        grad_w = diff.T @ feature_rows / max(1, len(feature_rows))
        grad_b = np.sum(diff, axis=0) / max(1, len(feature_rows))
        grad_w += 6e-3 * weights
        weights -= 0.22 * grad_w
        bias -= 0.22 * grad_b
    return weights, bias


def fit_selector_payload() -> Dict[str, object]:
    benchmark_names = resolve_phase3_diagnostic_family(TRAIN_FAMILY_NAME)
    feature_blocks = []
    label_blocks = []
    weight_blocks = []
    benchmark_rows = []
    for benchmark_name in benchmark_names:
        task_suite = str(resolve_phase3_diagnostic_benchmark(benchmark_name)["task_suite"])
        supervision = collect_router_supervision(task_suite, benchmark_name)
        feature_blocks.append(supervision["features"])
        label_blocks.append(supervision["labels"])
        weight_blocks.append(supervision["sample_weight"])
        benchmark_rows.append(
            {
                "benchmark_name": benchmark_name,
                "task_suite": task_suite,
                "label_counts": {
                    class_name: int(np.sum(supervision["labels"] == class_idx))
                    for class_idx, class_name in enumerate(CLASS_NAMES)
                },
                "mean_sample_weight": float(np.mean(supervision["sample_weight"])),
                "task_rows": supervision["supervision_rows"],
            }
        )
    X = np.vstack(feature_blocks)
    y = np.concatenate(label_blocks)
    sample_weight = np.concatenate(weight_blocks)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std <= 1e-8, 1.0, X_std)
    X_norm = (X - X_mean) / X_std
    normalized_weights, bias = fit_router_weights(X_norm, y, sample_weight)
    scaled_weights = normalized_weights / X_std[None, :]
    scaled_bias = bias - (scaled_weights @ X_mean)
    return {
        "feature_names": list(TASK_REGIME_FEATURE_NAMES),
        "class_names": list(CLASS_NAMES),
        "weights": tuple(float(value) for value in scaled_weights.reshape(-1)),
        "bias": tuple(float(value) for value in scaled_bias),
        "benchmark_rows": benchmark_rows,
    }


def run_benchmark() -> Dict:
    selector_payload = fit_selector_payload()
    task_ranking_payload = load_task_ranking_payload()
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
        "task_specialist_clone_two_stage_selector_merge",
        "task_specialist_clone_learned_router_selector_merge",
    )
    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        print("\n" + "=" * 60)
        print(f"Phase 3 Learned Router Selector Benchmark: {benchmark_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(
                task_suite,
                selector_payload=selector_payload,
                task_ranking_payload=task_ranking_payload,
            ),
            result_filename=f"phase3_learned_router_selector_{benchmark_name}_{task_suite}.json",
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
        "task_ranking_payload_source": task_ranking_payload["source_benchmark"],
        "overall_best_policy": overall_best_policy,
        "policy_win_counts": policy_win_counts,
        "aggregate": aggregate,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    class_count = len(payload["selector_payload"]["class_names"])
    feature_names = payload["selector_payload"]["feature_names"]
    flat_weights = payload["selector_payload"]["weights"]
    weight_lines = []
    for class_idx, class_name in enumerate(payload["selector_payload"]["class_names"]):
        offset = class_idx * len(feature_names)
        parts = ", ".join(
            f"{name}={flat_weights[offset + idx]:+.3f}" for idx, name in enumerate(feature_names)
        )
        weight_lines.append(
            f"- `{class_name}`: {parts}, bias={payload['selector_payload']['bias'][class_idx]:+.3f}"
        )
    lines = [
        "# Phase 3 Learned Router Selector Benchmark",
        "",
        "Learned three-way router over regime features for boost, interference control, and sparse-boost routing.",
        "",
        f"- Train family: `{payload['train_family_name']}`",
        f"- Eval family: `{payload['eval_family_name']}`",
        f"- Probe policy for feature collection: `{payload['probe_policy']}`",
        f"- Task-ranking fallback payload: `{payload['task_ranking_payload_source']}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
        f"- Router classes: {', '.join(f'`{name}`' for name in payload['selector_payload']['class_names'])}",
        "",
        "## Router Weights",
        "",
    ]
    lines.extend(weight_lines)
    lines.extend(["", "## Training Targets", ""])
    for row in payload["selector_payload"]["benchmark_rows"]:
        label_parts = ", ".join(f"{name}={count}" for name, count in row["label_counts"].items())
        lines.append(
            f"- `{row['benchmark_name']}` / `{row['task_suite']}`: labels {label_parts}, mean sample weight {row['mean_sample_weight']:.2f}"
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
    print(f"Saved learned router selector benchmark JSON: {TARGET_JSON}")
    print(f"Saved learned router selector benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
