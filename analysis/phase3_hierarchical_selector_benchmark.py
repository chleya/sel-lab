# -*- coding: utf-8 -*-
"""
Fit region-specific ranking heads behind the existing selector segmentation and benchmark them on selector_full_map.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase3_benchmark_family import aggregate_policy_summaries, build_stress_benchmark_config
from analysis.phase3_task_ranking_selector_benchmark import (
    EVAL_FAMILY_NAME,
    TRAIN_FAMILY_NAME,
    collect_task_level_supervision,
    fit_pairwise_ranking_weights,
)
from core.phase3_ablation import AblationConfig, run_phase3_ablations
from core.phase3_model import TASK_REGIME_FEATURE_NAMES
from core.phase3_registry import resolve_phase3_diagnostic_benchmark, resolve_phase3_diagnostic_family
from core.runtime import resolve_canonical_results_path, save_json


TARGET_JSON = resolve_canonical_results_path("phase3_hierarchical_selector_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_HIERARCHICAL_SELECTOR_BENCHMARK.md"
TASK_RANKING_SOURCE_JSON = resolve_canonical_results_path("phase3_task_ranking_selector_benchmark.json")
TWO_STAGE_SOURCE_JSON = resolve_canonical_results_path("phase3_two_stage_selector_benchmark.json")
LEARNED_ROUTER_SOURCE_JSON = resolve_canonical_results_path("phase3_learned_router_selector_benchmark.json")
REGION_NAMES = ("default", "embedded", "sparse_embedded")
FEATURE_INDEX = {name: idx for idx, name in enumerate(TASK_REGIME_FEATURE_NAMES)}
DEFAULT_SELECTOR_CONFIG = AblationConfig()


def load_json_payload(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_payloads() -> Dict[str, Dict[str, object]]:
    task_ranking = load_json_payload(Path(TASK_RANKING_SOURCE_JSON))
    two_stage = load_json_payload(Path(TWO_STAGE_SOURCE_JSON))
    learned_router = load_json_payload(Path(LEARNED_ROUTER_SOURCE_JSON))
    return {
        "task_ranking": task_ranking["selector_payload"],
        "two_stage": two_stage["selector_payload"],
        "learned_router": learned_router["selector_payload"],
        "sources": {
            "task_ranking": str(TASK_RANKING_SOURCE_JSON),
            "two_stage": str(TWO_STAGE_SOURCE_JSON),
            "learned_router": str(LEARNED_ROUTER_SOURCE_JSON),
        },
    }


def build_config(
    task_suite: str,
    selector_payload: Dict[str, object] | None = None,
    baseline_payloads: Dict[str, object] | None = None,
):
    selector_payload = selector_payload or {}
    baseline_payloads = baseline_payloads or {}
    default_head = selector_payload.get("default", {})
    embedded_head = selector_payload.get("embedded", {})
    sparse_head = selector_payload.get("sparse_embedded", {})
    task_ranking_payload = baseline_payloads.get("task_ranking", {})
    two_stage_payload = baseline_payloads.get("two_stage", {})
    learned_router_payload = baseline_payloads.get("learned_router", {})
    return build_stress_benchmark_config(
        task_suite,
        selector_task_ranking_fit_weights=tuple(
            task_ranking_payload.get("weights", (0.0,) * len(TASK_REGIME_FEATURE_NAMES))
        ),
        selector_task_ranking_fit_bias=float(task_ranking_payload.get("bias", 0.0)),
        selector_router_fit_weights=tuple(
            learned_router_payload.get("weights", (0.0,) * (3 * len(TASK_REGIME_FEATURE_NAMES)))
        ),
        selector_router_fit_bias=tuple(learned_router_payload.get("bias", (0.0, 0.0, 0.0))),
        selector_sparse_zero_ratio_floor=float(two_stage_payload.get("zero_ratio_floor", DEFAULT_SELECTOR_CONFIG.selector_sparse_zero_ratio_floor)),
        selector_sparse_conflict_delta_floor=float(
            two_stage_payload.get("conflict_delta_floor", DEFAULT_SELECTOR_CONFIG.selector_sparse_conflict_delta_floor)
        ),
        selector_hierarchical_default_fit_weights=tuple(
            default_head.get("weights", (0.0,) * len(TASK_REGIME_FEATURE_NAMES))
        ),
        selector_hierarchical_default_fit_bias=float(default_head.get("bias", 0.0)),
        selector_hierarchical_embedded_fit_weights=tuple(
            embedded_head.get("weights", (0.0,) * len(TASK_REGIME_FEATURE_NAMES))
        ),
        selector_hierarchical_embedded_fit_bias=float(embedded_head.get("bias", 0.0)),
        selector_hierarchical_sparse_fit_weights=tuple(
            sparse_head.get("weights", (0.0,) * len(TASK_REGIME_FEATURE_NAMES))
        ),
        selector_hierarchical_sparse_fit_bias=float(sparse_head.get("bias", 0.0)),
    )


def selector_region_from_feature_row(feature_row: np.ndarray) -> str:
    nonnegative_ratio = feature_row[FEATURE_INDEX["input_nonnegative_ratio"]]
    abs_mean = feature_row[FEATURE_INDEX["input_abs_mean"]]
    zero_ratio = feature_row[FEATURE_INDEX["input_zero_ratio"]]
    conflict_delta = feature_row[FEATURE_INDEX["conflict_delta"]]
    embedded_like = (
        nonnegative_ratio >= DEFAULT_SELECTOR_CONFIG.selector_input_nonnegative_threshold
        and abs_mean >= DEFAULT_SELECTOR_CONFIG.selector_input_abs_mean_threshold
    )
    sparse_embedded_like = (
        embedded_like
        and zero_ratio >= DEFAULT_SELECTOR_CONFIG.selector_sparse_zero_ratio_floor
        and conflict_delta >= DEFAULT_SELECTOR_CONFIG.selector_sparse_conflict_delta_floor
    )
    if sparse_embedded_like:
        return "sparse_embedded"
    if embedded_like:
        return "embedded"
    return "default"


def region_pair_scale(benchmark_name: str, region_name: str) -> float:
    scale = 1.0
    if "adversarial_gate" in benchmark_name:
        scale = 2.0
    if benchmark_name == "selector_adversarial_gate" and region_name == "embedded":
        scale *= 1.6
    if benchmark_name == "selector_sparse_adversarial_gate" and region_name == "sparse_embedded":
        scale *= 2.4
    return float(scale)


def fit_region_head(features: np.ndarray, utilities: np.ndarray, pair_scale: np.ndarray) -> Dict[str, object]:
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
    region_feature_blocks = {name: [] for name in REGION_NAMES}
    region_target_blocks = {name: [] for name in REGION_NAMES}
    region_scale_blocks = {name: [] for name in REGION_NAMES}
    benchmark_rows = []
    for benchmark_name in benchmark_names:
        task_suite = str(resolve_phase3_diagnostic_benchmark(benchmark_name)["task_suite"])
        supervision = collect_task_level_supervision(task_suite, benchmark_name)
        features = supervision["features"]
        utilities = supervision["utilities"]
        region_counts = {name: 0 for name in REGION_NAMES}
        task_rows = []
        for idx, (feature_row, utility) in enumerate(zip(features, utilities)):
            region_name = selector_region_from_feature_row(feature_row)
            pair_scale = region_pair_scale(benchmark_name, region_name)
            region_feature_blocks[region_name].append(feature_row)
            region_target_blocks[region_name].append(float(utility))
            region_scale_blocks[region_name].append(pair_scale)
            region_counts[region_name] += 1
            task_rows.append(
                {
                    "feature_index": idx,
                    "region": region_name,
                    "utility_margin": float(utility),
                    "pair_scale": float(pair_scale),
                }
            )
        benchmark_rows.append(
            {
                "benchmark_name": benchmark_name,
                "task_suite": task_suite,
                "region_counts": region_counts,
                "mean_utility": float(np.mean(utilities)),
                "task_rows": task_rows,
            }
        )
    region_payloads = {}
    for region_name in REGION_NAMES:
        features = np.vstack(region_feature_blocks[region_name]) if region_feature_blocks[region_name] else np.zeros(
            (0, len(TASK_REGIME_FEATURE_NAMES)),
            dtype=float,
        )
        utilities = np.array(region_target_blocks[region_name], dtype=float)
        pair_scale = np.array(region_scale_blocks[region_name], dtype=float)
        region_payloads[region_name] = fit_region_head(features, utilities, pair_scale)
    return {
        "feature_names": list(TASK_REGIME_FEATURE_NAMES),
        "region_names": list(REGION_NAMES),
        "region_payloads": region_payloads,
        "benchmark_rows": benchmark_rows,
    }


def run_benchmark() -> Dict:
    selector_payload = fit_selector_payload()
    baseline_payloads = load_baseline_payloads()
    benchmark_names = resolve_phase3_diagnostic_family(EVAL_FAMILY_NAME)
    rows: List[Dict] = []
    policy_win_counts: Dict[str, int] = {}
    policies = (
        "adapt_only",
        "task_specialist_clone_limited_merge",
        "task_specialist_clone_current_path_boost_merge",
        "task_specialist_clone_regime_switch_merge",
        "task_specialist_clone_signature_selector_merge",
        "task_specialist_clone_task_ranking_selector_merge",
        "task_specialist_clone_two_stage_selector_merge",
        "task_specialist_clone_learned_router_selector_merge",
        "task_specialist_clone_hierarchical_selector_merge",
    )
    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        print("\n" + "=" * 60)
        print(f"Phase 3 Hierarchical Selector Benchmark: {benchmark_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(
                task_suite,
                selector_payload["region_payloads"],
                baseline_payloads=baseline_payloads,
            ),
            result_filename=f"phase3_hierarchical_selector_{benchmark_name}_{task_suite}.json",
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
        "# Phase 3 Hierarchical Selector Benchmark",
        "",
        "Hierarchical segmented selector with region-specific ranking heads over `default`, `embedded`, and `sparse_embedded` regions.",
        "",
        f"- Train family: `{payload['train_family_name']}`",
        f"- Eval family: `{payload['eval_family_name']}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
        f"- Task-ranking payload source: `{payload['baseline_payload_sources']['task_ranking']}`",
        f"- Two-stage payload source: `{payload['baseline_payload_sources']['two_stage']}`",
        f"- Learned-router payload source: `{payload['baseline_payload_sources']['learned_router']}`",
        "",
        "## Region Heads",
        "",
    ]
    for region_name in payload["selector_payload"]["region_names"]:
        head = payload["selector_payload"]["region_payloads"][region_name]
        weight_parts = ", ".join(
            f"{name}={value:+.3f}"
            for name, value in zip(payload["selector_payload"]["feature_names"], head["weights"])
        )
        lines.append(
            f"- `{region_name}`: samples={head['sample_count']}, mean utility={head['mean_utility']:+.3f}, "
            f"weights {weight_parts}, bias={head['bias']:+.3f}"
        )
    lines.extend(["", "## Training Regions", ""])
    for row in payload["selector_payload"]["benchmark_rows"]:
        counts = ", ".join(f"{name}={count}" for name, count in row["region_counts"].items())
        lines.append(
            f"- `{row['benchmark_name']}` / `{row['task_suite']}`: mean utility {row['mean_utility']:+.3f}, regions {counts}"
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
    print(f"Saved hierarchical selector benchmark JSON: {TARGET_JSON}")
    print(f"Saved hierarchical selector benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
