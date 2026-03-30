# -*- coding: utf-8 -*-
"""
Benchmark a two-stage selector that routes embedded-like regimes before falling back to task-level ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase3_benchmark_family import aggregate_policy_summaries, build_stress_benchmark_config
from core.phase3_ablation import run_phase3_ablations
from core.phase3_registry import resolve_phase3_diagnostic_benchmark, resolve_phase3_diagnostic_family
from core.runtime import resolve_canonical_results_path, save_json


TRAIN_SOURCE_JSON = resolve_canonical_results_path("phase3_task_ranking_selector_benchmark.json")
TARGET_JSON = resolve_canonical_results_path("phase3_two_stage_selector_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_TWO_STAGE_SELECTOR_BENCHMARK.md"
FAMILY_NAME = "selector_full_map"


def load_task_ranking_payload() -> Dict[str, object]:
    payload = json.loads(Path(TRAIN_SOURCE_JSON).read_text(encoding="utf-8"))
    selector_payload = payload["selector_payload"]
    return {
        "feature_names": selector_payload["feature_names"],
        "weights": tuple(selector_payload["weights"]),
        "bias": float(selector_payload["bias"]),
        "source_benchmark": str(TRAIN_SOURCE_JSON),
    }


def build_config(task_suite: str, selector_payload: Dict[str, object]):
    return build_stress_benchmark_config(
        task_suite,
        selector_task_ranking_fit_weights=selector_payload["weights"],
        selector_task_ranking_fit_bias=selector_payload["bias"],
    )


def run_benchmark() -> Dict:
    selector_payload = load_task_ranking_payload()
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
        "task_specialist_clone_task_ranking_selector_merge",
        "task_specialist_clone_two_stage_selector_merge",
    )
    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        print("\n" + "=" * 60)
        print(f"Phase 3 Two-Stage Selector Benchmark: {benchmark_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(task_suite, selector_payload),
            result_filename=f"phase3_two_stage_selector_{benchmark_name}_{task_suite}.json",
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
        "family_name": FAMILY_NAME,
        "selector_payload": selector_payload,
        "overall_best_policy": overall_best_policy,
        "policy_win_counts": policy_win_counts,
        "aggregate": aggregate,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Two-Stage Selector Benchmark",
        "",
        "Two-stage selector using embedded-gate routing with task-ranking fallback.",
        "",
        f"- Family: `{payload['family_name']}`",
        f"- Source task-ranking payload: `{payload['selector_payload']['source_benchmark']}`",
        f"- Overall best policy: `{payload['overall_best_policy']}`",
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
    print(f"Saved two-stage selector benchmark JSON: {TARGET_JSON}")
    print(f"Saved two-stage selector benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
