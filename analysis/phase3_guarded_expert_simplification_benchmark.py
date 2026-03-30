# -*- coding: utf-8 -*-
"""
Test whether the guarded-expert default-region learner is actually necessary.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase3_benchmark_family import aggregate_policy_summaries
from analysis.phase3_guarded_expert_selector_benchmark import build_config, fit_selector_payload
from analysis.phase3_hierarchical_selector_benchmark import load_baseline_payloads
from core.phase3_ablation import run_phase3_ablations
from core.phase3_registry import resolve_phase3_diagnostic_benchmark, resolve_phase3_diagnostic_family
from core.runtime import resolve_canonical_results_path, save_json


TRAIN_FAMILY_NAME = "selector_full_map"
EVAL_FAMILY_NAME = "selector_full_map"
TARGET_JSON = resolve_canonical_results_path("phase3_guarded_expert_simplification_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_GUARDED_EXPERT_SIMPLIFICATION_BENCHMARK.md"


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
        "task_specialist_clone_guarded_expert_boost_default_merge",
        "task_specialist_clone_guarded_expert_merge_default_merge",
    )
    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        print("\n" + "=" * 60)
        print(f"Phase 3 Guarded Expert Simplification Benchmark: {benchmark_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(
                task_suite,
                selector_payload=selector_payload,
                baseline_payloads=baseline_payloads,
            ),
            result_filename=f"phase3_guarded_expert_simplification_{benchmark_name}_{task_suite}.json",
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
        "# Phase 3 Guarded Expert Simplification Benchmark",
        "",
        "Checks whether the guarded-expert default-region quadratic learner is actually needed, or whether a hardcoded default fallback is enough once the sparse and embedded branches are preserved.",
        "",
        f"- Train family: `{payload['train_family_name']}`",
        f"- Eval family: `{payload['eval_family_name']}`",
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
    print(f"Saved guarded expert simplification benchmark JSON: {TARGET_JSON}")
    print(f"Saved guarded expert simplification benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
