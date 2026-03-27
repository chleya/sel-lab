# -*- coding: utf-8 -*-
"""
Shared helpers for Phase 3 stress-map benchmark families.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Callable, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase3_ablation import AblationConfig, run_phase3_ablations
from core.phase3_registry import resolve_phase3_diagnostic_benchmark, resolve_phase3_diagnostic_family


SummaryDict = Dict[str, Dict[str, float]]
RowBuilder = Callable[[str, str, Dict[str, object], SummaryDict, str, Dict], Dict]


def build_stress_benchmark_config(task_suite: str, **overrides: float) -> AblationConfig:
    base = replace(
        AblationConfig(),
        task_suite=task_suite,
        learning_rate=0.02,
        epochs_per_task=25,
        runs=3,
    )
    if overrides:
        base = replace(base, **overrides)
    if task_suite.startswith("digits_pairs") or task_suite in {"feature_shift_embedded", "feature_shift_sparse_embedded"}:
        return replace(
            base,
            input_size=64,
            hidden_size=32,
            specialist_merge_scale=0.02,
            specialist_distill_strength=0.50,
            current_path_lr_boost=2.50,
        )
    return replace(
        base,
        specialist_merge_scale=0.05,
        specialist_distill_strength=0.35,
        current_path_lr_boost=2.00,
    )


def build_summary_row(result: Dict, policies: Sequence[str]) -> SummaryDict:
    return {policy: result["policies"][policy]["summary"] for policy in policies}


def aggregate_policy_summaries(
    rows: Sequence[Dict],
    policy_names: Iterable[str],
    *,
    summary_key: str = "summaries",
) -> Dict[str, Dict[str, float]]:
    aggregate: Dict[str, Dict[str, float]] = {}
    for policy_name in policy_names:
        policy_rows = [row for row in rows if policy_name in row[summary_key]]
        aggregate[policy_name] = {
            "benchmarks_present": len(policy_rows),
            "mean_avg_accuracy_delta_vs_adapt_only": float(
                sum(row[summary_key][policy_name]["avg_accuracy_delta_vs_adapt_only"] for row in policy_rows)
                / len(policy_rows)
            ),
            "mean_current_accuracy_delta_vs_adapt_only": float(
                sum(row[summary_key][policy_name]["current_accuracy_delta_vs_adapt_only"] for row in policy_rows)
                / len(policy_rows)
            ),
            "mean_prior_accuracy_delta_vs_adapt_only": float(
                sum(row[summary_key][policy_name]["prior_accuracy_delta_vs_adapt_only"] for row in policy_rows)
                / len(policy_rows)
            ),
            "mean_forgetting_delta_vs_adapt_only": float(
                sum(row[summary_key][policy_name]["forgetting_delta_vs_adapt_only"] for row in policy_rows)
                / len(policy_rows)
            ),
            "mean_avg_accuracy_gain_vs_fixed": float(
                sum(row[summary_key][policy_name]["avg_accuracy_gain_vs_fixed"] for row in policy_rows)
                / len(policy_rows)
            ),
        }
    return aggregate


def run_policy_benchmark_family(
    *,
    family_name: str,
    policies: Sequence[str] | None,
    result_prefix: str,
    title: str,
    config_overrides: Dict[str, float] | None = None,
    row_builder: RowBuilder | None = None,
) -> Dict:
    benchmark_names = resolve_phase3_diagnostic_family(family_name)
    config_overrides = config_overrides or {}
    rows: List[Dict] = []
    policy_win_counts: Dict[str, int] = {}

    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        benchmark_policies = tuple(spec["policies"]) if policies is None else tuple(policies)
        print("\n" + "=" * 60)
        print(f"{title}: {benchmark_name} ({task_suite})")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_stress_benchmark_config(task_suite, **config_overrides),
            result_filename=f"{result_prefix}_{benchmark_name}_{task_suite}.json",
            policies=benchmark_policies,
        )
        summaries = build_summary_row(result, benchmark_policies)
        candidate_policies = tuple(policy for policy in benchmark_policies if policy != "adapt_only")
        best_policy = max(
            candidate_policies,
            key=lambda name: summaries[name]["avg_accuracy_delta_vs_adapt_only"],
        )
        policy_win_counts[best_policy] = policy_win_counts.get(best_policy, 0) + 1
        if row_builder is None:
            row = {
                "benchmark_name": benchmark_name,
                "task_suite": task_suite,
                "benchmark_intent": spec["benchmark_intent"],
                "best_policy": best_policy,
                "policies": list(benchmark_policies),
                "summaries": summaries,
            }
        else:
            row = row_builder(benchmark_name, task_suite, spec, summaries, best_policy, result)
        rows.append(row)

    aggregate_policy_names = sorted({policy for row in rows for policy in row["policies"]})
    aggregate = aggregate_policy_summaries(rows, aggregate_policy_names)
    for policy_name, row in aggregate.items():
        row["benchmark_wins"] = policy_win_counts.get(policy_name, 0)

    overall_best_policy = max(
        (name for name in aggregate if name != "adapt_only"),
        key=lambda name: aggregate[name]["mean_avg_accuracy_delta_vs_adapt_only"],
    )
    return {
        "family_name": family_name,
        "benchmark_names": list(benchmark_names),
        "overall_best_policy": overall_best_policy,
        "policy_win_counts": policy_win_counts,
        "aggregate": aggregate,
        "rows": rows,
    }
