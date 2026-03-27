# -*- coding: utf-8 -*-
"""
Compare tuned Phase 3 reuse policies across a broader digits benchmark family.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase3_ablation import AblationConfig, run_phase3_ablations
from core.runtime import resolve_canonical_results_path, save_json


DIGITS_SUITES = (
    "digits_pairs",
    "digits_pairs_noisy",
    "digits_pairs_permuted",
)
POLICY_ORDER = (
    "task_specialist_clone_distill_merge",
    "task_specialist_clone_confidence_gate_merge",
    "task_specialist_clone_limited_merge",
)
TARGET_JSON = resolve_canonical_results_path("phase3_digits_benchmark_transfer.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_DIGITS_BENCHMARK_TRANSFER.md"


def build_config(task_suite: str) -> AblationConfig:
    return replace(
        AblationConfig(),
        input_size=64,
        hidden_size=32,
        task_suite=task_suite,
        specialist_merge_scale=0.02,
        specialist_distill_strength=0.50,
        learning_rate=0.02,
        epochs_per_task=35,
    )


def pick_best_policy(summaries: Dict[str, Dict]) -> tuple[str, Dict, List[str]]:
    best_value = max(summaries[name]["avg_accuracy_delta_vs_adapt_only"] for name in POLICY_ORDER)
    tied = [
        name
        for name in POLICY_ORDER
        if abs(summaries[name]["avg_accuracy_delta_vs_adapt_only"] - best_value) <= 1e-12
    ]
    best_name = tied[0]
    return best_name, summaries[best_name], tied


def run_digits_benchmark_transfer() -> Dict:
    rows: List[Dict] = []
    policy_win_counts = {name: 0 for name in POLICY_ORDER}

    for task_suite in DIGITS_SUITES:
        print("\n" + "=" * 60)
        print(f"Phase 3 Digits Benchmark Transfer: suite={task_suite}")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(task_suite),
            result_filename=f"phase3_{task_suite}_results.json",
            policies=(
                "adapt_only",
                "task_specialist_clone_limited_merge",
                "task_specialist_clone_confidence_gate_merge",
                "task_specialist_clone_distill_merge",
            ),
        )
        summaries = {
            "adapt_only": result["policies"]["adapt_only"]["summary"],
            "task_specialist_clone_limited_merge": result["policies"]["task_specialist_clone_limited_merge"]["summary"],
            "task_specialist_clone_confidence_gate_merge": result["policies"]["task_specialist_clone_confidence_gate_merge"]["summary"],
            "task_specialist_clone_distill_merge": result["policies"]["task_specialist_clone_distill_merge"]["summary"],
        }
        best_name, best_summary, tied = pick_best_policy(summaries)
        policy_win_counts[best_name] += 1
        rows.append(
            {
                "task_suite": task_suite,
                "best_reuse_policy": best_name,
                "tied_best_reuse_policies": tied,
                "adapt_only": summaries["adapt_only"],
                "task_specialist_clone_limited_merge": summaries["task_specialist_clone_limited_merge"],
                "task_specialist_clone_confidence_gate_merge": summaries["task_specialist_clone_confidence_gate_merge"],
                "task_specialist_clone_distill_merge": summaries["task_specialist_clone_distill_merge"],
                "avg_accuracy_delta_vs_adapt_only": best_summary["avg_accuracy_delta_vs_adapt_only"],
                "forgetting_delta_vs_adapt_only": best_summary["forgetting_delta_vs_adapt_only"],
                "avg_accuracy_gain_vs_fixed": best_summary["avg_accuracy_gain_vs_fixed"],
                "current_accuracy_delta_vs_adapt_only": best_summary["current_accuracy_delta_vs_adapt_only"],
                "prior_accuracy_delta_vs_adapt_only": best_summary["prior_accuracy_delta_vs_adapt_only"],
            }
        )

    aggregate = {}
    for policy_name in POLICY_ORDER:
        aggregate[policy_name] = {
            "mean_avg_accuracy_delta_vs_adapt_only": float(
                sum(row[policy_name]["avg_accuracy_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_forgetting_delta_vs_adapt_only": float(
                sum(row[policy_name]["forgetting_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_avg_accuracy_gain_vs_fixed": float(
                sum(row[policy_name]["avg_accuracy_gain_vs_fixed"] for row in rows) / len(rows)
            ),
            "suite_wins": policy_win_counts[policy_name],
        }

    overall_best_policy = max(
        POLICY_ORDER,
        key=lambda name: aggregate[name]["mean_avg_accuracy_delta_vs_adapt_only"],
    )
    hardest_suite = min(rows, key=lambda row: row[overall_best_policy]["avg_accuracy_gain_vs_fixed"])

    return {
        "task_suites": list(DIGITS_SUITES),
        "merge_scale": 0.02,
        "distill_strength": 0.50,
        "policy_order": list(POLICY_ORDER),
        "overall_best_policy": overall_best_policy,
        "hardest_suite_for_overall_best": hardest_suite["task_suite"],
        "aggregate": aggregate,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Digits Benchmark Transfer",
        "",
        f"- Digits benchmark family: `{', '.join(payload['task_suites'])}`",
        f"- Shared tuned configuration: merge scale `{payload['merge_scale']:.2f}`, distill strength `{payload['distill_strength']:.2f}`",
        f"- Overall best reuse policy by mean avg delta vs adapt_only: `{payload['overall_best_policy']}`",
        f"- Hardest suite for that policy vs fixed: `{payload['hardest_suite_for_overall_best']}`",
        "",
        "## Aggregate Policy Comparison",
        "",
    ]
    for policy_name in payload["policy_order"]:
        aggregate = payload["aggregate"][policy_name]
        lines.append(
            f"- `{policy_name}`: mean avg delta {aggregate['mean_avg_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"mean forgetting delta {aggregate['mean_forgetting_delta_vs_adapt_only']:+.1%}, "
            f"mean gain vs fixed {aggregate['mean_avg_accuracy_gain_vs_fixed']:+.1%}, "
            f"suite wins {aggregate['suite_wins']}"
        )
    lines.extend(
        [
            "",
            "## Suite Comparison",
            "",
            "| Task Suite | Best Reuse Policy | Avg Delta vs adapt_only | Forgetting Delta | Gain vs Fixed | Tied Best Policies |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| {row['task_suite']} | {row['best_reuse_policy']} | {row['avg_accuracy_delta_vs_adapt_only']:+.1%} | "
            f"{row['forgetting_delta_vs_adapt_only']:+.1%} | {row['avg_accuracy_gain_vs_fixed']:+.1%} | "
            f"{', '.join(row['tied_best_reuse_policies'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = run_digits_benchmark_transfer()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved digits benchmark transfer JSON: {TARGET_JSON}")
    print(f"Saved digits benchmark transfer markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
