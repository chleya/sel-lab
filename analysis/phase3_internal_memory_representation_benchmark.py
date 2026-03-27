# -*- coding: utf-8 -*-
"""
Compare compressed archived-memory representations under a fixed low budget.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase3_ablation import AblationConfig, run_phase3_ablations
from core.runtime import resolve_canonical_results_path, save_json


DIGITS_SUITES = ("digits_pairs", "digits_pairs_noisy", "digits_pairs_permuted")
REPRESENTATION_CASES = (
    ("full", 0, "recency"),
    ("recency_2", 2, "recency"),
    ("prototype_2", 2, "prototype"),
)
QUANT_BITS = 4
TARGET_JSON = resolve_canonical_results_path("phase3_internal_memory_representation_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_INTERNAL_MEMORY_REPRESENTATION_BENCHMARK.md"
FULL_BUDGET_RESULTS = PROJECT_ROOT / "results" / "phase3_archived_specialist_quantization_benchmark.json"
RECENCY_BUDGET_RESULTS = PROJECT_ROOT / "results" / "phase3_internal_memory_budget_benchmark.json"


def build_config(task_suite: str, selection_mode: str) -> AblationConfig:
    return replace(
        AblationConfig(),
        input_size=64,
        hidden_size=32,
        task_suite=task_suite,
        specialist_merge_scale=0.02,
        specialist_distill_strength=0.50,
        current_path_lr_boost=2.50,
        archived_specialist_quantization_bits=QUANT_BITS,
        archived_specialist_max_count=2,
        archived_specialist_selection_mode=selection_mode,
        learning_rate=0.02,
        epochs_per_task=35,
    )


def _row_by_suite(payload: Dict, task_suite: str) -> Dict:
    for row in payload["rows"]:
        if row["task_suite"] == task_suite:
            return row
    raise KeyError(f"Missing task suite {task_suite}")


def load_full_summary(task_suite: str) -> Dict:
    with FULL_BUDGET_RESULTS.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    row = _row_by_suite(payload, task_suite)
    return row["4"]


def load_recency_budget_summary(task_suite: str) -> Dict:
    with RECENCY_BUDGET_RESULTS.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    row = _row_by_suite(payload, task_suite)
    return row["2"]


def run_benchmark() -> Dict:
    rows: List[Dict] = []
    for task_suite in DIGITS_SUITES:
        suite_row: Dict[str, Dict] = {
            "task_suite": task_suite,
            "full": load_full_summary(task_suite),
            "recency_2": load_recency_budget_summary(task_suite),
        }
        print("\n" + "=" * 60)
        print(
            "Phase 3 Internal Memory Representation Benchmark: "
            f"suite={task_suite}, bits={QUANT_BITS}, budget=2, mode=prototype"
        )
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(task_suite, "prototype"),
            result_filename=f"phase3_internal_memory_representation_prototype_2_{task_suite}_results.json",
            policies=("adapt_only", "task_specialist_clone_current_path_boost_merge"),
        )
        suite_row["prototype_2"] = result["policies"]["task_specialist_clone_current_path_boost_merge"]["summary"]
        rows.append(suite_row)

    aggregate = {}
    for case_name, _, _ in REPRESENTATION_CASES:
        aggregate[case_name] = {
            "mean_avg_accuracy_delta_vs_adapt_only": float(
                sum(row[case_name]["avg_accuracy_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_current_accuracy_delta_vs_adapt_only": float(
                sum(row[case_name]["current_accuracy_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_forgetting_delta_vs_adapt_only": float(
                sum(row[case_name]["forgetting_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_avg_accuracy_gain_vs_fixed": float(
                sum(row[case_name]["avg_accuracy_gain_vs_fixed"] for row in rows) / len(rows)
            ),
            "mean_avg_accuracy_delta_vs_full_budget": float(
                sum(row[case_name]["final_avg_accuracy"] - row["full"]["final_avg_accuracy"] for row in rows)
                / len(rows)
            ),
        }
    best_case = max(aggregate, key=lambda key: aggregate[key]["mean_avg_accuracy_gain_vs_fixed"])
    return {
        "task_suites": list(DIGITS_SUITES),
        "quant_bits": QUANT_BITS,
        "representation_cases": [
            {"name": case_name, "memory_budget": memory_budget, "selection_mode": selection_mode}
            for case_name, memory_budget, selection_mode in REPRESENTATION_CASES
        ],
        "best_case_by_fixed_gain": best_case,
        "aggregate": aggregate,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Internal Memory Representation Benchmark",
        "",
        f"- Digits suites: `{', '.join(payload['task_suites'])}`",
        f"- Archived specialist precision: `{payload['quant_bits']}`-bit",
        f"- Best case by mean gain vs fixed: `{payload['best_case_by_fixed_gain']}`",
        "",
        "## Aggregate Summary",
        "",
    ]
    for case in payload["representation_cases"]:
        name = case["name"]
        row = payload["aggregate"][name]
        lines.append(
            f"- `{name}` (budget `{case['memory_budget']}`, mode `{case['selection_mode']}`): "
            f"mean avg delta vs `adapt_only` {row['mean_avg_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"mean current delta {row['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"mean forgetting delta {row['mean_forgetting_delta_vs_adapt_only']:+.1%}, "
            f"mean gain vs fixed {row['mean_avg_accuracy_gain_vs_fixed']:+.1%}, "
            f"mean delta vs full budget {row['mean_avg_accuracy_delta_vs_full_budget']:+.1%}"
        )
    lines.extend(["", "## Suite Breakdown", ""])
    for row in payload["rows"]:
        lines.append(f"### {row['task_suite']}")
        lines.append("")
        for case in payload["representation_cases"]:
            name = case["name"]
            summary = row[name]
            lines.append(
                f"- `{name}`: avg delta vs `adapt_only` {summary['avg_accuracy_delta_vs_adapt_only']:+.1%}, "
                f"current delta {summary['current_accuracy_delta_vs_adapt_only']:+.1%}, "
                f"forgetting delta {summary['forgetting_delta_vs_adapt_only']:+.1%}, "
                f"gain vs fixed {summary['avg_accuracy_gain_vs_fixed']:+.1%}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = run_benchmark()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved internal-memory representation benchmark JSON: {TARGET_JSON}")
    print(f"Saved internal-memory representation markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
