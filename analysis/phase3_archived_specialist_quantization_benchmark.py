# -*- coding: utf-8 -*-
"""
Stress-test archived-specialist precision on the broader digits family.
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
QUANT_BITS = (4, 2)
TARGET_JSON = resolve_canonical_results_path("phase3_archived_specialist_quantization_benchmark.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_ARCHIVED_SPECIALIST_QUANTIZATION_BENCHMARK.md"
BASELINE_RESULT_FILES = {
    "digits_pairs": "phase3_current_path_boost_digits_pairs_results.json",
    "digits_pairs_noisy": "phase3_current_path_boost_digits_pairs_noisy_results.json",
    "digits_pairs_permuted": "phase3_current_path_boost_digits_pairs_permuted_results.json",
}


def build_config(task_suite: str, quant_bits: int) -> AblationConfig:
    return replace(
        AblationConfig(),
        input_size=64,
        hidden_size=32,
        task_suite=task_suite,
        specialist_merge_scale=0.02,
        specialist_distill_strength=0.50,
        current_path_lr_boost=2.50,
        archived_specialist_quantization_bits=quant_bits,
        learning_rate=0.02,
        epochs_per_task=35,
    )


def load_baseline_summary(task_suite: str) -> Dict:
    path = PROJECT_ROOT / "results" / BASELINE_RESULT_FILES[task_suite]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["policies"]["task_specialist_clone_current_path_boost_merge"]["summary"]


def run_benchmark() -> Dict:
    rows: List[Dict] = []
    for task_suite in DIGITS_SUITES:
        suite_row: Dict[str, Dict] = {
            "task_suite": task_suite,
            "0": load_baseline_summary(task_suite),
        }
        for quant_bits in QUANT_BITS:
            print("\n" + "=" * 60)
            print(
                "Phase 3 Archived Specialist Quantization Benchmark: "
                f"suite={task_suite}, bits={quant_bits}"
            )
            print("=" * 60)
            result = run_phase3_ablations(
                config=build_config(task_suite, quant_bits),
                result_filename=f"phase3_current_path_boost_quant_{quant_bits}_{task_suite}_results.json",
                policies=("adapt_only", "task_specialist_clone_current_path_boost_merge"),
            )
            suite_row[str(quant_bits)] = result["policies"]["task_specialist_clone_current_path_boost_merge"]["summary"]
        rows.append(suite_row)

    aggregate = {}
    baseline_key = "0"
    for quant_bits in (0,) + QUANT_BITS:
        key = str(quant_bits)
        aggregate[key] = {
            "mean_avg_accuracy_delta_vs_adapt_only": float(
                sum(row[key]["avg_accuracy_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_current_accuracy_delta_vs_adapt_only": float(
                sum(row[key]["current_accuracy_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_forgetting_delta_vs_adapt_only": float(
                sum(row[key]["forgetting_delta_vs_adapt_only"] for row in rows) / len(rows)
            ),
            "mean_avg_accuracy_gain_vs_fixed": float(
                sum(row[key]["avg_accuracy_gain_vs_fixed"] for row in rows) / len(rows)
            ),
            "mean_avg_accuracy_delta_vs_unquantized": float(
                sum(row[key]["final_avg_accuracy"] - row[baseline_key]["final_avg_accuracy"] for row in rows) / len(rows)
            ),
        }

    best_bits = max(aggregate, key=lambda key: aggregate[key]["mean_avg_accuracy_gain_vs_fixed"])
    return {
        "task_suites": list(DIGITS_SUITES),
        "quant_bits": [0, *QUANT_BITS],
        "leading_policy": "task_specialist_clone_current_path_boost_merge",
        "best_bits_by_fixed_gain": int(best_bits),
        "aggregate": aggregate,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Archived Specialist Quantization Benchmark",
        "",
        "- Goal: stress-test how much precision archived specialists need under the current leading mechanism.",
        f"- Leading policy: `{payload['leading_policy']}`",
        f"- Digits suites: `{', '.join(payload['task_suites'])}`",
        f"- Quantization bits tested: `{', '.join(str(bits) for bits in payload['quant_bits'])}`",
        f"- Best bits by mean gain vs fixed in this band: `{payload['best_bits_by_fixed_gain']}`",
        "",
        "## Aggregate Summary",
        "",
    ]
    for key in [str(bits) for bits in payload["quant_bits"]]:
        row = payload["aggregate"][key]
        lines.append(
            f"- `{key}`-bit archived specialists: mean avg delta vs `adapt_only` {row['mean_avg_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"mean current delta {row['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"mean forgetting delta {row['mean_forgetting_delta_vs_adapt_only']:+.1%}, "
            f"mean gain vs fixed {row['mean_avg_accuracy_gain_vs_fixed']:+.1%}, "
            f"mean delta vs unquantized {row['mean_avg_accuracy_delta_vs_unquantized']:+.1%}"
        )
    lines.extend(["", "## Suite Breakdown", ""])
    for row in payload["rows"]:
        lines.append(f"### {row['task_suite']}")
        lines.append("")
        for bits in payload["quant_bits"]:
            summary = row[str(bits)]
            lines.append(
                f"- `{bits}`-bit: avg delta vs `adapt_only` {summary['avg_accuracy_delta_vs_adapt_only']:+.1%}, "
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
    print(f"Saved archived-specialist quantization benchmark JSON: {TARGET_JSON}")
    print(f"Saved archived-specialist quantization benchmark markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
