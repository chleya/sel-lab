# -*- coding: utf-8 -*-
"""
Compare Phase 3 baseline vs main reuse policy across richer task suites.
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
from core.runtime import resolve_results_path, save_json


TASK_SUITES = ("default", "rotated", "feature_shift", "noisy")
TARGET_JSON = resolve_results_path("phase3_suite_transfer.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_SUITE_TRANSFER.md"


def build_config(task_suite: str) -> AblationConfig:
    return replace(AblationConfig(), task_suite=task_suite, specialist_merge_scale=0.05)


def run_transfer() -> Dict:
    rows: List[Dict] = []
    for task_suite in TASK_SUITES:
        print("\n" + "=" * 60)
        print(f"Phase 3 Suite Transfer: suite={task_suite}")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(task_suite),
            result_filename=f"phase3_suite_transfer_{task_suite}.json",
            policies=("adapt_only", "task_specialist_clone_limited_merge"),
        )
        baseline = result["policies"]["adapt_only"]["summary"]
        specialist = result["policies"]["task_specialist_clone_limited_merge"]["summary"]
        rows.append(
            {
                "task_suite": task_suite,
                "adapt_only": baseline,
                "task_specialist_clone_limited_merge": specialist,
                "avg_accuracy_delta_vs_adapt_only": specialist["avg_accuracy_delta_vs_adapt_only"],
                "forgetting_delta_vs_adapt_only": specialist["forgetting_delta_vs_adapt_only"],
                "current_accuracy_delta_vs_adapt_only": specialist["current_accuracy_delta_vs_adapt_only"],
                "prior_accuracy_delta_vs_adapt_only": specialist["prior_accuracy_delta_vs_adapt_only"],
                "unit_delta_vs_adapt_only": specialist["unit_delta_vs_adapt_only"],
            }
        )

    best_accuracy = max(rows, key=lambda row: row["avg_accuracy_delta_vs_adapt_only"])
    best_forgetting = max(rows, key=lambda row: row["forgetting_delta_vs_adapt_only"])
    return {
        "task_suites": list(TASK_SUITES),
        "merge_scale": 0.05,
        "best_by_accuracy": best_accuracy,
        "best_by_forgetting": best_forgetting,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    best_accuracy = payload["best_by_accuracy"]
    best_forgetting = payload["best_by_forgetting"]
    lines = [
        "# Phase 3 Suite Transfer",
        "",
        f"- Specialist merge scale: `{payload['merge_scale']:.2f}`",
        f"- Best suite by average accuracy delta: `{best_accuracy['task_suite']}` ({best_accuracy['avg_accuracy_delta_vs_adapt_only']:+.1%} vs adapt_only)",
        f"- Best suite by forgetting delta: `{best_forgetting['task_suite']}` ({best_forgetting['forgetting_delta_vs_adapt_only']:+.1%} vs adapt_only)",
        "",
        "| Task Suite | Avg Delta vs adapt_only | Forgetting Delta | Current Delta | Prior Delta | Unit Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['task_suite']} | {row['avg_accuracy_delta_vs_adapt_only']:+.1%} | "
            f"{row['forgetting_delta_vs_adapt_only']:+.1%} | {row['current_accuracy_delta_vs_adapt_only']:+.1%} | "
            f"{row['prior_accuracy_delta_vs_adapt_only']:+.1%} | {row['unit_delta_vs_adapt_only']:+.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = run_transfer()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved suite transfer JSON: {TARGET_JSON}")
    print(f"Saved suite transfer markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
