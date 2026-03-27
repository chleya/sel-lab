# -*- coding: utf-8 -*-
"""
Refine task_specialist_clone_limited_merge around the promising merge-scale band.
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
from core.runtime import resolve_exploratory_results_path, save_json


MERGE_SCALES = (0.05, 0.10, 0.15, 0.20)
TARGET_JSON = resolve_exploratory_results_path("phase3_specialist_merge_sweep.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_SPECIALIST_MERGE_SWEEP.md"


def build_config(scale: float) -> AblationConfig:
    return replace(AblationConfig(), specialist_merge_scale=scale)


def run_sweep() -> Dict:
    rows: List[Dict] = []
    for scale in MERGE_SCALES:
        print("\n" + "=" * 60)
        print(f"Phase 3 Specialist Merge Sweep: scale={scale:.2f}")
        print("=" * 60)
        result = run_phase3_ablations(
            config=build_config(scale),
            result_filename=f"phase3_specialist_merge_scale_{str(scale).replace('.', '_')}.json",
            policies=("adapt_only", "task_specialist_clone_limited_merge"),
        )
        baseline = result["policies"]["adapt_only"]["summary"]
        specialist = result["policies"]["task_specialist_clone_limited_merge"]["summary"]
        rows.append(
            {
                "merge_scale": scale,
                "adapt_only": baseline,
                "task_specialist_clone_limited_merge": specialist,
                "avg_accuracy_delta_vs_adapt_only": specialist["avg_accuracy_delta_vs_adapt_only"],
                "forgetting_delta_vs_adapt_only": specialist["forgetting_delta_vs_adapt_only"],
                "current_accuracy_delta_vs_adapt_only": specialist["current_accuracy_delta_vs_adapt_only"],
                "prior_accuracy_delta_vs_adapt_only": specialist["prior_accuracy_delta_vs_adapt_only"],
                "unit_delta_vs_adapt_only": specialist["unit_delta_vs_adapt_only"],
            }
        )

    best = max(rows, key=lambda row: row["avg_accuracy_delta_vs_adapt_only"])
    lowest_forgetting = max(rows, key=lambda row: row["forgetting_delta_vs_adapt_only"])
    payload = {
        "merge_scales": list(MERGE_SCALES),
        "best_by_accuracy": best,
        "best_by_forgetting": lowest_forgetting,
        "rows": rows,
    }
    return payload


def render_markdown(payload: Dict) -> str:
    best = payload["best_by_accuracy"]
    best_forgetting = payload["best_by_forgetting"]
    lines = [
        "# Phase 3 Specialist Merge Sweep",
        "",
        "- Refined local sweep around the current promising merge-scale region.",
        f"- Best average-accuracy scale: `{best['merge_scale']:.2f}` ({best['avg_accuracy_delta_vs_adapt_only']:+.1%} vs adapt_only)",
        f"- Best forgetting scale: `{best_forgetting['merge_scale']:.2f}` ({best_forgetting['forgetting_delta_vs_adapt_only']:+.1%} forgetting delta vs adapt_only)",
        "",
        "| Merge Scale | Avg Delta vs adapt_only | Forgetting Delta | Current Delta | Prior Delta | Unit Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['merge_scale']:.2f} | {row['avg_accuracy_delta_vs_adapt_only']:+.1%} | "
            f"{row['forgetting_delta_vs_adapt_only']:+.1%} | {row['current_accuracy_delta_vs_adapt_only']:+.1%} | "
            f"{row['prior_accuracy_delta_vs_adapt_only']:+.1%} | {row['unit_delta_vs_adapt_only']:+.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = run_sweep()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved specialist sweep JSON: {TARGET_JSON}")
    print(f"Saved specialist sweep markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
