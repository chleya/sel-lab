# -*- coding: utf-8 -*-
"""
Joint sweep of specialist merge scale and distillation strength on digits_pairs.
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
from core.runtime import resolve_exploratory_results_path, save_json


TARGET_JSON = resolve_exploratory_results_path("phase3_distill_joint_sweep.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_DISTILL_JOINT_SWEEP.md"
MERGE_SCALES = (0.02, 0.05, 0.1)
DISTILL_STRENGTHS = (0.25, 0.5, 0.75)


def build_base_config() -> AblationConfig:
    return replace(
        AblationConfig(),
        input_size=64,
        hidden_size=32,
        task_suite="digits_pairs",
        learning_rate=0.02,
        epochs_per_task=35,
    )


def run_joint_sweep() -> Dict:
    rows: List[Dict] = []
    best_avg = None
    best_fixed_gap = None
    for merge_scale in MERGE_SCALES:
        for distill_strength in DISTILL_STRENGTHS:
            config = replace(
                build_base_config(),
                specialist_merge_scale=merge_scale,
                specialist_distill_strength=distill_strength,
            )
            result = run_phase3_ablations(
                config=config,
                result_filename=(
                    f"phase3_distill_joint_m{str(merge_scale).replace('.', '_')}"
                    f"_d{str(distill_strength).replace('.', '_')}.json"
                ),
                policies=("adapt_only", "task_specialist_clone_distill_merge"),
            )
            adapt = result["policies"]["adapt_only"]["summary"]
            distill = result["policies"]["task_specialist_clone_distill_merge"]["summary"]
            row = {
                "merge_scale": merge_scale,
                "distill_strength": distill_strength,
                "adapt_only": adapt,
                "task_specialist_clone_distill_merge": distill,
                "avg_accuracy_delta_vs_adapt_only": distill["avg_accuracy_delta_vs_adapt_only"],
                "forgetting_delta_vs_adapt_only": distill["forgetting_delta_vs_adapt_only"],
                "avg_accuracy_gain_vs_fixed": distill["avg_accuracy_gain_vs_fixed"],
            }
            rows.append(row)
            if best_avg is None or row["avg_accuracy_delta_vs_adapt_only"] > best_avg["avg_accuracy_delta_vs_adapt_only"]:
                best_avg = row
            if best_fixed_gap is None or row["avg_accuracy_gain_vs_fixed"] > best_fixed_gap["avg_accuracy_gain_vs_fixed"]:
                best_fixed_gap = row
    return {
        "task_suite": "digits_pairs",
        "best_avg_vs_adapt_only": best_avg,
        "closest_to_fixed": best_fixed_gap,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    best_avg = payload["best_avg_vs_adapt_only"]
    best_fixed = payload["closest_to_fixed"]
    lines = [
        "# Phase 3 Distill Joint Sweep",
        "",
        "- Benchmark: `digits_pairs`",
        f"- Best avg delta vs `adapt_only`: merge `{best_avg['merge_scale']:.2f}`, distill `{best_avg['distill_strength']:.2f}`, "
        f"avg delta {best_avg['avg_accuracy_delta_vs_adapt_only']:+.1%}, forgetting delta {best_avg['forgetting_delta_vs_adapt_only']:+.1%}",
        f"- Closest to fixed: merge `{best_fixed['merge_scale']:.2f}`, distill `{best_fixed['distill_strength']:.2f}`, "
        f"avg gain vs fixed {best_fixed['avg_accuracy_gain_vs_fixed']:+.1%}",
        "",
        "## Grid",
        "",
    ]
    for row in payload["rows"]:
        lines.append(
            f"- merge `{row['merge_scale']:.2f}`, distill `{row['distill_strength']:.2f}`: "
            f"avg vs adapt {row['avg_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"forgetting vs adapt {row['forgetting_delta_vs_adapt_only']:+.1%}, "
            f"vs fixed {row['avg_accuracy_gain_vs_fixed']:+.1%}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = run_joint_sweep()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved distill joint sweep JSON: {TARGET_JSON}")
    print(f"Saved distill joint sweep markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
