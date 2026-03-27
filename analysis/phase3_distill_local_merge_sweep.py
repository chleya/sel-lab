# -*- coding: utf-8 -*-
"""
Local merge-scale sweep around the current best distill configuration on digits_pairs.
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


TARGET_JSON = resolve_exploratory_results_path("phase3_distill_local_merge_sweep.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_DISTILL_LOCAL_MERGE_SWEEP.md"
MERGE_SCALES = (0.01, 0.02, 0.03)
DISTILL_STRENGTH = 0.5


def build_base_config() -> AblationConfig:
    return replace(
        AblationConfig(),
        input_size=64,
        hidden_size=32,
        task_suite="digits_pairs",
        learning_rate=0.02,
        epochs_per_task=35,
        specialist_distill_strength=DISTILL_STRENGTH,
    )


def run_sweep() -> Dict:
    rows: List[Dict] = []
    best_avg = None
    best_fixed = None
    for merge_scale in MERGE_SCALES:
        config = replace(build_base_config(), specialist_merge_scale=merge_scale)
        result = run_phase3_ablations(
            config=config,
            result_filename=f"phase3_distill_local_merge_{str(merge_scale).replace('.', '_')}.json",
            policies=("adapt_only", "task_specialist_clone_distill_merge"),
        )
        distill = result["policies"]["task_specialist_clone_distill_merge"]["summary"]
        row = {
            "merge_scale": merge_scale,
            "distill_strength": DISTILL_STRENGTH,
            "task_specialist_clone_distill_merge": distill,
            "avg_accuracy_delta_vs_adapt_only": distill["avg_accuracy_delta_vs_adapt_only"],
            "forgetting_delta_vs_adapt_only": distill["forgetting_delta_vs_adapt_only"],
            "avg_accuracy_gain_vs_fixed": distill["avg_accuracy_gain_vs_fixed"],
        }
        rows.append(row)
        if best_avg is None or row["avg_accuracy_delta_vs_adapt_only"] > best_avg["avg_accuracy_delta_vs_adapt_only"]:
            best_avg = row
        if best_fixed is None or row["avg_accuracy_gain_vs_fixed"] > best_fixed["avg_accuracy_gain_vs_fixed"]:
            best_fixed = row
    return {
        "task_suite": "digits_pairs",
        "distill_strength": DISTILL_STRENGTH,
        "best_avg_vs_adapt_only": best_avg,
        "closest_to_fixed": best_fixed,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    best_avg = payload["best_avg_vs_adapt_only"]
    best_fixed = payload["closest_to_fixed"]
    lines = [
        "# Phase 3 Distill Local Merge Sweep",
        "",
        "- Benchmark: `digits_pairs`",
        f"- Distill strength: `{payload['distill_strength']:.2f}`",
        f"- Best avg delta vs `adapt_only`: merge `{best_avg['merge_scale']:.2f}`, avg delta {best_avg['avg_accuracy_delta_vs_adapt_only']:+.1%}, forgetting delta {best_avg['forgetting_delta_vs_adapt_only']:+.1%}",
        f"- Closest to fixed: merge `{best_fixed['merge_scale']:.2f}`, avg gain vs fixed {best_fixed['avg_accuracy_gain_vs_fixed']:+.1%}",
        "",
        "## Sweep",
        "",
    ]
    for row in payload["rows"]:
        lines.append(
            f"- merge `{row['merge_scale']:.2f}`: avg vs adapt {row['avg_accuracy_delta_vs_adapt_only']:+.1%}, "
            f"forgetting vs adapt {row['forgetting_delta_vs_adapt_only']:+.1%}, "
            f"vs fixed {row['avg_accuracy_gain_vs_fixed']:+.1%}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = run_sweep()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved distill local merge sweep JSON: {TARGET_JSON}")
    print(f"Saved distill local merge sweep markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
