# -*- coding: utf-8 -*-
"""
Sweep distillation strength for the digits-pairs Phase 3 benchmark.
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


TARGET_JSON = resolve_exploratory_results_path("phase3_distill_sweep.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_DISTILL_SWEEP.md"
STRENGTHS = (0.15, 0.25, 0.35, 0.5, 0.75)


def build_base_config() -> AblationConfig:
    return replace(
        AblationConfig(),
        input_size=64,
        hidden_size=32,
        task_suite="digits_pairs",
        specialist_merge_scale=0.05,
        learning_rate=0.02,
        epochs_per_task=35,
    )


def run_sweep() -> Dict:
    rows: List[Dict] = []
    best = None
    for strength in STRENGTHS:
        config = replace(build_base_config(), specialist_distill_strength=strength)
        result = run_phase3_ablations(
            config=config,
            result_filename=f"phase3_distill_strength_{str(strength).replace('.', '_')}.json",
            policies=("adapt_only", "task_specialist_clone_confidence_gate_merge", "task_specialist_clone_distill_merge"),
        )
        confidence = result["policies"]["task_specialist_clone_confidence_gate_merge"]["summary"]
        distill = result["policies"]["task_specialist_clone_distill_merge"]["summary"]
        row = {
            "distill_strength": strength,
            "task_specialist_clone_confidence_gate_merge": confidence,
            "task_specialist_clone_distill_merge": distill,
            "avg_accuracy_delta_vs_confidence_gate": (
                distill["final_avg_accuracy"] - confidence["final_avg_accuracy"]
            ),
            "forgetting_delta_vs_confidence_gate": (
                confidence["mean_forgetting"] - distill["mean_forgetting"]
            ),
        }
        rows.append(row)
        if best is None or row["avg_accuracy_delta_vs_confidence_gate"] > best["avg_accuracy_delta_vs_confidence_gate"]:
            best = row
    return {
        "task_suite": "digits_pairs",
        "best": best,
        "rows": rows,
    }


def render_markdown(payload: Dict) -> str:
    best = payload["best"]
    lines = [
        "# Phase 3 Distill Sweep",
        "",
        "- Benchmark: `digits_pairs`",
        f"- Best distill strength by avg accuracy delta vs `task_specialist_clone_confidence_gate_merge`: `{best['distill_strength']:.2f}`",
        f"- Best avg delta vs confidence gate: {best['avg_accuracy_delta_vs_confidence_gate']:+.1%}",
        f"- Best forgetting delta vs confidence gate: {best['forgetting_delta_vs_confidence_gate']:+.1%}",
        "",
        "## Sweep",
        "",
    ]
    for row in payload["rows"]:
        lines.append(
            f"- `{row['distill_strength']:.2f}`: avg {row['avg_accuracy_delta_vs_confidence_gate']:+.1%}, "
            f"forgetting {row['forgetting_delta_vs_confidence_gate']:+.1%}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = run_sweep()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved distill sweep JSON: {TARGET_JSON}")
    print(f"Saved distill sweep markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
