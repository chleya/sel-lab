# -*- coding: utf-8 -*-
"""
Evaluate Phase 3 main policies on a more realistic digits-pair sequential benchmark.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase3_ablation import AblationConfig, run_phase3_ablations
from core.runtime import resolve_canonical_results_path, save_json


TARGET_JSON = resolve_canonical_results_path("phase3_digits_transfer.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_DIGITS_TRANSFER.md"


def build_config() -> AblationConfig:
    return replace(
        AblationConfig(),
        input_size=64,
        hidden_size=32,
        task_suite="digits_pairs",
        specialist_merge_scale=0.02,
        specialist_distill_strength=0.50,
        learning_rate=0.02,
        epochs_per_task=35,
    )


def run_digits_transfer() -> Dict:
    result = run_phase3_ablations(
        config=build_config(),
        result_filename="phase3_digits_pairs_results.json",
        policies=(
            "adapt_only",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_confidence_gate_merge",
            "task_specialist_clone_distill_merge",
        ),
    )
    baseline = result["policies"]["adapt_only"]["summary"]
    specialist = result["policies"]["task_specialist_clone_limited_merge"]["summary"]
    gated = result["policies"]["task_specialist_clone_confidence_gate_merge"]["summary"]
    distilled = result["policies"]["task_specialist_clone_distill_merge"]["summary"]
    candidate_rows = [
        ("task_specialist_clone_distill_merge", distilled),
        ("task_specialist_clone_confidence_gate_merge", gated),
        ("task_specialist_clone_limited_merge", specialist),
    ]
    best_value = max(item[1]["avg_accuracy_delta_vs_adapt_only"] for item in candidate_rows)
    best_name, best_summary = next(
        item
        for item in candidate_rows
        if abs(item[1]["avg_accuracy_delta_vs_adapt_only"] - best_value) <= 1e-12
    )
    return {
        "task_suite": "digits_pairs",
        "merge_scale": 0.02,
        "distill_strength": 0.50,
        "best_reuse_policy": best_name,
        "adapt_only": baseline,
        "task_specialist_clone_limited_merge": specialist,
        "task_specialist_clone_confidence_gate_merge": gated,
        "task_specialist_clone_distill_merge": distilled,
        "tied_best_reuse_policies": [
            name
            for name, summary in candidate_rows
            if abs(summary["avg_accuracy_delta_vs_adapt_only"] - best_value) <= 1e-12
        ],
        "avg_accuracy_delta_vs_adapt_only": best_summary["avg_accuracy_delta_vs_adapt_only"],
        "forgetting_delta_vs_adapt_only": best_summary["forgetting_delta_vs_adapt_only"],
        "current_accuracy_delta_vs_adapt_only": best_summary["current_accuracy_delta_vs_adapt_only"],
        "prior_accuracy_delta_vs_adapt_only": best_summary["prior_accuracy_delta_vs_adapt_only"],
        "unit_delta_vs_adapt_only": best_summary["unit_delta_vs_adapt_only"],
    }


def render_markdown(payload: Dict) -> str:
    return "\n".join(
        [
            "# Phase 3 Digits Transfer",
            "",
            "- Benchmark: sequential digits-pair binary tasks `(0,1) -> (2,3) -> (4,5) -> (6,7)`",
            f"- Tuned specialist merge scale: `{payload['merge_scale']:.2f}`",
            f"- Tuned distill strength: `{payload['distill_strength']:.2f}`",
            f"- Best reuse policy in this comparison: `{payload['best_reuse_policy']}`",
            f"- Tied best reuse policies at reported precision: `{', '.join(payload['tied_best_reuse_policies'])}`",
            f"- Best avg delta vs adapt_only: {payload['avg_accuracy_delta_vs_adapt_only']:+.1%}",
            f"- Best forgetting delta vs adapt_only: {payload['forgetting_delta_vs_adapt_only']:+.1%}",
            f"- Best current-task delta vs adapt_only: {payload['current_accuracy_delta_vs_adapt_only']:+.1%}",
            f"- Best prior-task delta vs adapt_only: {payload['prior_accuracy_delta_vs_adapt_only']:+.1%}",
            f"- Best unit delta vs adapt_only: {payload['unit_delta_vs_adapt_only']:+.1f}",
            "",
            "## Candidate Comparison",
            "",
            f"- `task_specialist_clone_limited_merge`: avg {payload['task_specialist_clone_limited_merge']['avg_accuracy_delta_vs_adapt_only']:+.1%}, forgetting {payload['task_specialist_clone_limited_merge']['forgetting_delta_vs_adapt_only']:+.1%}, current {payload['task_specialist_clone_limited_merge']['current_accuracy_delta_vs_adapt_only']:+.1%}, prior {payload['task_specialist_clone_limited_merge']['prior_accuracy_delta_vs_adapt_only']:+.1%}",
            f"- `task_specialist_clone_confidence_gate_merge`: avg {payload['task_specialist_clone_confidence_gate_merge']['avg_accuracy_delta_vs_adapt_only']:+.1%}, forgetting {payload['task_specialist_clone_confidence_gate_merge']['forgetting_delta_vs_adapt_only']:+.1%}, current {payload['task_specialist_clone_confidence_gate_merge']['current_accuracy_delta_vs_adapt_only']:+.1%}, prior {payload['task_specialist_clone_confidence_gate_merge']['prior_accuracy_delta_vs_adapt_only']:+.1%}",
            f"- `task_specialist_clone_distill_merge`: avg {payload['task_specialist_clone_distill_merge']['avg_accuracy_delta_vs_adapt_only']:+.1%}, forgetting {payload['task_specialist_clone_distill_merge']['forgetting_delta_vs_adapt_only']:+.1%}, current {payload['task_specialist_clone_distill_merge']['current_accuracy_delta_vs_adapt_only']:+.1%}, prior {payload['task_specialist_clone_distill_merge']['prior_accuracy_delta_vs_adapt_only']:+.1%}",
            "",
        ]
    )


def main() -> None:
    payload = run_digits_transfer()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved digits transfer JSON: {TARGET_JSON}")
    print(f"Saved digits transfer markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
