# -*- coding: utf-8 -*-
"""
Sweep the sparse-route thresholds for the two-stage selector and rank candidates on selector_full_map.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.phase3_benchmark_family import aggregate_policy_summaries, build_stress_benchmark_config
from core.phase3_ablation import run_phase3_ablations
from core.phase3_registry import resolve_phase3_diagnostic_benchmark, resolve_phase3_diagnostic_family
from core.runtime import resolve_canonical_results_path, save_json


TRAIN_SOURCE_JSON = resolve_canonical_results_path("phase3_task_ranking_selector_benchmark.json")
TARGET_JSON = resolve_canonical_results_path("phase3_two_stage_selector_sweep.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_TWO_STAGE_SELECTOR_SWEEP.md"
FAMILY_NAME = "selector_full_map"
FOCUSED_BENCHMARKS = (
    "plasticity_stress",
    "mixed_regime_stress",
    "selector_adversarial_gate",
    "selector_sparse_adversarial_gate",
)
SWEEP_CANDIDATES = (
    (0.385, -0.005),
    (0.365, -0.005),
    (0.405, -0.005),
    (0.385, -0.015),
    (0.385, 0.005),
)


def load_task_ranking_payload() -> Dict[str, object]:
    payload = json.loads(Path(TRAIN_SOURCE_JSON).read_text(encoding="utf-8"))
    selector_payload = payload["selector_payload"]
    return {
        "feature_names": selector_payload["feature_names"],
        "weights": tuple(selector_payload["weights"]),
        "bias": float(selector_payload["bias"]),
        "source_benchmark": str(TRAIN_SOURCE_JSON),
    }


def build_config(task_suite: str, selector_payload: Dict[str, object], *, zero_ratio_floor: float, conflict_delta_floor: float):
    return build_stress_benchmark_config(
        task_suite,
        selector_task_ranking_fit_weights=selector_payload["weights"],
        selector_task_ranking_fit_bias=selector_payload["bias"],
        selector_sparse_zero_ratio_floor=zero_ratio_floor,
        selector_sparse_conflict_delta_floor=conflict_delta_floor,
    )


def evaluate_candidate(selector_payload: Dict[str, object], *, zero_ratio_floor: float, conflict_delta_floor: float) -> Dict[str, object]:
    benchmark_names = FOCUSED_BENCHMARKS
    rows: List[Dict] = []
    policies = (
        "adapt_only",
        "task_specialist_clone_two_stage_selector_merge",
    )
    for benchmark_name in benchmark_names:
        spec = resolve_phase3_diagnostic_benchmark(benchmark_name)
        task_suite = str(spec["task_suite"])
        result = run_phase3_ablations(
            config=build_config(
                task_suite,
                selector_payload,
                zero_ratio_floor=zero_ratio_floor,
                conflict_delta_floor=conflict_delta_floor,
            ),
            result_filename=(
                "phase3_two_stage_selector_sweep_"
                f"zr_{zero_ratio_floor:.3f}_cd_{conflict_delta_floor:.3f}_{benchmark_name}_{task_suite}.json"
            ),
            policies=policies,
        )
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "task_suite": task_suite,
                "summaries": {name: result["policies"][name]["summary"] for name in policies},
            }
        )
    aggregate = aggregate_policy_summaries(rows, policies)
    target = aggregate["task_specialist_clone_two_stage_selector_merge"]
    embedded_gain = next(
        row["summaries"]["task_specialist_clone_two_stage_selector_merge"]["avg_accuracy_gain_vs_fixed"]
        for row in rows
        if row["benchmark_name"] == "selector_adversarial_gate"
    )
    sparse_gain = next(
        row["summaries"]["task_specialist_clone_two_stage_selector_merge"]["avg_accuracy_gain_vs_fixed"]
        for row in rows
        if row["benchmark_name"] == "selector_sparse_adversarial_gate"
    )
    feasible = embedded_gain >= 0.0 and target["mean_avg_accuracy_gain_vs_fixed"] >= 0.05
    score = (
        (1000.0 if feasible else 0.0)
        + 8.0 * sparse_gain
        + 2.0 * embedded_gain
        + target["mean_avg_accuracy_gain_vs_fixed"]
    )
    return {
        "zero_ratio_floor": zero_ratio_floor,
        "conflict_delta_floor": conflict_delta_floor,
        "aggregate": target,
        "embedded_gain_vs_fixed": embedded_gain,
        "sparse_gain_vs_fixed": sparse_gain,
        "feasible": feasible,
        "score": score,
    }


def run_sweep() -> Dict[str, object]:
    selector_payload = load_task_ranking_payload()
    candidates = []
    for zero_ratio_floor, conflict_delta_floor in SWEEP_CANDIDATES:
        print("\n" + "=" * 60)
        print(
            "Phase 3 Two-Stage Selector Sweep: "
            f"zero_ratio_floor={zero_ratio_floor:.3f}, conflict_delta_floor={conflict_delta_floor:.3f}"
        )
        print("=" * 60)
        candidates.append(
            evaluate_candidate(
                selector_payload,
                zero_ratio_floor=zero_ratio_floor,
                conflict_delta_floor=conflict_delta_floor,
            )
        )
    ranked = sorted(candidates, key=lambda row: (row["score"], row["sparse_gain_vs_fixed"]), reverse=True)
    best = ranked[0]
    return {
        "family_name": FAMILY_NAME,
        "focused_benchmarks": list(FOCUSED_BENCHMARKS),
        "selector_payload_source": selector_payload["source_benchmark"],
        "grid": {
            "candidates": [
                {
                    "zero_ratio_floor": zero_ratio_floor,
                    "conflict_delta_floor": conflict_delta_floor,
                }
                for zero_ratio_floor, conflict_delta_floor in SWEEP_CANDIDATES
            ],
        },
        "best_candidate": best,
        "candidates": ranked,
    }


def render_markdown(payload: Dict[str, object]) -> str:
    lines = [
        "# Phase 3 Two-Stage Selector Sweep",
        "",
        "Sparse-route threshold sweep for the two-stage selector on `selector_full_map`.",
        "",
        f"- Family: `{payload['family_name']}`",
        f"- Focused benchmarks: {', '.join(f'`{name}`' for name in payload['focused_benchmarks'])}",
        f"- Source task-ranking payload: `{payload['selector_payload_source']}`",
        "",
        "## Best Candidate",
        "",
    ]
    best = payload["best_candidate"]
    lines.extend(
        [
            f"- `zero_ratio_floor={best['zero_ratio_floor']:.3f}`",
            f"- `conflict_delta_floor={best['conflict_delta_floor']:.3f}`",
            f"- aggregate gain vs fixed `{best['aggregate']['mean_avg_accuracy_gain_vs_fixed']:+.1%}`",
            f"- aggregate delta vs adapt_only `{best['aggregate']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%}`",
            f"- embedded gain vs fixed `{best['embedded_gain_vs_fixed']:+.1%}`",
            f"- sparse gain vs fixed `{best['sparse_gain_vs_fixed']:+.1%}`",
            f"- feasible `{best['feasible']}`",
        ]
    )
    lines.extend(["", "## Top Candidates", ""])
    for row in payload["candidates"][:10]:
        lines.append(
            f"- `zr={row['zero_ratio_floor']:.3f}`, `cd={row['conflict_delta_floor']:.3f}`: "
            f"aggregate {row['aggregate']['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs fixed, "
            f"embedded {row['embedded_gain_vs_fixed']:+.1%}, "
            f"sparse {row['sparse_gain_vs_fixed']:+.1%}, "
            f"feasible={row['feasible']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = run_sweep()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved two-stage selector sweep JSON: {TARGET_JSON}")
    print(f"Saved two-stage selector sweep markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
