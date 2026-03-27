# -*- coding: utf-8 -*-
"""
Analyze how current Phase 3 true-reuse candidates compare against adapt_only.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import load_json_result, resolve_exploratory_results_path, save_json

TARGET_JSON = resolve_exploratory_results_path("phase3_gap_analysis.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_GAP_ANALYSIS.md"

BASELINE = "adapt_only"
TRUE_REUSE_CANDIDATES = (
    "best_clone_limited_adapt",
    "benefit_limited_clone",
    "best_clone_delayed_limited_adapt",
    "capped_delayed_limited_clone",
    "task_specialist_clone",
    "task_specialist_clone_limited_merge",
    "task_specialist_clone_confidence_gate_merge",
    "task_specialist_clone_routed_merge",
    "task_specialist_clone_soft_route_merge",
    "task_specialist_clone_budgeted_merge",
    "task_specialist_clone_anchor_merge",
    "task_specialist_clone_distance_anchor_merge",
    "task_specialist_clone_distill_merge",
    "task_specialist_clone_freeze_source",
)


def load_results() -> Dict:
    return load_json_result("phase3_ablation_results.json")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_task_deltas(
    baseline_runs: List[List[Dict]],
    candidate_runs: List[List[Dict]],
) -> List[Dict]:
    task_count = len(baseline_runs[0])
    rows: List[Dict] = []
    for task_id in range(task_count):
        baseline_rows = [run[task_id] for run in baseline_runs]
        candidate_rows = [run[task_id] for run in candidate_runs]
        rows.append(
            {
                "task_id": task_id,
                "current_accuracy_delta": mean(
                    c["current_accuracy"] - b["current_accuracy"]
                    for b, c in zip(baseline_rows, candidate_rows)
                ),
                "prior_accuracy_delta": mean(
                    c["prior_avg_accuracy"] - b["prior_avg_accuracy"]
                    for b, c in zip(baseline_rows, candidate_rows)
                ),
                "forgetting_delta": mean(
                    c["mean_forgetting"] - b["mean_forgetting"]
                    for b, c in zip(baseline_rows, candidate_rows)
                ),
                "unit_delta": mean(
                    c["unit_count"] - b["unit_count"]
                    for b, c in zip(baseline_rows, candidate_rows)
                ),
            }
        )
    return rows


def classify_gap(summary: Dict) -> List[str]:
    findings: List[str] = []
    if summary["avg_accuracy_delta_vs_adapt_only"] >= 0:
        findings.append("matches_or_exceeds_adapt_only")
    else:
        findings.append("below_adapt_only")

    if summary["current_accuracy_delta_vs_adapt_only"] < summary["prior_accuracy_delta_vs_adapt_only"]:
        findings.append("current_task_gap_is_larger_than_retention_gap")
    else:
        findings.append("retention_gap_is_larger_or_equal")

    if summary["forgetting_delta_vs_adapt_only"] < 0:
        findings.append("forgets_more_than_adapt_only")
    else:
        findings.append("forgets_less_or_equal_than_adapt_only")

    if summary["unit_delta_vs_adapt_only"] > 0:
        findings.append("uses_more_units_than_adapt_only")
    else:
        findings.append("does_not_expand_beyond_adapt_only")

    return findings


def build_payload(data: Dict) -> Dict:
    baseline = data["policies"][BASELINE]
    candidates: Dict[str, Dict] = {}
    for name in TRUE_REUSE_CANDIDATES:
        if name not in data["policies"]:
            continue
        candidate = data["policies"][name]
        candidates[name] = {
            "summary": candidate["summary"],
            "task_deltas_vs_adapt_only": summarize_task_deltas(
                baseline["runs"],
                candidate["runs"],
            ),
            "gap_classification": classify_gap(candidate["summary"]),
        }

    closest = max(
        candidates.items(),
        key=lambda item: item[1]["summary"]["avg_accuracy_delta_vs_adapt_only"],
    )
    lowest_forgetting = min(
        candidates.items(),
        key=lambda item: item[1]["summary"]["mean_forgetting"],
    )
    return {
        "baseline": BASELINE,
        "closest_true_reuse": {
            "name": closest[0],
            "avg_accuracy_delta_vs_adapt_only": closest[1]["summary"]["avg_accuracy_delta_vs_adapt_only"],
            "forgetting_delta_vs_adapt_only": closest[1]["summary"]["forgetting_delta_vs_adapt_only"],
        },
        "lowest_forgetting_true_reuse": {
            "name": lowest_forgetting[0],
            "mean_forgetting": lowest_forgetting[1]["summary"]["mean_forgetting"],
            "avg_accuracy_delta_vs_adapt_only": lowest_forgetting[1]["summary"]["avg_accuracy_delta_vs_adapt_only"],
        },
        "candidates": candidates,
    }


def render_markdown(payload: Dict) -> str:
    closest = payload["closest_true_reuse"]
    lowest = payload["lowest_forgetting_true_reuse"]
    lines = [
        "# Phase 3 Gap Analysis",
        "",
        f"- Baseline: `{payload['baseline']}`",
        f"- Leading true-reuse policy: `{closest['name']}` ({closest['avg_accuracy_delta_vs_adapt_only']:+.1%} vs adapt_only, forgetting delta {closest['forgetting_delta_vs_adapt_only']:+.1%})",
        f"- Lowest-forgetting true-reuse policy: `{lowest['name']}` (mean forgetting {lowest['mean_forgetting']:.1%}, avg delta {lowest['avg_accuracy_delta_vs_adapt_only']:+.1%})",
        "",
    ]
    for name, candidate in payload["candidates"].items():
        summary = candidate["summary"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Avg accuracy delta vs adapt_only: {summary['avg_accuracy_delta_vs_adapt_only']:+.1%}",
                f"- Current-task delta vs adapt_only: {summary['current_accuracy_delta_vs_adapt_only']:+.1%}",
                f"- Prior-task delta vs adapt_only: {summary['prior_accuracy_delta_vs_adapt_only']:+.1%}",
                f"- Forgetting delta vs adapt_only: {summary['forgetting_delta_vs_adapt_only']:+.1%}",
                f"- Unit delta vs adapt_only: {summary['unit_delta_vs_adapt_only']:+.1f}",
                f"- Classification: {', '.join(candidate['gap_classification'])}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = build_payload(load_results())
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved gap analysis JSON: {TARGET_JSON}")
    print(f"Saved gap analysis markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
