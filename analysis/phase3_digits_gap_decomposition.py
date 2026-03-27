# -*- coding: utf-8 -*-
"""
Decompose the broader-digits fixed gap into current-task and prior-task components.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase3_registry import PHASE3_DIAGNOSTIC_SUITES, resolve_phase3_policy_set
from core.runtime import load_json_result, resolve_canonical_results_path, save_json


BROADER_DIGITS_SUITES = PHASE3_DIAGNOSTIC_SUITES["broader_digits"]
BASELINE_RESULT_FILES = {
    "digits_pairs": "phase3_digits_pairs_results.json",
    "digits_pairs_noisy": "phase3_source_aware_digits_pairs_noisy_results.json",
    "digits_pairs_permuted": "phase3_source_aware_digits_pairs_permuted_results.json",
}
POLICY_RESULT_FILES = {
    "adapt_only": BASELINE_RESULT_FILES,
    "task_specialist_clone_limited_merge": BASELINE_RESULT_FILES,
    "task_specialist_clone_distill_merge": BASELINE_RESULT_FILES,
    "task_specialist_clone_role_separation_merge": {
        "digits_pairs": "phase3_role_sep_digits_pairs_results.json",
        "digits_pairs_noisy": "phase3_role_sep_digits_pairs_noisy_results.json",
        "digits_pairs_permuted": "phase3_role_sep_digits_pairs_permuted_results.json",
    },
    "task_specialist_clone_source_aware_merge": {
        "digits_pairs": "phase3_source_aware_digits_pairs_results.json",
        "digits_pairs_noisy": "phase3_source_aware_digits_pairs_noisy_results.json",
        "digits_pairs_permuted": "phase3_source_aware_digits_pairs_permuted_results.json",
    },
    "task_specialist_clone_asymmetric_retention_merge": {
        "digits_pairs": "phase3_asymmetric_retention_digits_pairs_results.json",
        "digits_pairs_noisy": "phase3_asymmetric_retention_digits_pairs_noisy_results.json",
        "digits_pairs_permuted": "phase3_asymmetric_retention_digits_pairs_permuted_results.json",
    },
    "task_specialist_clone_current_path_boost_merge": {
        "digits_pairs": "phase3_current_path_boost_digits_pairs_results.json",
        "digits_pairs_noisy": "phase3_current_path_boost_digits_pairs_noisy_results.json",
        "digits_pairs_permuted": "phase3_current_path_boost_digits_pairs_permuted_results.json",
    },
}
POLICIES = resolve_phase3_policy_set("digits_gap_core")
TARGET_JSON = resolve_canonical_results_path("phase3_digits_gap_decomposition.json")
TARGET_MD = PROJECT_ROOT / "PHASE3_DIGITS_GAP_DECOMPOSITION.md"


def load_result(name: str) -> Dict:
    return load_json_result(name)


def classify_gap(row: Dict) -> str:
    current_gap = abs(row["current_accuracy_gap_vs_fixed"])
    prior_gap = abs(row["prior_accuracy_gap_vs_fixed"])
    if current_gap > prior_gap + 1e-9:
        return "current_task_gap_dominates"
    if prior_gap > current_gap + 1e-9:
        return "prior_task_gap_dominates"
    return "current_and_prior_gaps_are_tied"


def build_payload() -> Dict:
    suite_rows: List[Dict] = []
    aggregate: Dict[str, Dict] = {}

    for suite_name, baseline_filename in BASELINE_RESULT_FILES.items():
        baseline_data = load_result(baseline_filename)
        fixed_summary = baseline_data["fixed"]["summary"]
        suite_policy_rows: Dict[str, Dict] = {}
        for policy_name in POLICIES:
            filename = POLICY_RESULT_FILES[policy_name][suite_name]
            data = load_result(filename)
            if policy_name not in data["policies"]:
                continue
            summary = data["policies"][policy_name]["summary"]
            row = {
                "final_avg_accuracy": summary["final_avg_accuracy"],
                "final_current_accuracy": summary["final_current_accuracy"],
                "final_prior_avg_accuracy": summary["final_prior_avg_accuracy"],
                "avg_accuracy_gap_vs_fixed": summary["final_avg_accuracy"] - fixed_summary["final_avg_accuracy"],
                "current_accuracy_gap_vs_fixed": summary["final_current_accuracy"] - fixed_summary["final_current_accuracy"],
                "prior_accuracy_gap_vs_fixed": summary["final_prior_avg_accuracy"] - fixed_summary["final_prior_avg_accuracy"],
                "forgetting_delta_vs_fixed": summary["mean_forgetting"] - fixed_summary["mean_forgetting"],
                "dominant_gap": None,
            }
            row["dominant_gap"] = classify_gap(row)
            suite_policy_rows[policy_name] = row
            aggregate.setdefault(
                policy_name,
                {
                    "mean_avg_accuracy_gap_vs_fixed": 0.0,
                    "mean_current_accuracy_gap_vs_fixed": 0.0,
                    "mean_prior_accuracy_gap_vs_fixed": 0.0,
                    "mean_forgetting_delta_vs_fixed": 0.0,
                    "suite_count": 0,
                },
            )
            aggregate_row = aggregate[policy_name]
            aggregate_row["mean_avg_accuracy_gap_vs_fixed"] += row["avg_accuracy_gap_vs_fixed"]
            aggregate_row["mean_current_accuracy_gap_vs_fixed"] += row["current_accuracy_gap_vs_fixed"]
            aggregate_row["mean_prior_accuracy_gap_vs_fixed"] += row["prior_accuracy_gap_vs_fixed"]
            aggregate_row["mean_forgetting_delta_vs_fixed"] += row["forgetting_delta_vs_fixed"]
            aggregate_row["suite_count"] += 1

        suite_rows.append(
            {
                "task_suite": suite_name,
                "fixed": fixed_summary,
                "policies": suite_policy_rows,
            }
        )

    for policy_name, row in aggregate.items():
        count = max(1, row.pop("suite_count"))
        row["mean_avg_accuracy_gap_vs_fixed"] /= count
        row["mean_current_accuracy_gap_vs_fixed"] /= count
        row["mean_prior_accuracy_gap_vs_fixed"] /= count
        row["mean_forgetting_delta_vs_fixed"] /= count
        row["dominant_gap"] = classify_gap(
            {
                "current_accuracy_gap_vs_fixed": row["mean_current_accuracy_gap_vs_fixed"],
                "prior_accuracy_gap_vs_fixed": row["mean_prior_accuracy_gap_vs_fixed"],
            }
        )

    best_policy = max(
        (name for name in aggregate if name != "adapt_only"),
        key=lambda name: aggregate[name]["mean_avg_accuracy_gap_vs_fixed"],
    )
    return {
        "task_suites": list(BROADER_DIGITS_SUITES),
        "policies": list(POLICIES),
        "best_policy_by_fixed_gap": best_policy,
        "aggregate": aggregate,
        "rows": suite_rows,
    }


def render_markdown(payload: Dict) -> str:
    lines = [
        "# Phase 3 Digits Gap Decomposition",
        "",
        f"- Digits suites: `{', '.join(payload['task_suites'])}`",
        f"- Best policy by mean avg gap vs fixed: `{payload['best_policy_by_fixed_gap']}`",
        "",
        "## Aggregate Gap Summary",
        "",
    ]
    for policy_name in payload["policies"]:
        if policy_name not in payload["aggregate"]:
            continue
        row = payload["aggregate"][policy_name]
        lines.append(
            f"- `{policy_name}`: avg gap {row['mean_avg_accuracy_gap_vs_fixed']:+.1%}, "
            f"current gap {row['mean_current_accuracy_gap_vs_fixed']:+.1%}, "
            f"prior gap {row['mean_prior_accuracy_gap_vs_fixed']:+.1%}, "
            f"forgetting delta {row['mean_forgetting_delta_vs_fixed']:+.1%}, "
            f"classification `{row['dominant_gap']}`"
        )
    lines.extend(
        [
            "",
            "## Suite Breakdown",
            "",
        ]
    )
    for suite in payload["rows"]:
        lines.append(f"### {suite['task_suite']}")
        lines.append("")
        for policy_name in payload["policies"]:
            if policy_name not in suite["policies"]:
                continue
            row = suite["policies"][policy_name]
            lines.append(
                f"- `{policy_name}`: avg gap {row['avg_accuracy_gap_vs_fixed']:+.1%}, "
                f"current gap {row['current_accuracy_gap_vs_fixed']:+.1%}, "
                f"prior gap {row['prior_accuracy_gap_vs_fixed']:+.1%}, "
                f"forgetting delta {row['forgetting_delta_vs_fixed']:+.1%}, "
                f"classification `{row['dominant_gap']}`"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = build_payload()
    save_json(payload, TARGET_JSON)
    TARGET_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved digits gap decomposition JSON: {TARGET_JSON}")
    print(f"Saved digits gap decomposition markdown: {TARGET_MD}")


if __name__ == "__main__":
    main()
