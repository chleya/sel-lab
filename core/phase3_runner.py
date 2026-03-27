"""
Runner and summary helpers for phase-3 structure-reuse ablations.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from core.phase3_common import create_task, save_phase3_ablation_results
from core.phase3_model import AblationConfig, FixedNetwork, ReusePolicyNetwork
from core.phase3_registry import (
    PHASE3_DIAGNOSTIC_SUITES,
    build_phase3_policy_metadata,
    diagnostic_family_for_task_suite,
    resolve_phase3_policies,
)

def summarize_policy_runs(results: List[List[Dict]]) -> Dict:
    final_rows = [run[-1] for run in results]
    final_avg_accuracy = float(np.mean([row["avg_accuracy"] for row in final_rows]))
    final_current_accuracy = float(np.mean([row["current_accuracy"] for row in final_rows]))
    final_prior_accuracy = float(np.mean([row["prior_avg_accuracy"] for row in final_rows]))
    final_units = float(np.mean([row["unit_count"] for row in final_rows]))
    final_frozen = float(np.mean([row["frozen_units"] for row in final_rows]))
    mean_forgetting = float(np.mean([row["mean_forgetting"] for row in final_rows]))
    mean_backward_transfer = float(np.mean([row["mean_backward_transfer"] for row in final_rows]))

    return {
        "final_avg_accuracy": final_avg_accuracy,
        "final_current_accuracy": final_current_accuracy,
        "final_prior_avg_accuracy": final_prior_accuracy,
        "mean_forgetting": mean_forgetting,
        "mean_backward_transfer": mean_backward_transfer,
        "final_units": final_units,
        "final_frozen_units": final_frozen,
    }


def compare_against_fixed(policy_summary: Dict, fixed_summary: Dict) -> Dict:
    return {
        "avg_accuracy_gain_vs_fixed": float(
            policy_summary["final_avg_accuracy"] - fixed_summary["final_avg_accuracy"]
        ),
        "forgetting_delta_vs_fixed": float(
            fixed_summary["mean_forgetting"] - policy_summary["mean_forgetting"]
        ),
        "backward_transfer_gain_vs_fixed": float(
            policy_summary["mean_backward_transfer"] - fixed_summary["mean_backward_transfer"]
        ),
    }


def compare_against_reference(policy_summary: Dict, reference_summary: Dict, prefix: str) -> Dict:
    return {
        f"avg_accuracy_delta_vs_{prefix}": float(
            policy_summary["final_avg_accuracy"] - reference_summary["final_avg_accuracy"]
        ),
        f"forgetting_delta_vs_{prefix}": float(
            reference_summary["mean_forgetting"] - policy_summary["mean_forgetting"]
        ),
        f"backward_transfer_delta_vs_{prefix}": float(
            policy_summary["mean_backward_transfer"] - reference_summary["mean_backward_transfer"]
        ),
        f"current_accuracy_delta_vs_{prefix}": float(
            policy_summary["final_current_accuracy"] - reference_summary["final_current_accuracy"]
        ),
        f"prior_accuracy_delta_vs_{prefix}": float(
            policy_summary["final_prior_avg_accuracy"] - reference_summary["final_prior_avg_accuracy"]
        ),
        f"unit_delta_vs_{prefix}": float(
            policy_summary["final_units"] - reference_summary["final_units"]
        ),
    }


def run_phase3_ablations(
    config: AblationConfig | None = None,
    result_filename: str = "phase3_ablation_results.json",
    policies: Sequence[str] | None = None,
) -> Dict:
    config = config or AblationConfig()
    policies = resolve_phase3_policies(config, policies)
    policy_metadata = build_phase3_policy_metadata(policies)
    diagnostic_family = diagnostic_family_for_task_suite(config.task_suite)

    print("\n" + "=" * 60)
    print("Phase 3: Structure Reuse Ablations")
    print("=" * 60)

    fixed_runs: List[List[Dict]] = []
    policy_runs: Dict[str, List[List[Dict]]] = {policy: [] for policy in policies}

    for run in range(config.runs):
        print(f"\n--- Run {run + 1}/{config.runs} ---")
        seed = run * 100 + 42
        fixed = FixedNetwork(config, seed=seed)
        learners = {
            policy: ReusePolicyNetwork(config, policy=policy, seed=seed)
            for policy in policies
        }

        fixed_task_rows: List[Dict] = []
        policy_task_rows: Dict[str, List[Dict]] = {policy: [] for policy in policies}
        fixed_initial_by_task: Dict[int, float] = {}
        policy_initial_by_task: Dict[str, Dict[int, float]] = {policy: {} for policy in policies}

        for task_id in range(config.num_tasks):
            X, y = create_task(task_id, task_suite=config.task_suite)

            for epoch in range(config.epochs_per_task):
                for i in range(len(X)):
                    fixed.learn(X[i], y[i], lr=config.learning_rate)
                    for learner in learners.values():
                        learner.learn(X[i], y[i], lr=config.learning_rate)

                if (epoch + 1) % config.evolve_interval == 0:
                    for learner in learners.values():
                        learner.evolve()

            fixed_scores = {}
            policy_scores: Dict[str, Dict[int, float]] = {policy: {} for policy in policies}
            for prev_task in range(task_id + 1):
                X_test, y_test = create_task(prev_task, task_suite=config.task_suite)
                fixed_scores[prev_task] = fixed.accuracy(X_test, y_test)
                for policy, learner in learners.items():
                    policy_scores[policy][prev_task] = learner.accuracy(X_test, y_test)

            if task_id not in fixed_initial_by_task:
                fixed_initial_by_task[task_id] = fixed_scores[task_id]
            for policy in policies:
                if task_id not in policy_initial_by_task[policy]:
                    policy_initial_by_task[policy][task_id] = policy_scores[policy][task_id]

            fixed_forgetting = [
                fixed_initial_by_task[prev_task] - fixed_scores[prev_task]
                for prev_task in range(task_id)
            ]
            fixed_task_rows.append(
                {
                    "task_id": task_id,
                    "current_accuracy": fixed_scores[task_id],
                    "avg_accuracy": float(np.mean(list(fixed_scores.values()))),
                    "prior_avg_accuracy": float(
                        np.mean([fixed_scores[t] for t in range(task_id)]) if task_id > 0 else fixed_scores[task_id]
                    ),
                    "mean_forgetting": float(np.mean(fixed_forgetting)) if fixed_forgetting else 0.0,
                    "mean_backward_transfer": float(-np.mean(fixed_forgetting)) if fixed_forgetting else 0.0,
                    "unit_count": 1,
                    "frozen_units": 0,
                }
            )

            for policy, learner in learners.items():
                forgetting = [
                    policy_initial_by_task[policy][prev_task] - policy_scores[policy][prev_task]
                    for prev_task in range(task_id)
                ]
                policy_task_rows[policy].append(
                    {
                        "task_id": task_id,
                        "current_accuracy": policy_scores[policy][task_id],
                        "avg_accuracy": float(np.mean(list(policy_scores[policy].values()))),
                        "prior_avg_accuracy": float(
                            np.mean([policy_scores[policy][t] for t in range(task_id)])
                            if task_id > 0
                            else policy_scores[policy][task_id]
                        ),
                        "mean_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
                        "mean_backward_transfer": float(-np.mean(forgetting)) if forgetting else 0.0,
                        "unit_count": learner.active_units,
                        "frozen_units": learner.frozen_units,
                    }
                )
                learner.register_task_outcome(policy_task_rows[policy][-1])
                learner.reset_after_task()

            print(
                f"  Task {task_id}: "
                f"fixed={fixed_scores[task_id]:.0%}, "
                + ", ".join(f"{policy}={policy_scores[policy][task_id]:.0%}" for policy in policies)
            )

        fixed_runs.append(fixed_task_rows)
        for policy in policies:
            policy_runs[policy].append(policy_task_rows[policy])

    fixed_summary = summarize_policy_runs(fixed_runs)
    policy_summaries = {}
    for policy in policies:
        summary = summarize_policy_runs(policy_runs[policy])
        summary.update(compare_against_fixed(summary, fixed_summary))
        policy_summaries[policy] = summary
    adapt_summary = policy_summaries["adapt_only"]
    for policy in policies:
        if policy == "adapt_only":
            continue
        policy_summaries[policy].update(compare_against_reference(policy_summaries[policy], adapt_summary, "adapt_only"))
    policy_summaries["adapt_only"].update(
        {
            "avg_accuracy_delta_vs_adapt_only": 0.0,
            "forgetting_delta_vs_adapt_only": 0.0,
            "backward_transfer_delta_vs_adapt_only": 0.0,
            "current_accuracy_delta_vs_adapt_only": 0.0,
            "prior_accuracy_delta_vs_adapt_only": 0.0,
            "unit_delta_vs_adapt_only": 0.0,
        }
    )

    payload = {
        "config": {
            "input_size": config.input_size,
            "hidden_size": config.hidden_size,
            "output_size": config.output_size,
            "runs": config.runs,
            "epochs_per_task": config.epochs_per_task,
            "num_tasks": config.num_tasks,
            "learning_rate": config.learning_rate,
            "capped_clone_max_units": config.capped_clone_max_units,
            "tension_threshold": config.tension_threshold,
            "evolve_interval": config.evolve_interval,
            "low_tension_ratio": config.low_tension_ratio,
            "clone_margin_ratio": config.clone_margin_ratio,
            "trend_window": config.trend_window,
            "trend_max_increase": config.trend_max_increase,
            "benefit_score_threshold": config.benefit_score_threshold,
            "benefit_reopen_margin": config.benefit_reopen_margin,
            "limited_adapt_scale": config.limited_adapt_scale,
            "specialist_merge_scale": config.specialist_merge_scale,
            "retention_merge_bonus": config.retention_merge_bonus,
            "specialist_confidence_threshold": config.specialist_confidence_threshold,
            "specialist_min_gate_scale": config.specialist_min_gate_scale,
            "specialist_route_min_scale": config.specialist_route_min_scale,
            "specialist_softmax_temperature": config.specialist_softmax_temperature,
            "specialist_budget_scale": config.specialist_budget_scale,
            "specialist_source_agreement_bonus": config.specialist_source_agreement_bonus,
            "specialist_source_conflict_floor": config.specialist_source_conflict_floor,
            "specialist_anchor_strength": config.specialist_anchor_strength,
            "specialist_distance_anchor_bonus": config.specialist_distance_anchor_bonus,
            "specialist_distill_strength": config.specialist_distill_strength,
            "specialist_retention_strength": config.specialist_retention_strength,
            "current_path_lr_boost": config.current_path_lr_boost,
            "archived_specialist_quantization_bits": config.archived_specialist_quantization_bits,
            "archived_specialist_max_count": config.archived_specialist_max_count,
            "archived_specialist_selection_mode": config.archived_specialist_selection_mode,
            "delayed_adapt_tasks": config.delayed_adapt_tasks,
            "freeze_windows": list(config.freeze_windows),
            "task_suite": config.task_suite,
            "diagnostic_family": diagnostic_family,
        },
        "policy_metadata": policy_metadata,
        "diagnostic_suites": PHASE3_DIAGNOSTIC_SUITES,
        "fixed": {
            "summary": fixed_summary,
            "runs": fixed_runs,
        },
        "policies": {
            policy: {
                "summary": policy_summaries[policy],
                "runs": policy_runs[policy],
            }
            for policy in policies
        },
    }

    print("\n" + "=" * 60)
    print("Phase 3 Ablation Summary")
    print("=" * 60)
    print(
        f"Fixed: avg={fixed_summary['final_avg_accuracy']:.1%}, "
        f"forgetting={fixed_summary['mean_forgetting']:.1%}"
    )
    for policy in policies:
        summary = policy_summaries[policy]
        print(
            f"{policy}: avg={summary['final_avg_accuracy']:.1%}, "
            f"gain={summary['avg_accuracy_gain_vs_fixed']:+.1%}, "
            f"forgetting={summary['mean_forgetting']:.1%}, "
            f"bwt={summary['mean_backward_transfer']:+.1%}"
        )

    target = save_phase3_ablation_results(payload, filename=result_filename)
    print(f"\nResults saved: {target}")
    return payload


if __name__ == "__main__":
    run_phase3_ablations()
