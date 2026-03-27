# -*- coding: utf-8 -*-
"""
Generate a unified experiment report from current SEL-Lab results.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import load_json_result, resolve_canonical_results_path, resolve_results_path, save_json
from orchestrator.experiment_runner import run_experiments


CANONICAL_RESULT_FILES = {
    "phase2": "phase2_results.json",
    "phase2_hard": "phase2_hard_results.json",
    "phase2_improved": "phase2_improved_results.json",
    "phase2_ablation": "phase2_ablation_results.json",
    "phase3": "phase3_results.json",
    "phase3_ablation": "phase3_ablation_results.json",
    "phase3_digits": "phase3_digits_transfer.json",
    "phase3_digits_benchmark": "phase3_digits_benchmark_transfer.json",
    "phase3_role_separation": "phase3_role_separation_benchmark.json",
    "phase3_source_aware": "phase3_source_aware_benchmark.json",
    "phase3_asymmetric_retention": "phase3_asymmetric_retention_benchmark.json",
    "phase3_digits_gap": "phase3_digits_gap_decomposition.json",
    "phase3_diagnostic_suite": "phase3_diagnostic_suite.json",
    "phase3_interference_diagnostic": "phase3_interference_diagnostic.json",
    "phase3_mode_switch_diagnostic": "phase3_mode_switch_diagnostic.json",
    "phase3_dual_mode_update_diagnostic": "phase3_dual_mode_update_diagnostic.json",
    "phase3_diagnostic_benchmark_family": "phase3_diagnostic_benchmark_family.json",
    "phase3_regime_switch_oracle": "phase3_regime_switch_oracle.json",
    "phase3_regime_switch_benchmark": "phase3_regime_switch_benchmark.json",
    "phase3_diagnostic_selector_benchmark": "phase3_diagnostic_selector_benchmark.json",
    "phase3_fingerprint_selector_benchmark": "phase3_fingerprint_selector_benchmark.json",
    "phase3_selector_adversarial_benchmark": "phase3_selector_adversarial_benchmark.json",
    "phase3_current_path_boost": "phase3_current_path_boost_benchmark.json",
    "phase3_dual_path_boost": "phase3_dual_path_boost_benchmark.json",
    "phase3_dual_path_decoupled": "phase3_dual_path_decoupled_benchmark.json",
    "phase3_external_memory_boost": "phase3_external_memory_boost_benchmark.json",
    "phase3_external_memory_routed": "phase3_external_memory_routed_benchmark.json",
    "phase3_two_speed_boost": "phase3_two_speed_boost_benchmark.json",
    "phase3_decoupled_objective_boost": "phase3_decoupled_objective_boost_benchmark.json",
    "phase3_archived_specialist_quant": "phase3_archived_specialist_quantization_benchmark.json",
    "phase3_quantized_memory_path": "phase3_quantized_memory_path_benchmark.json",
    "phase3_internal_memory_budget": "phase3_internal_memory_budget_benchmark.json",
    "phase3_internal_memory_selection": "phase3_internal_memory_selection_benchmark.json",
    "phase3_internal_memory_representation": "phase3_internal_memory_representation_benchmark.json",
    "phase3_internal_memory_weight_prototype": "phase3_internal_memory_weight_prototype_benchmark.json",
    "phase3_internal_memory_key_value": "phase3_internal_memory_key_value_benchmark.json",
    "phase4": "phase4_results.json",
    "phase4_real": "phase4_real_results.json",
    "phase4_repair": "phase4_repair_sweep.json",
    "phase4_expansion": "phase4_expansion_sweep.json",
}

def load_canonical_results() -> Dict[str, Dict]:
    return {
        summary_key: load_json_result(filename)
        for summary_key, filename in CANONICAL_RESULT_FILES.items()
    }


def build_summary() -> Dict:
    core_summary = run_experiments(3)
    canonical_results = load_canonical_results()

    return {
        "core": {
            "mean_test_accuracy": core_summary["mean_test_accuracy"],
            "std_test_accuracy": core_summary["std_test_accuracy"],
            "max_test_accuracy": core_summary["max_test_accuracy"],
            "mean_modules": core_summary["mean_modules"],
            "runs": core_summary["results"],
        },
        **canonical_results,
    }


def render_markdown(summary: Dict) -> str:
    core = summary["core"]
    phase2 = summary["phase2"]
    phase2_hard = summary["phase2_hard"]
    phase2_improved = summary["phase2_improved"]
    phase2_ablation = summary["phase2_ablation"]
    phase3 = summary["phase3"]
    phase3_ablation = summary["phase3_ablation"]
    phase3_digits = summary["phase3_digits"]
    phase3_digits_benchmark = summary["phase3_digits_benchmark"]
    phase3_role_separation = summary["phase3_role_separation"]
    phase3_source_aware = summary["phase3_source_aware"]
    phase3_asymmetric_retention = summary["phase3_asymmetric_retention"]
    phase3_digits_gap = summary["phase3_digits_gap"]
    phase3_diagnostic_suite = summary["phase3_diagnostic_suite"]
    phase3_interference_diagnostic = summary["phase3_interference_diagnostic"]
    phase3_mode_switch_diagnostic = summary["phase3_mode_switch_diagnostic"]
    phase3_dual_mode_update_diagnostic = summary["phase3_dual_mode_update_diagnostic"]
    phase3_diagnostic_benchmark_family = summary["phase3_diagnostic_benchmark_family"]
    phase3_regime_switch_oracle = summary["phase3_regime_switch_oracle"]
    phase3_regime_switch_benchmark = summary["phase3_regime_switch_benchmark"]
    phase3_diagnostic_selector_benchmark = summary["phase3_diagnostic_selector_benchmark"]
    phase3_fingerprint_selector_benchmark = summary["phase3_fingerprint_selector_benchmark"]
    phase3_selector_adversarial_benchmark = summary["phase3_selector_adversarial_benchmark"]
    phase3_current_path_boost = summary["phase3_current_path_boost"]
    phase3_dual_path_boost = summary["phase3_dual_path_boost"]
    phase3_dual_path_decoupled = summary["phase3_dual_path_decoupled"]
    phase3_external_memory_boost = summary["phase3_external_memory_boost"]
    phase3_external_memory_routed = summary["phase3_external_memory_routed"]
    phase3_two_speed_boost = summary["phase3_two_speed_boost"]
    phase3_decoupled_objective_boost = summary["phase3_decoupled_objective_boost"]
    phase3_archived_specialist_quant = summary["phase3_archived_specialist_quant"]
    phase3_quantized_memory_path = summary["phase3_quantized_memory_path"]
    phase3_internal_memory_budget = summary["phase3_internal_memory_budget"]
    phase3_internal_memory_selection = summary["phase3_internal_memory_selection"]
    phase3_internal_memory_representation = summary["phase3_internal_memory_representation"]
    phase3_internal_memory_weight_prototype = summary["phase3_internal_memory_weight_prototype"]
    phase3_internal_memory_key_value = summary["phase3_internal_memory_key_value"]
    phase4 = summary["phase4"]
    phase4_real = summary["phase4_real"]
    phase4_repair = summary["phase4_repair"]["best"]
    phase4_expansion = summary["phase4_expansion"]["best_expanding"]
    policy_items = phase3_ablation["policies"].items()
    best_phase3_policy_name, best_phase3_policy = max(
        policy_items,
        key=lambda item: item[1]["summary"]["avg_accuracy_gain_vs_fixed"],
    )
    lowest_forgetting_name, lowest_forgetting_policy = min(
        phase3_ablation["policies"].items(),
        key=lambda item: item[1]["summary"]["mean_forgetting"],
    )
    degenerate_trigger_candidates = {
        "low_tension_clone",
        "margin_clone",
        "margin_limited_clone",
        "trend_limited_clone",
    }
    closest_degenerate_name, _ = max(
        [
            item
            for item in phase3_ablation["policies"].items()
            if item[0] in degenerate_trigger_candidates
        ],
        key=lambda item: item[1]["summary"]["avg_accuracy_delta_vs_adapt_only"],
    )
    true_reuse_candidates = {
        "best_clone",
        "best_clone_limited_adapt",
        "benefit_limited_clone",
        "best_clone_delayed_limited_adapt",
        "best_clone_freeze_1",
        "best_clone_freeze_2",
        "best_clone_freeze_3",
        "task_specialist_clone",
        "task_specialist_clone_limited_merge",
        "task_specialist_clone_confidence_gate_merge",
        "task_specialist_clone_routed_merge",
        "task_specialist_clone_soft_route_merge",
        "task_specialist_clone_budgeted_merge",
        "task_specialist_clone_anchor_merge",
        "task_specialist_clone_distance_anchor_merge",
        "task_specialist_clone_distill_merge",
        "task_specialist_clone_current_path_boost_merge",
        "task_specialist_clone_freeze_source",
        "capped_delayed_limited_clone",
    }
    closest_true_reuse_name, closest_true_reuse_policy = max(
        [
            item
            for item in phase3_ablation["policies"].items()
            if item[0] in true_reuse_candidates
        ],
        key=lambda item: item[1]["summary"]["avg_accuracy_delta_vs_adapt_only"],
    )
    true_reuse_name, true_reuse_policy = max(
        [
            item
            for item in phase3_ablation["policies"].items()
            if item[0] in true_reuse_candidates
        ],
        key=lambda item: item[1]["summary"]["avg_accuracy_gain_vs_fixed"],
    )
    simple_phase2_name, simple_phase2_policy = max(
        phase2_ablation["tasks"]["simple"]["policies"].items(),
        key=lambda item: item[1]["summary"]["final_gain_vs_fixed"],
    )
    hard_phase2_name, hard_phase2_policy = max(
        phase2_ablation["tasks"]["hard"]["policies"].items(),
        key=lambda item: item[1]["summary"]["final_gain_vs_fixed"],
    )
    current_path_boost_hardest = min(
        phase3_current_path_boost["rows"],
        key=lambda row: row["task_specialist_clone_current_path_boost_merge"]["avg_accuracy_gain_vs_fixed"],
    )
    broader_digits_leader = phase3_current_path_boost["aggregate"]["task_specialist_clone_current_path_boost_merge"]

    lines = [
        "# Unified SEL-Lab Report",
        "",
        "## Executive Summary",
        "",
        f"- Core SEL trainer is stable on the simple classification benchmark: mean test accuracy {core['mean_test_accuracy']:.1%} with std {core['std_test_accuracy']:.1%}.",
        f"- Phase 2 remains mixed: simple task advantage {phase2['advantage']:+.1%}, hard task advantage {phase2_hard['advantage']:+.1%}, knowledge-reuse variant {phase2_improved['advantage']:+.1%}.",
        f"- Phase 3 is the clearest success: incremental-learning advantage {phase3['advantage']:+.1%} with lower forgetting ({phase3['forgetting_evolving']:+.1%} vs {phase3['forgetting_fixed']:+.1%}).",
        f"- Phase 4 is now repaired: digits benchmark advantage {phase4['advantage']:+.1%}, real-entrypoint advantage {phase4_real['advantage']:+.1%}.",
        "",
        "## Research Position",
        "",
        "- The project's main claim is now narrower than older static reports suggest.",
        "- SEL currently looks most defensible as a continual-learning and structure-reuse framework.",
        "- Single-task benchmarks are treated as mechanism screens and boundary checks, not as the main justification for the project.",
        "",
        "## Key Findings",
        "",
        "1. Structural evolution is not universally beneficial under all dynamics settings.",
        f"   Phase 2 knowledge reuse currently underperforms by {abs(phase2_improved['advantage']):.1%}.",
        f"   The current Phase 2 mechanism leaders are `{simple_phase2_name}` on the simple task ({simple_phase2_policy['summary']['final_gain_vs_fixed']:+.1%}) and `{hard_phase2_name}` on the hard task ({hard_phase2_policy['summary']['final_gain_vs_fixed']:+.1%}).",
        "2. Incremental learning benefits from evolution much more clearly than single-task learning.",
        f"   Phase 3 improves cross-task average accuracy from {phase3['final_avg_fixed']:.1%} to {phase3['final_avg_evolving']:.1%}.",
        "   The current strong baseline is "
        f"`adapt_only`, but the broader-digits leading mechanism is now `task_specialist_clone_current_path_boost_merge` at {broader_digits_leader['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only` and {broader_digits_leader['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs fixed.",
        "3. Phase 4 failure was initially caused by bad optimization dynamics, not by the evolution idea itself.",
        f"   The best repair sweep setting was `{phase4_repair['name']}` with final gap {phase4_repair['final_accuracy_gap']:+.1%}.",
        "4. Explicit expansion in Phase 4 remains a mechanism under test rather than a settled win.",
        f"   Best expanding setup: threshold {phase4_expansion['threshold']:.2f}, final gap {phase4_expansion['final_accuracy_gap']:+.1%}.",
        "",
        "## Phase 3 Ablations",
        "",
        f"- Main baseline: `adapt_only` ({phase3_ablation['policies']['adapt_only']['summary']['avg_accuracy_gain_vs_fixed']:+.1%} vs fixed).",
        f"- Closest true-reuse policy vs `adapt_only`: `{closest_true_reuse_name}` ({closest_true_reuse_policy['summary']['avg_accuracy_delta_vs_adapt_only']:+.1%} avg accuracy, {closest_true_reuse_policy['summary']['forgetting_delta_vs_adapt_only']:+.1%} forgetting delta).",
        f"- Lowest-forgetting policy: `{lowest_forgetting_name}` (mean forgetting {lowest_forgetting_policy['summary']['mean_forgetting']:.1%}).",
        f"- Strongest true reuse candidate: `{true_reuse_name}` ({true_reuse_policy['summary']['avg_accuracy_gain_vs_fixed']:+.1%} vs fixed).",
        f"- Degenerate trigger variants like `{closest_degenerate_name}` are retained in the full matrix, but not treated as main challengers because they collapse toward `adapt_only` behavior.",
        "- Current reading: the full synthetic ablation matrix still shows specialist/distillation reuse as a meaningful positive family, but it is no longer the canonical boundary story.",
        "- Canonical boundary story now lives in broader digits, where current-task plasticity changes displaced low-merge distillation as the leading mechanism.",
        "",
        "## Phase 3 Digits Boundary",
        "",
        f"- Base digits benchmark: sequential digits-pair transfer still has a tuned narrow-point winner `{phase3_digits['best_reuse_policy']}`.",
        f"- Base digits narrow-point delta vs `adapt_only`: {phase3_digits['avg_accuracy_delta_vs_adapt_only']:+.1%} average accuracy and {phase3_digits['forgetting_delta_vs_adapt_only']:+.1%} forgetting delta.",
        f"- Base digits tuned distill configuration: merge scale {phase3_digits['merge_scale']:.2f}, distill strength {phase3_digits.get('distill_strength', 0.0):.2f}.",
        f"- Earlier broader-digits robustness baseline before plasticity changes: `{phase3_digits_benchmark['overall_best_policy']}` winning {phase3_digits_benchmark['aggregate'][phase3_digits_benchmark['overall_best_policy']]['suite_wins']} suites.",
        f"- Current broader-digits leader: `task_specialist_clone_current_path_boost_merge` with mean avg delta {broader_digits_leader['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only`, mean current-task delta {broader_digits_leader['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, mean gain vs fixed {broader_digits_leader['mean_avg_accuracy_gain_vs_fixed']:+.1%}, and suite wins {broader_digits_leader['suite_wins']}.",
        f"- Negative role-separation check: `task_specialist_clone_role_separation_merge` reached mean avg delta {phase3_role_separation['aggregate']['task_specialist_clone_role_separation_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only` and won {phase3_role_separation['aggregate']['task_specialist_clone_role_separation_merge']['suite_wins']} suites.",
        f"- Negative source-aware check: `task_specialist_clone_source_aware_merge` reached mean avg delta {phase3_source_aware['aggregate']['task_specialist_clone_source_aware_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only` and won {phase3_source_aware['aggregate']['task_specialist_clone_source_aware_merge']['suite_wins']} suites.",
        f"- Negative asymmetric-update check: `task_specialist_clone_asymmetric_retention_merge` reached mean avg delta {phase3_asymmetric_retention['aggregate']['task_specialist_clone_asymmetric_retention_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only` and won {phase3_asymmetric_retention['aggregate']['task_specialist_clone_asymmetric_retention_merge']['suite_wins']} suites.",
        f"- Positive current-path-plasticity check: `task_specialist_clone_current_path_boost_merge` reaches mean avg delta {broader_digits_leader['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only`, mean current-task delta {broader_digits_leader['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, and wins {broader_digits_leader['suite_wins']} suites.",
        f"- Negative dual-path split check: `task_specialist_clone_dual_path_boost_merge` reaches mean avg delta {phase3_dual_path_boost['aggregate']['task_specialist_clone_dual_path_boost_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only`, mean current-task delta {phase3_dual_path_boost['aggregate']['task_specialist_clone_dual_path_boost_merge']['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, mean gain vs fixed {phase3_dual_path_boost['aggregate']['task_specialist_clone_dual_path_boost_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%}, and wins {phase3_dual_path_boost['aggregate']['task_specialist_clone_dual_path_boost_merge']['suite_wins']} suites.",
        f"- Negative decoupled-loss split check: `task_specialist_clone_dual_path_decoupled_merge` reaches mean avg delta {phase3_dual_path_decoupled['aggregate']['task_specialist_clone_dual_path_decoupled_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only`, mean current-task delta {phase3_dual_path_decoupled['aggregate']['task_specialist_clone_dual_path_decoupled_merge']['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, mean gain vs fixed {phase3_dual_path_decoupled['aggregate']['task_specialist_clone_dual_path_decoupled_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%}, and wins {phase3_dual_path_decoupled['aggregate']['task_specialist_clone_dual_path_decoupled_merge']['suite_wins']} suites.",
        f"- Negative external-memory split check: `task_specialist_clone_external_memory_boost_merge` reaches mean avg delta {phase3_external_memory_boost['aggregate']['task_specialist_clone_external_memory_boost_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only`, mean current-task delta {phase3_external_memory_boost['aggregate']['task_specialist_clone_external_memory_boost_merge']['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, mean gain vs fixed {phase3_external_memory_boost['aggregate']['task_specialist_clone_external_memory_boost_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%}, and wins {phase3_external_memory_boost['aggregate']['task_specialist_clone_external_memory_boost_merge']['suite_wins']} suites.",
        f"- Negative routed-memory query check: `task_specialist_clone_external_memory_routed_merge` reaches mean avg delta {phase3_external_memory_routed['aggregate']['task_specialist_clone_external_memory_routed_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only`, mean current-task delta {phase3_external_memory_routed['aggregate']['task_specialist_clone_external_memory_routed_merge']['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, mean gain vs fixed {phase3_external_memory_routed['aggregate']['task_specialist_clone_external_memory_routed_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%}, and wins {phase3_external_memory_routed['aggregate']['task_specialist_clone_external_memory_routed_merge']['suite_wins']} suites.",
        f"- Negative two-speed optimizer check: `task_specialist_clone_two_speed_boost_merge` reaches mean avg delta {phase3_two_speed_boost['aggregate']['task_specialist_clone_two_speed_boost_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only`, mean current-task delta {phase3_two_speed_boost['aggregate']['task_specialist_clone_two_speed_boost_merge']['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, mean gain vs fixed {phase3_two_speed_boost['aggregate']['task_specialist_clone_two_speed_boost_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%}, and wins {phase3_two_speed_boost['aggregate']['task_specialist_clone_two_speed_boost_merge']['suite_wins']} suites.",
        f"- Negative decoupled-objective optimizer check: `task_specialist_clone_decoupled_objective_boost_merge` reaches mean avg delta {phase3_decoupled_objective_boost['aggregate']['task_specialist_clone_decoupled_objective_boost_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only`, mean current-task delta {phase3_decoupled_objective_boost['aggregate']['task_specialist_clone_decoupled_objective_boost_merge']['mean_current_accuracy_delta_vs_adapt_only']:+.1%}, mean gain vs fixed {phase3_decoupled_objective_boost['aggregate']['task_specialist_clone_decoupled_objective_boost_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%}, and wins {phase3_decoupled_objective_boost['aggregate']['task_specialist_clone_decoupled_objective_boost_merge']['suite_wins']} suites.",
        f"- Positive archived-specialist quantization check: keeping the leading mechanism but quantizing only archived specialists improves mean gain vs fixed from {phase3_archived_specialist_quant['aggregate']['0']['mean_avg_accuracy_gain_vs_fixed']:+.1%} unquantized to {phase3_archived_specialist_quant['aggregate']['4']['mean_avg_accuracy_gain_vs_fixed']:+.1%} at `4-bit` and {phase3_archived_specialist_quant['aggregate']['2']['mean_avg_accuracy_gain_vs_fixed']:+.1%} at `2-bit`.",
        f"- Negative replacement check under quantization: even at `4-bit`, `task_specialist_clone_external_memory_boost_merge` still trails quantized internal specialists ({phase3_quantized_memory_path['aggregate']['task_specialist_clone_external_memory_boost_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs {phase3_quantized_memory_path['aggregate']['task_specialist_clone_current_path_boost_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs fixed).",
        f"- Negative internal-memory budget check: under `4-bit` archived-specialist quantization, limiting archive budget from full to `2` drops mean gain vs fixed from {phase3_internal_memory_budget['aggregate']['0']['mean_avg_accuracy_gain_vs_fixed']:+.1%} to {phase3_internal_memory_budget['aggregate']['2']['mean_avg_accuracy_gain_vs_fixed']:+.1%}, and budget `1` drops it further to {phase3_internal_memory_budget['aggregate']['1']['mean_avg_accuracy_gain_vs_fixed']:+.1%}.",
        f"- Negative archive-selection check: at budget `2`, switching from recency to coverage selection stays effectively flat ({phase3_internal_memory_selection['aggregate']['recency_2']['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs {phase3_internal_memory_selection['aggregate']['coverage_2']['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs fixed).",
        f"- Negative archive-representation check: at budget `2`, a simple prototype compression also stays flat ({phase3_internal_memory_representation['aggregate']['prototype_2']['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs {phase3_internal_memory_representation['aggregate']['recency_2']['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs fixed).",
        f"- Negative weight-space archive compression check: at budget `2`, weight prototypes also stay flat ({phase3_internal_memory_weight_prototype['aggregate']['weight_prototype_2']['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs {phase3_internal_memory_weight_prototype['aggregate']['recency_2']['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs fixed).",
        f"- Negative key-value retrieval check: at budget `2`, sample-conditioned key-value retrieval also stays flat ({phase3_internal_memory_key_value['aggregate']['key_value_2']['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs {phase3_internal_memory_key_value['aggregate']['recency_2']['mean_avg_accuracy_gain_vs_fixed']:+.1%} vs fixed).",
        f"- Hardest suite for `task_specialist_clone_current_path_boost_merge`: `{current_path_boost_hardest['task_suite']}` with gain vs fixed {current_path_boost_hardest['task_specialist_clone_current_path_boost_merge']['avg_accuracy_gain_vs_fixed']:+.1%}.",
        f"- Gap decomposition: `{phase3_digits_gap['best_policy_by_fixed_gap']}` is now best by mean fixed-gap, and the dominant remaining deficit is `{phase3_digits_gap['aggregate'][phase3_digits_gap['best_policy_by_fixed_gap']]['dominant_gap']}`.",
        f"- New compact diagnostic suite: `plasticity_boost` wins {phase3_diagnostic_suite['family_win_counts'].get('plasticity_boost', 0)} of {len(phase3_diagnostic_suite['rows'])} stress tests, while `specialist_reuse` wins {phase3_diagnostic_suite['family_win_counts'].get('specialist_reuse', 0)}.",
        f"- Diagnostic read: `plasticity_stress` is won by `{phase3_diagnostic_suite['rows'][0]['best_policy']}`, `interference_stress` is won by `{phase3_diagnostic_suite['rows'][1]['best_policy']}`, and `retrieval_ambiguity_stress` is won by `{phase3_diagnostic_suite['rows'][2]['best_policy']}`.",
        f"- Stronger interference diagnostic family: `{phase3_interference_diagnostic['overall_best_policy']}` is best over `feature_shift` and `feature_shift_noisy`; low-merge control stays ahead of both raw specialist cloning and current-path boost when interference pressure is increased.",
        f"- Structured mode-switch check: `{phase3_mode_switch_diagnostic['overall_best_policy']}` still wins the mixed diagnostic trio; `task_specialist_clone_mode_switch_boost_merge` partially recovers interference/forgetting pressure relative to plain boost, but remains behind `limited_merge` on `feature_shift_noisy` and behind plain `current_path_boost` on the digits plasticity stresses.",
        f"- Explicit dual-mode update check: `{phase3_dual_mode_update_diagnostic['overall_best_policy']}` still wins the mixed diagnostic trio; `task_specialist_clone_dual_mode_update_merge` behaves almost identically to `mode_switch_boost`, so switching archived specialists to retention-only under conflict is still not enough to unify the frontier.",
        f"- Formal diagnostic benchmark family: `plasticity_boost` wins {phase3_diagnostic_benchmark_family['family_win_counts'].get('plasticity_boost', 0)} stresses, `specialist_reuse` wins {phase3_diagnostic_benchmark_family['family_win_counts'].get('specialist_reuse', 0)}, and the newer hybrid candidates win 0; the mixed variants still do not create a new family-level leader.",
        f"- Regime-switch oracle: benchmark-conditioned switching between `current_path_boost` and `limited_merge` reaches mean avg delta {phase3_regime_switch_oracle['aggregate']['regime_switch_oracle']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only` and mean gain vs fixed {phase3_regime_switch_oracle['aggregate']['regime_switch_oracle']['mean_avg_accuracy_gain_vs_fixed']:+.1%}, improving over plain `current_path_boost` ({phase3_regime_switch_oracle['aggregate']['task_specialist_clone_current_path_boost_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%}, {phase3_regime_switch_oracle['aggregate']['task_specialist_clone_current_path_boost_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%}).",
        f"- Explicit regime-switch benchmark: `task_specialist_clone_regime_switch_merge` reaches mean avg delta {phase3_regime_switch_benchmark['aggregate']['task_specialist_clone_regime_switch_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only`, mean gain vs fixed {phase3_regime_switch_benchmark['aggregate']['task_specialist_clone_regime_switch_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%}, and wins {phase3_regime_switch_benchmark['aggregate']['task_specialist_clone_regime_switch_merge']['benchmark_wins']} stresses.",
        f"- Negative diagnostic-driven selector check: `task_specialist_clone_diagnostic_selector_merge` falls back to mean avg delta {phase3_diagnostic_selector_benchmark['aggregate']['task_specialist_clone_diagnostic_selector_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only`, mean gain vs fixed {phase3_diagnostic_selector_benchmark['aggregate']['task_specialist_clone_diagnostic_selector_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%}, and fails to match the hardcoded regime switch.",
        f"- Positive fingerprint-driven selector check: `task_specialist_clone_fingerprint_selector_merge` reaches mean avg delta {phase3_fingerprint_selector_benchmark['aggregate']['task_specialist_clone_fingerprint_selector_merge']['mean_avg_accuracy_delta_vs_adapt_only']:+.1%} vs `adapt_only`, mean gain vs fixed {phase3_fingerprint_selector_benchmark['aggregate']['task_specialist_clone_fingerprint_selector_merge']['mean_avg_accuracy_gain_vs_fixed']:+.1%}, and matches the hardcoded regime-switch frontier.",
        f"- Adversarial selector benchmark check: on `feature_shift_embedded`, hardcoded `task_specialist_clone_regime_switch_merge` stays on the stronger interference-control frontier at {phase3_selector_adversarial_benchmark['policies']['task_specialist_clone_regime_switch_merge']['avg_accuracy_gain_vs_fixed']:+.1%} vs fixed, while both `task_specialist_clone_diagnostic_selector_merge` and `task_specialist_clone_fingerprint_selector_merge` fall back to the weaker boost frontier at {phase3_selector_adversarial_benchmark['policies']['task_specialist_clone_fingerprint_selector_merge']['avg_accuracy_gain_vs_fixed']:+.1%}.",
        "- Current read: lightweight distillation still matters on the original `digits_pairs` narrow point, but the canonical broader-digits story has shifted beyond a single always-on mechanism. Current-path plasticity is still the strongest single-policy lever, and low-merge specialist reuse is still the interference-control baseline, but an explicit regime switch between those two now outperforms either one alone on the formal stress map. That is the first positive evidence that the plasticity/interference split can be exploited at a higher control level even though local hybrid tweaks failed. The selector story is now more precise: benchmark-hardcoded switching works, simple scalar conflict/confidence diagnostics do not, richer task fingerprints are sufficient on the current formal family, but those fingerprints still fail under adversarial regime ambiguity. The next gap is not whether regime selection matters; it is whether we can learn a selector that tracks causal task pressure rather than superficial input statistics. The earlier negative checks still matter: a minimal explicit current/memory path split does not beat the single-path `current_path_boost` recipe, decoupling task loss so it only trains the current path still fails to create a better plasticity/retention frontier, a truly separate external memory bank does not surpass the single-network boost baseline, routed retrieval over that memory bank still does not replace the baseline, a two-speed optimizer split is effectively identical to the baseline, even a simple task-loss-vs-consistency objective split is still effectively identical to the baseline, a more structured conflict-aware mode switch still trades away too much plasticity to unify the plasticity/interference frontier, and an explicit dual-mode update rule still lands on the same tradeoff. New compressed-memory checks refine the rest of the picture further: quantizing archived specialists to `4-bit` or even `2-bit` slightly improves the frontier, but quantized external memory still loses to quantized internal specialist reuse, aggressive archive-count compression is negative, a simple coverage-style archive selector does not recover that loss, a simple prototype-style archive compression does not recover it, weight-space prototypes do not recover it, and sample-conditioned key-value retrieval does not recover it either.",
        "",
        "## Phase 2 Mechanism Bench",
        "",
        f"- Simple-task leader: `{simple_phase2_name}` ({simple_phase2_policy['summary']['final_gain_vs_fixed']:+.1%} vs fixed, AUC {simple_phase2_policy['summary']['auc_gain_vs_fixed']:+.1%}).",
        f"- Hard-task leader: `{hard_phase2_name}` ({hard_phase2_policy['summary']['final_gain_vs_fixed']:+.1%} vs fixed, AUC {hard_phase2_policy['summary']['auc_gain_vs_fixed']:+.1%}).",
        "- Current reading: single-task structure changes remain unstable, and the mechanism bench is more useful as a filter for failure modes than as evidence of a broad win.",
        "",
        "## Current Interpretation",
        "",
        "- The strongest area is Phase 3, where the project shows a repeatable gain under sequential learning pressure.",
        "- Phase 4 is a useful validation line, but its mechanism story is still less settled than Phase 3.",
        "- The weakest area remains Phase 2 knowledge reuse on hard single-task classification.",
        "",
        "## Recommended Next Step",
        "",
        "- Focus on Phase 3 regime selection rather than more local hybrids. The next credible step is a learned selector that can survive adversarial regime ambiguity; simple scalar diagnostics and simple input fingerprints are still too brittle. Treat memory-split and light-query variants as lower priority.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict) -> Tuple[Path, Path]:
    json_path = resolve_canonical_results_path("unified_summary.json")
    save_json(summary, json_path)

    md_path = PROJECT_ROOT / "UNIFIED_REPORT.md"
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, md_path


def main():
    summary = build_summary()
    json_path, md_path = write_outputs(summary)
    print(f"Saved unified JSON summary: {json_path}")
    print(f"Saved unified markdown report: {md_path}")


if __name__ == "__main__":
    main()
