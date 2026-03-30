# -*- coding: utf-8 -*-
"""
Canonical policy and diagnostic registry for Phase 3.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence


PHASE3_POLICY_ORDER = (
    "adapt_only",
    "random_clone",
    "low_tension_clone",
    "best_clone",
    "margin_clone",
    "trend_limited_clone",
    "best_clone_limited_adapt",
    "benefit_limited_clone",
    "best_clone_delayed_limited_adapt",
    "capped_delayed_limited_clone",
    "task_specialist_clone",
    "task_specialist_clone_limited_merge",
    "task_specialist_clone_retention_merge",
    "task_specialist_clone_confidence_gate_merge",
    "task_specialist_clone_source_aware_merge",
    "task_specialist_clone_routed_merge",
    "task_specialist_clone_soft_route_merge",
    "task_specialist_clone_budgeted_merge",
    "task_specialist_clone_anchor_merge",
    "task_specialist_clone_distance_anchor_merge",
    "task_specialist_clone_distill_merge",
    "task_specialist_clone_current_path_boost_merge",
    "task_specialist_clone_regime_switch_merge",
    "task_specialist_clone_diagnostic_selector_merge",
    "task_specialist_clone_fingerprint_selector_merge",
    "task_specialist_clone_signature_selector_merge",
    "task_specialist_clone_fitted_selector_merge",
    "task_specialist_clone_prototype_selector_merge",
    "task_specialist_clone_dynamics_selector_merge",
    "task_specialist_clone_outcome_selector_merge",
    "task_specialist_clone_ranking_selector_merge",
    "task_specialist_clone_task_ranking_selector_merge",
    "task_specialist_clone_constrained_task_ranking_selector_merge",
    "task_specialist_clone_two_stage_selector_merge",
    "task_specialist_clone_learned_router_selector_merge",
    "task_specialist_clone_hierarchical_selector_merge",
    "task_specialist_clone_hierarchical_quadratic_selector_merge",
    "task_specialist_clone_hierarchical_sparse_gate_selector_merge",
    "task_specialist_clone_guarded_router_selector_merge",
    "task_specialist_clone_guarded_expert_selector_merge",
    "task_specialist_clone_guarded_expert_boost_default_merge",
    "task_specialist_clone_guarded_expert_merge_default_merge",
    "task_specialist_clone_guarded_expert_linear_default_merge",
    "task_specialist_clone_confidence_boost_merge",
    "task_specialist_clone_mode_switch_boost_merge",
    "task_specialist_clone_dual_mode_update_merge",
    "task_specialist_clone_decoupled_objective_boost_merge",
    "task_specialist_clone_two_speed_boost_merge",
    "task_specialist_clone_external_memory_boost_merge",
    "task_specialist_clone_external_memory_routed_merge",
    "task_specialist_clone_current_path_boost_retention_split_merge",
    "task_specialist_clone_dual_path_boost_merge",
    "task_specialist_clone_dual_path_decoupled_merge",
    "task_specialist_clone_asymmetric_retention_merge",
    "task_specialist_clone_role_separation_merge",
    "task_specialist_clone_freeze_source",
    "margin_limited_clone",
)

PHASE3_POLICY_SPECS: Dict[str, Dict[str, str]] = {
    "adapt_only": {
        "family": "baseline",
        "forward_change": "none",
        "update_rule_change": "single active path only",
        "memory_change": "none",
        "benchmark_intent": "reference policy",
    },
    "random_clone": {
        "family": "growth_control",
        "forward_change": "extra random unit",
        "update_rule_change": "default shared task loss",
        "memory_change": "none",
        "benchmark_intent": "degenerate growth baseline",
    },
    "low_tension_clone": {
        "family": "growth_control",
        "forward_change": "clone low-tension unit",
        "update_rule_change": "default shared task loss",
        "memory_change": "none",
        "benchmark_intent": "trigger quality check",
    },
    "best_clone": {
        "family": "clone_reuse",
        "forward_change": "clone strongest current source",
        "update_rule_change": "default shared task loss",
        "memory_change": "none",
        "benchmark_intent": "plain reuse baseline",
    },
    "margin_clone": {
        "family": "growth_control",
        "forward_change": "margin-gated clone",
        "update_rule_change": "default shared task loss",
        "memory_change": "none",
        "benchmark_intent": "trigger quality check",
    },
    "trend_limited_clone": {
        "family": "growth_control",
        "forward_change": "trend-gated clone",
        "update_rule_change": "limited clone adaptation",
        "memory_change": "none",
        "benchmark_intent": "trigger quality check",
    },
    "best_clone_limited_adapt": {
        "family": "clone_reuse",
        "forward_change": "clone strongest current source",
        "update_rule_change": "limited clone adaptation",
        "memory_change": "none",
        "benchmark_intent": "core reuse baseline",
    },
    "benefit_limited_clone": {
        "family": "clone_reuse",
        "forward_change": "benefit-gated clone",
        "update_rule_change": "limited clone adaptation",
        "memory_change": "none",
        "benchmark_intent": "reuse gating check",
    },
    "best_clone_delayed_limited_adapt": {
        "family": "clone_reuse",
        "forward_change": "clone strongest current source",
        "update_rule_change": "delay then limited adapt",
        "memory_change": "none",
        "benchmark_intent": "plasticity-retention tradeoff",
    },
    "capped_delayed_limited_clone": {
        "family": "clone_reuse",
        "forward_change": "bounded clone pool",
        "update_rule_change": "delay then limited adapt",
        "memory_change": "capacity cap",
        "benchmark_intent": "bounded reuse check",
    },
    "task_specialist_clone": {
        "family": "specialist_reuse",
        "forward_change": "task specialist clone",
        "update_rule_change": "limited specialist adaptation",
        "memory_change": "none",
        "benchmark_intent": "specialist baseline",
    },
    "task_specialist_clone_limited_merge": {
        "family": "specialist_reuse",
        "forward_change": "low merge from archived specialists",
        "update_rule_change": "limited specialist adaptation",
        "memory_change": "specialist residual path",
        "benchmark_intent": "robustness baseline",
    },
    "task_specialist_clone_retention_merge": {
        "family": "specialist_reuse",
        "forward_change": "distance-weighted specialist merge",
        "update_rule_change": "limited specialist adaptation",
        "memory_change": "retention bonus",
        "benchmark_intent": "retention bonus check",
    },
    "task_specialist_clone_confidence_gate_merge": {
        "family": "routing_gating",
        "forward_change": "confidence-gated specialist merge",
        "update_rule_change": "limited specialist adaptation",
        "memory_change": "specialist residual path",
        "benchmark_intent": "gating check",
    },
    "task_specialist_clone_source_aware_merge": {
        "family": "routing_gating",
        "forward_change": "source-aware specialist gate",
        "update_rule_change": "limited specialist adaptation",
        "memory_change": "specialist residual path",
        "benchmark_intent": "routing conflict check",
    },
    "task_specialist_clone_routed_merge": {
        "family": "routing_gating",
        "forward_change": "single routed specialist merge",
        "update_rule_change": "limited specialist adaptation",
        "memory_change": "specialist retrieval",
        "benchmark_intent": "routing sparsity check",
    },
    "task_specialist_clone_soft_route_merge": {
        "family": "routing_gating",
        "forward_change": "softmax routed specialist merge",
        "update_rule_change": "limited specialist adaptation",
        "memory_change": "specialist retrieval",
        "benchmark_intent": "routing softness check",
    },
    "task_specialist_clone_budgeted_merge": {
        "family": "routing_gating",
        "forward_change": "budgeted specialist merge",
        "update_rule_change": "limited specialist adaptation",
        "memory_change": "specialist retrieval",
        "benchmark_intent": "routing budget check",
    },
    "task_specialist_clone_anchor_merge": {
        "family": "retention_control",
        "forward_change": "confidence-gated specialist merge",
        "update_rule_change": "anchor regularization",
        "memory_change": "anchored specialist state",
        "benchmark_intent": "retention regularization check",
    },
    "task_specialist_clone_distance_anchor_merge": {
        "family": "retention_control",
        "forward_change": "confidence-gated specialist merge",
        "update_rule_change": "distance-scaled anchor regularization",
        "memory_change": "anchored specialist state",
        "benchmark_intent": "retention distance check",
    },
    "task_specialist_clone_distill_merge": {
        "family": "retention_control",
        "forward_change": "confidence-gated specialist merge",
        "update_rule_change": "distillation regularization",
        "memory_change": "anchored specialist state",
        "benchmark_intent": "narrow benchmark sharpen check",
    },
    "task_specialist_clone_current_path_boost_merge": {
        "family": "plasticity_boost",
        "forward_change": "low specialist merge",
        "update_rule_change": "boost active current path",
        "memory_change": "specialist residual path",
        "benchmark_intent": "current-task plasticity leader",
    },
    "task_specialist_clone_regime_switch_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by stress regime",
        "update_rule_change": "benchmark-conditioned regime selection",
        "memory_change": "specialist residual path",
        "benchmark_intent": "explicit regime-selection check",
    },
    "task_specialist_clone_diagnostic_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by task diagnostics",
        "update_rule_change": "conflict/confidence-driven regime selection",
        "memory_change": "specialist residual path",
        "benchmark_intent": "diagnostic-driven regime-selection check",
    },
    "task_specialist_clone_fingerprint_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by task fingerprint",
        "update_rule_change": "input-distribution-driven regime selection",
        "memory_change": "specialist residual path",
        "benchmark_intent": "task-fingerprint regime-selection check",
    },
    "task_specialist_clone_signature_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by multi-signal task signature",
        "update_rule_change": "combined online task-signature regime selection",
        "memory_change": "specialist residual path",
        "benchmark_intent": "task-signature regime-selection check",
    },
    "task_specialist_clone_fitted_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by fitted task-signature score",
        "update_rule_change": "linear fitted regime selection over task-signature features",
        "memory_change": "specialist residual path",
        "benchmark_intent": "fitted task-signature regime-selection check",
    },
    "task_specialist_clone_prototype_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by prototype task-signature retrieval",
        "update_rule_change": "nearest-exemplar regime selection over standardized task-signature features",
        "memory_change": "selector prototype bank",
        "benchmark_intent": "prototype task-signature regime-selection check",
    },
    "task_specialist_clone_dynamics_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by task learning dynamics",
        "update_rule_change": "fitted regime selection over richer online task dynamics and task-signature features",
        "memory_change": "selector fit over regime features",
        "benchmark_intent": "dynamic regime-feature selector check",
    },
    "task_specialist_clone_outcome_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by outcome-supervised regime score",
        "update_rule_change": "fit regime score directly to benchmark outcome margins over richer online regime features",
        "memory_change": "selector fit over outcome supervision",
        "benchmark_intent": "outcome-supervised regime-selection check",
    },
    "task_specialist_clone_ranking_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by ranked regime preference score",
        "update_rule_change": "pairwise preference fitting over structured selector-map regime targets",
        "memory_change": "selector fit over ranking supervision",
        "benchmark_intent": "structured ranking regime-selection check",
    },
    "task_specialist_clone_task_ranking_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by task-level ranked regime preference score",
        "update_rule_change": "pairwise preference fitting over task-level structured selector targets",
        "memory_change": "selector fit over task-level ranking supervision",
        "benchmark_intent": "task-level structured ranking regime-selection check",
    },
    "task_specialist_clone_constrained_task_ranking_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by constrained task-level ranked regime preference score",
        "update_rule_change": "task-level ranking with explicit sparse-gate protection",
        "memory_change": "selector fit over constrained task-level ranking supervision",
        "benchmark_intent": "constrained task-level structured ranking regime-selection check",
    },
    "task_specialist_clone_two_stage_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by a two-stage selector",
        "update_rule_change": "embedded-gate routing followed by task-level ranking fallback",
        "memory_change": "mixture-style selector over signature and task-ranking routes",
        "benchmark_intent": "two-stage regime-selection check",
    },
    "task_specialist_clone_learned_router_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by a learned multi-route selector",
        "update_rule_change": "fit a three-way router over regime features for boost, interference control, and sparse-boost routing",
        "memory_change": "learned router over structured selector routes",
        "benchmark_intent": "learned multi-route regime-selection check",
    },
    "task_specialist_clone_hierarchical_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by hierarchical segmented selector heads",
        "update_rule_change": "route tasks into sparse, embedded, and default regions, then fit a separate ranking head per region",
        "memory_change": "hierarchical region-specific selector heads",
        "benchmark_intent": "hierarchical segmented regime-selection check",
    },
    "task_specialist_clone_hierarchical_quadratic_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by hierarchical segmented quadratic selector heads",
        "update_rule_change": "route tasks into sparse, embedded, and default regions, then fit a separate quadratic ranking head per region",
        "memory_change": "hierarchical non-linear region-specific selector heads",
        "benchmark_intent": "hierarchical segmented non-linear regime-selection check",
    },
    "task_specialist_clone_hierarchical_sparse_gate_selector_merge": {
        "family": "regime_switch",
        "forward_change": "switch between low-merge specialist reuse and current-path boost by sparse-gate-aware hierarchical segmented heads",
        "update_rule_change": "route tasks into sparse, embedded, and default regions, then fit quadratic region heads with explicit sparse-gate protection",
        "memory_change": "hierarchical sparse-gate-aware selector heads",
        "benchmark_intent": "hierarchical sparse-gate-aware regime-selection check",
    },
    "task_specialist_clone_guarded_router_selector_merge": {
        "family": "regime_switch",
        "forward_change": "preserve a hard sparse boost branch while learning the remaining segmented routes",
        "update_rule_change": "hard-route sparse embedded tasks to boost and fit quadratic heads only for default and embedded regions",
        "memory_change": "guarded segmented router with preserved sparse branch",
        "benchmark_intent": "guarded sparse-branch regime-selection check",
    },
    "task_specialist_clone_guarded_expert_selector_merge": {
        "family": "regime_switch",
        "forward_change": "preserve hard sparse and embedded expert branches while learning only the default segmented route",
        "update_rule_change": "hard-route sparse tasks to boost, embedded tasks to signature logic, and fit a quadratic head only for the default region",
        "memory_change": "guarded expert router with preserved sparse and embedded branches",
        "benchmark_intent": "guarded expert regime-selection check",
    },
    "task_specialist_clone_guarded_expert_boost_default_merge": {
        "family": "regime_switch",
        "forward_change": "preserve hard sparse and embedded expert branches and hard-route the default region to boost",
        "update_rule_change": "hard-route sparse tasks to boost, embedded tasks to signature logic, and default tasks to boost",
        "memory_change": "guarded expert router with no learned default branch",
        "benchmark_intent": "guarded expert boost-default simplification check",
    },
    "task_specialist_clone_guarded_expert_merge_default_merge": {
        "family": "regime_switch",
        "forward_change": "preserve hard sparse and embedded expert branches and hard-route the default region to interference control",
        "update_rule_change": "hard-route sparse tasks to boost, embedded tasks to signature logic, and default tasks to low-merge interference control",
        "memory_change": "guarded expert router with no learned default branch",
        "benchmark_intent": "guarded expert merge-default simplification check",
    },
    "task_specialist_clone_guarded_expert_linear_default_merge": {
        "family": "regime_switch",
        "forward_change": "preserve hard sparse and embedded expert branches while learning a linear default-region fallback",
        "update_rule_change": "hard-route sparse tasks to boost, embedded tasks to signature logic, and fit a linear head only for the default region",
        "memory_change": "guarded expert router with linear default fallback",
        "benchmark_intent": "guarded expert linear-default regime-selection check",
    },
    "task_specialist_clone_confidence_boost_merge": {
        "family": "plasticity_boost",
        "forward_change": "confidence-gated low specialist merge",
        "update_rule_change": "boost active current path",
        "memory_change": "specialist residual path",
        "benchmark_intent": "hybrid plasticity/interference check",
    },
    "task_specialist_clone_mode_switch_boost_merge": {
        "family": "plasticity_boost",
        "forward_change": "low specialist merge with conflict-aware boost fallback",
        "update_rule_change": "boost current path, downshift under specialist conflict",
        "memory_change": "specialist residual path",
        "benchmark_intent": "dual-mode plasticity/interference check",
    },
    "task_specialist_clone_dual_mode_update_merge": {
        "family": "optimizer_split",
        "forward_change": "low specialist merge with conflict-aware boost fallback",
        "update_rule_change": "downshift current boost and switch archived specialists to retention-only under conflict",
        "memory_change": "anchored specialist residual path",
        "benchmark_intent": "explicit dual-mode update check",
    },
    "task_specialist_clone_decoupled_objective_boost_merge": {
        "family": "optimizer_split",
        "forward_change": "low specialist merge",
        "update_rule_change": "task loss for current path, consistency for old path",
        "memory_change": "specialist residual path",
        "benchmark_intent": "objective split check",
    },
    "task_specialist_clone_two_speed_boost_merge": {
        "family": "optimizer_split",
        "forward_change": "low specialist merge",
        "update_rule_change": "boost current path, slow old path",
        "memory_change": "specialist residual path",
        "benchmark_intent": "two-speed optimizer check",
    },
    "task_specialist_clone_external_memory_boost_merge": {
        "family": "memory_split",
        "forward_change": "external memory merge",
        "update_rule_change": "boost current path",
        "memory_change": "external archived specialists",
        "benchmark_intent": "memory-bank split check",
    },
    "task_specialist_clone_external_memory_routed_merge": {
        "family": "memory_split",
        "forward_change": "routed external memory merge",
        "update_rule_change": "boost current path",
        "memory_change": "external archived specialists",
        "benchmark_intent": "memory query check",
    },
    "task_specialist_clone_current_path_boost_retention_split_merge": {
        "family": "optimizer_split",
        "forward_change": "low specialist merge",
        "update_rule_change": "boost current path, retention-only old path",
        "memory_change": "specialist residual path",
        "benchmark_intent": "retention split check",
    },
    "task_specialist_clone_dual_path_boost_merge": {
        "family": "memory_split",
        "forward_change": "dual current/memory path merge",
        "update_rule_change": "boost current path, memory-only old path",
        "memory_change": "internal memory path",
        "benchmark_intent": "minimal dual-path check",
    },
    "task_specialist_clone_dual_path_decoupled_merge": {
        "family": "memory_split",
        "forward_change": "dual current/memory path merge",
        "update_rule_change": "task loss only on current path",
        "memory_change": "internal memory path",
        "benchmark_intent": "loss-decoupled dual-path check",
    },
    "task_specialist_clone_asymmetric_retention_merge": {
        "family": "retention_control",
        "forward_change": "confidence-gated specialist merge",
        "update_rule_change": "retention-only updates on old specialists",
        "memory_change": "anchored specialist state",
        "benchmark_intent": "asymmetric update check",
    },
    "task_specialist_clone_role_separation_merge": {
        "family": "retention_control",
        "forward_change": "low specialist merge",
        "update_rule_change": "specialists learn only on own task",
        "memory_change": "archival specialist role split",
        "benchmark_intent": "role separation check",
    },
    "task_specialist_clone_freeze_source": {
        "family": "clone_reuse",
        "forward_change": "task specialist clone",
        "update_rule_change": "freeze source after clone",
        "memory_change": "none",
        "benchmark_intent": "source freezing check",
    },
    "margin_limited_clone": {
        "family": "growth_control",
        "forward_change": "margin-gated clone",
        "update_rule_change": "limited clone adaptation",
        "memory_change": "none",
        "benchmark_intent": "trigger quality check",
    },
}

PHASE3_POLICY_SETS = {
    "full_matrix": PHASE3_POLICY_ORDER,
    "broader_digits_boundary": (
        "adapt_only",
        "task_specialist_clone_limited_merge",
        "task_specialist_clone_distill_merge",
        "task_specialist_clone_current_path_boost_merge",
    ),
    "digits_gap_core": (
        "adapt_only",
        "task_specialist_clone_limited_merge",
        "task_specialist_clone_distill_merge",
        "task_specialist_clone_role_separation_merge",
        "task_specialist_clone_source_aware_merge",
        "task_specialist_clone_asymmetric_retention_merge",
        "task_specialist_clone_current_path_boost_merge",
    ),
}

PHASE3_DIAGNOSTIC_SUITES = {
    "transfer_basics": ("default", "rotated", "feature_shift", "noisy"),
    "broader_digits": ("digits_pairs", "digits_pairs_noisy", "digits_pairs_permuted"),
    "plasticity_stress": ("digits_pairs", "digits_pairs_noisy", "digits_pairs_permuted"),
    "interference_stress": ("feature_shift", "feature_shift_noisy"),
}

PHASE3_DIAGNOSTIC_BENCHMARKS: Dict[str, Dict[str, object]] = {
    "plasticity_stress": {
        "task_suite": "digits_pairs_noisy",
        "policies": (
            "adapt_only",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_distill_merge",
            "task_specialist_clone_current_path_boost_merge",
        ),
        "benchmark_intent": "stress current-task learning under noisy sequential digits",
    },
    "interference_stress": {
        "task_suite": "feature_shift",
        "policies": (
            "adapt_only",
            "task_specialist_clone",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_current_path_boost_merge",
        ),
        "benchmark_intent": "separate raw specialist interference from low-merge control",
    },
    "interference_stress_strong": {
        "task_suite": "feature_shift_noisy",
        "policies": (
            "adapt_only",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_current_path_boost_merge",
            "task_specialist_clone_mode_switch_boost_merge",
            "task_specialist_clone_dual_mode_update_merge",
        ),
        "benchmark_intent": "stress interference control under stronger feature-shift noise",
    },
    "retrieval_ambiguity_stress": {
        "task_suite": "digits_pairs_permuted",
        "policies": (
            "adapt_only",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_source_aware_merge",
            "task_specialist_clone_routed_merge",
            "task_specialist_clone_current_path_boost_merge",
        ),
        "benchmark_intent": "test whether routing/retrieval helps under ambiguous digit remapping",
    },
    "selector_generalization_gate": {
        "task_suite": "feature_shift_nonnegative",
        "policies": (
            "adapt_only",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_current_path_boost_merge",
            "task_specialist_clone_regime_switch_merge",
            "task_specialist_clone_diagnostic_selector_merge",
            "task_specialist_clone_fingerprint_selector_merge",
            "task_specialist_clone_signature_selector_merge",
            "task_specialist_clone_fitted_selector_merge",
        ),
        "benchmark_intent": "check whether selector logic overreacts to digits-like nonnegative inputs without true adversarial embedding",
    },
    "mixed_regime_stress": {
        "task_suite": "feature_shift_mixed",
        "policies": (
            "adapt_only",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_current_path_boost_merge",
            "task_specialist_clone_regime_switch_merge",
            "task_specialist_clone_diagnostic_selector_merge",
            "task_specialist_clone_fingerprint_selector_merge",
            "task_specialist_clone_signature_selector_merge",
            "task_specialist_clone_fitted_selector_merge",
            "task_specialist_clone_prototype_selector_merge",
        ),
        "benchmark_intent": "stress selector behavior when acquisition and interference pressure are mixed within the same suite",
    },
    "selector_adversarial_gate": {
        "task_suite": "feature_shift_embedded",
        "policies": (
            "adapt_only",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_current_path_boost_merge",
            "task_specialist_clone_regime_switch_merge",
            "task_specialist_clone_diagnostic_selector_merge",
            "task_specialist_clone_fingerprint_selector_merge",
            "task_specialist_clone_signature_selector_merge",
            "task_specialist_clone_fitted_selector_merge",
        ),
        "benchmark_intent": "stress selector robustness when interference is embedded in digits-like nonnegative inputs",
    },
    "selector_sparse_adversarial_gate": {
        "task_suite": "feature_shift_sparse_embedded",
        "policies": (
            "adapt_only",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_current_path_boost_merge",
            "task_specialist_clone_regime_switch_merge",
            "task_specialist_clone_diagnostic_selector_merge",
            "task_specialist_clone_fingerprint_selector_merge",
            "task_specialist_clone_signature_selector_merge",
            "task_specialist_clone_fitted_selector_merge",
            "task_specialist_clone_prototype_selector_merge",
        ),
        "benchmark_intent": "stress selector robustness under sparse embedded interference cues rather than dense digits-like statistics",
    },
}

PHASE3_DIAGNOSTIC_FAMILIES: Dict[str, tuple[str, ...]] = {
    "core_stress_map": (
        "plasticity_stress",
        "interference_stress",
        "retrieval_ambiguity_stress",
    ),
    "expanded_stress_map": (
        "plasticity_stress",
        "interference_stress",
        "interference_stress_strong",
        "retrieval_ambiguity_stress",
    ),
    "selector_gate_map": (
        "selector_generalization_gate",
        "selector_adversarial_gate",
        "selector_sparse_adversarial_gate",
    ),
    "selector_full_map": (
        "plasticity_stress",
        "interference_stress",
        "interference_stress_strong",
        "retrieval_ambiguity_stress",
        "mixed_regime_stress",
        "selector_generalization_gate",
        "selector_adversarial_gate",
        "selector_sparse_adversarial_gate",
    ),
}


def _dynamic_freeze_policy_spec(policy_name: str) -> Dict[str, str]:
    if policy_name == "best_clone_freeze_source":
        freeze_note = "freeze clone source for one task"
    else:
        freeze_note = f"freeze clone source for {policy_name.rsplit('_', 1)[1]} task windows"
    return {
        "family": "clone_reuse",
        "forward_change": "clone strongest current source",
        "update_rule_change": freeze_note,
        "memory_change": "none",
        "benchmark_intent": "freeze-window check",
    }


def get_phase3_policy_spec(policy_name: str) -> Dict[str, str]:
    if policy_name in PHASE3_POLICY_SPECS:
        return dict(PHASE3_POLICY_SPECS[policy_name])
    if policy_name == "best_clone_freeze_source" or policy_name.startswith("best_clone_freeze_"):
        return _dynamic_freeze_policy_spec(policy_name)
    raise KeyError(f"Unknown Phase 3 policy: {policy_name}")


def build_phase3_policy_metadata(policies: Iterable[str]) -> Dict[str, Dict[str, str]]:
    return {policy_name: get_phase3_policy_spec(policy_name) for policy_name in policies}


def group_phase3_policies_by_family(policies: Iterable[str]) -> Dict[str, tuple[str, ...]]:
    grouped: Dict[str, list[str]] = {}
    for policy_name in policies:
        family = get_phase3_policy_spec(policy_name)["family"]
        grouped.setdefault(family, []).append(policy_name)
    return {
        family_name: tuple(policy_names)
        for family_name, policy_names in grouped.items()
    }


def resolve_phase3_policies(config, policies: Sequence[str] | None = None) -> list[str]:
    if policies is not None:
        return list(policies)
    default_policies = list(PHASE3_POLICY_ORDER)
    default_policies.extend(f"best_clone_freeze_{window}" for window in config.freeze_windows)
    return default_policies


def resolve_phase3_policy_set(name: str) -> tuple[str, ...]:
    return tuple(PHASE3_POLICY_SETS[name])


def diagnostic_family_for_task_suite(task_suite: str) -> str | None:
    for family_name, suites in PHASE3_DIAGNOSTIC_SUITES.items():
        if task_suite in suites:
            return family_name
    return None


def resolve_phase3_diagnostic_benchmark(name: str) -> Dict[str, object]:
    return dict(PHASE3_DIAGNOSTIC_BENCHMARKS[name])


def resolve_phase3_diagnostic_family(name: str) -> tuple[str, ...]:
    return tuple(PHASE3_DIAGNOSTIC_FAMILIES[name])
