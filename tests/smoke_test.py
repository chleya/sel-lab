# -*- coding: utf-8 -*-
"""
SEL-Lab smoke tests for the current architecture skeleton.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase4_failure_analysis import analyze_phase4_failure
from core.phase3_ablation import AblationConfig, run_phase3_ablations
from analysis.phase3_benchmark_family import build_stress_benchmark_config, run_policy_benchmark_family
from analysis.generate_unified_report import CANONICAL_RESULT_FILES
from core.phase3_policies import (
    BASE_CLONE_POLICY_HANDLERS,
    CLONE_POLICY_FAMILY_HANDLERS,
    CLONE_POLICY_HANDLERS,
    SPECIAL_EVOLVE_HANDLERS,
)
from core.phase3_registry import (
    PHASE3_POLICY_ORDER,
    get_phase3_policy_spec,
    group_phase3_policies_by_family,
    resolve_phase3_diagnostic_benchmark,
    resolve_phase3_diagnostic_family,
)
from core.phase2_common import Phase2Config
from core.phase2_ablation import Phase2AblationConfig, run_phase2_ablation
from core.runtime import history_to_records, resolve_canonical_results_path, resolve_smoke_results_path, split_train_test
from core.sel_core import SELConfig, SELNetwork, SELTrainer, create_task
from orchestrator.experiment_runner import run_experiments

ENTRYPOINTS = [
    ["core/phase2.py"],
    ["core/phase2_hard.py"],
    ["core/phase2_improved.py"],
    ["core/phase4_mnist.py"],
    ["core/phase4_real.py"],
]


@pytest.fixture
def training_result():
    config = SELConfig(input_size=4, output_size=2, initial_modules=3, learning_rate=0.1, epochs=20)
    X, y = create_task("simple_classification")
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    trainer = SELTrainer(config)
    return trainer.train(X_train, y_train, X_test, y_test)


def test_training_main_path(training_result):
    print("\n[TEST 1] Training main path...")
    result = training_result

    assert "final_train_accuracy" in result
    assert "final_test_accuracy" in result
    assert "network" in result
    assert "evolution_log" in result
    assert "topology_history" in result
    assert result["final_train_accuracy"] > 0

    print(f"  Train Accuracy: {result['final_train_accuracy']:.1%}")
    print(f"  Test Accuracy: {result['final_test_accuracy']:.1%}")
    print("  [PASS] Training main path works!")


def test_serialize_deserialize(training_result):
    print("\n[TEST 2] Serialize/Deserialize...")
    network = training_result["network"]
    original_modules = len(network.modules)
    original_weights = [module.weights.copy() for module in network.modules]

    serialized = network.serialize()
    assert "modules" in serialized
    assert len(serialized["modules"]) == original_modules

    restored = SELNetwork(SELConfig(input_size=4, output_size=2, initial_modules=1))
    restored.deserialize(serialized)

    assert len(restored.modules) == original_modules
    for index, module in enumerate(restored.modules):
        assert np.allclose(module.weights, original_weights[index])

    print(f"  Modules restored: {len(restored.modules)}")
    print("  [PASS] Serialize/Deserialize works!")


def test_visualization_data_boundary():
    print("\n[TEST 3] Visualization data boundary...")
    config = SELConfig(input_size=4, output_size=2, initial_modules=3, epochs=10)
    X, y = create_task("simple_classification")
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    trainer = SELTrainer(config)
    trainer.train(X_train, y_train, X_test, y_test)

    history = history_to_records(trainer.metrics)
    assert len(history) > 0
    assert "test_accuracy" in history[0]
    assert "avg_loss" in history[0]
    assert len(trainer.topology_history) == len(trainer.metrics)
    assert "module_names" in trainer.topology_history[0]

    print("  [PASS] Visualization history conversion works!")


def test_orchestrator_summary():
    print("\n[TEST 4] Orchestrator summary...")
    summary = run_experiments(2)
    assert summary["runs"] == 2
    assert len(summary["results"]) == 2
    assert "mean_modules" in summary
    assert "std_test_accuracy" in summary
    print("  [PASS] Orchestrator summary works!")


def test_script_entrypoints():
    print("\n[TEST 5] Script entrypoints...")
    for args in ENTRYPOINTS:
        completed = subprocess.run(
            [sys.executable, *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            f"{' '.join(args)} failed with code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
        print(f"  [PASS] {' '.join(args)}")


def test_phase4_analysis_respects_age_decay():
    print("\n[TEST 6] Phase 4 analysis dynamics...")
    baseline = analyze_phase4_failure(epochs=3, seed=42, age_decay=0.05)
    slower = analyze_phase4_failure(epochs=3, seed=42, age_decay=0.005)

    baseline_lr = baseline["findings"]["final_effective_lr"]
    slower_lr = slower["findings"]["final_effective_lr"]
    assert slower_lr > baseline_lr, (
        f"expected slower age decay to keep a higher effective LR, got {slower_lr} <= {baseline_lr}"
    )
    print("  [PASS] Phase 4 analysis uses configured age decay!")


def test_phase3_ablation_matrix():
    print("\n[TEST 7] Phase 3 ablation matrix...")
    result = run_phase3_ablations(
        AblationConfig(
            runs=1,
            epochs_per_task=5,
            evolve_interval=5,
            freeze_windows=(1, 2),
        ),
        result_filename="smoke/smoke_phase3_ablation_results.json",
    )

    policy_keys = set(result["policies"].keys())
    assert "adapt_only" in policy_keys
    assert "margin_clone" in policy_keys
    assert "margin_limited_clone" in policy_keys
    assert "trend_limited_clone" in policy_keys
    assert "best_clone_limited_adapt" in policy_keys
    assert "benefit_limited_clone" in policy_keys
    assert "best_clone_delayed_limited_adapt" in policy_keys
    assert "capped_delayed_limited_clone" in policy_keys
    assert "task_specialist_clone" in policy_keys
    assert "task_specialist_clone_limited_merge" in policy_keys
    assert "task_specialist_clone_retention_merge" in policy_keys
    assert "task_specialist_clone_confidence_gate_merge" in policy_keys
    assert "task_specialist_clone_source_aware_merge" in policy_keys
    assert "task_specialist_clone_routed_merge" in policy_keys
    assert "task_specialist_clone_soft_route_merge" in policy_keys
    assert "task_specialist_clone_budgeted_merge" in policy_keys
    assert "task_specialist_clone_anchor_merge" in policy_keys
    assert "task_specialist_clone_distance_anchor_merge" in policy_keys
    assert "task_specialist_clone_distill_merge" in policy_keys
    assert "task_specialist_clone_current_path_boost_merge" in policy_keys
    assert "task_specialist_clone_regime_switch_merge" in policy_keys
    assert "task_specialist_clone_diagnostic_selector_merge" in policy_keys
    assert "task_specialist_clone_fingerprint_selector_merge" in policy_keys
    assert "task_specialist_clone_signature_selector_merge" in policy_keys
    assert "task_specialist_clone_fitted_selector_merge" in policy_keys
    assert "task_specialist_clone_prototype_selector_merge" in policy_keys
    assert "task_specialist_clone_dynamics_selector_merge" in policy_keys
    assert "task_specialist_clone_outcome_selector_merge" in policy_keys
    assert "task_specialist_clone_ranking_selector_merge" in policy_keys
    assert "task_specialist_clone_task_ranking_selector_merge" in policy_keys
    assert "task_specialist_clone_constrained_task_ranking_selector_merge" in policy_keys
    assert "task_specialist_clone_two_stage_selector_merge" in policy_keys
    assert "task_specialist_clone_learned_router_selector_merge" in policy_keys
    assert "task_specialist_clone_hierarchical_selector_merge" in policy_keys
    assert "task_specialist_clone_hierarchical_quadratic_selector_merge" in policy_keys
    assert "task_specialist_clone_hierarchical_sparse_gate_selector_merge" in policy_keys
    assert "task_specialist_clone_guarded_router_selector_merge" in policy_keys
    assert "task_specialist_clone_guarded_expert_selector_merge" in policy_keys
    assert "task_specialist_clone_guarded_expert_boost_default_merge" in policy_keys
    assert "task_specialist_clone_guarded_expert_merge_default_merge" in policy_keys
    assert "task_specialist_clone_guarded_expert_linear_default_merge" in policy_keys
    assert "task_specialist_clone_confidence_boost_merge" in policy_keys
    assert "task_specialist_clone_mode_switch_boost_merge" in policy_keys
    assert "task_specialist_clone_dual_mode_update_merge" in policy_keys
    assert "task_specialist_clone_decoupled_objective_boost_merge" in policy_keys
    assert "task_specialist_clone_two_speed_boost_merge" in policy_keys
    assert "task_specialist_clone_external_memory_boost_merge" in policy_keys
    assert "task_specialist_clone_external_memory_routed_merge" in policy_keys
    assert "task_specialist_clone_current_path_boost_retention_split_merge" in policy_keys
    assert "task_specialist_clone_dual_path_boost_merge" in policy_keys
    assert "task_specialist_clone_dual_path_decoupled_merge" in policy_keys
    assert "task_specialist_clone_asymmetric_retention_merge" in policy_keys
    assert "task_specialist_clone_role_separation_merge" in policy_keys
    assert "task_specialist_clone_freeze_source" in policy_keys
    assert "best_clone_freeze_1" in policy_keys
    assert "best_clone_freeze_2" in policy_keys
    assert "best_clone_freeze_source" not in policy_keys
    assert resolve_smoke_results_path("smoke_phase3_ablation_results.json").exists()

    assert result["config"]["diagnostic_family"] == "transfer_basics"
    assert result["policy_metadata"]["task_specialist_clone_current_path_boost_merge"]["family"] == "plasticity_boost"
    assert "broader_digits" in result["diagnostic_suites"]

    limited = result["policies"]["best_clone_limited_adapt"]["summary"]
    assert "avg_accuracy_gain_vs_fixed" in limited
    assert "mean_forgetting" in limited
    print("  [PASS] Phase 3 ablation emits the current mechanism matrix!")


def test_phase3_task_suite_support():
    print("\n[TEST 8] Phase 3 task suites...")
    result = run_phase3_ablations(
        AblationConfig(
            runs=1,
            epochs_per_task=5,
            evolve_interval=5,
            task_suite="feature_shift",
        ),
        result_filename="smoke/smoke_phase3_feature_shift_results.json",
        policies=("adapt_only", "task_specialist_clone_limited_merge"),
    )
    assert result["config"]["task_suite"] == "feature_shift"
    assert result["config"]["diagnostic_family"] == "transfer_basics"
    assert "task_specialist_clone_limited_merge" in result["policies"]
    assert resolve_smoke_results_path("smoke_phase3_feature_shift_results.json").exists()
    print("  [PASS] Phase 3 supports richer task suites!")

    mixed = run_phase3_ablations(
        AblationConfig(
            runs=1,
            epochs_per_task=3,
            evolve_interval=3,
            task_suite="feature_shift_mixed",
        ),
        result_filename="smoke/smoke_phase3_feature_shift_mixed_results.json",
        policies=("adapt_only", "task_specialist_clone_regime_switch_merge"),
    )
    assert mixed["config"]["task_suite"] == "feature_shift_mixed"
    assert resolve_smoke_results_path("smoke_phase3_feature_shift_mixed_results.json").exists()
    print("  [PASS] Phase 3 mixed-regime suite works!")


def test_phase3_digits_pair_suite():
    print("\n[TEST 9] Phase 3 digits-pair suite...")
    result = run_phase3_ablations(
        AblationConfig(
            input_size=64,
            hidden_size=16,
            runs=1,
            epochs_per_task=3,
            evolve_interval=3,
            task_suite="digits_pairs",
        ),
        result_filename="smoke/smoke_phase3_digits_pairs_results.json",
        policies=("adapt_only", "task_specialist_clone_limited_merge"),
    )
    assert result["config"]["task_suite"] == "digits_pairs"
    assert result["config"]["input_size"] == 64
    assert result["config"]["diagnostic_family"] == "broader_digits"
    assert resolve_smoke_results_path("smoke_phase3_digits_pairs_results.json").exists()
    print("  [PASS] Phase 3 digits-pair suite works!")


def test_phase3_digits_variant_suite():
    print("\n[TEST 10] Phase 3 digits variant suite...")
    result = run_phase3_ablations(
        AblationConfig(
            input_size=64,
            hidden_size=16,
            runs=1,
            epochs_per_task=3,
            evolve_interval=3,
            task_suite="digits_pairs_noisy",
        ),
        result_filename="smoke/smoke_phase3_digits_pairs_noisy_results.json",
        policies=("adapt_only", "task_specialist_clone_distill_merge"),
    )
    assert result["config"]["task_suite"] == "digits_pairs_noisy"
    assert "task_specialist_clone_distill_merge" in result["policies"]
    assert resolve_smoke_results_path("smoke_phase3_digits_pairs_noisy_results.json").exists()
    print("  [PASS] Phase 3 digits variants work!")


def test_phase3_registry_policy_consistency():
    print("\n[TEST 11] Phase 3 registry/policy consistency...")
    pure_preclone_policies = {
        "adapt_only",
        "random_clone",
        "trend_limited_clone",
        "margin_clone",
    }
    default_clone_skeleton_policies = {
        "best_clone",
        "low_tension_clone",
    }
    registry_policy_names = {
        policy_name
        for policy_name in PHASE3_POLICY_ORDER
        if policy_name not in pure_preclone_policies
        and not policy_name.startswith("best_clone_freeze_")
        and policy_name not in default_clone_skeleton_policies
    }
    handler_policy_names = set(BASE_CLONE_POLICY_HANDLERS.keys())

    missing_handlers = registry_policy_names - handler_policy_names
    unexpected_handlers = handler_policy_names - registry_policy_names
    assert not missing_handlers, f"missing Phase 3 handlers for: {sorted(missing_handlers)}"
    assert not unexpected_handlers, f"unexpected Phase 3 handlers for: {sorted(unexpected_handlers)}"
    assert set(SPECIAL_EVOLVE_HANDLERS.keys()) == {
        "adapt_only",
        "random_clone",
        "benefit_limited_clone",
        "trend_limited_clone",
        "margin_clone",
    }
    assert get_phase3_policy_spec("best_clone")["family"] == "clone_reuse"
    assert get_phase3_policy_spec("low_tension_clone")["family"] == "growth_control"
    for policy_name in default_clone_skeleton_policies:
        assert policy_name not in handler_policy_names

    grouped_policy_names = group_phase3_policies_by_family(BASE_CLONE_POLICY_HANDLERS.keys())
    assert tuple(grouped_policy_names["plasticity_boost"]) == (
        "task_specialist_clone_current_path_boost_merge",
        "task_specialist_clone_confidence_boost_merge",
        "task_specialist_clone_mode_switch_boost_merge",
    )
    assert "task_specialist_clone_signature_selector_merge" in grouped_policy_names["regime_switch"]
    assert "task_specialist_clone_fitted_selector_merge" in grouped_policy_names["regime_switch"]
    assert "task_specialist_clone_dynamics_selector_merge" in grouped_policy_names["regime_switch"]
    assert "task_specialist_clone_outcome_selector_merge" in grouped_policy_names["regime_switch"]

    flattened = {}
    for family_name, family_handlers in CLONE_POLICY_FAMILY_HANDLERS.items():
        for policy_name, handler in family_handlers.items():
            expected_family = get_phase3_policy_spec(policy_name)["family"]
            assert expected_family == family_name, (
                f"policy {policy_name} grouped under {family_name}, expected {expected_family}"
            )
            flattened[policy_name] = handler

    assert set(flattened.keys()) == handler_policy_names
    assert set(CLONE_POLICY_HANDLERS.keys()) == handler_policy_names
    print("  [PASS] Phase 3 registry and policy layer stay aligned!")


def test_unified_report_phase3_canonical_read():
    print("\n[TEST 12] Unified report Phase 3 canonical read...")
    report_path = PROJECT_ROOT / "UNIFIED_REPORT.md"
    report_text = report_path.read_text(encoding="utf-8")

    assert "broader-digits leading mechanism is now `task_specialist_clone_current_path_boost_merge`" in report_text
    assert "Base digits benchmark: sequential digits-pair transfer still has a tuned narrow-point winner `task_specialist_clone_distill_merge`." in report_text
    assert "Current broader-digits leader: `task_specialist_clone_current_path_boost_merge`" in report_text
    assert resolve_canonical_results_path("unified_summary.json").exists()
    print("  [PASS] Unified report reflects the current Phase 3 main line!")


def test_canonical_result_manifest():
    print("\n[TEST 13] Canonical result manifest...")
    for summary_key, filename in CANONICAL_RESULT_FILES.items():
        canonical_path = resolve_canonical_results_path(filename)
        root_path = PROJECT_ROOT / "results" / filename
        result_path = canonical_path if canonical_path.exists() else root_path
        assert result_path.exists(), f"missing canonical result for {summary_key}: {filename}"
    assert CANONICAL_RESULT_FILES["phase3_current_path_boost"] == "phase3_current_path_boost_benchmark.json"
    assert CANONICAL_RESULT_FILES["phase3_digits"] == "phase3_digits_transfer.json"
    print("  [PASS] Canonical result manifest resolves to existing files!")


def test_phase2_mechanism_bench():
    print("\n[TEST 14] Phase 2 mechanism bench...")
    result = run_phase2_ablation(
        Phase2AblationConfig(
            simple=Phase2Config(runs=1, epochs=10),
            hard=Phase2Config(output_size=3, learning_rate=0.03, runs=1, epochs=10),
        ),
        result_filename="smoke/smoke_phase2_ablation_results.json",
    )
    simple_policies = set(result["tasks"]["simple"]["policies"].keys())
    hard_policies = set(result["tasks"]["hard"]["policies"].keys())
    expected = {
        "adapt_only",
        "random_growth",
        "best_clone",
        "low_tension_clone",
        "best_clone_limited_adapt",
    }
    assert simple_policies == expected
    assert hard_policies == expected
    assert "final_gain_vs_fixed" in result["tasks"]["hard"]["policies"]["best_clone"]["summary"]
    assert resolve_smoke_results_path("smoke_phase2_ablation_results.json").exists()
    print("  [PASS] Phase 2 mechanism bench emits policy comparisons!")


def test_phase3_benchmark_family_helper():
    print("\n[TEST 15] Phase 3 benchmark-family helper...")
    feature_shift_config = build_stress_benchmark_config(
        "feature_shift",
        selector_conflict_threshold=0.30,
    )
    digits_config = build_stress_benchmark_config(
        "digits_pairs_noisy",
        selector_conflict_threshold=0.30,
    )
    assert feature_shift_config.input_size == 4
    assert feature_shift_config.current_path_lr_boost == 2.00
    assert digits_config.input_size == 64
    assert digits_config.current_path_lr_boost == 2.50

    payload = run_policy_benchmark_family(
        family_name="core_stress_map",
        policies=(
            "adapt_only",
            "task_specialist_clone_limited_merge",
            "task_specialist_clone_current_path_boost_merge",
        ),
        result_prefix="smoke/smoke_phase3_benchmark_helper",
        title="Phase 3 Benchmark Family Helper Smoke",
        config_overrides={
            "runs": 1,
            "epochs_per_task": 3,
            "evolve_interval": 3,
        },
    )
    assert payload["family_name"] == "core_stress_map"
    assert len(payload["rows"]) == 3
    assert "task_specialist_clone_current_path_boost_merge" in payload["aggregate"]
    assert payload["aggregate"]["adapt_only"]["benchmarks_present"] == 3
    selector_gate_family = resolve_phase3_diagnostic_family("selector_gate_map")
    assert selector_gate_family == (
        "selector_generalization_gate",
        "selector_adversarial_gate",
        "selector_sparse_adversarial_gate",
    )
    assert resolve_phase3_diagnostic_benchmark("selector_adversarial_gate")["task_suite"] == "feature_shift_embedded"
    assert resolve_phase3_diagnostic_benchmark("mixed_regime_stress")["task_suite"] == "feature_shift_mixed"
    assert resolve_phase3_diagnostic_benchmark("selector_sparse_adversarial_gate")["task_suite"] == "feature_shift_sparse_embedded"
    assert resolve_smoke_results_path("smoke_phase3_benchmark_helper_plasticity_stress_digits_pairs_noisy.json").exists()
    print("  [PASS] Phase 3 benchmark-family helper works!")


def main():
    print("=" * 60)
    print("SEL-Lab Smoke Test Suite")
    print("=" * 60)

    training = training_result.__wrapped__()

    tests = [
        lambda: test_training_main_path(training),
        lambda: test_serialize_deserialize(training),
        test_visualization_data_boundary,
        test_orchestrator_summary,
        test_script_entrypoints,
        test_phase4_analysis_respects_age_decay,
        test_phase3_ablation_matrix,
        test_phase3_task_suite_support,
        test_phase3_digits_pair_suite,
        test_phase3_digits_variant_suite,
        test_phase3_registry_policy_consistency,
        test_unified_report_phase3_canonical_read,
        test_canonical_result_manifest,
        test_phase2_mechanism_bench,
        test_phase3_benchmark_family_helper,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
