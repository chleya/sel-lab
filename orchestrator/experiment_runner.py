# -*- coding: utf-8 -*-
"""
Top-level experiment orchestration for the SEL core.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import (
    resolve_results_path,
    save_json,
    split_train_test,
    summarize_scalar_runs,
)
from core.sel_core import SELConfig, SELTrainer, create_task


def build_default_config(input_size: int, output_size: int) -> SELConfig:
    return SELConfig(
        input_size=input_size,
        output_size=output_size,
        initial_modules=3,
        learning_rate=0.1,
        epochs=100,
        random_seed=42,
    )


def run_repeated_experiment(
    base_config: SELConfig,
    n_runs: int = 5,
    task_type: str = "simple_classification",
) -> Dict:
    X, y = create_task(task_type)
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    print("\n" + "=" * 60)
    print("SEL-Lab Experiments")
    print("=" * 60)
    print(f"Data: {len(X_train)} train, {len(X_test)} test samples")

    all_results: List[Dict] = []
    for run in range(n_runs):
        base_seed = base_config.random_seed if base_config.random_seed is not None else 42
        config = replace(base_config, random_seed=base_seed + run)
        trainer = SELTrainer(config)
        result = trainer.train(X_train, y_train, X_test, y_test)

        run_result = {
            "run": run + 1,
            "seed": config.random_seed,
            "train_accuracy": result["final_train_accuracy"],
            "test_accuracy": result["final_test_accuracy"],
            "modules": result["final_modules"],
            "changes": result["total_structural_changes"],
            "evolution_events": len(result["evolution_log"]),
        }
        all_results.append(run_result)

        print(
            f"Run {run + 1}/{n_runs}: "
            f"Train={run_result['train_accuracy']:.1%}, "
            f"Test={run_result['test_accuracy']:.1%}, "
            f"Modules={run_result['modules']}, "
            f"EvolutionEvents={run_result['evolution_events']}"
        )

    test_stats = summarize_scalar_runs(all_results, "test_accuracy")
    module_stats = summarize_scalar_runs(all_results, "modules")

    return {
        "task_type": task_type,
        "runs": n_runs,
        "config": {
            "input_size": base_config.input_size,
            "output_size": base_config.output_size,
            "initial_modules": base_config.initial_modules,
            "learning_rate": base_config.learning_rate,
            "epochs": base_config.epochs,
            "tension_threshold": base_config.tension_threshold,
            "random_seed": base_config.random_seed,
        },
        "mean_test_accuracy": test_stats["mean"],
        "std_test_accuracy": test_stats["std"],
        "max_test_accuracy": test_stats["max"],
        "mean_modules": module_stats["mean"],
        "results": all_results,
    }


def run_experiments(n_runs: int = 5, config: Optional[SELConfig] = None) -> Dict:
    if config is None:
        X, y = create_task("simple_classification")
        config = build_default_config(input_size=X.shape[1], output_size=y.shape[1])
    return run_repeated_experiment(config, n_runs=n_runs)


def save_results(summary: Dict, filepath: str | None = None):
    target = resolve_results_path("sel_experiment_results.json") if filepath is None else filepath
    saved = save_json(summary, target)
    print(f"\nResults saved to: {saved}")
    return str(saved)


def main():
    summary = run_experiments(n_runs=5)
    save_results(summary)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Mean Test Accuracy: {summary['mean_test_accuracy']:.1%}")
    print(f"  Max Test Accuracy: {summary['max_test_accuracy']:.1%}")
    print(f"  Mean Module Count: {summary['mean_modules']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
