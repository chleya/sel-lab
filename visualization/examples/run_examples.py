#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization example runs using the shared runtime abstractions.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import history_to_records, results_dir, split_train_test
from core.sel_core import SELConfig, SELTrainer, create_task
from visualization.experiment_comparator import ExperimentComparator
from visualization.metrics_panel import MetricsPanel
from visualization.sel_visualizer import VisualizationConfig
from visualization.topology_animator import TopologyAnimator


def example_output_dir() -> str:
    return str(results_dir("visualization"))


def train_once(config: SELConfig) -> SELTrainer:
    X, y = create_task("simple_classification")
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    trainer = SELTrainer(config)
    trainer.train(X_train, y_train, X_test, y_test)
    return trainer


def example_1_basic_animation():
    print("\n" + "=" * 50)
    print("Example 1: Topology Animation")
    print("=" * 50)

    trainer = train_once(
        SELConfig(input_size=4, output_size=2, initial_modules=3, learning_rate=0.1, epochs=50)
    )
    print(f"Final Test Accuracy: {trainer.metrics[-1].test_accuracy:.1%}")

    animator = TopologyAnimator(
        VisualizationConfig(output_dir=example_output_dir(), frame_interval=300)
    )
    for metrics in trainer.topology_history:
        animator.add_history(metrics)
    animator.create_animation()
    animator.save_animation("example_topology", format="gif")
    print("Saved: example_topology.gif")
    animator.close_all()


def example_2_metrics_dashboard():
    print("\n" + "=" * 50)
    print("Example 2: Metrics Dashboard")
    print("=" * 50)

    trainer = train_once(SELConfig(input_size=4, output_size=2, initial_modules=3, epochs=100))
    panel = MetricsPanel(VisualizationConfig(output_dir=example_output_dir()))
    panel.add_training_history(history_to_records(trainer.metrics))
    panel.save_dashboard("example_dashboard")
    print("Saved: example_dashboard.png")
    panel.close_all()


def example_3_experiment_comparison():
    print("\n" + "=" * 50)
    print("Example 3: Experiment Comparison")
    print("=" * 50)

    comparator = ExperimentComparator(VisualizationConfig(output_dir=example_output_dir()))
    for run in range(3):
        config = SELConfig(
            input_size=4,
            output_size=2,
            initial_modules=3,
            learning_rate=0.05 + run * 0.05,
            epochs=50,
            random_seed=42 + run,
        )
        trainer = train_once(config)
        history = history_to_records(trainer.metrics)
        comparator.add_experiment(f"LR={config.learning_rate}", history)
        print(f"  LR={config.learning_rate}: {history[-1]['test_accuracy']:.1%}")

    comparator.save_comparison("example_comparison")
    print("Saved: example_comparison.png")
    comparator.close_all()


def example_4_interactive():
    print("\n" + "=" * 50)
    print("Example 4: Interactive Visualization")
    print("=" * 50)
    print("Note: Interactive mode requires GUI environment")
    print("Showing static snapshot instead...")

    trainer = train_once(SELConfig(input_size=4, output_size=2, initial_modules=3, epochs=50))
    animator = TopologyAnimator(VisualizationConfig(output_dir=example_output_dir()))
    for metrics in trainer.topology_history:
        animator.add_history(metrics)

    fig = animator.create_static_snapshot(epoch=25)
    animator.save_figure(fig, "example_snapshot_epoch25")
    print("Saved: example_snapshot_epoch25.png")
    animator.close_all()


def main():
    print("\n" + "=" * 50)
    print("SEL Visualization Examples")
    print("=" * 50)

    examples = [
        ("Topology Animation", example_1_basic_animation),
        ("Metrics Dashboard", example_2_metrics_dashboard),
        ("Experiment Comparison", example_3_experiment_comparison),
        ("Interactive Demo", example_4_interactive),
    ]

    for i, (name, func) in enumerate(examples):
        print(f"\n[{i + 1}/{len(examples)}] {name}")
        try:
            func()
        except Exception as exc:
            print(f"  Error: {exc}")

    print("\n" + "=" * 50)
    print("All examples completed!")
    print(f"Check {example_output_dir()} for outputs")
    print("=" * 50)


if __name__ == "__main__":
    main()
