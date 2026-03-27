#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization entrypoint built on top of the shared runtime layer.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import history_to_records, results_dir, split_train_test
from core.sel_core import SELConfig, SELTrainer, create_task
from visualization import ExperimentComparator, MetricsPanel, TopologyAnimator
from visualization.sel_visualizer import VisualizationConfig


def default_output_dir():
    return str(results_dir("visualization"))


def train_demo(config: SELConfig):
    X, y = create_task("simple_classification")
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    trainer = SELTrainer(config)
    trainer.train(X_train, y_train, X_test, y_test)
    return trainer


def run_demo(mode: str = "full"):
    print("\n" + "=" * 60)
    print("SEL-Lab Visualization Demo")
    print("=" * 60)

    output_dir = default_output_dir()
    if mode in ["full", "animation"]:
        print("\n[1/3] Training network for animation...")
        run_topology_demo(output_dir)
    if mode in ["full", "metrics"]:
        print("\n[2/3] Creating metrics dashboard...")
        run_metrics_demo(output_dir)
    if mode in ["full", "comparison"]:
        print("\n[3/3] Running experiment comparison...")
        run_comparison_demo(output_dir)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"Output: {output_dir}")
    print("=" * 60)


def run_topology_demo(output_dir: str):
    trainer = train_demo(
        SELConfig(input_size=4, output_size=2, initial_modules=3, epochs=50, tension_threshold=0.7)
    )
    animator = TopologyAnimator(VisualizationConfig(output_dir=output_dir, frame_interval=300))
    for metrics in trainer.topology_history:
        animator.add_history(metrics)

    print("  Saving topology animation...")
    animator.create_animation()
    animator.save_animation("topology_evolution", format="gif")

    print("  Saving static snapshots...")
    for epoch in [0, 25, 49]:
        fig = animator.create_static_snapshot(epoch)
        animator.save_figure(fig, f"topology_epoch_{epoch}")
    animator.close_all()


def run_metrics_demo(output_dir: str):
    trainer = train_demo(SELConfig(input_size=4, output_size=2, initial_modules=3, epochs=100))
    panel = MetricsPanel(VisualizationConfig(output_dir=output_dir))
    panel.add_training_history(history_to_records(trainer.metrics))
    print("  Saving metrics dashboard...")
    panel.save_dashboard("training_dashboard")
    panel.close_all()


def run_comparison_demo(output_dir: str):
    comparator = ExperimentComparator(VisualizationConfig(output_dir=output_dir))
    print("  Running 3 experiments...")
    for i in range(3):
        trainer = train_demo(
            SELConfig(
                input_size=4,
                output_size=2,
                initial_modules=3 + i,
                epochs=50,
                random_seed=42 + i,
            )
        )
        history = history_to_records(trainer.metrics)
        comparator.add_experiment(f"Run {i + 1}", history)
        print(f"    Run {i + 1}: Test Acc={history[-1]['test_accuracy']:.1%}")

    print("  Saving comparison dashboard...")
    comparator.save_comparison("experiment_comparison")
    comparator.close_all()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SEL Visualization")
    parser.add_argument(
        "--mode",
        choices=["full", "animation", "metrics", "comparison"],
        default="full",
        help="Demo mode",
    )
    args = parser.parse_args()
    run_demo(args.mode)


if __name__ == "__main__":
    main()
