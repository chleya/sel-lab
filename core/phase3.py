# -*- coding: utf-8 -*-
"""
SEL-Lab Phase 3 - Incremental Learning & Task Transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase3_common import Phase3Config, create_task, save_phase3_results, summarize_phase3_runs
from exploration.structure_insights import get_structure_recommendation


class FixedNetwork:
    """Fixed-structure baseline for incremental learning."""

    def __init__(self, config: Phase3Config, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.W1 = self.rng.normal(0.0, 0.5, size=(config.input_size, config.hidden_size))
        self.W2 = self.rng.normal(0.0, 0.5, size=(config.hidden_size, config.output_size))
        self.feedback = self.rng.normal(0.0, 0.5, size=(config.output_size, config.hidden_size))

    def forward(self, x):
        return np.tanh(x @ self.W1) @ self.W2

    def learn(self, x, target, lr: float = 0.01):
        h = np.tanh(x @ self.W1)
        out = h @ self.W2
        error = target - out
        fb_error = self.feedback.T @ error
        self.W2 += lr * np.outer(h, error)
        self.W1 += lr * np.outer(x, fb_error)
        return float(np.mean(error ** 2))

    def predict(self, x):
        return int(np.argmax(self.forward(x)))

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)


class EvolvingNetwork:
    """Incremental learner with knowledge reuse."""

    def __init__(self, config: Phase3Config, seed: int | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.units: List[Dict] = []
        self.add_unit()

    def add_unit(self, clone_from: int = -1):
        if 0 <= clone_from < len(self.units):
            source = self.units[clone_from]
            unit = {
                "W1": source["W1"].copy() + self.rng.normal(0.0, 0.1, size=source["W1"].shape),
                "W2": source["W2"].copy() + self.rng.normal(0.0, 0.1, size=source["W2"].shape),
                "feedback": source["feedback"].copy()
                + self.rng.normal(0.0, 0.1, size=source["feedback"].shape),
                "active": True,
                "tension": source["tension"],
                "age": 0,
            }
        else:
            unit = {
                "W1": self.rng.normal(0.0, 0.5, size=(self.config.input_size, self.config.hidden_size)),
                "W2": self.rng.normal(0.0, 0.5, size=(self.config.hidden_size, self.config.output_size)),
                "feedback": self.rng.normal(
                    0.0, 0.5, size=(self.config.output_size, self.config.hidden_size)
                ),
                "active": True,
                "tension": 0.5,
                "age": 0,
            }
        self.units.append(unit)
        return len(self.units) - 1

    def forward(self, x):
        outputs = []
        for unit in self.units:
            if unit["active"]:
                h = np.tanh(x @ unit["W1"])
                outputs.append(h @ unit["W2"])
        if outputs:
            return np.mean(outputs, axis=0)
        return np.zeros(self.config.output_size)

    def learn(self, x, target, lr: float = 0.01):
        out = self.forward(x)
        error = target - out
        for unit in self.units:
            if not unit["active"]:
                continue
            h = np.tanh(x @ unit["W1"])
            fb_error = unit["feedback"].T @ error
            unit_lr = lr * (1.0 / (1.0 + unit["age"] * 0.1))
            unit["W2"] += unit_lr * np.outer(h, error)
            unit["W1"] += unit_lr * np.outer(x, fb_error)
            unit["age"] += 1
            unit["tension"] = 0.9 * unit["tension"] + 0.1 * float(np.mean(error ** 2))
        return float(np.mean(error ** 2))

    def evolve(self):
        active = [unit for unit in self.units if unit["active"]]
        if not active:
            return []
        avg_tension = float(np.mean([unit["tension"] for unit in active]))
        changes = []
        if avg_tension > 0.15 and len(self.units) < 12:
            source_idx = min(
                [idx for idx, unit in enumerate(self.units) if unit["active"]],
                key=lambda idx: self.units[idx]["tension"],
            )
            self.add_unit(clone_from=source_idx)
            changes.append(f"add(clone_from={source_idx})")
        return changes

    def reset_tension(self):
        for unit in self.units:
            unit["tension"] = 0.3

    def predict(self, x):
        return int(np.argmax(self.forward(x)))

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)

    @property
    def active_units(self):
        return sum(1 for unit in self.units if unit["active"])


def run_phase3():
    print("\n" + "=" * 60)
    print("Phase 3: Incremental Learning & Task Transfer")
    print("=" * 60)

    config = Phase3Config()
    results = []

    for run in range(config.runs):
        print(f"\n--- Run {run + 1}/{config.runs} ---")
        seed = run * 100 + 42
        fixed = FixedNetwork(config, seed=seed)
        evolving = EvolvingNetwork(config, seed=seed)
        task_results = []

        for task_id in range(config.num_tasks):
            X, y = create_task(task_id, task_suite=config.task_suite)
            print(f"  Task {task_id}: ", end="")

            for epoch in range(config.epochs_per_task):
                for i in range(len(X)):
                    fixed.learn(X[i], y[i])
                    evolving.learn(X[i], y[i])
                if (epoch + 1) % 10 == 0:
                    evolving.evolve()

            all_accuracies = {}
            for prev_task in range(task_id + 1):
                X_test, y_test = create_task(prev_task, task_suite=config.task_suite)
                all_accuracies[f"task_{prev_task}"] = {
                    "fixed": fixed.accuracy(X_test, y_test),
                    "evolving": evolving.accuracy(X_test, y_test),
                }

            avg_fixed = float(np.mean([all_accuracies[f"task_{t}"]["fixed"] for t in range(task_id + 1)]))
            avg_evolving = float(
                np.mean([all_accuracies[f"task_{t}"]["evolving"] for t in range(task_id + 1)])
            )
            current_fixed = all_accuracies[f"task_{task_id}"]["fixed"]
            current_evolving = all_accuracies[f"task_{task_id}"]["evolving"]
            unit_count = evolving.active_units
            structure_recommendation = get_structure_recommendation(unit_count)

            print(f"Fixed={current_fixed:.0%}, Evolving={current_evolving:.0%}, Avg={avg_evolving:.0%}, Units={unit_count}")
            print(f"  Structure: {structure_recommendation}")
            task_results.append(
                {
                    "task_id": task_id,
                    "current_fixed": current_fixed,
                    "current_evolving": current_evolving,
                    "avg_fixed": avg_fixed,
                    "avg_evolving": avg_evolving,
                    "unit_count": unit_count,
                }
            )
            evolving.reset_tension()

        results.append(task_results)

    summary = summarize_phase3_runs(results)

    print("\n" + "=" * 60)
    print("Phase 3 Results Summary")
    print("=" * 60)
    print("Final Average Accuracy (all tasks):")
    print(f"  Fixed:   {summary['final_avg_fixed']:.1%}")
    print(f"  Evolving: {summary['final_avg_evolving']:.1%}")
    print("\nForgetting (first_task - avg_all):")
    print(f"  Fixed:   {summary['forgetting_fixed']:+.1%}")
    print(f"  Evolving: {summary['forgetting_evolving']:+.1%}")
    print(f"\nIncremental Learning Advantage: {summary['advantage']:+.1%}")
    print(f"\n{'[SUCCESS] Evolution helps incremental learning!' if summary['decision'] == 'success' else '[NEUTRAL] Mixed results'}")

    payload = {
        "num_tasks": config.num_tasks,
        "task_suite": config.task_suite,
        **summary,
        "results": results,
    }
    target = save_phase3_results(payload)
    print(f"\nResults saved: {target}")
    return payload


if __name__ == "__main__":
    run_phase3()
