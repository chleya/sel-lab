# -*- coding: utf-8 -*-
"""
SEL-Lab Phase 1 Implementation
Forward-only learning verification with a fixed DFA baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import resolve_canonical_results_path, resolve_results_path, save_json, split_train_test
from core.sel_core import create_task


@dataclass
class Phase1Config:
    input_size: int = 4
    hidden_size: int = 8
    output_size: int = 2
    learning_rate: float = 0.05
    runs: int = 10
    epochs: int = 100


class Phase1Network:
    """Stable DFA baseline used for Phase 1 verification."""

    def __init__(self, config: Phase1Config, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.config = config
        self.W1 = self.rng.normal(
            0.0, np.sqrt(2.0 / config.input_size), size=(config.input_size, config.hidden_size)
        )
        self.W2 = self.rng.normal(
            0.0, np.sqrt(2.0 / config.hidden_size), size=(config.hidden_size, config.output_size)
        )
        self.feedback = self.rng.normal(0.0, 0.1, size=(config.output_size, config.hidden_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1)
        return h @ self.W2

    def forward_learning(self, x: np.ndarray, target: np.ndarray) -> float:
        h = np.tanh(x @ self.W1)
        out = h @ self.W2
        error = target - out
        fb_signal = self.feedback.T @ error

        self.W2 += self.config.learning_rate * np.outer(h, error)
        self.W1 += self.config.learning_rate * np.outer(x, fb_signal)
        self.W1 = np.clip(self.W1, -2.0, 2.0)
        self.W2 = np.clip(self.W2, -2.0, 2.0)
        return float(np.mean(np.abs(error)))

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)

    @property
    def param_count(self) -> int:
        return self.W1.size + self.W2.size


class Phase1Experiment:
    """Phase 1 experiment runner."""

    def __init__(self, config: Phase1Config | None = None):
        self.config = config or Phase1Config()
        self.history = {"acc": [], "loss": []}
        self.last_result: Dict | None = None

    def run(self, verbose: bool = True) -> Dict:
        if verbose:
            print("\n" + "=" * 60)
            print("Phase 1: Forward-Only Learning Verification")
            print("Method: Direct Feedback Alignment (DFA)")
            print("Constraints: No Backpropagation")
            print("=" * 60)

        X, y = create_task("simple_classification")
        X_train, y_train, X_test, y_test = split_train_test(X, y)

        all_results = []
        for run in range(self.config.runs):
            seed = run * 100 + 42
            network = Phase1Network(self.config, seed=seed)
            run_results = {"run": run + 1, "seed": seed, "accuracies": [], "losses": []}

            if verbose:
                print(f"\n--- Run {run + 1}/{self.config.runs} (seed={seed}) ---")

            for epoch in range(self.config.epochs):
                epoch_loss = 0.0
                for i in range(len(X_train)):
                    epoch_loss += network.forward_learning(X_train[i], y_train[i])
                avg_loss = epoch_loss / len(X_train)
                acc = network.accuracy(X_test, y_test)

                run_results["accuracies"].append(acc)
                run_results["losses"].append(avg_loss)
                self.history["acc"].append(acc)
                self.history["loss"].append(avg_loss)

                if verbose and (epoch + 1) % 25 == 0:
                    print(f"Epoch {epoch + 1}: Acc={acc:.1%}, Loss={avg_loss:.4f}")

            final_acc = run_results["accuracies"][-1]
            if verbose:
                print(f"Final: Acc={final_acc:.1%}")
            all_results.append(run_results)

        final_accs = [result["accuracies"][-1] for result in all_results]
        slope = 0.0
        if len(self.history["acc"]) >= 2:
            x = np.arange(len(self.history["acc"]))
            slope = float(np.polyfit(x, self.history["acc"], 1)[0])

        evaluation = {
            "final_accuracy": float(np.mean(final_accs)),
            "std_accuracy": float(np.std(final_accs)),
            "min_accuracy": float(np.min(final_accs)),
            "max_accuracy": float(np.max(final_accs)),
            "trend_slope": slope,
            "improving": bool(slope > 0.001),
            "stable": bool(np.std(final_accs) < 0.2),
            "runs_above_80": sum(1 for acc in final_accs if acc > 0.8),
            "total_runs": len(final_accs),
        }

        success = (
            evaluation["final_accuracy"] > 0.75
            and evaluation["runs_above_80"] >= evaluation["total_runs"] // 2
        )
        if verbose:
            print("\n" + "=" * 60)
            print("Results Summary")
            print("=" * 60)
            print(
                f"Final Accuracy: {evaluation['final_accuracy']:.1%} "
                f"(+/- {evaluation['std_accuracy']:.1%})"
            )
            print(f"Range: {evaluation['min_accuracy']:.1%} - {evaluation['max_accuracy']:.1%}")
            print(
                f"Trend: {evaluation['trend_slope']:.5f} "
                f"({'improving' if evaluation['improving'] else 'stable'})"
            )
            print(f"Runs > 80%: {evaluation['runs_above_80']}/{evaluation['total_runs']}")
            print(
                "\nDecision: "
                f"{'[SUCCESS] Forward-only learning works!' if success else '[NEEDS WORK] Refine learning'}"
            )

        self.last_result = {
            "results": all_results,
            "evaluation": evaluation,
            "decision": "success" if success else "refine",
        }
        return self.last_result

    def save_results(self, filepath: str | None = None):
        result = self.last_result if self.last_result is not None else self.run(verbose=False)
        target = resolve_canonical_results_path("phase1_results.json") if filepath is None else filepath
        save_json(result, target)
        print(f"\nResults saved: {target}")


def main():
    print("\n" + "=" * 60)
    print("SEL-Lab Phase 1 Implementation")
    print("=" * 60)

    config = Phase1Config(
        input_size=4,
        output_size=2,
        hidden_size=8,
        learning_rate=0.05,
        runs=10,
        epochs=100,
    )
    experiment = Phase1Experiment(config)
    experiment.run(verbose=True)
    experiment.save_results()
    print("\nPhase 1 Complete!")


if __name__ == "__main__":
    main()
