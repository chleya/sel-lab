# -*- coding: utf-8 -*-
"""
Shared building blocks for Phase 2 experiment variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import resolve_canonical_results_path, resolve_results_path, save_json, split_train_test


@dataclass
class Phase2Config:
    input_size: int = 4
    output_size: int = 2
    hidden_size: int = 8
    learning_rate: float = 0.05
    runs: int = 3
    epochs: int = 50


def create_simple_task(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(100, 4))
    y = np.zeros((100, 2))
    for i in range(100):
        if X[i, 0] + X[i, 1] > 0:
            y[i, 0] = 1
        else:
            y[i, 1] = 1
    return X, y


def create_hard_task(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 2.0, size=(200, 4))
    y = np.zeros((200, 3))
    for i in range(200):
        score = np.sin(X[i, 0] * 2) + np.cos(X[i, 1] * 2) + X[i, 2] * X[i, 3]
        if score > 1.0:
            y[i, 0] = 1
        elif score < -1.0:
            y[i, 1] = 1
        else:
            y[i, 2] = 1
    return X, y


class FixedNetwork:
    """Shared fixed-structure DFA baseline."""

    def __init__(self, config: Phase2Config, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.W1 = self.rng.normal(0.0, 0.5, size=(config.input_size, config.hidden_size))
        self.W2 = self.rng.normal(0.0, 0.5, size=(config.hidden_size, config.output_size))
        self.feedback = self.rng.normal(0.0, 0.5, size=(config.output_size, config.hidden_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.W1) @ self.W2

    def learn(self, x: np.ndarray, target: np.ndarray, lr: float = 0.01) -> float:
        h = np.tanh(x @ self.W1)
        out = h @ self.W2
        error = target - out
        fb_error = self.feedback.T @ error
        self.W2 += lr * np.outer(h, error)
        self.W1 += lr * np.outer(x, fb_error)
        return float(np.mean(error ** 2))

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)


class EvolvingNetwork:
    """Shared evolving network with optional knowledge reuse."""

    def __init__(
        self,
        config: Phase2Config,
        seed: int | None = None,
        use_knowledge_reuse: bool = False,
        policy: str = "random_growth",
        low_tension_ratio: float = 0.75,
        limited_adapt_scale: float = 0.35,
    ):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.use_knowledge_reuse = use_knowledge_reuse
        self.policy = policy
        self.low_tension_ratio = low_tension_ratio
        self.limited_adapt_scale = limited_adapt_scale
        self.units: List[Dict] = []
        self.max_active_units = 0
        for _ in range(2 if config.output_size == 2 else 1):
            self.add_unit()

    def add_unit(self, clone_from: int = -1) -> int:
        if self.use_knowledge_reuse and 0 <= clone_from < len(self.units):
            source = self.units[clone_from]
            unit = {
                "W1": source["W1"].copy() + self.rng.normal(0.0, 0.1, size=source["W1"].shape),
                "W2": source["W2"].copy() + self.rng.normal(0.0, 0.1, size=source["W2"].shape),
                "feedback": source["feedback"].copy()
                + self.rng.normal(0.0, 0.1, size=source["feedback"].shape),
                "active": True,
                "tension": source["tension"],
                "age": 0,
                "lr_scale": 1.0,
            }
        else:
            unit = {
                "W1": self.rng.normal(0.0, 0.5, size=(self.config.input_size, self.config.hidden_size)),
                "W2": self.rng.normal(0.0, 0.5, size=(self.config.hidden_size, self.config.output_size)),
                "feedback": self.rng.normal(
                    0.0, 0.5, size=(self.config.output_size, self.config.hidden_size)
                ),
                "active": True,
                "tension": 0.5 if self.use_knowledge_reuse else 0.0,
                "age": 0,
                "lr_scale": 1.0,
            }
        self.units.append(unit)
        self.max_active_units = max(self.max_active_units, self.active_units)
        return len(self.units) - 1

    @property
    def active_units(self) -> int:
        return sum(1 for unit in self.units if unit["active"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        outputs = []
        for unit in self.units:
            if unit["active"]:
                h = np.tanh(x @ unit["W1"])
                outputs.append(h @ unit["W2"])
        if outputs:
            return np.mean(outputs, axis=0)
        return np.zeros(self.config.output_size)

    def learn(self, x: np.ndarray, target: np.ndarray, lr: float = 0.01) -> float:
        out = self.forward(x)
        error = target - out
        for unit in self.units:
            if not unit["active"]:
                continue
            h = np.tanh(x @ unit["W1"])
            fb_error = unit["feedback"].T @ error
            unit_lr = lr * unit["lr_scale"]
            if self.use_knowledge_reuse:
                unit_lr *= 1.0 / (1.0 + unit["age"] * 0.1)
            unit["W2"] += unit_lr * np.outer(h, error)
            unit["W1"] += unit_lr * np.outer(x, fb_error)
            unit["age"] += 1
            unit["tension"] = 0.9 * unit["tension"] + 0.1 * float(np.mean(error ** 2))
        return float(np.mean(error ** 2))

    def evolve(self, add_threshold: float, remove_threshold: float, max_units: int = 10) -> List[str]:
        active = [unit for unit in self.units if unit["active"]]
        if not active:
            return []

        changes: List[str] = []
        avg_tension = float(np.mean([unit["tension"] for unit in active]))
        if avg_tension > add_threshold and self.active_units < max_units:
            active_indices = [idx for idx, unit in enumerate(self.units) if unit["active"]]
            if self.policy == "adapt_only":
                pass
            elif self.policy == "random_growth" or not self.use_knowledge_reuse:
                self.add_unit()
                changes.append("add")
            else:
                source_idx = min(active_indices, key=lambda idx: self.units[idx]["tension"])
                source_tension = self.units[source_idx]["tension"]
                if (
                    self.policy == "low_tension_clone"
                    and source_tension >= avg_tension * self.low_tension_ratio
                ):
                    pass
                else:
                    new_idx = self.add_unit(clone_from=source_idx)
                    changes.append(f"add(clone_from={source_idx})")
                    if self.policy == "best_clone_limited_adapt":
                        self.units[new_idx]["lr_scale"] = self.limited_adapt_scale
                        changes.append(f"limited_adapt({new_idx})")
        elif avg_tension < remove_threshold and self.active_units > 3:
            active_indices = [idx for idx, unit in enumerate(self.units) if unit["active"]]
            worst_idx = max(active_indices, key=lambda idx: self.units[idx]["tension"])
            self.units[worst_idx]["active"] = False
            changes.append(f"remove({worst_idx})")

        self.max_active_units = max(self.max_active_units, self.active_units)
        return changes

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)


def run_phase2_suite(
    *,
    title: str,
    config: Phase2Config,
    task_fn,
    evolving_builder,
    result_filename: str,
    success_threshold: float,
    extra_payload: Dict | None = None,
) -> Dict:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    X, y = task_fn()
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    results = []

    for run in range(config.runs):
        print(f"\n--- Run {run + 1}/{config.runs} ---")
        seed = run * 100 + 42
        fixed = FixedNetwork(config, seed=seed)
        evolving = evolving_builder(config, seed)
        fixed_accs: List[float] = []
        evolving_accs: List[float] = []

        for epoch in range(config.epochs):
            for i in range(len(X_train)):
                fixed.learn(X_train[i], y_train[i])
                evolving.learn(X_train[i], y_train[i])

            if (epoch + 1) % 10 == 0:
                evolving.evolve(add_threshold=0.15, remove_threshold=0.02, max_units=10)

            fixed_acc = fixed.accuracy(X_test, y_test)
            evolving_acc = evolving.accuracy(X_test, y_test)
            fixed_accs.append(fixed_acc)
            evolving_accs.append(evolving_acc)

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1}: "
                    f"Fixed={fixed_acc:.1%}, Evolving={evolving_acc:.1%}, Units={evolving.active_units}"
                )

        results.append(
            {
                "fixed_final": fixed_accs[-1],
                "evolving_final": evolving_accs[-1],
                "fixed_auc": float(np.mean(fixed_accs)),
                "evolving_auc": float(np.mean(evolving_accs)),
                "max_units": evolving.max_active_units,
            }
        )
        print(f"Final: Fixed={fixed_accs[-1]:.1%}, Evolving={evolving_accs[-1]:.1%}")

    fixed_final = float(np.mean([result["fixed_final"] for result in results]))
    evolving_final = float(np.mean([result["evolving_final"] for result in results]))
    fixed_auc = float(np.mean([result["fixed_auc"] for result in results]))
    evolving_auc = float(np.mean([result["evolving_auc"] for result in results]))
    advantage = evolving_final - fixed_final
    success = advantage > success_threshold

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Final Accuracy: Fixed={fixed_final:.1%}, Evolving={evolving_final:.1%}")
    print(f"AUC (Avg): Fixed={fixed_auc:.1%}, Evolving={evolving_auc:.1%}")
    print(f"Evolution Advantage: {advantage:+.1%}")
    print(f"\n{'[SUCCESS] Evolution provides advantage!' if success else '[NEUTRAL] No clear advantage'}")

    payload = {
        "fixed_final": fixed_final,
        "evolving_final": evolving_final,
        "fixed_auc": fixed_auc,
        "evolving_auc": evolving_auc,
        "advantage": float(advantage),
        "decision": "success" if success else "neutral",
        "results": results,
    }
    if extra_payload:
        payload.update(extra_payload)

    if "/" in result_filename or "\\" in result_filename:
        target = resolve_results_path(result_filename)
    else:
        target = resolve_canonical_results_path(result_filename)
    save_json(payload, target)
    print(f"\nResults saved: {target}")
    return payload
