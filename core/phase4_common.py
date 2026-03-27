# -*- coding: utf-8 -*-
"""
Shared classifiers, data loading, and execution helpers for Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import resolve_canonical_results_path, resolve_results_path, save_json


@dataclass
class Phase4Config:
    input_size: int
    hidden_size: int
    output_size: int = 10
    learning_rate: float = 0.05
    runs: int = 3
    epochs: int = 30
    age_decay: float = 0.05
    initial_tension: float = 0.5
    clone_noise_scale: float = 0.1


class DFAClassifier:
    def __init__(self, config: Phase4Config, seed: int | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / config.input_size)
        scale2 = np.sqrt(2.0 / config.hidden_size)
        self.W1 = self.rng.normal(0.0, scale1, size=(config.input_size, config.hidden_size))
        self.W2 = self.rng.normal(0.0, scale2, size=(config.hidden_size, config.output_size))
        self.feedback = self.rng.normal(0.0, 0.1, size=(config.output_size, config.hidden_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.W1) @ self.W2

    def learn(self, x: np.ndarray, target: np.ndarray) -> float:
        h = np.tanh(x @ self.W1)
        out = h @ self.W2
        error = target - out
        fb_error = self.feedback.T @ error
        self.W2 += self.config.learning_rate * np.outer(h, error)
        self.W1 += self.config.learning_rate * np.outer(x, fb_error)
        self.W1 = np.clip(self.W1, -5, 5)
        self.W2 = np.clip(self.W2, -5, 5)
        return float(np.mean(error ** 2))

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)


class EvolvingDFAClassifier:
    def __init__(self, config: Phase4Config, seed: int | None = None, max_units: int = 8):
        self.config = config
        self.max_units = max_units
        self.rng = np.random.default_rng(seed)
        self.units: List[Dict] = []
        self.max_active_units = 0
        self.add_unit()

    def add_unit(self, clone_from: int = -1) -> int:
        scale1 = np.sqrt(2.0 / self.config.input_size)
        scale2 = np.sqrt(2.0 / self.config.hidden_size)
        if 0 <= clone_from < len(self.units):
            source = self.units[clone_from]
            unit = {
                "W1": source["W1"].copy()
                + self.rng.normal(0.0, self.config.clone_noise_scale, size=source["W1"].shape),
                "W2": source["W2"].copy()
                + self.rng.normal(0.0, self.config.clone_noise_scale, size=source["W2"].shape),
                "feedback": source["feedback"].copy()
                + self.rng.normal(0.0, self.config.clone_noise_scale, size=source["feedback"].shape),
                "tension": source["tension"],
                "age": 0,
                "active": True,
            }
        else:
            unit = {
                "W1": self.rng.normal(0.0, scale1, size=(self.config.input_size, self.config.hidden_size)),
                "W2": self.rng.normal(0.0, scale2, size=(self.config.hidden_size, self.config.output_size)),
                "feedback": self.rng.normal(
                    0.0, 0.1, size=(self.config.output_size, self.config.hidden_size)
                ),
                "tension": self.config.initial_tension,
                "age": 0,
                "active": True,
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

    def learn(self, x: np.ndarray, target: np.ndarray) -> float:
        out = self.forward(x)
        error = target - out
        for unit in self.units:
            if not unit["active"]:
                continue
            h = np.tanh(x @ unit["W1"])
            fb_error = unit["feedback"].T @ error
            lr = self.config.learning_rate * (1.0 / (1.0 + unit["age"] * self.config.age_decay))
            unit["W2"] += lr * np.outer(h, error)
            unit["W1"] += lr * np.outer(x, fb_error)
            unit["W1"] = np.clip(unit["W1"], -5, 5)
            unit["W2"] = np.clip(unit["W2"], -5, 5)
            unit["age"] += 1
            unit["tension"] = 0.9 * unit["tension"] + 0.1 * float(np.mean(error ** 2))
        return float(np.mean(error ** 2))

    def evolve(self, threshold: float = 0.3) -> List[str]:
        active = [unit for unit in self.units if unit["active"]]
        if not active:
            return []
        avg_tension = float(np.mean([unit["tension"] for unit in active]))
        changes: List[str] = []
        if avg_tension > threshold and self.active_units < self.max_units:
            active_indices = [idx for idx, unit in enumerate(self.units) if unit["active"]]
            source_idx = min(active_indices, key=lambda idx: self.units[idx]["tension"])
            self.add_unit(clone_from=source_idx)
            changes.append(f"add(clone_from={source_idx})")
        self.max_active_units = max(self.max_active_units, self.active_units)
        return changes

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)


def one_hot(labels: np.ndarray, classes: int = 10) -> np.ndarray:
    encoded = np.zeros((len(labels), classes))
    for i, label in enumerate(labels):
        encoded[i, int(label)] = 1
    return encoded


def load_digits_dataset(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = one_hot(digits.target, classes=10)
    n = min(n_samples, len(X))
    return X[:n], y[:n]


def load_real_mnist_or_digits() -> Tuple[np.ndarray, np.ndarray, str]:
    prefer_openml = os.environ.get("SEL_LAB_USE_OPENML", "").lower() in {"1", "true", "yes"}
    if prefer_openml:
        try:
            from sklearn.datasets import fetch_openml

            mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
            X = mnist.data.astype(np.float32) / 255.0
            y = one_hot(mnist.target.astype(int), classes=10)
            return X, y, "MNIST"
        except Exception:
            pass

    X, y = load_digits_dataset(n_samples=1797)
    return X, y, "digits"


def split_indices(n: int, n_train: int, n_test: int | None = None, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    if n_test is None:
        n_test = n - n_train
    return indices[:n_train], indices[n_train : n_train + n_test]


def run_phase4_suite(
    *,
    title: str,
    config: Phase4Config,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    result_filename: str,
    dataset_name: str,
    evolution_threshold: float = 0.3,
) -> Dict:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    results = []
    for run in range(config.runs):
        print(f"\n--- Run {run + 1}/{config.runs} ---")
        seed = run * 100 + 42
        fixed = DFAClassifier(config, seed=seed)
        evolving = EvolvingDFAClassifier(config, seed=seed)
        fixed_accs: List[float] = []
        evolving_accs: List[float] = []

        for epoch in range(config.epochs):
            perm = np.random.default_rng(seed + epoch).permutation(len(X_train))
            for i in perm:
                fixed.learn(X_train[i], y_train[i])
                evolving.learn(X_train[i], y_train[i])
            evolving.evolve(threshold=evolution_threshold)

            fixed_acc = fixed.accuracy(X_test, y_test)
            evolving_acc = evolving.accuracy(X_test, y_test)
            fixed_accs.append(fixed_acc)
            evolving_accs.append(evolving_acc)
            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}: Fixed={fixed_acc:.1%}, "
                    f"Evolving={evolving_acc:.1%}, Units={evolving.active_units}"
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

    fixed_final = float(np.mean([r["fixed_final"] for r in results]))
    evolving_final = float(np.mean([r["evolving_final"] for r in results]))
    fixed_auc = float(np.mean([r["fixed_auc"] for r in results]))
    evolving_auc = float(np.mean([r["evolving_auc"] for r in results]))
    advantage = evolving_final - fixed_final
    success = advantage > 0

    print("\n" + "=" * 60)
    print("Phase 4 Results Summary")
    print("=" * 60)
    print(f"Final Test Accuracy: Fixed={fixed_final:.1%}, Evolving={evolving_final:.1%}")
    print(f"AUC: Fixed={fixed_auc:.1%}, Evolving={evolving_auc:.1%}")
    print(f"Evolution Advantage: {advantage:+.1%}")
    print(f"\n{'[SUCCESS] Evolution provides advantage!' if success else '[NEUTRAL] No clear advantage'}")

    payload = {
        "dataset": dataset_name,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "fixed_final": fixed_final,
        "evolving_final": evolving_final,
        "fixed_auc": fixed_auc,
        "evolving_auc": evolving_auc,
        "advantage": float(advantage),
        "decision": "success" if success else "neutral",
        "results": results,
    }
    if "/" in result_filename or "\\" in result_filename:
        target = resolve_results_path(result_filename)
    else:
        target = resolve_canonical_results_path(result_filename)
    save_json(payload, target)
    print(f"\nResults saved: {target}")
    return payload
