# -*- coding: utf-8 -*-
"""
Mechanism analysis for why Phase 4 evolution underperforms.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase4_common import (
    DFAClassifier,
    EvolvingDFAClassifier,
    Phase4Config,
    load_digits_dataset,
    split_indices,
)
from core.runtime import resolve_exploratory_results_path, save_json


def _effective_learning_rate(config: Phase4Config, active_units: List[Dict]) -> float:
    if not active_units:
        return 0.0
    return float(
        np.mean(
            [
                config.learning_rate * (1.0 / (1.0 + unit["age"] * config.age_decay))
                for unit in active_units
            ]
        )
    )


def analyze_phase4_failure(
    epochs: int = 15,
    seed: int = 42,
    evolution_threshold: float = 0.3,
    age_decay: float = 0.05,
    initial_tension: float = 0.5,
    result_filename: str | None = None,
) -> Dict:
    X, y = load_digits_dataset(n_samples=1000)
    train_idx, test_idx = split_indices(len(X), n_train=int(len(X) * 0.7), seed=seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    config = Phase4Config(
        input_size=X.shape[1],
        hidden_size=32,
        learning_rate=0.05,
        runs=1,
        epochs=epochs,
        age_decay=age_decay,
        initial_tension=initial_tension,
    )
    fixed = DFAClassifier(config, seed=seed)
    evolving = EvolvingDFAClassifier(config, seed=seed)

    history: List[Dict] = []
    expansion_epochs: List[int] = []

    for epoch in range(epochs):
        perm = np.random.default_rng(seed + epoch).permutation(len(X_train))
        fixed_epoch_losses = []
        evolving_epoch_losses = []

        for i in perm:
            fixed_epoch_losses.append(fixed.learn(X_train[i], y_train[i]))
            evolving_epoch_losses.append(evolving.learn(X_train[i], y_train[i]))

        changes = evolving.evolve(threshold=evolution_threshold)
        if changes:
            expansion_epochs.append(epoch)

        active_units = [unit for unit in evolving.units if unit["active"]]
        avg_tension = float(np.mean([unit["tension"] for unit in active_units])) if active_units else 0.0
        avg_age = float(np.mean([unit["age"] for unit in active_units])) if active_units else 0.0
        effective_lr = _effective_learning_rate(config, active_units)
        fixed_acc = fixed.accuracy(X_test, y_test)
        evolving_acc = evolving.accuracy(X_test, y_test)

        record = {
            "epoch": epoch,
            "fixed_loss": float(np.mean(fixed_epoch_losses)),
            "evolving_loss": float(np.mean(evolving_epoch_losses)),
            "fixed_acc": fixed_acc,
            "evolving_acc": evolving_acc,
            "accuracy_gap": fixed_acc - evolving_acc,
            "avg_tension": avg_tension,
            "active_units": evolving.active_units,
            "avg_unit_age": avg_age,
            "effective_learning_rate": effective_lr,
            "changes": changes,
        }
        history.append(record)

    tensions = [row["avg_tension"] for row in history]
    gaps = [row["accuracy_gap"] for row in history]
    effective_lrs = [row["effective_learning_rate"] for row in history]

    findings = {
        "triggered_expansion": bool(expansion_epochs),
        "expansion_epochs": expansion_epochs,
        "max_tension": float(np.max(tensions)) if tensions else 0.0,
        "mean_tension": float(np.mean(tensions)) if tensions else 0.0,
        "threshold": evolution_threshold,
        "final_fixed_acc": history[-1]["fixed_acc"],
        "final_evolving_acc": history[-1]["evolving_acc"],
        "final_accuracy_gap": history[-1]["accuracy_gap"],
        "mean_accuracy_gap": float(np.mean(gaps)) if gaps else 0.0,
        "initial_effective_lr": effective_lrs[0] if effective_lrs else 0.0,
        "final_effective_lr": effective_lrs[-1] if effective_lrs else 0.0,
        "mean_effective_lr": float(np.mean(effective_lrs)) if effective_lrs else 0.0,
        "dominant_failure_mode": _infer_failure_mode(history, evolution_threshold),
    }

    payload = {
        "config": {
            "epochs": epochs,
            "seed": seed,
            "threshold": evolution_threshold,
            "age_decay": age_decay,
            "initial_tension": initial_tension,
        },
        "findings": findings,
        "history": history,
    }
    if result_filename is not None:
        target = resolve_exploratory_results_path(result_filename)
        save_json(payload, target)
        print(f"Saved analysis: {target}")
    return payload


def _infer_failure_mode(history: List[Dict], threshold: float) -> str:
    if not history:
        return "no_data"
    max_tension = max(row["avg_tension"] for row in history)
    final_units = history[-1]["active_units"]
    final_gap = history[-1]["accuracy_gap"]
    lr_ratio = history[-1]["effective_learning_rate"] / max(history[0]["effective_learning_rate"], 1e-8)

    if max_tension < threshold and final_units == 1:
        return "under_triggered_evolution"
    if final_units > 1 and final_gap > 0.1:
        return "cloning_without_benefit"
    if lr_ratio < 0.5:
        return "age_decay_too_aggressive"
    return "capacity_or_update_mismatch"


def main():
    payload = analyze_phase4_failure(result_filename="phase4_failure_analysis.json")
    findings = payload["findings"]
    print("\nPhase 4 Failure Analysis")
    print(f"  Triggered expansion: {findings['triggered_expansion']}")
    print(f"  Mean tension: {findings['mean_tension']:.3f}")
    print(f"  Final accuracy gap: {findings['final_accuracy_gap']:+.1%}")
    print(f"  Effective LR: {findings['initial_effective_lr']:.4f} -> {findings['final_effective_lr']:.4f}")
    print(f"  Failure mode: {findings['dominant_failure_mode']}")


if __name__ == "__main__":
    main()
