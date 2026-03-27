# -*- coding: utf-8 -*-
"""
Shared task generators and reporting helpers for Phase 3 incremental learning.
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

from core.runtime import resolve_canonical_results_path, resolve_results_path, save_json


@dataclass
class Phase3Config:
    input_size: int = 4
    output_size: int = 2
    hidden_size: int = 8
    learning_rate: float = 0.03
    runs: int = 5
    epochs_per_task: int = 50
    num_tasks: int = 4
    task_suite: str = "default"


def _embed_feature_shift_inputs(X: np.ndarray) -> np.ndarray:
    features = []
    for dim in range(X.shape[1]):
        v = X[:, dim]
        pos = np.maximum(v, 0.0)
        neg = np.maximum(-v, 0.0)
        abs_v = np.abs(v)
        sq = v**2
        features.extend(
            [
                pos,
                neg,
                abs_v,
                sq,
                np.maximum(v - 1.0, 0.0),
                np.maximum(-v - 1.0, 0.0),
                np.maximum(v - 2.0, 0.0),
                np.maximum(-v - 2.0, 0.0),
                np.exp(-abs_v),
                np.exp(-0.5 * sq),
                np.abs(v - 1.0),
                np.abs(v + 1.0),
                np.maximum(1.0 - abs_v, 0.0),
                np.maximum(2.0 - abs_v, 0.0),
                pos * abs_v,
                neg * abs_v,
            ]
        )
    return np.stack(features, axis=1).astype(np.float32)


def _embed_sparse_conflict_inputs(X: np.ndarray) -> np.ndarray:
    features = []
    for dim in range(X.shape[1]):
        v = X[:, dim]
        abs_v = np.abs(v)
        sparse_core = np.maximum(abs_v - 1.25, 0.0)
        features.extend(
            [
                np.maximum(v, 0.0),
                np.maximum(-v, 0.0),
                abs_v,
                v**2,
                np.maximum(v - 1.5, 0.0),
                np.maximum(-v - 1.5, 0.0),
                np.maximum(0.75 - abs_v, 0.0),
                sparse_core,
                np.exp(-abs_v),
                np.exp(-2.0 * sparse_core),
                (v > 0.5).astype(np.float32),
                (v < -0.5).astype(np.float32),
                (abs_v < 0.35).astype(np.float32),
                np.abs(v - 1.5),
                np.abs(v + 1.5),
                sparse_core * abs_v,
            ]
        )
    return np.stack(features, axis=1).astype(np.float32)


def _create_digits_pair_task(task_id: int, seed: int, task_suite: str) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits

    rng = np.random.default_rng(seed + task_id)
    digits = load_digits()
    pair_map = ((0, 1), (2, 3), (4, 5), (6, 7))
    digit_a, digit_b = pair_map[task_id % len(pair_map)]
    mask = np.isin(digits.target, [digit_a, digit_b])
    X_all = digits.data[mask].astype(np.float32) / 16.0
    labels = digits.target[mask]

    class_a = X_all[labels == digit_a]
    class_b = X_all[labels == digit_b]
    n_per_class = min(80, len(class_a), len(class_b))
    idx_a = rng.permutation(len(class_a))[:n_per_class]
    idx_b = rng.permutation(len(class_b))[:n_per_class]
    X = np.concatenate([class_a[idx_a], class_b[idx_b]], axis=0)
    y = np.zeros((len(X), 2))
    y[:n_per_class, 0] = 1
    y[n_per_class:, 1] = 1

    if task_suite == "digits_pairs_noisy":
        X = np.clip(X + rng.normal(0.0, 0.20, size=X.shape), 0.0, 1.0)
    elif task_suite == "digits_pairs_permuted":
        permutation = rng.permutation(X.shape[1])
        X = X[:, permutation]
    elif task_suite not in {"digits_pairs", "digits_pairs_noisy", "digits_pairs_permuted"}:
        raise ValueError(f"Unknown Phase 3 digits task suite: {task_suite}")

    shuffle = rng.permutation(len(X))
    return X[shuffle], y[shuffle]


def create_task(task_id: int, seed: int = 42, task_suite: str = "default") -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + task_id)
    if task_suite.startswith("digits_pairs"):
        return _create_digits_pair_task(task_id, seed=seed, task_suite=task_suite)

    X = rng.normal(0.0, 2.0, size=(100, 4))
    y = np.zeros((100, 2))

    for i in range(100):
        if task_suite == "default":
            if task_id == 0:
                positive = X[i, 0] + X[i, 1] > 0
            elif task_id == 1:
                positive = X[i, 0] * X[i, 1] > 0
            elif task_id == 2:
                positive = np.sin(X[i, 0]) + np.cos(X[i, 1]) > 0
            else:
                positive = X[i, 0] ** 2 + X[i, 1] ** 2 > 2
        elif task_suite == "rotated":
            if task_id == 0:
                positive = 0.8 * X[i, 0] - 0.6 * X[i, 1] > 0
            elif task_id == 1:
                positive = (X[i, 0] - X[i, 1]) * (X[i, 0] + X[i, 1]) > 0
            elif task_id == 2:
                positive = np.sin(0.7 * X[i, 0] + 0.4 * X[i, 1]) > 0
            else:
                positive = (0.5 * X[i, 0] - 0.8 * X[i, 1]) ** 2 + (0.6 * X[i, 0] + 0.3 * X[i, 1]) ** 2 > 2.5
        elif task_suite in {
            "feature_shift",
            "feature_shift_noisy",
            "feature_shift_nonnegative",
            "feature_shift_embedded",
            "feature_shift_mixed",
            "feature_shift_sparse_embedded",
        }:
            if task_suite == "feature_shift_nonnegative":
                X[i] = np.abs(X[i])
            if task_id == 0:
                score = X[i, 0] + 0.5 * X[i, 2]
                if task_suite == "feature_shift_noisy":
                    score += 0.30 * X[i, 1] - 0.20 * abs(X[i, 3])
                elif task_suite == "feature_shift_mixed":
                    score += 0.25 * X[i, 1]
                    score += 0.15 * np.sign(X[i, 0] * X[i, 2])
                positive = score > 0
            elif task_id == 1:
                score = X[i, 1] * X[i, 2]
                if task_suite == "feature_shift_noisy":
                    score -= 0.35 * X[i, 0]
                elif task_suite == "feature_shift_mixed":
                    score -= 0.20 * abs(X[i, 0])
                    score += 0.25 * X[i, 3]
                positive = score > 0
            elif task_id == 2:
                score = np.sin(X[i, 2]) + np.cos(X[i, 3])
                if task_suite == "feature_shift_noisy":
                    score += 0.25 * X[i, 1]
                elif task_suite == "feature_shift_mixed":
                    score += 0.35 * X[i, 0]
                    score -= 0.20 * abs(X[i, 1] - X[i, 2])
                positive = score > 0
            else:
                score = X[i, 2] ** 2 + 0.5 * X[i, 3] ** 2
                if task_suite == "feature_shift_noisy":
                    score -= 0.50 * abs(X[i, 0])
                elif task_suite == "feature_shift_mixed":
                    score -= 0.35 * abs(X[i, 0])
                    score += 0.30 * X[i, 1] * X[i, 2]
                positive = score > 2
        elif task_suite == "noisy":
            if task_id == 0:
                score = X[i, 0] + X[i, 1] + 0.35 * X[i, 2]
                positive = score > 0
            elif task_id == 1:
                score = X[i, 0] * X[i, 1] - 0.4 * X[i, 2]
                positive = score > 0
            elif task_id == 2:
                score = np.sin(X[i, 0]) + np.cos(X[i, 1]) + 0.3 * X[i, 3]
                positive = score > 0
            else:
                score = X[i, 0] ** 2 + X[i, 1] ** 2 - 0.7 * abs(X[i, 2])
                positive = score > 2
        else:
            raise ValueError(f"Unknown Phase 3 task suite: {task_suite}")

        if positive:
            y[i, 0] = 1
        else:
            y[i, 1] = 1

    if task_suite == "feature_shift_embedded":
        return _embed_feature_shift_inputs(X), y
    if task_suite == "feature_shift_sparse_embedded":
        return _embed_sparse_conflict_inputs(X), y
    return X, y


def summarize_phase3_runs(results: List[List[Dict]]) -> Dict:
    final_avg_fixed = float(np.mean([run[-1]["avg_fixed"] for run in results]))
    final_avg_evolving = float(np.mean([run[-1]["avg_evolving"] for run in results]))
    first_task_fixed = float(np.mean([run[0]["current_fixed"] for run in results]))
    first_task_evolving = float(np.mean([run[0]["current_evolving"] for run in results]))
    forgetting_fixed = first_task_fixed - final_avg_fixed
    forgetting_evolving = first_task_evolving - final_avg_evolving
    advantage = final_avg_evolving - final_avg_fixed
    success = advantage > 0 and abs(forgetting_evolving) < abs(forgetting_fixed)

    return {
        "final_avg_fixed": final_avg_fixed,
        "final_avg_evolving": final_avg_evolving,
        "forgetting_fixed": float(forgetting_fixed),
        "forgetting_evolving": float(forgetting_evolving),
        "advantage": float(advantage),
        "decision": "success" if success else "neutral",
    }


def save_phase3_results(payload: Dict) -> Path:
    target = resolve_canonical_results_path("phase3_results.json")
    save_json(payload, target)
    return target


def save_phase3_ablation_results(payload: Dict, filename: str = "phase3_ablation_results.json") -> Path:
    if "/" in filename or "\\" in filename:
        target = resolve_results_path(filename)
    else:
        target = resolve_canonical_results_path(filename)
    save_json(payload, target)
    return target
