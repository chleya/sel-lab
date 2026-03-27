# -*- coding: utf-8 -*-
"""
SEL-Lab Phase 4 - real MNIST entrypoint with digits fallback.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase4_common import (
    Phase4Config,
    load_real_mnist_or_digits,
    run_phase4_suite,
    split_indices,
)


def run_phase4_real():
    X, y, dataset_name = load_real_mnist_or_digits()
    if dataset_name == "MNIST":
        train_idx, test_idx = split_indices(len(X), n_train=10000, n_test=2000, seed=42)
        config = Phase4Config(
            input_size=784,
            hidden_size=128,
            learning_rate=0.01,
            runs=3,
            epochs=15,
            age_decay=0.005,
        )
    else:
        train_idx, test_idx = split_indices(len(X), n_train=int(len(X) * 0.7), seed=42)
        config = Phase4Config(
            input_size=X.shape[1],
            hidden_size=32,
            learning_rate=0.05,
            runs=3,
            epochs=15,
            age_decay=0.005,
        )

    return run_phase4_suite(
        title="Phase 4: Real MNIST Test",
        config=config,
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        result_filename="phase4_real_results.json",
        dataset_name=dataset_name,
        evolution_threshold=0.05,
    )


if __name__ == "__main__":
    run_phase4_real()
