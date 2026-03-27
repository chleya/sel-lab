# -*- coding: utf-8 -*-
"""
SEL-Lab Phase 4 - digits/MNIST-scale image classification test.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase4_common import Phase4Config, load_digits_dataset, run_phase4_suite, split_indices


def run_phase4():
    X, y = load_digits_dataset(n_samples=1000)
    train_idx, test_idx = split_indices(len(X), n_train=int(len(X) * 0.7), seed=42)
    return run_phase4_suite(
        title="Phase 4: MNIST Real-World Test",
        config=Phase4Config(
            input_size=X.shape[1],
            hidden_size=32,
            learning_rate=0.05,
            runs=3,
            epochs=30,
            age_decay=0.005,
        ),
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        result_filename="phase4_results.json",
        dataset_name="digits",
        evolution_threshold=0.05,
    )


if __name__ == "__main__":
    run_phase4()
