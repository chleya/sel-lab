# -*- coding: utf-8 -*-
"""
Phase 2 mechanism bench for single-task structure-reuse policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase2_common import (
    EvolvingNetwork,
    FixedNetwork,
    Phase2Config,
    create_hard_task,
    create_simple_task,
)
from core.runtime import resolve_results_path, save_json, split_train_test


@dataclass
class Phase2AblationConfig:
    simple: Phase2Config = field(default_factory=lambda: Phase2Config(runs=3, epochs=50))
    hard: Phase2Config = field(
        default_factory=lambda: Phase2Config(output_size=3, learning_rate=0.03, runs=3, epochs=80)
    )
    add_threshold: float = 0.15
    remove_threshold: float = 0.02
    max_units: int = 10
    low_tension_ratio: float = 0.75
    limited_adapt_scale: float = 0.35


def summarize_policy_runs(results: List[Dict]) -> Dict:
    return {
        "final_accuracy": float(np.mean([row["final_accuracy"] for row in results])),
        "auc_accuracy": float(np.mean([row["auc_accuracy"] for row in results])),
        "max_units": float(np.mean([row["max_units"] for row in results])),
    }


def run_policy_on_task(
    *,
    config: Phase2Config,
    task_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    policy: str,
    use_knowledge_reuse: bool,
    add_threshold: float,
    remove_threshold: float,
    max_units: int,
    low_tension_ratio: float,
    limited_adapt_scale: float,
) -> List[Dict]:
    X, y = task_fn()
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    runs = []

    for run in range(config.runs):
        seed = run * 100 + 42
        learner = EvolvingNetwork(
            config,
            seed=seed,
            use_knowledge_reuse=use_knowledge_reuse,
            policy=policy,
            low_tension_ratio=low_tension_ratio,
            limited_adapt_scale=limited_adapt_scale,
        )
        accs: List[float] = []

        for epoch in range(config.epochs):
            for i in range(len(X_train)):
                learner.learn(X_train[i], y_train[i], lr=config.learning_rate)
            if (epoch + 1) % 10 == 0:
                learner.evolve(
                    add_threshold=add_threshold,
                    remove_threshold=remove_threshold,
                    max_units=max_units,
                )
            accs.append(learner.accuracy(X_test, y_test))

        runs.append(
            {
                "final_accuracy": accs[-1],
                "auc_accuracy": float(np.mean(accs)),
                "max_units": learner.max_active_units,
            }
        )

    return runs


def run_fixed_baseline(config: Phase2Config, task_fn: Callable[[], Tuple[np.ndarray, np.ndarray]]) -> Dict:
    X, y = task_fn()
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    runs = []

    for run in range(config.runs):
        seed = run * 100 + 42
        learner = FixedNetwork(config, seed=seed)
        accs: List[float] = []
        for _ in range(config.epochs):
            for i in range(len(X_train)):
                learner.learn(X_train[i], y_train[i], lr=config.learning_rate)
            accs.append(learner.accuracy(X_test, y_test))
        runs.append(
            {
                "final_accuracy": accs[-1],
                "auc_accuracy": float(np.mean(accs)),
                "max_units": 1,
            }
        )

    return {
        "summary": summarize_policy_runs(runs),
        "runs": runs,
    }


def run_phase2_ablation(
    config: Phase2AblationConfig | None = None,
    result_filename: str = "phase2_ablation_results.json",
) -> Dict:
    config = config or Phase2AblationConfig()
    tasks = {
        "simple": {
            "config": config.simple,
            "task_fn": create_simple_task,
        },
        "hard": {
            "config": config.hard,
            "task_fn": create_hard_task,
        },
    }
    policies = {
        "adapt_only": {"policy": "adapt_only", "use_knowledge_reuse": True},
        "random_growth": {"policy": "random_growth", "use_knowledge_reuse": False},
        "best_clone": {"policy": "best_clone", "use_knowledge_reuse": True},
        "low_tension_clone": {"policy": "low_tension_clone", "use_knowledge_reuse": True},
        "best_clone_limited_adapt": {
            "policy": "best_clone_limited_adapt",
            "use_knowledge_reuse": True,
        },
    }

    print("\n" + "=" * 60)
    print("Phase 2: Mechanism Bench")
    print("=" * 60)

    payload = {"config": {}, "tasks": {}}
    for task_name, task_spec in tasks.items():
        task_config = task_spec["config"]
        task_fn = task_spec["task_fn"]
        print(f"\n--- Task: {task_name} ---")

        fixed = run_fixed_baseline(task_config, task_fn)
        fixed_summary = fixed["summary"]
        task_payload = {"fixed": fixed, "policies": {}}

        print(
            f"Fixed: final={fixed_summary['final_accuracy']:.1%}, "
            f"auc={fixed_summary['auc_accuracy']:.1%}"
        )

        for name, spec in policies.items():
            runs = run_policy_on_task(
                config=task_config,
                task_fn=task_fn,
                policy=spec["policy"],
                use_knowledge_reuse=spec["use_knowledge_reuse"],
                add_threshold=config.add_threshold,
                remove_threshold=config.remove_threshold,
                max_units=config.max_units,
                low_tension_ratio=config.low_tension_ratio,
                limited_adapt_scale=config.limited_adapt_scale,
            )
            summary = summarize_policy_runs(runs)
            summary["final_gain_vs_fixed"] = summary["final_accuracy"] - fixed_summary["final_accuracy"]
            summary["auc_gain_vs_fixed"] = summary["auc_accuracy"] - fixed_summary["auc_accuracy"]
            task_payload["policies"][name] = {
                "summary": summary,
                "runs": runs,
            }
            print(
                f"{name}: final={summary['final_accuracy']:.1%}, "
                f"gain={summary['final_gain_vs_fixed']:+.1%}, "
                f"auc={summary['auc_accuracy']:.1%}, "
                f"units={summary['max_units']:.1f}"
            )

        payload["tasks"][task_name] = task_payload

    payload["config"] = {
        "simple": {
            "runs": config.simple.runs,
            "epochs": config.simple.epochs,
            "learning_rate": config.simple.learning_rate,
        },
        "hard": {
            "runs": config.hard.runs,
            "epochs": config.hard.epochs,
            "learning_rate": config.hard.learning_rate,
        },
        "add_threshold": config.add_threshold,
        "remove_threshold": config.remove_threshold,
        "max_units": config.max_units,
        "low_tension_ratio": config.low_tension_ratio,
        "limited_adapt_scale": config.limited_adapt_scale,
    }

    target = resolve_results_path(result_filename)
    save_json(payload, target)
    print(f"\nResults saved: {target}")
    return payload


if __name__ == "__main__":
    run_phase2_ablation()
