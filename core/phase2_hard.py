# -*- coding: utf-8 -*-
"""
SEL-Lab Phase 2b - Hard Task Structural Evolution test.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase2_common import EvolvingNetwork, Phase2Config, create_hard_task, run_phase2_suite


def run_phase2_hard():
    return run_phase2_suite(
        title="Phase 2b: Hard Task - Structural Evolution Advantage",
        config=Phase2Config(output_size=3, learning_rate=0.03, runs=5, epochs=100),
        task_fn=create_hard_task,
        evolving_builder=lambda config, seed: EvolvingNetwork(config, seed=seed, use_knowledge_reuse=False),
        result_filename="phase2_hard_results.json",
        success_threshold=0.05,
        extra_payload={"task": "3-class XOR-like"},
    )


if __name__ == "__main__":
    run_phase2_hard()
