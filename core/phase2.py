# -*- coding: utf-8 -*-
"""
SEL-Lab Phase 2 - Structural Evolution Advantage.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase2_common import EvolvingNetwork, Phase2Config, create_simple_task, run_phase2_suite


def run_phase2():
    return run_phase2_suite(
        title="Phase 2: Structural Evolution Advantage",
        config=Phase2Config(),
        task_fn=create_simple_task,
        evolving_builder=lambda config, seed: EvolvingNetwork(config, seed=seed, use_knowledge_reuse=False),
        result_filename="phase2_results.json",
        success_threshold=0.0,
    )


if __name__ == "__main__":
    run_phase2()
