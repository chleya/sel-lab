# -*- coding: utf-8 -*-
"""
Compatibility wrapper for phase-3 structure-reuse ablations.
"""

from __future__ import annotations

from core.phase3_model import AblationConfig, FixedNetwork, ReusePolicyNetwork
from core.phase3_runner import (
    compare_against_fixed,
    compare_against_reference,
    run_phase3_ablations,
    summarize_policy_runs,
)

__all__ = [
    "AblationConfig",
    "FixedNetwork",
    "ReusePolicyNetwork",
    "summarize_policy_runs",
    "compare_against_fixed",
    "compare_against_reference",
    "run_phase3_ablations",
]


if __name__ == "__main__":
    run_phase3_ablations()
