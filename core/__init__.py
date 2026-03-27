# -*- coding: utf-8 -*-
"""
SEL-Lab Framework
Structural Evolution Learning Laboratory

项目目标：
- 无反向传播的前向学习
- 基于局部动力学的结构演化
- 适合边缘计算的自我演化系统

核心原则：
1. No backpropagation required
2. Learning as dynamics, not minimization
3. Structure as a first-class entity
4. Evaluation by adaptability
"""

__version__ = "1.0.0"
__author__ = "SEL-Lab"

from .sel_core import (
    SELConfig,
    SELModule,
    SELNetwork,
    SELTrainer,
    TrainingMetrics
)

from .environment import (
    Environment,
    SequencePredictionEnvironment
)

from .runtime import (
    dataclass_to_dict,
    ensure_dir,
    history_to_records,
    project_root,
    resolve_results_path,
    results_dir,
    save_json,
    split_train_test,
    summarize_scalar_runs,
)

__all__ = [
    'SELConfig',
    'SELModule', 
    'SELNetwork',
    'SELTrainer',
    'TrainingMetrics',
    'Environment',
    'SequencePredictionEnvironment',
    'dataclass_to_dict',
    'ensure_dir',
    'history_to_records',
    'project_root',
    'resolve_results_path',
    'results_dir',
    'save_json',
    'split_train_test',
    'summarize_scalar_runs',
]
