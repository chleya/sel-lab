# -*- coding: utf-8 -*-
"""
SEL + EGGROLL 双演化边缘智能框架

核心模块：
- SEL: 结构演化学习
- EGGROLL: 低秩参数优化
- 多模态融合
- 边缘计算
"""

__version__ = "0.1.0"

from .sel import SELModule
from .sel_enhanced import SELNetwork, SELModule as SELModuleEnhanced, SELConfig
from .eggroll import EGGROLLOptimizer
from .fusion import FusionCenter, MultimodalFusion
from .edge import EdgeNode, EdgeSwarm, EdgeNodeConfig
from .optimized_edge import OptimizedEdgeNode, OptimizedEdgeSwarm, OptimizedEdgeNodeConfig
from .cerebellar_edge import CerebellarEdgeNode, CerebellarEdgeSwarm, CerebellarConfig
from .datasets import load_dataset, DatasetConfig, SKABDataset, OccupancyDataset
from .utils import generate_sensor_data, evaluate_performance, save_model, load_model

__all__ = [
    # 基础模块
    "SELModule",
    "SELModuleEnhanced",
    "SELNetwork",
    "SELConfig",
    "EGGROLLOptimizer",
    # 融合模块
    "FusionCenter",
    "MultimodalFusion",
    # 边缘计算
    "EdgeNode",
    "EdgeSwarm",
    "EdgeNodeConfig",
    # 优化版本
    "OptimizedEdgeNode",
    "OptimizedEdgeSwarm",
    "OptimizedEdgeNodeConfig",
    # 类小脑版本
    "CerebellarEdgeNode",
    "CerebellarEdgeSwarm",
    "CerebellarConfig",
    # 数据集
    "load_dataset",
    "DatasetConfig",
    "SKABDataset",
    "OccupancyDataset",
    # 工具函数
    "generate_sensor_data",
    "evaluate_performance",
    "save_model",
    "load_model"
]
