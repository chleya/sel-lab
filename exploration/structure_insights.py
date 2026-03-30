# -*- coding: utf-8 -*-
"""
SEL-Lab Structure Insights (Simplified)
简化版结构洞察 - 保留核心发现，去除过度工程

核心洞察:
1. 结构复杂度与智能能力存在非线性关系
2. 存在"最优复杂度"，不是越复杂越好
3. SEL 需要简约性约束防止过度增长

使用方式:
    from exploration.structure_insights import analyze_structure_efficiency
    efficiency = analyze_structure_efficiency(sel_network)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


def compute_structure_complexity(modules: List) -> Dict[str, float]:
    """
    计算结构复杂度 - 简化版
    
    Returns:
        {
            'module_count': 模块数量,
            'total_parameters': 总参数量,
            'complexity_score': 综合复杂度得分
        }
    """
    n_modules = len(modules)
    total_params = sum(m.weights.size for m in modules)
    
    # 简化复杂度计算
    complexity = n_modules * np.log1p(total_params / max(n_modules, 1))
    
    return {
        'module_count': n_modules,
        'total_parameters': total_params,
        'complexity_score': complexity
    }


def assess_task_complexity(X: np.ndarray, y: np.ndarray) -> float:
    """
    评估任务复杂度
    
    Args:
        X: 输入数据
        y: 目标数据
    
    Returns:
        任务复杂度分数 (0-1)
    """
    # 基于数据特征和标签分布评估复杂度
    n_samples, n_features = X.shape
    n_classes = y.shape[1] if len(y.shape) > 1 else len(np.unique(y))
    
    # 特征复杂度
    feature_complexity = min(1.0, n_features / 20.0)
    
    # 样本复杂度
    sample_complexity = min(1.0, n_samples / 500.0)
    
    # 类别复杂度
    class_complexity = min(1.0, n_classes / 10.0)
    
    # 数据分布复杂度（基于特征方差）
    if n_features > 0:
        feature_var = np.var(X, axis=0)
        var_complexity = min(1.0, np.mean(feature_var) / 5.0)
    else:
        var_complexity = 0.0
    
    # 综合复杂度
    complexity = 0.3 * feature_complexity + 0.2 * sample_complexity + 0.3 * class_complexity + 0.2 * var_complexity
    return float(np.clip(complexity, 0.0, 1.0))


def get_optimal_module_range(task_complexity: float) -> Tuple[int, int]:
    """
    根据任务复杂度获取最优模块范围
    
    Args:
        task_complexity: 任务复杂度 (0-1)
    
    Returns:
        (min_modules, max_modules) 元组
    """
    if task_complexity < 0.3:
        # 简单任务
        return (1, 3)
    elif task_complexity < 0.7:
        # 中等任务
        return (3, 5)
    else:
        # 复杂任务
        return (4, 6)


def analyze_structure_efficiency(
    sel_network,
    performance: Optional[float] = None,
    task_complexity: Optional[float] = None
) -> Dict:
    """
    分析结构效率
    
    Args:
        sel_network: SELNetwork 实例
        performance: 可选的性能指标 (0-1)
        task_complexity: 可选的任务复杂度 (0-1)
    
    Returns:
        {
            'complexity': 结构复杂度,
            'performance': 性能,
            'efficiency': 效率 = 性能 / 复杂度,
            'is_optimal': 是否在最优范围内,
            'task_complexity': 任务复杂度,
            'optimal_range': 最优模块范围
        }
    """
    complexity_metrics = compute_structure_complexity(sel_network.modules)
    complexity = complexity_metrics['complexity_score']
    n_modules = complexity_metrics['module_count']
    
    # 如果没有提供性能，使用占位符
    perf = performance if performance is not None else 0.5
    
    # 如果没有提供任务复杂度，使用默认值
    task_complexity = task_complexity if task_complexity is not None else 0.5
    
    # 获取最优模块范围
    min_optimal, max_optimal = get_optimal_module_range(task_complexity)
    
    # 计算效率
    efficiency = perf / (complexity + 1e-6)
    
    # 基于任务复杂度判断是否在最优范围
    is_optimal = min_optimal <= n_modules <= max_optimal
    
    # 生成推荐
    if is_optimal:
        recommendation = 'OK - Optimal structure for task complexity'
    elif n_modules < min_optimal:
        recommendation = f'Consider adding modules (optimal range: {min_optimal}-{max_optimal})'
    else:
        recommendation = f'Consider pruning modules (optimal range: {min_optimal}-{max_optimal})'
    
    return {
        'complexity': complexity,
        'performance': perf,
        'efficiency': efficiency,
        'module_count': n_modules,
        'is_optimal': is_optimal,
        'task_complexity': task_complexity,
        'optimal_range': (min_optimal, max_optimal),
        'recommendation': recommendation
    }


def get_structure_recommendation(n_modules: int, task_complexity: float = 0.5) -> str:
    """
    基于我们的研究发现给出建议
    
    Args:
        n_modules: 当前模块数量
        task_complexity: 任务复杂度 (0-1)
    
    Returns:
        建议字符串
    """
    min_optimal, max_optimal = get_optimal_module_range(task_complexity)
    
    if n_modules < min_optimal:
        return f"Module count is low. Consider adding modules for this task complexity (optimal: {min_optimal}-{max_optimal})."
    elif min_optimal <= n_modules <= max_optimal:
        return f"✅ Optimal range for task complexity. Good balance of complexity and capability."
    elif max_optimal < n_modules <= max_optimal + 2:
        return f"⚠️ Getting complex. Monitor for diminishing returns (optimal: {min_optimal}-{max_optimal})."
    else:
        return f"❌ Likely over-complex. Consider pruning based on 'structure = intelligence' finding (optimal: {min_optimal}-{max_optimal})."


# 保留验证实验的简化版
if __name__ == "__main__":
    print("Structure Insights Module")
    print("=" * 50)
    print("\nCore insight from our research:")
    print("  Structure complexity ↔ Intelligence capability")
    print("  Optimal range depends on task complexity")
    print("  Simple tasks: 1-3 modules")
    print("  Medium tasks: 3-5 modules")
    print("  Complex tasks: 4-6 modules")
    
    print("\nTask complexity assessment examples:")
    # 创建不同复杂度的测试数据
    import numpy as np
    
    # 简单任务
    X_simple = np.random.normal(0, 1, size=(100, 2))
    y_simple = np.zeros((100, 2))
    y_simple[X_simple[:, 0] > 0, 0] = 1
    y_simple[X_simple[:, 0] <= 0, 1] = 1
    complexity_simple = assess_task_complexity(X_simple, y_simple)
    print(f"  Simple task complexity: {complexity_simple:.2f}")
    
    # 中等任务
    X_medium = np.random.normal(0, 1, size=(200, 4))
    y_medium = np.zeros((200, 2))
    y_medium[X_medium[:, 0] + X_medium[:, 1] > 0, 0] = 1
    y_medium[X_medium[:, 0] + X_medium[:, 1] <= 0, 1] = 1
    complexity_medium = assess_task_complexity(X_medium, y_medium)
    print(f"  Medium task complexity: {complexity_medium:.2f}")
    
    # 复杂任务
    X_complex = np.random.normal(0, 1, size=(300, 8))
    y_complex = np.zeros((300, 3))
    for i in range(300):
        if X_complex[i, 0] + X_complex[i, 1] > 0 and X_complex[i, 2] > 0:
            y_complex[i, 0] = 1
        elif X_complex[i, 3] + X_complex[i, 4] > 0:
            y_complex[i, 1] = 1
        else:
            y_complex[i, 2] = 1
    complexity_complex = assess_task_complexity(X_complex, y_complex)
    print(f"  Complex task complexity: {complexity_complex:.2f}")
    
    print("\nRecommendations for different module counts and task complexities:")
    for complexity in [0.2, 0.5, 0.8]:
        print(f"\nTask complexity: {complexity:.1f}")
        for n in [1, 3, 5, 7]:
            print(f"  {n} modules: {get_structure_recommendation(n, complexity)}")
