# -*- coding: utf-8 -*-
"""
验证"结构即智能"发现 - 快速测试最优配置

测试:
1. 原始 SEL 默认配置 (约 10 模块)
2. 最优配置 (3 模块 + 限制增长)

预期: 如果"结构即智能"正确，3模块应该表现更好或相当
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sel_core import SELNetwork, SELConfig
from core.runtime import resolve_results_path, save_json


def run_quick_phase3_experiment(
    config: SELConfig,
    experiment_name: str,
    n_tasks: int = 3,
    epochs_per_task: int = 20
) -> Dict:
    """快速运行 Phase 3 实验"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {experiment_name}")
    print(f"{'='*60}")
    print(f"Config: max_modules={config.max_modules}, initial_modules={config.initial_modules}")
    
    # 创建简单的连续学习任务
    np.random.seed(42)
    tasks = []
    for i in range(n_tasks):
        X = np.random.randn(50, 8)
        # 每个任务有不同的决策边界
        if i == 0:
            y = np.array([[1, 0] if x[0] > 0 else [0, 1] for x in X])
        elif i == 1:
            y = np.array([[1, 0] if x[1] > 0 else [0, 1] for x in X])
        else:
            y = np.array([[1, 0] if x[0] + x[1] > 0 else [0, 1] for x in X])
        tasks.append((X, y))
    
    # 创建 SEL
    sel = SELNetwork(config)
    
    # 训练并记录
    results = {
        'task_performances': [],
        'module_counts': [],
        'final_modules': 0,
        'avg_performance': 0.0,
        'training_time': 0.0
    }
    
    start_time = time.time()
    
    for task_idx, (X, y) in enumerate(tasks):
        print(f"\nTask {task_idx + 1}/{n_tasks}")
        
        # 训练
        for epoch in range(epochs_per_task):
            for i in range(len(X)):
                sel.forward_learning(X[i:i+1], y[i:i+1])
        
        # 评估
        predictions = []
        for i in range(len(X)):
            pred = sel.forward(X[i:i+1])
            predictions.append(pred)
        predictions = np.array(predictions)
        
        # Fix shape mismatch
        if len(predictions.shape) > 2:
            predictions = predictions.reshape(predictions.shape[0], -1)
        if len(y.shape) > 2:
            y = y.reshape(y.shape[0], -1)
            
        pred_labels = predictions.argmax(axis=1) if predictions.shape[1] > 1 else (predictions > 0.5).astype(int).flatten()
        true_labels = y.argmax(axis=1) if y.shape[1] > 1 else (y > 0.5).astype(int).flatten()
        
        accuracy = np.mean(pred_labels == true_labels)
        
        results['task_performances'].append(accuracy)
        results['module_counts'].append(len(sel.modules))
        
        print(f"  Accuracy: {accuracy:.2%}, Modules: {len(sel.modules)}")
    
    results['training_time'] = time.time() - start_time
    results['final_modules'] = len(sel.modules)
    results['avg_performance'] = np.mean(results['task_performances'])
    results['final_performance'] = results['task_performances'][-1]
    
    print(f"\nSummary:")
    print(f"  Avg Performance: {results['avg_performance']:.2%}")
    print(f"  Final Performance: {results['final_performance']:.2%}")
    print(f"  Final Modules: {results['final_modules']}")
    print(f"  Training Time: {results['training_time']:.2f}s")
    
    return results


def verify_optimal_config():
    """验证最优配置"""
    
    print("=" * 70)
    print(" VERIFYING 'STRUCTURE = INTELLIGENCE' HYPOTHESIS ")
    print("=" * 70)
    print("\nTesting if 3-module config outperforms default config...")
    
    # 配置 1: 原始默认配置 (允许增长)
    default_config = SELConfig(
        input_size=8,
        output_size=2,
        initial_modules=3,
        max_modules=10,  # 允许增长到10
        learning_rate=0.1,
        epochs=20
    )
    
    # 配置 2: 最优配置 (限制为3模块)
    optimal_config = SELConfig(
        input_size=8,
        output_size=2,
        initial_modules=3,
        max_modules=3,   # 限制为3，不增长
        learning_rate=0.1,
        epochs=20
    )
    
    # 运行实验
    print("\n" + "=" * 70)
    print(" EXPERIMENT 1: Default Config (max_modules=10)")
    print("=" * 70)
    default_results = run_quick_phase3_experiment(
        default_config, 
        "Default Config",
        n_tasks=3,
        epochs_per_task=20
    )
    
    print("\n" + "=" * 70)
    print(" EXPERIMENT 2: Optimal Config (max_modules=3)")
    print("=" * 70)
    optimal_results = run_quick_phase3_experiment(
        optimal_config,
        "Optimal Config (3 modules)",
        n_tasks=3,
        epochs_per_task=20
    )
    
    # 对比结果
    print("\n" + "=" * 70)
    print(" COMPARISON RESULTS ")
    print("=" * 70)
    
    performance_diff = optimal_results['avg_performance'] - default_results['avg_performance']
    module_diff = optimal_results['final_modules'] - default_results['final_modules']
    time_diff = optimal_results['training_time'] - default_results['training_time']
    
    print(f"\nDefault Config:")
    print(f"  Avg Performance: {default_results['avg_performance']:.2%}")
    print(f"  Final Modules: {default_results['final_modules']}")
    print(f"  Training Time: {default_results['training_time']:.2f}s")
    
    print(f"\nOptimal Config (3 modules):")
    print(f"  Avg Performance: {optimal_results['avg_performance']:.2%}")
    print(f"  Final Modules: {optimal_results['final_modules']}")
    print(f"  Training Time: {optimal_results['training_time']:.2f}s")
    
    print(f"\nDifference (Optimal - Default):")
    print(f"  Performance: {performance_diff:+.2%} {'✅ BETTER' if performance_diff > 0 else '❌ WORSE'}")
    print(f"  Modules: {module_diff:+d} {'✅ SIMPLER' if module_diff < 0 else '❌ MORE COMPLEX'}")
    print(f"  Time: {time_diff:+.2f}s {'✅ FASTER' if time_diff < 0 else '❌ SLOWER'}")
    
    # 结论
    print("\n" + "=" * 70)
    print(" CONCLUSION ")
    print("=" * 70)
    
    if performance_diff > -0.05:  # 允许5%的误差
        print("✅ 'STRUCTURE = INTELLIGENCE' HYPOTHESIS SUPPORTED!")
        print("   Simpler structure performs similarly or better.")
        print("   This validates our discovery!")
        hypothesis_supported = True
    else:
        print("❌ 'STRUCTURE = INTELLIGENCE' HYPOTHESIS NOT SUPPORTED")
        print("   Simpler structure performs significantly worse.")
        print("   Our discovery may not generalize to this task.")
        hypothesis_supported = False
    
    # 保存结果
    results = {
        'hypothesis_supported': hypothesis_supported,
        'performance_diff': performance_diff,
        'module_diff': module_diff,
        'time_diff': time_diff,
        'default_results': default_results,
        'optimal_results': optimal_results
    }
    
    save_path = resolve_results_path("verify_optimal_config_results.json")
    save_json(results, save_path)
    print(f"\nDetailed results saved to: {save_path}")
    
    return results


if __name__ == "__main__":
    verify_optimal_config()
