# -*- coding: utf-8 -*-
"""
Phase 4 age_decay 参数优化脚本
测试不同 age_decay 和 evolution_threshold 组合的结构扩展效果
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase4_common import Phase4Config, load_real_mnist_or_digits, split_indices, run_phase4_suite


def run_age_decay_optimization():
    """优化 age_decay 和 evolution_threshold 参数"""
    # 加载数据集
    X, y, dataset_name = load_real_mnist_or_digits()
    n = len(X)
    train_idx, test_idx = split_indices(n, n_train=int(n * 0.8))
    
    # 测试的参数组合
    age_decay_values = [0.001, 0.002, 0.003, 0.004, 0.005]
    evolution_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
    
    results = []
    
    for age_decay in age_decay_values:
        for threshold in evolution_thresholds:
            print(f"\n{'='*70}")
            print(f"Testing age_decay={age_decay}, threshold={threshold}")
            print(f"{'='*70}")
            
            # 配置：使用更小的学习率，让模型收敛更慢，tension 有更多机会积累
            config = Phase4Config(
                input_size=X.shape[1],
                hidden_size=100,
                learning_rate=0.01,  # 减小学习率
                runs=3,
                epochs=50,  # 增加训练轮数
                age_decay=age_decay,
                initial_tension=0.5,
                clone_noise_scale=0.1
            )
            
            # 运行测试
            result = run_phase4_suite(
                title=f"Phase 4 - age_decay={age_decay}, threshold={threshold}",
                config=config,
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                result_filename=f"phase4_optimize_{age_decay:.4f}_{threshold:.2f}.json",
                dataset_name=dataset_name,
                evolution_threshold=threshold
            )
            
            # 记录结果
            results.append({
                "age_decay": age_decay,
                "evolution_threshold": threshold,
                "fixed_final": result["fixed_final"],
                "evolving_final": result["evolving_final"],
                "advantage": result["advantage"],
                "decision": result["decision"],
                "max_units": max(r["max_units"] for r in result["results"])
            })
    
    # 保存优化结果
    optimization_results = {
        "dataset": dataset_name,
        "optimization_results": results
    }
    
    output_path = PROJECT_ROOT / "results" / "phase4_age_decay_optimization.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(optimization_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("Age Decay Optimization Summary")
    print(f"{'='*70}")
    print("age_decay | threshold | fixed_final | evolving_final | advantage | max_units | decision")
    print("-" * 70)
    
    # 排序结果，优先考虑 max_units > 1 的配置
    sorted_results = sorted(results, key=lambda r: (-r['max_units'], -r['advantage']))
    
    for r in sorted_results:
        print(f"{r['age_decay']:.4f} | {r['evolution_threshold']:.2f} | {r['fixed_final']:.3f} | {r['evolving_final']:.3f} | {r['advantage']:+.3f} | {r['max_units']:d} | {r['decision']}")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_age_decay_optimization()
