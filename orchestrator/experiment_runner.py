# -*- coding: utf-8 -*-
"""
SEL-Lab Orchestrator
实验运行器
"""

import sys
import os
import json
from typing import Dict, List

# 添加 core 目录到路径
core_path = os.path.join(os.path.dirname(__file__), '..', 'core')
sys.path.insert(0, core_path)

import numpy as np
from sel_core import SELTrainer, SELConfig, create_task


def run_experiments(n_runs: int = 5) -> Dict:
    """
    运行多次实验
    
    Args:
        n_runs: 实验次数
        
    Returns:
        实验结果汇总
    """
    print("\n" + "=" * 60)
    print("SEL-Lab Experiments")
    print("=" * 60)
    
    # 创建数据
    X, y = create_task("simple_classification")
    split = len(X) // 2
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    print(f"Data: {len(X_train)} train, {len(X_test)} test samples")
    
    # 配置
    config = SELConfig(
        input_size=X_train.shape[1],
        output_size=y_train.shape[1],
        initial_modules=3,
        learning_rate=0.1,
        epochs=100
    )
    
    # 运行多次实验
    all_results = []
    
    for run in range(n_runs):
        trainer = SELTrainer(config)
        result = trainer.train(X_train, y_train, X_test, y_test)
        
        all_results.append({
            'run': run + 1,
            'train_accuracy': result['final_train_accuracy'],
            'test_accuracy': result['final_test_accuracy'],
            'modules': result['final_modules'],
            'changes': result['total_structural_changes']
        })
        
        print(f"Run {run+1}/{n_runs}: Train={result['final_train_accuracy']:.1%}, "
              f"Test={result['final_test_accuracy']:.1%}, Modules={result['final_modules']}")
    
    # 汇总
    test_accs = [r['test_accuracy'] for r in all_results]
    
    summary = {
        'runs': n_runs,
        'mean_test_accuracy': float(np.mean(test_accs)),
        'std_test_accuracy': float(np.std(test_accs)),
        'max_test_accuracy': float(np.max(test_accs)),
        'results': all_results
    }
    
    return summary


def save_results(summary: Dict, filepath: str = None):
    """保存结果"""
    if filepath is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, 'sel_experiment_results.json')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    """主函数"""
    # 运行实验
    summary = run_experiments(n_runs=5)
    
    # 保存
    save_results(summary)
    
    # 输出汇总
    print(f"\n" + "=" * 60)
    print("Summary:")
    print(f"  Mean Test Accuracy: {summary['mean_test_accuracy']:.1%}")
    print(f"  Max Test Accuracy: {summary['max_test_accuracy']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
