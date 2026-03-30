# -*- coding: utf-8 -*-
"""
P0-2 & P1-2: 重跑基础消融实验（修复剪枝 bug 后）

根据审查报告要求：
1. 重跑 phase3_ablation 和 phase3_digits_* 系列
2. 将 runs 提升到 5
3. 输出均值±标准差
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.phase3_registry_core import PHASE3_POLICY_ORDER
from core.phase3_common import Phase3Config, create_task, save_phase3_results, summarize_phase3_runs
from core.sel_core import SELNetwork


def run_single_experiment(config: Phase3Config, seed: int):
    """运行单次实验"""
    np.random.seed(seed)
    
    # 创建任务序列
    tasks = [create_task(config, task_id) for task_id in range(config.num_tasks)]
    
    # 初始化网络
    evolving = SELNetwork(config)
    fixed = SELNetwork(config)
    fixed.freeze_structure()
    
    results = []
    all_accuracies = {}
    
    for task_id, (X_train, y_train, X_test, y_test) in enumerate(tasks):
        # 训练
        for epoch in range(config.epochs_per_task):
            evolving.forward_learning(X_train, y_train)
            fixed.forward_learning(X_train, y_train)
        
        # 测试所有任务
        task_accuracies = {}
        for prev_task_id in range(task_id + 1):
            _, _, X_prev_test, y_prev_test = tasks[prev_task_id]
            
            evolving_acc = evolving.accuracy(X_prev_test, y_prev_test)
            fixed_acc = fixed.accuracy(X_prev_test, y_prev_test)
            
            task_accuracies[f"task_{prev_task_id}"] = {
                "evolving": evolving_acc,
                "fixed": fixed_acc
            }
        
        all_accuracies[f"task_{task_id}"] = task_accuracies
        
        # 记录结果
        avg_evolving = np.mean([
            all_accuracies[f"task_{t}"][f"task_{t}"]["evolving"] 
            for t in range(task_id + 1)
        ])
        avg_fixed = np.mean([
            all_accuracies[f"task_{t}"][f"task_{t}"]["fixed"] 
            for t in range(task_id + 1)
        ])
        
        results.append({
            "task_id": task_id,
            "current_evolving": task_accuracies[f"task_{task_id}"]["evolving"],
            "current_fixed": task_accuracies[f"task_{task_id}"]["fixed"],
            "avg_evolving": avg_evolving,
            "avg_fixed": avg_fixed,
            "unit_count": evolving.active_units
        })
    
    return results


def run_ablation_with_runs(config: Phase3Config, runs: int = 5):
    """运行多次实验并计算统计量"""
    all_results = []
    
    for run in range(runs):
        print(f"\n=== Run {run + 1}/{runs} ===")
        seed = config.random_seed + run * 1000
        results = run_single_experiment(config, seed)
        all_results.append(results)
    
    # 计算统计量
    summary = {
        "runs": runs,
        "num_tasks": config.num_tasks,
        "final_avg_evolving_mean": np.mean([r[-1]["avg_evolving"] for r in all_results]),
        "final_avg_evolving_std": np.std([r[-1]["avg_evolving"] for r in all_results]),
        "final_avg_fixed_mean": np.mean([r[-1]["avg_fixed"] for r in all_results]),
        "final_avg_fixed_std": np.std([r[-1]["avg_fixed"] for r in all_results]),
    }
    
    # 计算遗忘率
    forgetting_rates = []
    for run_results in all_results:
        initial_acc = run_results[0]["current_evolving"]
        final_acc = run_results[-1]["current_evolving"]
        forgetting = initial_acc - final_acc
        forgetting_rates.append(forgetting)
    
    summary["forgetting_rate_mean"] = np.mean(forgetting_rates)
    summary["forgetting_rate_std"] = np.std(forgetting_rates)
    
    return summary, all_results


def main():
    """主函数"""
    print("=" * 70)
    print("P0-2 & P1-2: 重跑基础消融实验（修复剪枝 bug 后）")
    print("=" * 70)
    
    # 配置
    config = Phase3Config(
        num_tasks=5,
        epochs_per_task=50,
        input_size=40,
        output_size=2,
        hidden_size=64,
        learning_rate=0.03,
        runs=5
    )
    
    runs = config.runs  # P1-2: 提升到 5
    
    print(f"\n配置:")
    print(f"  任务数: {config.num_tasks}")
    print(f"  Epochs per task: {config.epochs_per_task}")
    print(f"  Runs: {runs}")
    print(f"  输入维度: {config.input_size}")
    print(f"  隐藏层大小: {config.hidden_size}")
    
    # 运行实验
    print("\n开始实验...")
    summary, all_results = run_ablation_with_runs(config, runs)
    
    # 输出结果
    print("\n" + "=" * 70)
    print("实验结果:")
    print("=" * 70)
    print(f"\n最终平均准确率:")
    print(f"  Evolving: {summary['final_avg_evolving_mean']:.4f} ± {summary['final_avg_evolving_std']:.4f}")
    print(f"  Fixed:    {summary['final_avg_fixed_mean']:.4f} ± {summary['final_avg_fixed_std']:.4f}")
    print(f"  差异:     {summary['final_avg_evolving_mean'] - summary['final_avg_fixed_mean']:.4f}")
    
    print(f"\n遗忘率:")
    print(f"  Mean: {summary['forgetting_rate_mean']:.4f} ± {summary['forgetting_rate_std']:.4f}")
    
    # 保存结果
    results_path = "results/canonical/phase3_ablation_fixed_results.json"
    save_phase3_results(summary, all_results, results_path)
    print(f"\n结果已保存到: {results_path}")
    
    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
