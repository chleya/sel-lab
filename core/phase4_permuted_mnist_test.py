# -*- coding: utf-8 -*-
"""
Phase 4 Permuted-MNIST 测试脚本 - 持续学习版本
使用 EvolvingDFAClassifier（带隐藏层）替代 SEL
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase4_common import Phase4Config, DFAClassifier, EvolvingDFAClassifier, one_hot


def load_mnist_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 MNIST 数据集"""
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(int)
        
        # 分割训练集和测试集
        n = len(X)
        train_idx = np.arange(60000)
        test_idx = np.arange(60000, n)
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"Failed to load MNIST: {e}")
        # 回退到 digits 数据集
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target
        
        n = len(X)
        train_idx = np.arange(int(n * 0.8))
        test_idx = np.arange(int(n * 0.8), n)
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        return X_train, y_train, X_test, y_test


def create_permuted_mnist(X: np.ndarray, y: np.ndarray, num_tasks: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """创建 Permuted-MNIST 任务序列"""
    tasks = []
    y_onehot = one_hot(y, classes=10)
    
    for i in range(num_tasks):
        rng = np.random.default_rng(42 + i)
        permutation = rng.permutation(X.shape[1])
        X_permuted = X[:, permutation]
        tasks.append((X_permuted, y_onehot))
    return tasks


def evaluate_permuted_mnist():
    """评估 Permuted-MNIST 基准 - 正确的持续学习版本"""
    print("=" * 70)
    print("Evaluating Permuted-MNIST Benchmark (Continual Learning)")
    print("=" * 70)
    
    # 加载数据集
    X_train, y_train, X_test, y_test = load_mnist_dataset()
    
    # 减小数据集大小以加速测试
    X_train = X_train[:10000]  # 只使用10000个训练样本
    y_train = y_train[:10000]
    X_test = X_test[:2000]     # 只使用2000个测试样本
    y_test = y_test[:2000]
    
    print(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test")
    
    # 创建 Permuted-MNIST 任务
    num_tasks = 5  # 使用5个任务进行标准测试
    train_tasks = create_permuted_mnist(X_train, y_train, num_tasks)
    test_tasks = create_permuted_mnist(X_test, y_test, num_tasks)
    
    print(f"Created {len(train_tasks)} Permuted-MNIST tasks")
    
    # 测试固定网络（DFA）- 持续学习版本
    print("\nTesting Fixed Network (DFA) - Continual Learning")
    fixed_results = []
    fixed_initial_accs = {}  # 记录每个任务的初始准确率
    
    # 创建单个DFA网络用于所有任务
    config = Phase4Config(
        input_size=train_tasks[0][0].shape[1],
        hidden_size=100,
        output_size=10,
        learning_rate=0.01,
        epochs=50
    )
    fixed_network = DFAClassifier(config, seed=42)
    
    for task_idx, (X_task, y_task) in enumerate(train_tasks):
        print(f"  Training on Task {task_idx}...")
        
        # 在当前任务上继续训练（不是新建网络）
        for epoch in range(config.epochs):
            perm = np.random.permutation(len(X_task))
            for i in perm:
                fixed_network.learn(X_task[i], y_task[i])
        
        # 测试当前任务并记录初始准确率
        current_test_X, current_test_y = test_tasks[task_idx]
        current_acc = fixed_network.accuracy(current_test_X, current_test_y)
        fixed_initial_accs[task_idx] = current_acc
        
        # 测试所有已见任务（计算真实遗忘）
        backward_accs = []
        forgetting_list = []
        for prev_idx in range(task_idx + 1):
            prev_test_X, prev_test_y = test_tasks[prev_idx]
            acc = fixed_network.accuracy(prev_test_X, prev_test_y)
            backward_accs.append(acc)
            # 真实遗忘 = 初始准确率 - 当前准确率
            forgetting = fixed_initial_accs[prev_idx] - acc
            forgetting_list.append(forgetting)
        
        avg_backward_acc = np.mean(backward_accs)
        avg_forgetting = np.mean(forgetting_list) if task_idx > 0 else 0.0
        
        fixed_results.append({
            "task": task_idx,
            "current_accuracy": current_acc,
            "backward_accuracy": avg_backward_acc,
            "forgetting": avg_forgetting,
            "per_task_forgetting": forgetting_list
        })
        
        print(f"  Task {task_idx}: Current={current_acc:.1%}, Backward={avg_backward_acc:.1%}, Forgetting={avg_forgetting:+.1%}")
    
    # 测试演化网络（EvolvingDFA）- 持续学习版本
    print("\nTesting Evolving Network (EvolvingDFA) - Continual Learning")
    evolving_results = []
    evolving_initial_accs = {}  # 记录每个任务的初始准确率
    
    # 创建单个EvolvingDFA网络用于所有任务
    evolving_config = Phase4Config(
        input_size=train_tasks[0][0].shape[1],
        hidden_size=100,
        output_size=10,
        learning_rate=0.01,
        epochs=50,
        age_decay=0.001,  # 使用优化后的 age_decay
        initial_tension=0.5,
        clone_noise_scale=0.1
    )
    evolving_network = EvolvingDFAClassifier(evolving_config, seed=42, max_units=10)
    
    for task_idx, (X_task, y_task) in enumerate(train_tasks):
        print(f"  Training on Task {task_idx}...")
        
        # 在当前任务上继续训练（不是新建网络）
        for epoch in range(evolving_config.epochs):
            perm = np.random.permutation(len(X_task))
            for i in perm:
                evolving_network.learn(X_task[i], y_task[i])
            
            # 每个epoch结束后进行演化
            if epoch % 5 == 0:  # 每5个epoch演化一次
                evolving_network.evolve()
        
        # 测试当前任务并记录初始准确率
        current_test_X, current_test_y = test_tasks[task_idx]
        current_acc = evolving_network.accuracy(current_test_X, current_test_y)
        evolving_initial_accs[task_idx] = current_acc
        
        # 测试所有已见任务（计算真实遗忘）
        backward_accs = []
        forgetting_list = []
        for prev_idx in range(task_idx + 1):
            prev_test_X, prev_test_y = test_tasks[prev_idx]
            acc = evolving_network.accuracy(prev_test_X, prev_test_y)
            backward_accs.append(acc)
            # 真实遗忘 = 初始准确率 - 当前准确率
            forgetting = evolving_initial_accs[prev_idx] - acc
            forgetting_list.append(forgetting)
        
        avg_backward_acc = np.mean(backward_accs)
        avg_forgetting = np.mean(forgetting_list) if task_idx > 0 else 0.0
        
        evolving_results.append({
            "task": task_idx,
            "current_accuracy": current_acc,
            "backward_accuracy": avg_backward_acc,
            "forgetting": avg_forgetting,
            "per_task_forgetting": forgetting_list,
            "final_units": len([u for u in evolving_network.units if u["active"]]),
            "total_units": len(evolving_network.units)
        })
        
        active_units = len([u for u in evolving_network.units if u["active"]])
        print(f"  Task {task_idx}: Current={current_acc:.1%}, Backward={avg_backward_acc:.1%}, Forgetting={avg_forgetting:+.1%}, Units={active_units}/{len(evolving_network.units)}")
    
    # 保存结果
    results = {
        "dataset": "Permuted-MNIST",
        "num_tasks": num_tasks,
        "fixed_results": fixed_results,
        "evolving_results": evolving_results
    }
    
    output_path = PROJECT_ROOT / "results" / "phase4_permuted_mnist_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    fixed_final = fixed_results[-1]
    evolving_final = evolving_results[-1]
    
    print(f"Fixed Network (DFA):")
    print(f"  Final Current Accuracy: {fixed_final['current_accuracy']:.1%}")
    print(f"  Final Backward Accuracy: {fixed_final['backward_accuracy']:.1%}")
    print(f"  Final Forgetting: {fixed_final['forgetting']:+.1%}")
    
    print(f"\nEvolving Network (EvolvingDFA):")
    print(f"  Final Current Accuracy: {evolving_final['current_accuracy']:.1%}")
    print(f"  Final Backward Accuracy: {evolving_final['backward_accuracy']:.1%}")
    print(f"  Final Forgetting: {evolving_final['forgetting']:+.1%}")
    print(f"  Final Active Units: {evolving_final['final_units']}/{evolving_final['total_units']}")
    
    print(f"\nAdvantage (EvolvingDFA - DFA):")
    print(f"  Current Accuracy: {evolving_final['current_accuracy'] - fixed_final['current_accuracy']:+.1%}")
    print(f"  Backward Accuracy: {evolving_final['backward_accuracy'] - fixed_final['backward_accuracy']:+.1%}")
    print(f"  Forgetting: {evolving_final['forgetting'] - fixed_final['forgetting']:+.1%}")


def main():
    """主函数"""
    evaluate_permuted_mnist()


if __name__ == "__main__":
    main()
