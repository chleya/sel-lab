# -*- coding: utf-8 -*-
"""
Phase 4 Permuted-MNIST 测试脚本
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

from core.sel_core import SELConfig, SELTrainer
from core.phase4_common import Phase4Config, DFAClassifier, EvolvingDFAClassifier, one_hot, split_indices


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
    """评估 Permuted-MNIST 基准"""
    print("=" * 70)
    print("Evaluating Permuted-MNIST Benchmark")
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
    num_tasks = 3  # 减少任务数量以加速测试
    train_tasks = create_permuted_mnist(X_train, y_train, num_tasks)
    test_tasks = create_permuted_mnist(X_test, y_test, num_tasks)
    
    print(f"Created {len(train_tasks)} Permuted-MNIST tasks")
    
    # 测试固定网络
    print("\nTesting Fixed Network (DFA)")
    fixed_results = []
    for task_idx, (X_task, y_task) in enumerate(train_tasks):
        config = Phase4Config(
            input_size=X_task.shape[1],
            hidden_size=100,
            output_size=10,
            learning_rate=0.01,
            epochs=50
        )
        
        # 训练
        fixed = DFAClassifier(config, seed=42 + task_idx)
        for epoch in range(config.epochs):
            perm = np.random.permutation(len(X_task))
            for i in perm:
                fixed.learn(X_task[i], y_task[i])
        
        # 测试当前任务
        current_test_X, current_test_y = test_tasks[task_idx]
        current_acc = fixed.accuracy(current_test_X, current_test_y)
        
        # 测试所有之前的任务
        backward_accs = []
        for prev_idx in range(task_idx + 1):
            prev_test_X, prev_test_y = test_tasks[prev_idx]
            acc = fixed.accuracy(prev_test_X, prev_test_y)
            backward_accs.append(acc)
        
        avg_backward_acc = np.mean(backward_accs)
        forgetting = current_acc - avg_backward_acc if task_idx > 0 else 0.0
        
        fixed_results.append({
            "task": task_idx,
            "current_accuracy": current_acc,
            "backward_accuracy": avg_backward_acc,
            "forgetting": forgetting
        })
        
        print(f"Task {task_idx}: Current={current_acc:.1%}, Backward={avg_backward_acc:.1%}, Forgetting={forgetting:+.1%}")
    
    # 测试演化网络
    print("\nTesting Evolving Network (SEL)")
    evolving_results = []
    for task_idx, (X_task, y_task) in enumerate(train_tasks):
        config = SELConfig(
            input_size=X_task.shape[1],
            output_size=10,
            initial_modules=3,
            learning_rate=0.01,
            epochs=50,
            max_modules=5
        )
        
        # 训练
        trainer = SELTrainer(config)
        result = trainer.train(X_task, y_task, task_id=task_idx)
        
        # 测试当前任务
        current_test_X, current_test_y = test_tasks[task_idx]
        current_acc = trainer.network.accuracy(current_test_X, current_test_y)
        
        # 测试所有之前的任务
        backward_accs = []
        for prev_idx in range(task_idx + 1):
            prev_test_X, prev_test_y = test_tasks[prev_idx]
            acc = trainer.network.accuracy(prev_test_X, prev_test_y)
            backward_accs.append(acc)
        
        avg_backward_acc = np.mean(backward_accs)
        forgetting = current_acc - avg_backward_acc if task_idx > 0 else 0.0
        
        evolving_results.append({
            "task": task_idx,
            "current_accuracy": current_acc,
            "backward_accuracy": avg_backward_acc,
            "forgetting": forgetting,
            "final_modules": result['final_modules']
        })
        
        print(f"Task {task_idx}: Current={current_acc:.1%}, Backward={avg_backward_acc:.1%}, Forgetting={forgetting:+.1%}, Modules={result['final_modules']}")
    
    # 保存结果
    results = {
        "dataset": "Permuted-MNIST",
        "fixed_results": fixed_results,
        "evolving_results": evolving_results
    }
    
    output_path = PROJECT_ROOT / "results" / "phase4_permuted_mnist_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """主函数"""
    evaluate_permuted_mnist()


if __name__ == "__main__":
    main()
