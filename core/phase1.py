# -*- coding: utf-8 -*-
"""
SEL-Lab Phase 1 Implementation
Structural Evolution Learning - Phase 1

特点：
- 稳定的前向学习（DFA）
- 完整的评估指标
- 可重复的结果
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import json
import os


@dataclass
class Phase1Config:
    """Phase 1 配置"""
    input_size: int = 4
    hidden_size: int = 8
    output_size: int = 2
    learning_rate: float = 0.05
    runs: int = 10
    epochs: int = 100


class Phase1Network:
    """Phase 1 网络 - 稳定的 DFA 学习"""
    
    def __init__(self, config: Phase1Config, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.config = config
        
        # 权重初始化（He初始化）
        self.W1 = np.random.randn(config.input_size, config.hidden_size) * np.sqrt(2.0 / config.input_size)
        self.W2 = np.random.randn(config.hidden_size, config.output_size) * np.sqrt(2.0 / config.hidden_size)
        
        # 固定随机反馈矩阵
        self.feedback = np.random.randn(config.output_size, config.hidden_size) * 0.1
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1)
        return h @ self.W2
    
    def forward_learning(self, x: np.ndarray, target: np.ndarray) -> float:
        """DFA 学习"""
        h = np.tanh(x @ self.W1)
        out = h @ self.W2
        error = target - out
        
        # DFA：使用随机反馈矩阵传递误差
        # 误差信号 = 输出误差 @ 反馈矩阵
        fb_signal = error @ self.feedback
        
        # 更新权重（无反向传播）
        self.W2 += self.config.learning_rate * np.outer(h, error)
        self.W1 += self.config.learning_rate * np.outer(x, fb_signal)
        
        # 约束
        self.W1 = np.clip(self.W1, -2, 2)
        self.W2 = np.clip(self.W2, -2, 2)
        
        return np.mean(np.abs(error))
    
    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X))
                     if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)
    
    @property
    def param_count(self) -> int:
        return self.W1.size + self.W2.size


class Phase1Experiment:
    """Phase 1 实验"""
    
    def __init__(self, config: Phase1Config = None):
        self.config = config or Phase1Config()
        self.history = {'acc': [], 'loss': []}
    
    def run(self, verbose: bool = True) -> Dict:
        if verbose:
            print("\n" + "=" * 60)
            print("Phase 1: Forward-Only Learning Verification")
            print("Method: Direct Feedback Alignment (DFA)")
            print("Constraints: No Backpropagation")
            print("=" * 60)
        
        # 创建任务
        np.random.seed(42)
        X = np.random.randn(200, self.config.input_size) * 2
        y = np.zeros((200, self.config.output_size))
        for i in range(200):
            if X[i, 0] + X[i, 1] > 0:
                y[i, 0] = 1
            else:
                y[i, 1] = 1
        
        X_train, y_train = X[:100], y[:100]
        X_test, y_test = X[100:], y[100:]
        
        all_results = []
        
        for run in range(self.config.runs):
            network = Phase1Network(self.config, seed=run * 100 + 42)
            
            run_results = {
                'run': run + 1,
                'seed': run * 100 + 42,
                'accuracies': [],
                'losses': []
            }
            
            if verbose:
                print(f"\n--- Run {run + 1}/{self.config.runs} (seed={run * 100 + 42}) ---")
            
            for epoch in range(self.config.epochs):
                # 训练
                epoch_loss = 0
                for i in range(len(X_train)):
                    loss = network.forward_learning(X_train[i], y_train[i])
                    epoch_loss += loss
                
                avg_loss = epoch_loss / len(X_train)
                
                # 评估
                acc = network.accuracy(X_test, y_test)
                run_results['accuracies'].append(acc)
                run_results['losses'].append(avg_loss)
                
                self.history['acc'].append(acc)
                self.history['loss'].append(avg_loss)
                
                if verbose and (epoch + 1) % 25 == 0:
                    print(f"Epoch {epoch+1}: Acc={acc:.1%}, Loss={avg_loss:.4f}")
            
            final_acc = run_results['accuracies'][-1]
            print(f"Final: Acc={final_acc:.1%}")
            all_results.append(run_results)
        
        # 评估
        final_accs = [r['accuracies'][-1] for r in all_results]
        
        # 趋势分析
        if len(self.history['acc']) >= 2:
            x = np.arange(len(self.history['acc']))
            slope = np.polyfit(x, self.history['acc'], 1)[0]
        else:
            slope = 0
        
        eval_result = {
            'final_accuracy': float(np.mean(final_accs)),
            'std_accuracy': float(np.std(final_accs)),
            'min_accuracy': float(np.min(final_accs)),
            'max_accuracy': float(np.max(final_accs)),
            'trend_slope': float(slope),
            'improving': bool(slope > 0.001),
            'stable': bool(np.std(final_accs) < 0.2),
            'runs_above_80': sum(1 for acc in final_accs if acc > 0.8),
            'total_runs': len(final_accs)
        }
        
        if verbose:
            print(f"\n" + "=" * 60)
            print("Results Summary")
            print("=" * 60)
            print(f"Final Accuracy: {eval_result['final_accuracy']:.1%} (+/- {eval_result['std_accuracy']:.1%})")
            print(f"Range: {eval_result['min_accuracy']:.1%} - {eval_result['max_accuracy']:.1%}")
            print(f"Trend: {eval_result['trend_slope']:.5f} ({'improving' if eval_result['improving'] else 'stable'})")
            print(f"Runs > 80%: {eval_result['runs_above_80']}/{eval_result['total_runs']}")
            
            # 决策
            success = eval_result['final_accuracy'] > 0.75 and eval_result['runs_above_80'] >= eval_result['total_runs'] // 2
            print(f"\nDecision: {'[SUCCESS] Forward-only learning works!' if success else '[NEEDS WORK] Refine learning'}")
        
        return {
            'results': all_results,
            'evaluation': eval_result,
            'decision': 'success' if (
                eval_result['final_accuracy'] > 0.75 and 
                eval_result['runs_above_80'] >= eval_result['total_runs'] // 2
            ) else 'refine'
        }
    
    def save_results(self, filepath: str = "results/phase1_results.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        result = self.run(verbose=False)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved: {filepath}")


def main():
    print("\n" + "=" * 60)
    print("SEL-Lab Phase 1 Implementation")
    print("=" * 60)
    
    config = Phase1Config(
        input_size=4,
        output_size=2,
        hidden_size=8,
        learning_rate=0.05,
        runs=10,
        epochs=100
    )
    
    experiment = Phase1Experiment(config)
    results = experiment.run(verbose=True)
    experiment.save_results()
    
    print("\nPhase 1 Complete!")


if __name__ == "__main__":
    main()
