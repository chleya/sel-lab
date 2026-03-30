# -*- coding: utf-8 -*-
"""
SEL-Lab Meta-Learning Exploration
元学习与自适应机制探索

探索方向:
1. 学习如何学习 - 元知识提取和迁移
2. 超参数自适应
3. 快速适应新任务（few-shot learning）
4. 任务无关的元表示
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from pathlib import Path
import sys
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import resolve_results_path, save_json


@dataclass
class MetaLearningConfig:
    """元学习配置"""
    input_size: int = 8
    hidden_size: int = 16
    output_size: int = 2
    
    # 元学习参数
    meta_lr: float = 0.01           # 元学习率
    inner_lr: float = 0.1           # 内循环学习率
    inner_steps: int = 5            # 内循环步数
    meta_batch_size: int = 4        # 任务批次大小
    
    # 自适应参数
    adapt_lr: bool = True           # 自适应学习率
    adapt_architecture: bool = True # 自适应架构
    
    # 任务分布
    n_tasks: int = 20               # 总任务数
    k_shot: int = 5                 # K-shot 学习
    query_size: int = 15            # 查询集大小


class TaskDistribution:
    """任务分布采样器 - 生成相关但不同的任务"""
    
    def __init__(self, config: MetaLearningConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # 生成任务原型（元知识的基础）
        self.task_prototypes = []
        for _ in range(config.n_tasks):
            # 每个任务有特定的权重模式
            prototype = {
                'W1': self.rng.normal(0, 0.5, (config.input_size, config.hidden_size)),
                'W2': self.rng.normal(0, 0.5, (config.hidden_size, config.output_size)),
                'noise_level': self.rng.uniform(0.1, 0.5),
                'task_type': self.rng.choice(['linear', 'xor', 'sin', 'classification'])
            }
            self.task_prototypes.append(prototype)
    
    def sample_task(self, task_id: Optional[int] = None) -> Dict:
        """采样一个任务"""
        if task_id is None:
            task_id = self.rng.integers(0, self.config.n_tasks)
        
        prototype = self.task_prototypes[task_id]
        cfg = self.config
        
        # 生成支持集（k-shot）
        X_support = self.rng.normal(0, 1, (cfg.k_shot, cfg.input_size))
        
        # 基于任务类型生成标签
        if prototype['task_type'] == 'linear':
            y_support = (X_support @ prototype['W1'] @ prototype['W2'] > 0).astype(float)
        elif prototype['task_type'] == 'xor':
            y_support = np.array([
                [1.0, 0.0] if (x[0] > 0) != (x[1] > 0) else [0.0, 1.0]
                for x in X_support
            ])
        else:
            # 默认分类
            logits = X_support @ prototype['W1'] @ prototype['W2']
            y_support = np.eye(cfg.output_size)[np.argmax(logits, axis=1)]
        
        # 添加噪声
        y_support += self.rng.normal(0, prototype['noise_level'], y_support.shape)
        
        # 生成查询集
        X_query = self.rng.normal(0, 1, (cfg.query_size, cfg.input_size))
        if prototype['task_type'] == 'linear':
            y_query = (X_query @ prototype['W1'] @ prototype['W2'] > 0).astype(float)
        elif prototype['task_type'] == 'xor':
            y_query = np.array([
                [1.0, 0.0] if (x[0] > 0) != (x[1] > 0) else [0.0, 1.0]
                for x in X_query
            ])
        else:
            logits = X_query @ prototype['W1'] @ prototype['W2']
            y_query = np.eye(cfg.output_size)[np.argmax(logits, axis=1)]
        
        return {
            'task_id': task_id,
            'task_type': prototype['task_type'],
            'X_support': X_support,
            'y_support': y_support,
            'X_query': X_query,
            'y_query': y_query
        }
    
    def sample_meta_batch(self, batch_size: Optional[int] = None) -> List[Dict]:
        """采样一个元批次"""
        size = batch_size or self.config.meta_batch_size
        return [self.sample_task() for _ in range(size)]


class MetaLearner:
    """元学习器 - MAML 风格的元学习"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        
        # 元参数（所有任务共享的初始化）
        self.theta = {
            'W1': np.random.randn(config.input_size, config.hidden_size) * 0.1,
            'W2': np.random.randn(config.hidden_size, config.output_size) * 0.1,
            'b1': np.zeros(config.hidden_size),
            'b2': np.zeros(config.output_size)
        }
        
        # 元优化器状态
        self.m = {k: np.zeros_like(v) for k, v in self.theta.items()}  # 动量
        self.v = {k: np.zeros_like(v) for k, v in self.theta.items()}  # 速度
        self.beta1, self.beta2 = 0.9, 0.999
        self.epsilon = 1e-8
        self.step_count = 0
        
        # 自适应学习率历史
        self.lr_history = []
        self.performance_history = []
    
    def forward(self, x: np.ndarray, params: Optional[Dict] = None) -> np.ndarray:
        """前向传播"""
        p = params or self.theta
        h = np.tanh(x @ p['W1'] + p['b1'])
        return h @ p['W2'] + p['b2']
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray, params: Optional[Dict] = None) -> float:
        """计算损失"""
        pred = self.forward(X, params)
        return np.mean((pred - y) ** 2)
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray, params: Optional[Dict] = None) -> Dict:
        """计算梯度"""
        p = params or self.theta
        
        # 前向
        z1 = X @ p['W1'] + p['b1']
        h = np.tanh(z1)
        z2 = h @ p['W2'] + p['b2']
        
        # 反向
        dz2 = 2 * (z2 - y) / len(X)
        dW2 = h.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        dh = dz2 @ p['W2'].T
        dz1 = dh * (1 - np.tanh(z1) ** 2)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        return {'W1': dW1, 'W2': dW2, 'b1': db1, 'b2': db2}
    
    def inner_loop(self, task: Dict, steps: Optional[int] = None) -> Tuple[Dict, List[float]]:
        """内循环适应 - 在单个任务上快速学习"""
        steps = steps or self.config.inner_steps
        
        # 克隆当前元参数
        task_params = {k: v.copy() for k, v in self.theta.items()}
        
        losses = []
        for step in range(steps):
            # 在支持集上计算梯度
            grads = self.compute_gradients(task['X_support'], task['y_support'], task_params)
            
            # 梯度下降更新
            for key in task_params:
                task_params[key] -= self.config.inner_lr * grads[key]
            
            # 记录损失
            loss = self.compute_loss(task['X_support'], task['y_support'], task_params)
            losses.append(loss)
        
        return task_params, losses
    
    def meta_update(self, meta_batch: List[Dict]):
        """元更新 - 基于多个任务的适应结果更新元参数"""
        self.step_count += 1
        
        # 收集所有任务的查询集损失梯度
        meta_gradients = {k: np.zeros_like(v) for k, v in self.theta.items()}
        query_losses = []
        
        for task in meta_batch:
            # 内循环适应
            adapted_params, _ = self.inner_loop(task)
            
            # 在查询集上评估
            query_loss = self.compute_loss(task['X_query'], task['y_query'], adapted_params)
            query_losses.append(query_loss)
            
            # 计算查询集梯度（二阶近似）
            query_grads = self.compute_gradients(task['X_query'], task['y_query'], adapted_params)
            
            # 累积元梯度
            for key in meta_gradients:
                meta_gradients[key] += query_grads[key] / len(meta_batch)
        
        # 自适应学习率
        current_lr = self.config.meta_lr
        if self.config.adapt_lr and len(self.performance_history) > 5:
            recent_perf = np.mean(self.performance_history[-5:])
            older_perf = np.mean(self.performance_history[-10:-5]) if len(self.performance_history) >= 10 else recent_perf
            
            if recent_perf > older_perf * 1.05:
                current_lr *= 1.1  # 增加学习率
            elif recent_perf < older_perf * 0.95:
                current_lr *= 0.9  # 减少学习率
        
        self.lr_history.append(current_lr)
        
        # Adam 优化器更新
        for key in self.theta:
            g = meta_gradients[key]
            
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[key] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[key] / (1 - self.beta2 ** self.step_count)
            
            self.theta[key] -= current_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        mean_query_loss = np.mean(query_losses)
        self.performance_history.append(mean_query_loss)
        
        return mean_query_loss
    
    def evaluate_few_shot(self, task: Dict, shots: Optional[List[int]] = None) -> Dict:
        """评估 few-shot 学习能力"""
        shots = shots or [1, 5, 10]
        results = {}
        
        for k in shots:
            if k <= len(task['X_support']):
                # 使用 k-shot
                few_shot_task = {
                    'X_support': task['X_support'][:k],
                    'y_support': task['y_support'][:k],
                    'X_query': task['X_query'],
                    'y_query': task['y_query']
                }
                
                # 快速适应
                adapted_params, _ = self.inner_loop(few_shot_task, steps=k)
                
                # 评估
                predictions = self.forward(task['X_query'], adapted_params)
                pred_labels = np.argmax(predictions, axis=1)
                true_labels = np.argmax(task['y_query'], axis=1)
                accuracy = np.mean(pred_labels == true_labels)
                
                results[f'{k}_shot'] = accuracy
        
        return results


def run_meta_learning_exploration(
    meta_iterations: int = 100,
    eval_interval: int = 10,
    verbose: bool = True
) -> Dict:
    """运行元学习探索实验"""
    
    print("=" * 60)
    print("Meta-Learning SEL Exploration")
    print("=" * 60)
    
    config = MetaLearningConfig(
        input_size=8,
        hidden_size=16,
        output_size=2,
        meta_lr=0.01,
        inner_lr=0.1,
        inner_steps=5,
        n_tasks=20,
        k_shot=5
    )
    
    # 创建任务分布
    task_dist = TaskDistribution(config, seed=42)
    
    # 创建元学习器
    meta_learner = MetaLearner(config)
    
    # 训练历史
    meta_losses = []
    few_shot_performances = []
    
    print(f"\nMeta-training for {meta_iterations} iterations...")
    
    for iteration in range(meta_iterations):
        # 采样元批次
        meta_batch = task_dist.sample_meta_batch()
        
        # 元更新
        meta_loss = meta_learner.meta_update(meta_batch)
        meta_losses.append(meta_loss)
        
        # 定期评估
        if iteration % eval_interval == 0:
            # 采样新任务评估 few-shot 能力
            test_task = task_dist.sample_task(task_id=0)  # 使用固定任务评估
            few_shot_results = meta_learner.evaluate_few_shot(test_task, shots=[1, 5, 10])
            few_shot_performances.append(few_shot_results)
            
            if verbose:
                print(f"  Iter {iteration}: meta_loss={meta_loss:.4f}, "
                      f"1-shot={few_shot_results.get('1_shot', 0):.2%}, "
                      f"5-shot={few_shot_results.get('5_shot', 0):.2%}")
    
    # 最终评估
    print("\n" + "=" * 60)
    print("Final Evaluation on New Tasks")
    print("=" * 60)
    
    final_results = []
    for task_id in range(5):  # 评估5个新任务
        test_task = task_dist.sample_task(task_id=task_id)
        few_shot_results = meta_learner.evaluate_few_shot(test_task, shots=[1, 5, 10])
        final_results.append(few_shot_results)
        
        print(f"Task {task_id}: "
              f"1-shot={few_shot_results.get('1_shot', 0):.2%}, "
              f"5-shot={few_shot_results.get('5_shot', 0):.2%}, "
              f"10-shot={few_shot_results.get('10_shot', 0):.2%}")
    
    # 汇总
    summary = {
        'mean_1_shot': np.mean([r.get('1_shot', 0) for r in final_results]),
        'mean_5_shot': np.mean([r.get('5_shot', 0) for r in final_results]),
        'mean_10_shot': np.mean([r.get('10_shot', 0) for r in final_results]),
        'final_meta_loss': meta_losses[-1],
        'meta_loss_progression': meta_losses
    }
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Mean 1-shot accuracy: {summary['mean_1_shot']:.2%}")
    print(f"  Mean 5-shot accuracy: {summary['mean_5_shot']:.2%}")
    print(f"  Mean 10-shot accuracy: {summary['mean_10_shot']:.2%}")
    print(f"  Final meta loss: {summary['final_meta_loss']:.4f}")
    print("=" * 60)
    
    results = {
        'config': config.__dict__,
        'meta_losses': meta_losses,
        'few_shot_performances': few_shot_performances,
        'final_evaluation': final_results,
        'summary': summary
    }
    
    # 保存结果
    save_path = resolve_results_path("meta_learning_exploration_results.json")
    save_json(results, save_path)
    print(f"\nResults saved to: {save_path}")
    
    return results


if __name__ == "__main__":
    run_meta_learning_exploration(meta_iterations=100, eval_interval=10)
