# -*- coding: utf-8 -*-
"""
EGGROLL (低秩参数优化) 模块

实现基于低秩扰动的参数优化算法。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class EGGROLLConfig:
    """EGGROLL 配置"""
    rank: int = 3  # 低秩扰动的秩
    sigma: float = 0.1  # 扰动强度
    population_size: int = 8  # 扰动种群大小
    learning_rate: float = 0.01  # 学习率
    max_iterations: int = 1000  # 最大迭代次数
    tolerance: float = 1e-6  # 收敛 tolerance


class EGGROLLOptimizer:
    """EGGROLL 优化器"""
    
    def __init__(self, in_size: int, out_size: int, config: Optional[EGGROLLConfig] = None, seed: Optional[int] = None):
        """初始化 EGGROLL 优化器
        
        Args:
            in_size: 输入维度
            out_size: 输出维度
            config: EGGROLL 配置
            seed: 随机种子
        """
        self.in_size = in_size
        self.out_size = out_size
        self.rng = np.random.default_rng(seed)
        self.config = config or EGGROLLConfig()
        
        # 权重矩阵
        self.W = self.rng.normal(0.0, np.sqrt(2.0 / in_size), (in_size, out_size))
        
        # 优化历史
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
    
    def generate_perturbation(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成低秩扰动
        
        Returns:
            (A, B) 扰动矩阵对
        """
        a, b = self.W.shape
        noise = self.rng.normal(0.0, 1.0, size=(a + b, self.config.rank))
        B = noise[:b]
        A = noise[b:]
        return A, B
    
    def evaluate_perturbation(self, X: np.ndarray, y: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
        """评估扰动
        
        Args:
            X: 输入数据
            y: 真实标签
            A: A 矩阵
            B: B 矩阵
            
        Returns:
            适应度分数
        """
        # 应用扰动
        W_perturbed = self.W + self.config.sigma * (A @ B.T)
        
        # 前向传播
        logits = X @ W_perturbed
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 计算损失
        loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
        
        # 适应度分数（损失的倒数）
        fitness = 1.0 / (loss + 1e-8)
        return fitness
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算损失
        
        Args:
            X: 输入数据
            y: 真实标签
            
        Returns:
            损失值
        """
        logits = X @ self.W
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
        return loss
    
    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率
        
        Args:
            X: 输入数据
            y: 真实标签
            
        Returns:
            准确率
        """
        logits = X @ self.W
        y_pred = np.argmax(logits, axis=1)
        y_true = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred == y_true)
        return accuracy
    
    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        """执行一步优化
        
        Args:
            X: 输入数据
            y: 真实标签
            
        Returns:
            损失值
        """
        # 生成扰动种群
        perturbations = []
        for _ in range(self.config.population_size):
            A, B = self.generate_perturbation()
            perturbations.append((A, B))
        
        # 评估适应度
        fitness_scores = []
        for A, B in perturbations:
            fitness = self.evaluate_perturbation(X, y, A, B)
            fitness_scores.append(fitness)
        
        # 融合更新
        avg_update = np.zeros_like(self.W)
        total_fitness = sum(fitness_scores)
        
        if total_fitness > 0:
            for (A, B), fitness in zip(perturbations, fitness_scores):
                perturbation = self.config.sigma * (A @ B.T)
                weight = fitness / total_fitness
                avg_update += perturbation * weight
            
            # 更新权重
            self.W += self.config.learning_rate * avg_update
            # 权重裁剪
            self.W = np.clip(self.W, -3.0, 3.0)
        
        # 计算当前损失和准确率
        loss = self.compute_loss(X, y)
        accuracy = self.compute_accuracy(X, y)
        
        # 记录历史
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        
        return loss
    
    def optimize(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> List[float]:
        """执行优化
        
        Args:
            X: 输入数据
            y: 真实标签
            epochs: 迭代次数
            
        Returns:
            损失历史
        """
        for epoch in range(epochs):
            loss = self.step(X, y)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                accuracy = self.accuracy_history[-1]
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # 收敛检查
            if len(self.loss_history) > 1:
                loss_diff = abs(self.loss_history[-1] - self.loss_history[-2])
                if loss_diff < self.config.tolerance:
                    print(f"Converged at epoch {epoch+1}")
                    break
        
        return self.loss_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测
        
        Args:
            X: 输入数据
            
        Returns:
            预测结果
        """
        logits = X @ self.W
        predictions = np.argmax(logits, axis=1)
        return predictions
    
    def get_weights(self) -> np.ndarray:
        """获取权重
        
        Returns:
            权重矩阵
        """
        return self.W.copy()
    
    def set_weights(self, weights: np.ndarray):
        """设置权重
        
        Args:
            weights: 新的权重矩阵
        """
        self.W = weights.copy()
    
    def reset(self):
        """重置优化器"""
        self.W = self.rng.normal(0.0, np.sqrt(2.0 / self.in_size), (self.in_size, self.out_size))
        self.loss_history = []
        self.accuracy_history = []
