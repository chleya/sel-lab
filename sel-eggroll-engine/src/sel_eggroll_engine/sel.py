# -*- coding: utf-8 -*-
"""
SEL (结构演化学习) 模块

实现直接反馈对齐的前向学习机制，无需反向传播。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class SELConfig:
    """SEL 配置"""
    alpha: float = 0.1  # 学习率
    beta: float = 0.9   # 动量
    dropout: float = 0.0  # 丢弃率
    activation: str = "softmax"  # 激活函数


class SELModule:
    """SEL 结构演化模块"""
    
    def __init__(self, in_size: int, out_size: int, config: Optional[SELConfig] = None, seed: Optional[int] = None):
        """初始化 SEL 模块
        
        Args:
            in_size: 输入维度
            out_size: 输出维度
            config: SEL 配置
            seed: 随机种子
        """
        self.in_size = in_size
        self.out_size = out_size
        self.rng = np.random.default_rng(seed)
        self.config = config or SELConfig()
        
        # 权重矩阵
        self.W = self.rng.normal(0.0, np.sqrt(2.0 / in_size), (in_size, out_size))
        
        # 激活值和误差
        self.activations: List[np.ndarray] = []
        self.errors: List[np.ndarray] = []
        
        # 动量
        self.momentum = np.zeros_like(self.W)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播
        
        Args:
            X: 输入数据，形状 (batch_size, in_size)
            
        Returns:
            输出数据，形状 (batch_size, out_size)
        """
        # 线性变换
        logits = X @ self.W
        
        # 激活函数
        if self.config.activation == "softmax":
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        elif self.config.activation == "sigmoid":
            output = 1.0 / (1.0 + np.exp(-logits))
        elif self.config.activation == "relu":
            output = np.maximum(0, logits)
        else:
            output = logits
        
        # 保存激活值
        self.activations.append(output)
        
        return output
    
    def compute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算误差
        
        Args:
            y_true: 真实标签
            y_pred: 预测值
            
        Returns:
            误差矩阵
        """
        if self.config.activation == "softmax":
            # 交叉熵误差
            error = y_true - y_pred
        else:
            # 均方误差
            error = y_true - y_pred
        
        self.errors.append(error)
        return error
    
    def update(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """更新权重
        
        Args:
            X: 输入数据
            y_true: 真实标签
            
        Returns:
            损失值
        """
        # 前向传播
        y_pred = self.forward(X)
        
        # 计算误差
        error = self.compute_error(y_true, y_pred)
        
        # 计算损失
        if self.config.activation == "softmax":
            loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
        else:
            loss = np.mean(np.square(error))
        
        # 直接反馈对齐：输入梯度 * 输出误差
        gradient = X.T @ error
        
        # 动量更新
        self.momentum = self.config.beta * self.momentum + (1 - self.config.beta) * gradient
        
        # 更新权重
        self.W += self.config.alpha * self.momentum
        
        # 清空缓存
        self.activations = []
        self.errors = []
        
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测
        
        Args:
            X: 输入数据
            
        Returns:
            预测结果
        """
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
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
        """重置模块"""
        self.W = self.rng.normal(0.0, np.sqrt(2.0 / self.in_size), (self.in_size, self.out_size))
        self.momentum = np.zeros_like(self.W)
        self.activations = []
        self.errors = []
