# -*- coding: utf-8 -*-
"""
SEL-Lab: Edge Computing Optimized
边缘计算优化版本
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import json
import os


@dataclass
class EdgeConfig:
    """边缘配置"""
    input_size: int = 4
    hidden_size: int = 8
    output_size: int = 2
    learning_rate: float = 0.05
    max_modules: int = 4
    prune_threshold: float = 0.01


class MicroModule:
    """极小模块"""
    def __init__(self, in_size: int, out_size: int):
        self.weights = np.random.randn(in_size, out_size) * 0.1
        self.local_tension = 0.0
        self.active = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.weights)
    
    def forward_learning(self, x: np.ndarray, error: np.ndarray, lr: float) -> float:
        self.weights += lr * np.outer(x, error)
        self.weights = np.clip(self.weights, -1, 1)
        self.local_tension = 0.9 * self.local_tension + 0.1 * np.mean(np.abs(error))
        return np.mean(np.abs(error))
    
    def prune(self, threshold: float = 0.01):
        """剪枝"""
        self.weights[np.abs(self.weights) < threshold] = 0
    
    @property
    def param_count(self) -> int:
        return np.sum(self.weights != 0)


class EdgeNetwork:
    """边缘网络 - 极简"""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.layers = []
        
        # 输入层 -> 隐藏层
        self.layers.append(MicroModule(config.input_size, config.hidden_size))
        # 隐藏层 -> 输出层
        self.layers.append(MicroModule(config.hidden_size, config.output_size))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x
        for layer in self.layers:
            if layer.active:
                a = layer.forward(a)
        return a
    
    def forward_learning(self, x: np.ndarray, target: np.ndarray, lr: float):
        """前向学习"""
        # 前向
        h = self.layers[0].forward(x)
        out = self.layers[1].forward(h)
        
        # 误差
        error_out = target - out
        
        # 更新输出层
        self.layers[1].forward_learning(h, error_out, lr)
        
        # 反向误差（简化版）
        error_hidden = error_out @ self.layers[1].weights.T
        
        # 更新隐藏层
        self.layers[0].forward_learning(x, error_hidden, lr)
    
    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X))
                     if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)
    
    @property
    def total_params(self) -> int:
        return sum(l.param_count for l in self.layers)
    
    def evolve(self) -> Dict:
        """自我演化"""
        changes = {}
        
        # 剪枝
        for i, layer in enumerate(self.layers):
            old_nonzero = np.sum(layer.weights != 0)
            layer.prune(self.config.prune_threshold)
            new_nonzero = np.sum(layer.weights != 0)
            
            if old_nonzero != new_nonzero:
                changes[f'layer_{i}'] = f'pruned: {old_nonzero} -> {new_nonzero}'
        
        # 权重扰动（高张力）
        for i, layer in enumerate(self.layers):
            if layer.local_tension > 0.5:
                noise = np.random.randn(*layer.weights.shape) * 0.01
                layer.weights += noise
                changes[f'layer_{i}'] = 'perturbed'
        
        return changes
    
    def serialize(self) -> Dict:
        weights_list = []
        for l in self.layers:
            # 转换为Python原生类型
            w = l.weights.tolist()
            weights_list.append(w)
        return {
            'weights': weights_list,
            'tensions': [float(l.local_tension) for l in self.layers]
        }
    
    def deserialize(self, data: Dict):
        for i, layer in enumerate(self.layers):
            layer.weights = np.array(data['weights'][i])
            layer.local_tension = data['tensions'][i]


class EdgeTrainer:
    """边缘训练器"""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.network = None
        self.history = []
    
    def train(self, X_train, y_train, X_test=None, y_test=None, epochs=100):
        self.network = EdgeNetwork(self.config)
        
        for epoch in range(epochs):
            idx = np.random.permutation(len(X_train))
            
            for i in idx:
                self.network.forward_learning(X_train[i], y_train[i], self.config.learning_rate)
            
            # 演化
            changes = self.network.evolve()
            
            # 评估
            train_acc = self.network.accuracy(X_train, y_train)
            test_acc = self.network.accuracy(X_test, y_test) if X_test is not None else 0
            
            self.history.append({
                'epoch': epoch,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'params': self.network.total_params
            })
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: Train={train_acc:.1%}, Test={test_acc:.1%}, Params={self.network.total_params}")
        
        return {
            'final_train': train_acc,
            'final_test': test_acc,
            'total_params': self.network.total_params,
            'history': self.history
        }
    
    def save(self, filepath: str):
        # 确保history中的numpy类型转换为Python原生类型
        history_clean = []
        for h in self.history:
            history_clean.append({
                'epoch': int(h['epoch']),
                'train_acc': float(h['train_acc']),
                'test_acc': float(h['test_acc']),
                'params': int(h['params'])
            })
        
        data = self.network.serialize()
        data['history'] = history_clean
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Model saved: {filepath}")
    
    def load(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.network.deserialize(data)
        self.history = data.get('history', [])


def demo():
    """演示"""
    print("\n" + "=" * 60)
    print("SEL-Lab Edge Computing Demo")
    print("=" * 60)
    
    config = EdgeConfig(input_size=4, hidden_size=8, output_size=2)
    
    np.random.seed(42)
    X = np.random.randn(200, 4) * 2
    y = np.zeros((200, 2))
    for i in range(200):
        if X[i, 0] + X[i, 1] > 0:
            y[i, 0] = 1
        else:
            y[i, 1] = 1
    
    X_train, y_train = X[:100], y[:100]
    X_test, y_test = X[100:], y[100:]
    
    trainer = EdgeTrainer(config)
    result = trainer.train(X_train, y_train, X_test, y_test, epochs=100)
    
    # 创建目录并保存
    os.makedirs("results", exist_ok=True)
    trainer.save("results/edge_model.json")
    
    print(f"\n" + "=" * 60)
    print("Final Results:")
    print(f"  Train Accuracy: {result['final_train']:.1%}")
    print(f"  Test Accuracy: {result['final_test']:.1%}")
    print(f"  Total Params: {result['total_params']}")
    print(f"  Model Size: ~{result['total_params'] * 4} bytes")
    print("=" * 60)
    
    # 推理速度
    import time
    start = time.time()
    for _ in range(1000):
        trainer.network.forward(X_test[0])
    elapsed = (time.time() - start) * 1000
    print(f"\nInference Speed: {elapsed:.1f}ms for 1000 samples")
    print(f"Per sample: {elapsed:.3f}ms")
    
    return result


if __name__ == "__main__":
    demo()
