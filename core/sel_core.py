# -*- coding: utf-8 -*-
"""
SEL Core Framework
Structural Evolution Learning - Core Implementation

核心特性：
- 前向学习（无反向传播）
- 模块化架构
- 张力驱动的结构演化
- 局部更新规则
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


# ============== 配置 ==============

@dataclass
class SELConfig:
    """
    SEL 配置
    
    Attributes:
        input_size: 输入维度
        output_size: 输出维度
        initial_modules: 初始模块数
        learning_rate: 学习率
        momentum: 动量系数
        tension_threshold: 张力阈值
        max_modules: 最大模块数
        mutation_rate: 变异率
        epochs: 训练轮数
    """
    input_size: int
    output_size: int
    initial_modules: int = 3
    learning_rate: float = 0.1
    momentum: float = 0.9
    tension_threshold: float = 0.5
    max_modules: int = 10
    mutation_rate: float = 0.2
    epochs: int = 100


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    train_accuracy: float
    test_accuracy: float
    avg_tension: float
    module_count: int
    structural_changes: int = 0


# ============== 核心模块 ==============

class SELModule:
    """
    SEL 可演化模块
    
    特性：
    - 前向学习（无反向传播）
    - 张力追踪
    - 局部更新
    - 权重约束
    """
    
    def __init__(self, name: str, in_size: int, out_size: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.name = name
        self.in_size = in_size
        self.out_size = out_size
        self.weights = np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)
        
        # 动量
        self.momentum = np.zeros((in_size, out_size))
        
        # 张力追踪
        self.local_tension = 0.0
        self.error_history: List[float] = []
        
        # DFA 反馈矩阵
        self.feedback_matrix = np.random.randn(out_size, out_size) * 0.1
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.tanh(x @ self.weights)
    
    def forward_learning(self, x: np.ndarray, output_error: np.ndarray,
                        lr: float = 0.1, momentum: float = 0.9) -> float:
        """
        前向学习
        
        无反向传播，只用输出误差
        
        Args:
            x: 输入
            output_error: 输出误差 (target - output)
            lr: 学习率
            momentum: 动量系数
            
        Returns:
            误差幅度
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # DFA 风格的误差变换
        fb_error = output_error @ self.feedback_matrix
        
        # 动量更新
        grad = np.dot(x.T, fb_error)
        self.momentum = momentum * self.momentum - lr * grad
        self.weights += self.momentum
        
        # 约束
        self.weights = np.clip(self.weights, -2, 2)
        
        # 更新张力
        error_mag = np.mean(np.abs(fb_error))
        self.error_history.append(error_mag)
        if len(self.error_history) > 10:
            self.error_history.pop(0)
        self.local_tension = 0.9 * self.local_tension + 0.1 * error_mag
        
        return error_mag
    
    def structural_adaptation(self, threshold: float = 0.7) -> Tuple[bool, str]:
        """
        基于张力的结构调整
        
        高张力 → 扰动权重（增加多样性）
        低张力 → 稳定
        
        Returns:
            (是否变化, 变化原因)
        """
        if self.local_tension > threshold:
            noise = np.random.randn(*self.weights.shape) * 0.01
            self.weights += noise
            self.local_tension *= 0.3
            return True, "perturbed"
        return False, "stable"
    
    def mutate_weights(self, rate: float = 0.2, magnitude: float = 0.1):
        """权重变异"""
        mask = np.random.random(self.weights.shape) < rate
        self.weights[mask] += np.random.randn(*self.weights.shape)[mask] * magnitude
    
    @property
    def weight_norm(self) -> float:
        return np.mean(np.abs(self.weights))
    
    @property
    def param_count(self) -> int:
        return self.weights.size
    
    def __repr__(self):
        return f"SELModule({self.name}, tension={self.local_tension:.3f}, params={self.param_count})"


class SELNetwork:
    """
    SEL 模块化网络
    
    特性：
    - 多模块并行
    - 前向学习
    - 动态结构演化
    """
    
    def __init__(self, config: SELConfig):
        self.config = config
        self.modules: List[SELModule] = []
        
        # 初始化模块
        for i in range(config.initial_modules):
            self.add_module(f"m{i}", seed=42 + i)
    
    def add_module(self, name: str, seed: int = None) -> SELModule:
        """添加模块"""
        m = SELModule(name, self.config.input_size, self.config.output_size, seed=seed)
        self.modules.append(m)
        return m
    
    def remove_module(self, index: int) -> Optional[SELModule]:
        """移除模块（保留至少一个）"""
        if len(self.modules) > 1 and 0 <= index < len(self.modules):
            return self.modules.pop(index)
        return None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播（所有模块并行）"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        outputs = [m.forward(x) for m in self.modules]
        return np.mean(outputs, axis=0)
    
    def forward_learning(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        前向学习所有模块
        
        无反向传播
        """
        output = self.forward(x)
        error = target - output
        total_error = 0
        
        for m in self.modules:
            err = m.forward_learning(x, error, 
                                   lr=self.config.learning_rate,
                                   momentum=self.config.momentum)
            total_error += err
        
        return total_error / len(self.modules)
    
    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X))
                     if self.predict(X[i]) == int(np.argmax(y[i])))
        return correct / len(X)
    
    def structural_evolution(self) -> List[Tuple]:
        """
        结构演化
        
        基于张力的增删模块
        """
        changes = []
        
        # 个体模块调整
        for i, m in enumerate(self.modules):
            changed, reason = m.structural_adaptation(self.config.tension_threshold)
            if changed:
                changes.append((m.name, reason))
        
        # 网络结构调整
        if len(self.modules) < self.config.max_modules:
            if np.random.random() < 0.05:
                self.add_module(f"new{len(self.modules)}")
                changes.append(("new", "added"))
        
        if len(self.modules) > 1 and np.random.random() < 0.05:
            idx = np.random.randint(0, len(self.modules))
            removed = self.remove_module(idx)
            if removed:
                changes.append((removed.name, "removed"))
        
        return changes
    
    @property
    def param_count(self) -> int:
        return sum(m.param_count for m in self.modules)
    
    def serialize(self) -> Dict:
        """序列化"""
        return {
            'modules': [m.weights.tolist() for m in self.modules],
            'tensions': [m.local_tension for m in self.modules]
        }
    
    def deserialize(self, data: Dict):
        """反序列化"""
        for i, m in enumerate(self.modules):
            if i < len(data['modules']):
                m.weights = np.array(data['modules'][i])
                m.local_tension = data['tensions'][i]
    
    def __repr__(self):
        return f"SELNetwork(modules={len(self.modules)}, params={self.param_count})"


# ============== 训练器 ==============

class SELTrainer:
    """
    SEL 训练器
    
    负责：
    - 训练循环
    - 指标记录
    - 实验管理
    """
    
    def __init__(self, config: SELConfig):
        self.config = config
        self.network: SELNetwork = None
        self.metrics: List[TrainingMetrics] = []
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray = None, y_test: np.ndarray = None) -> Dict:
        """
        训练网络
        
        Returns:
            训练结果统计
        """
        self.network = SELNetwork(self.config)
        self.metrics = []
        
        for epoch in range(self.config.epochs):
            # 打乱数据
            indices = np.random.permutation(len(X_train))
            
            epoch_errors = []
            
            for i in indices:
                err = self.network.forward_learning(X_train[i], y_train[i])
                epoch_errors.append(err)
            
            # 结构演化
            changes = self.network.structural_evolution()
            
            # 计算指标
            train_acc = self.network.accuracy(X_train, y_train)
            test_acc = 0.0
            if X_test is not None:
                test_acc = self.network.accuracy(X_test, y_test)
            
            avg_tension = np.mean([m.local_tension for m in self.network.modules])
            
            metric = TrainingMetrics(
                epoch=epoch,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                avg_tension=avg_tension,
                module_count=len(self.network.modules),
                structural_changes=len(changes)
            )
            self.metrics.append(metric)
        
        return {
            'final_train_accuracy': self.metrics[-1].train_accuracy,
            'final_test_accuracy': self.metrics[-1].test_accuracy,
            'final_modules': self.metrics[-1].module_count,
            'total_structural_changes': sum(m.structural_changes for m in self.metrics),
            'network': self.network
        }
    
    def get_results(self) -> Dict:
        """获取结果摘要"""
        if not self.metrics:
            return {}
        
        best_epoch = max(range(len(self.metrics)), 
                        key=lambda i: self.metrics[i].test_accuracy)
        best = self.metrics[best_epoch]
        
        return {
            'best_test_accuracy': best.test_accuracy,
            'best_epoch': best.epoch,
            'final_train_accuracy': self.metrics[-1].train_accuracy,
            'final_test_accuracy': self.metrics[-1].test_accuracy,
            'final_modules': self.metrics[-1].module_count,
            'total_changes': sum(m.structural_changes for m in self.metrics),
            'avg_tension': np.mean([m.avg_tension for m in self.metrics])
        }


# ============== 工具函数 ==============

def create_task(task_type: str = "simple_classification") -> Tuple[np.ndarray, np.ndarray]:
    """
    创建测试任务
    
    Args:
        task_type: 任务类型
        
    Returns:
        X, y
    """
    np.random.seed(42)
    
    if task_type == "simple_classification":
        # 简单线性分类
        X = np.random.randn(200, 4) * 2
        y = np.zeros((200, 2))
        for i in range(200):
            if X[i, 0] + X[i, 1] > 0:
                y[i, 0] = 1
            else:
                y[i, 1] = 1
        return X, y
    
    elif task_type == "xor":
        # XOR 问题
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
        return X, y
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def run_experiment(config: SELConfig = None,
                  task_type: str = "simple_classification") -> Dict:
    """
    运行实验
    
    快捷函数
    """
    if config is None:
        config = SELConfig(input_size=4, output_size=2)
    
    X, y = create_task(task_type)
    
    # 划分数据
    split = len(X) // 2
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # 训练
    trainer = SELTrainer(config)
    result = trainer.train(X_train, y_train, X_test, y_test)
    
    return result


# ============== 主函数 ==============

def main():
    """主函数 - 演示"""
    print("\n" + "=" * 60)
    print("SEL Framework Demo")
    print("=" * 60)
    
    # 配置
    config = SELConfig(
        input_size=4,
        output_size=2,
        initial_modules=3,
        learning_rate=0.1,
        epochs=100
    )
    
    # 运行实验
    result = run_experiment(config)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {result['final_train_accuracy']:.1%}")
    print(f"  Test Accuracy: {result['final_test_accuracy']:.1%}")
    print(f"  Final Modules: {result['final_modules']}")
    print(f"  Structural Changes: {result['total_structural_changes']}")
    
    print("\n" + "=" * 60)
    print("Status: SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()
