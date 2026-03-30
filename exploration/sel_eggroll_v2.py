# -*- coding: utf-8 -*-
"""
SEL + EGGROLL v2 - 改进版本

改进点：
1. 使用更强大的局部学习（类似 Hebbian + 误差修正）
2. 自适应学习率
3. 更好的集成策略：动态权重
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LowRankConfig:
    """低秩优化配置"""
    rank: int = 2
    sigma: float = 0.1
    population_size: int = 15
    eggroll_lr: float = 0.02
    eggroll_frequency: int = 5


def generate_low_rank_perturbation(W, rank, sigma, rng):
    """生成低秩扰动"""
    a, b = W.shape
    perturbation = rng.normal(0.0, 1.0, size=(a + b, rank))
    B = perturbation[:b]
    A = perturbation[b:]
    return A, B


def evaluate_perturbation(module, X, y, A, B, sigma):
    """评估扰动"""
    W_perturbed = module.W_main + sigma * (A @ B.T)
    y_pred = np.tanh(X @ W_perturbed)
    loss = float(np.mean((y_pred - y) ** 2))
    return 1.0 / (loss + 1e-8)


def fuse_high_rank_update(module, perturbations, fitness_scores, sigma, lr):
    """融合高秩更新"""
    if not perturbations or not fitness_scores:
        return
    
    avg_update = np.zeros_like(module.W_main)
    total_fitness = sum(fitness_scores)
    
    if total_fitness > 0:
        for (A, B), fitness in zip(perturbations, fitness_scores):
            perturbation = sigma * (A @ B.T)
            weight = fitness / total_fitness
            avg_update += perturbation * weight
        
        module.W_main += lr * avg_update
        module.W_main = np.clip(module.W_main, -2.0, 2.0)


class ImprovedSELModule:
    """
    改进的 SEL 模块
    
    使用 Hebbian 学习 + 误差修正的组合
    """
    
    def __init__(self, name, in_size, out_size, low_rank_config=None, rng=None, seed=None):
        self.rng = rng or np.random.default_rng(seed)
        self.name = name
        self.in_size = in_size
        self.out_size = out_size
        self.low_rank_config = low_rank_config or LowRankConfig()
        
        # 主权重
        self.W_main = self.rng.normal(0.0, np.sqrt(2.0 / in_size), 
                                       size=(in_size, out_size))
        
        # 动量
        self.velocity = np.zeros_like(self.W_main)
        
        # 自适应学习率状态
        self.lr_scale = 1.0
        self.prev_grad = None
        
        # 历史记录
        self.loss_history = []
        self.age = 0
    
    def forward(self, x):
        """前向传播"""
        x = np.atleast_2d(x)
        return np.tanh(x @ self.W_main)
    
    def local_learning(self, x, y, lr=0.1, momentum=0.9):
        """
        改进的局部学习
        
        结合：
        1. Hebbian 学习（相关性）
        2. 误差驱动修正
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        
        # 前向传播
        output = self.forward(x)
        
        # 误差
        error = y - output
        
        # Hebbian 项（输入-输出相关性）
        hebbian_grad = x.T @ output
        
        # 误差修正项
        error_grad = x.T @ error
        
        # 组合梯度（自适应权重）
        hebbian_weight = 0.3
        error_weight = 0.7
        
        grad = hebbian_weight * hebbian_grad + error_weight * error_grad
        
        # 自适应学习率调整
        if self.prev_grad is not None:
            # 如果梯度方向一致，增加学习率
            alignment = np.sum(grad * self.prev_grad) / (np.linalg.norm(grad) * np.linalg.norm(self.prev_grad) + 1e-8)
            if alignment > 0.5:
                self.lr_scale = min(self.lr_scale * 1.05, 2.0)
            else:
                self.lr_scale = max(self.lr_scale * 0.95, 0.5)
        
        self.prev_grad = grad.copy()
        
        # 应用动量更新
        effective_lr = lr * self.lr_scale
        self.velocity = momentum * self.velocity + effective_lr * grad
        self.W_main += self.velocity
        self.W_main = np.clip(self.W_main, -2.0, 2.0)
        
        # 记录
        loss = float(np.mean(error ** 2))
        self.loss_history.append(loss)
        if len(self.loss_history) > 20:
            self.loss_history.pop(0)
        
        self.age += 1
        return loss
    
    def eggroll_update(self, X, y):
        """EGGROLL 更新"""
        config = self.low_rank_config
        
        # 生成扰动
        perturbations = []
        for _ in range(config.population_size):
            A, B = generate_low_rank_perturbation(
                self.W_main, config.rank, config.sigma, self.rng
            )
            perturbations.append((A, B))
        
        # 评估
        fitness_scores = []
        for A, B in perturbations:
            fitness = evaluate_perturbation(self, X, y, A, B, config.sigma)
            fitness_scores.append(fitness)
        
        # 融合
        fuse_high_rank_update(
            self, perturbations, fitness_scores,
            config.sigma, config.eggroll_lr
        )
        
        return fitness_scores


class ImprovedSELWithEGGROLL:
    """改进的集成网络"""
    
    def __init__(self, input_size, output_size, initial_modules=3, max_modules=5,
                 low_rank_config=None, random_seed=42):
        self.rng = np.random.default_rng(random_seed)
        self.input_size = input_size
        self.output_size = output_size
        self.max_modules = max_modules
        self.low_rank_config = low_rank_config or LowRankConfig()
        
        self.modules = []
        for i in range(initial_modules):
            self.add_module(f"m{i}")
        
        self.current_epoch = 0
        self.epoch_loss_history = []
        
        # 动态权重
        self.sel_weight = 0.5
        self.eggroll_weight = 0.5
    
    def add_module(self, name):
        """添加模块"""
        module = ImprovedSELModule(
            name=name,
            in_size=self.input_size,
            out_size=self.output_size,
            low_rank_config=self.low_rank_config,
            rng=self.rng,
        )
        self.modules.append(module)
        return module
    
    def forward(self, x):
        """前向传播"""
        x = np.atleast_2d(x)
        outputs = [module.forward(x) for module in self.modules]
        return np.mean(outputs, axis=0)
    
    def train_step(self, X, y, lr=0.1, use_eggroll=True):
        """训练步骤"""
        output = self.forward(X)
        y = np.atleast_2d(y)
        error = y - output
        
        # 计算当前损失
        current_loss = float(np.mean(error ** 2))
        
        # 动态调整权重
        if len(self.epoch_loss_history) > 5:
            recent_improvement = self.epoch_loss_history[-5] - current_loss
            if recent_improvement > 0:
                # 有改进，增加 SEL 权重
                self.sel_weight = min(self.sel_weight + 0.05, 0.8)
            else:
                # 无改进，增加 EGGROLL 权重
                self.eggroll_weight = min(self.eggroll_weight + 0.05, 0.8)
            # 归一化
            total = self.sel_weight + self.eggroll_weight
            self.sel_weight /= total
            self.eggroll_weight /= total
        
        # SEL 学习
        total_loss = 0.0
        for module in self.modules:
            loss = module.local_learning(X, y, lr=lr * self.sel_weight)
            total_loss += loss
        
        avg_loss = total_loss / len(self.modules)
        self.epoch_loss_history.append(avg_loss)
        self.current_epoch += 1
        
        # EGGROLL 更新
        eggroll_fitnesses = []
        if use_eggroll and (self.current_epoch % self.low_rank_config.eggroll_frequency == 0):
            for module in self.modules:
                fitnesses = module.eggroll_update(X, y)
                eggroll_fitnesses.append(fitnesses)
        
        return avg_loss, eggroll_fitnesses


def test_improved_version():
    """测试改进版本"""
    print("\n" + "="*70)
    print("改进版本测试 (Improved SEL + EGGROLL)")
    print("="*70)
    
    # 创建螺旋数据（最具挑战性的任务）
    rng = np.random.default_rng(42)
    n_samples = 500
    n_classes = 3
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros((n_samples * n_classes, n_classes))
    
    for class_id in range(n_classes):
        ix = range(n_samples * class_id, n_samples * (class_id + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(class_id * 4, (class_id + 1) * 4, n_samples) + rng.normal(0, 0.2, n_samples)
        X[ix] = np.c_[r * np.sin(t * 2.0), r * np.cos(t * 2.0)]
        y[ix, class_id] = 1
    
    config = LowRankConfig(rank=2, sigma=0.15, population_size=20, 
                          eggroll_lr=0.02, eggroll_frequency=5)
    
    network = ImprovedSELWithEGGROLL(
        input_size=2, output_size=3,
        initial_modules=4, max_modules=6,
        low_rank_config=config
    )
    
    print(f"\n训练螺旋数据分类（改进版本）")
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print("-"*70)
    
    for epoch in range(200):
        loss, fitnesses = network.train_step(X, y, lr=0.1, use_eggroll=True)
        
        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {loss:.6f}, "
                  f"SEL_w={network.sel_weight:.2f}, EGG_w={network.eggroll_weight:.2f}", end="")
            if fitnesses:
                avg_fit = np.mean([np.mean(f) for f in fitnesses])
                print(f" | Fitness = {avg_fit:.2f}")
            else:
                print()
    
    print("\n" + "="*70)
    print("改进版本测试完成！")
    print("="*70)


if __name__ == "__main__":
    test_improved_version()
