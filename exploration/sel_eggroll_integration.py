# -*- coding: utf-8 -*-
"""
SEL + EGGROLL 集成实现

这个文件实现了将 SEL 的结构进化与 EGGROLL 的低秩参数优化结合的完整系统。

双层进化框架：
- 上层：SEL 结构进化（张力驱动，保持 3-5 模块）
- 下层：EGGROLL 低秩参数优化（高效进化策略）
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LowRankConfig:
    """低秩优化配置（EGGROLL 风格）"""
    rank: int = 1  # 低秩秩数
    sigma: float = 0.1  # 扰动标准差
    population_size: int = 10  # 种群大小
    eggroll_lr: float = 0.01  # EGGROLL 学习率
    eggroll_frequency: int = 5  # 每多少轮执行一次 EGGROLL 更新


def generate_low_rank_perturbation(
    W: np.ndarray,
    rank: int,
    sigma: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成低秩扰动（EGGROLL 风格）
    
    数学原理：
    扰动 = sigma * B @ A.T
    其中 A 和 B 是低秩矩阵
    
    Args:
        W: 主权重矩阵
        rank: 低秩秩数
        sigma: 扰动标准差
        rng: 随机数生成器
    
    Returns:
        (A, B): 低秩矩阵对
    """
    a, b = W.shape
    perturbation = rng.normal(0.0, 1.0, size=(a + b, rank))
    B = perturbation[:b]  # b × r
    A = perturbation[b:]  # a × r
    return A, B


def evaluate_perturbation(
    module,
    X: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    sigma: float
) -> float:
    """
    评估单个扰动的适应度
    
    Args:
        module: LowRankSELModule 实例
        X: 输入数据
        y: 目标数据
        A: 低秩矩阵 A
        B: 低秩矩阵 B
        sigma: 扰动标准差
    
    Returns:
        适应度分数（越高越好）
    """
    # 应用低秩扰动: W_perturbed = W + sigma * A @ B.T
    W_perturbed = module.W_main + sigma * (A @ B.T)
    
    # 前向传播
    y_pred = np.tanh(X @ W_perturbed)
    
    # 计算损失
    loss = float(np.mean((y_pred - y) ** 2))
    
    # 适应度 = 1 / (损失 + epsilon) - 损失越小，适应度越高
    fitness = 1.0 / (loss + 1e-8)
    return fitness


def fuse_high_rank_update(
    module,
    perturbations: List[Tuple[np.ndarray, np.ndarray]],
    fitness_scores: List[float],
    sigma: float,
    lr: float
):
    """
    融合高秩更新（EGGROLL 的核心创新）
    
    关键洞察：
    虽然单个扰动是低秩的，但多个低秩矩阵的和可以是高秩的！
    
    Args:
        module: LowRankSELModule 实例
        perturbations: 低秩扰动列表 [(A1, B1), (A2, B2), ...]
        fitness_scores: 对应每个扰动的适应度分数
        sigma: 扰动标准差
        lr: 学习率
    """
    if not perturbations or not fitness_scores:
        return
    
    avg_update = np.zeros_like(module.W_main)
    total_fitness = sum(fitness_scores)
    
    if total_fitness > 0:
        for (A, B), fitness in zip(perturbations, fitness_scores):
            # 低秩扰动: delta_W = sigma * A @ B.T
            perturbation = sigma * (A @ B.T)
            # 加权平均，按适应度加权
            weight = fitness / total_fitness
            avg_update += perturbation * weight
        
        # 更新主权重
        module.W_main += lr * avg_update
        module.W_main = np.clip(module.W_main, -2.0, 2.0)


class LowRankSELModule:
    """
    结合低秩学习的 SEL 模块
    
    这个模块保持了 SEL 的模块化设计，同时集成了 EGGROLL 风格的低秩扰动优化。
    """
    
    def __init__(
        self,
        name: str,
        in_size: int,
        out_size: int,
        low_rank_config: Optional[LowRankConfig] = None,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        self.rng = rng or np.random.default_rng(seed)
        self.name = name
        self.in_size = in_size
        self.out_size = out_size
        self.low_rank_config = low_rank_config or LowRankConfig()
        
        # 主权重矩阵（像传统 SEL 一样）
        self.W_main = self.rng.normal(0.0, np.sqrt(2.0 / in_size), 
                                       size=(in_size, out_size))
        
        # 反馈矩阵（DFA）
        self.feedback_matrix = self.rng.normal(0.0, 0.1, 
                                                size=(out_size, out_size))
        
        # 速度（动量）
        self.velocity = np.zeros_like(self.W_main)
        
        # 张力相关
        self.local_tension = 0.0
        self.age = 0
        self.loss_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self.improvement_history: List[float] = []
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播（保持 SEL 风格）"""
        x = np.atleast_2d(x)
        return np.tanh(x @ self.W_main)
    
    def project_feedback(self, output_error: np.ndarray) -> np.ndarray:
        """DFA 反馈投影"""
        output_error = np.atleast_2d(output_error)
        return (self.feedback_matrix.T @ output_error.T).T
    
    def forward_learning(
        self,
        x: np.ndarray,
        output_error: np.ndarray,
        lr: float = 0.1,
        momentum: float = 0.9,
        tension_window: int = 8,
        min_improvement: float = 1e-3,
    ) -> float:
        """
        正向学习（SEL 风格 + DFA）
        
        保持 SEL 的原始学习方式，同时为 EGGROLL 留出接口。
        """
        x = np.atleast_2d(x)
        output_error = np.atleast_2d(output_error)
        
        # DFA 学习（SEL 原始方式）
        fb_error = self.project_feedback(output_error)
        grad = x.T @ fb_error
        self.velocity = momentum * self.velocity + lr * grad
        self.W_main += self.velocity
        self.W_main = np.clip(self.W_main, -2.0, 2.0)
        
        # 记录信息用于张力计算
        loss = float(np.mean(output_error ** 2))
        grad_norm = float(np.linalg.norm(grad))
        
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        
        if len(self.loss_history) > 1:
            self.improvement_history.append(self.loss_history[-2] - self.loss_history[-1])
        
        # 维护历史窗口
        if len(self.loss_history) > tension_window:
            self.loss_history.pop(0)
        if len(self.grad_norm_history) > tension_window:
            self.grad_norm_history.pop(0)
        if len(self.improvement_history) > tension_window - 1:
            self.improvement_history.pop(0)
        
        self.age += 1
        return loss
    
    def eggroll_update(
        self,
        X: np.ndarray,
        y: np.ndarray
    ):
        """
        EGGROLL 风格的低秩进化（完整实现）
        
        这个方法执行完整的 EGGROLL 流程：
        1. 生成低秩扰动种群
        2. 评估每个扰动的适应度
        3. 根据适应度融合高秩更新
        """
        config = self.low_rank_config
        
        # 1. 生成低秩扰动种群
        perturbations: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(config.population_size):
            A, B = generate_low_rank_perturbation(
                self.W_main,
                config.rank,
                config.sigma,
                self.rng
            )
            perturbations.append((A, B))
        
        # 2. 评估每个扰动的适应度
        fitness_scores: List[float] = []
        for A, B in perturbations:
            fitness = evaluate_perturbation(self, X, y, A, B, config.sigma)
            fitness_scores.append(fitness)
        
        # 3. 融合高秩更新
        fuse_high_rank_update(
            self,
            perturbations,
            fitness_scores,
            config.sigma,
            config.eggroll_lr
        )
        
        return fitness_scores
    
    @property
    def param_count(self) -> int:
        return self.W_main.size + self.feedback_matrix.size


class SELWithEGGROLL:
    """
    SEL + EGGROLL 集成网络
    
    上层：SEL 结构进化（张力驱动，保持 3-5 模块）
    下层：EGGROLL 低秩参数优化（高效进化策略）
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        initial_modules: int = 3,
        max_modules: int = 5,
        low_rank_config: Optional[LowRankConfig] = None,
        random_seed: Optional[int] = 42,
    ):
        self.rng = np.random.default_rng(random_seed)
        self.input_size = input_size
        self.output_size = output_size
        self.max_modules = max_modules
        self.low_rank_config = low_rank_config or LowRankConfig()
        
        self.modules: List[LowRankSELModule] = []
        for i in range(initial_modules):
            self.add_module(f"m{i}")
        
        self.current_epoch = 0
        self.epoch_loss_history: List[float] = []
    
    def add_module(self, name: str):
        """添加模块（SEL 风格）"""
        module = LowRankSELModule(
            name=name,
            in_size=self.input_size,
            out_size=self.output_size,
            low_rank_config=self.low_rank_config,
            rng=self.rng,
        )
        self.modules.append(module)
        return module
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播（模块平均，SEL 风格）"""
        x = np.atleast_2d(x)
        outputs = [module.forward(x) for module in self.modules]
        return np.mean(outputs, axis=0)
    
    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        use_eggroll: bool = True
    ) -> Tuple[float, List[List[float]]]:
        """
        单次训练步骤
        
        结合了 SEL 的 DFA 学习和 EGGROLL 的低秩优化。
        
        Args:
            X: 批量输入数据
            y: 批量目标数据
            lr: 学习率
            use_eggroll: 是否使用 EGGROLL 更新
        
        Returns:
            (平均损失, 每个模块的 EGGROLL 适应度列表)
        """
        output = self.forward(X)
        y = np.atleast_2d(y)
        error = y - output
        
        # SEL 的 DFA 学习
        total_loss = 0.0
        for module in self.modules:
            loss = module.forward_learning(X, error, lr=lr)
            total_loss += loss
        
        avg_loss = total_loss / len(self.modules)
        self.epoch_loss_history.append(avg_loss)
        self.current_epoch += 1
        
        # EGGROLL 的低秩进化（按频率执行）
        eggroll_fitnesses: List[List[float]] = []
        if use_eggroll and (self.current_epoch % self.low_rank_config.eggroll_frequency == 0):
            for module in self.modules:
                fitnesses = module.eggroll_update(X, y)
                eggroll_fitnesses.append(fitnesses)
        
        return avg_loss, eggroll_fitnesses
    
    def structural_evolution(self):
        """
        结构进化（简化版 SEL 风格）
        
        保持在 3-5 模块的最优范围内。
        """
        changes = []
        
        if len(self.modules) < 3:
            # 添加模块
            self.add_module(f"m{len(self.modules)}")
            changes.append(("added", "structure optimization"))
        elif len(self.modules) > 5:
            # 移除模块
            removed = self.modules.pop()
            changes.append((removed.name, "pruned"))
        
        return changes


def demo_sel_eggroll_full():
    """演示完整的 SEL + EGGROLL 集成系统"""
    print("\n" + "=" * 70)
    print("SEL + EGGROLL 完整集成演示")
    print("=" * 70)
    
    # 配置
    input_size = 4
    output_size = 2
    low_rank_config = LowRankConfig(
        rank=1,
        sigma=0.1,
        population_size=10,
        eggroll_lr=0.01,
        eggroll_frequency=5
    )
    
    # 创建集成网络
    network = SELWithEGGROLL(
        input_size=input_size,
        output_size=output_size,
        initial_modules=3,
        max_modules=5,
        low_rank_config=low_rank_config,
    )
    
    print(f"\n✓ 初始化网络:")
    print(f"  输入维度: {input_size}")
    print(f"  输出维度: {output_size}")
    print(f"  初始模块数: 3")
    print(f"  最大模块数: 5")
    print(f"  低秩秩数: {low_rank_config.rank}")
    print(f"  种群大小: {low_rank_config.population_size}")
    print(f"  EGGROLL 频率: 每 {low_rank_config.eggroll_frequency} 轮")
    
    # 生成简单的测试数据
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 2.0, size=(100, 4))
    y = np.zeros((100, 2))
    for i in range(100):
        if X[i, 0] + X[i, 1] > 0:
            y[i, 0] = 1
        else:
            y[i, 1] = 1
    
    # 演示训练
    print(f"\n✓ 开始训练（完整集成）:")
    print("-" * 70)
    
    for epoch in range(20):
        avg_loss, eggroll_fitnesses = network.train_step(X, y, lr=0.1, use_eggroll=True)
        
        print(f"Epoch {epoch+1:2d}: 损失 = {avg_loss:.4f}, 模块数 = {len(network.modules)}", end="")
        
        # 显示 EGGROLL 更新信息
        if eggroll_fitnesses:
            avg_fitness = np.mean([np.mean(f) for f in eggroll_fitnesses])
            print(f" | EGGROLL: 平均适应度 = {avg_fitness:.2f}")
        else:
            print()
        
        # 结构进化
        changes = network.structural_evolution()
        if changes:
            print(f"  结构变化: {changes}")
    
    print("\n" + "=" * 70)
    print("完整集成演示完成！")
    print("=" * 70)
    print("\n核心功能:")
    print("1. 上层: SEL 结构进化 - 保持在 3-5 模块最优范围")
    print("2. 下层: EGGROLL 低秩优化 - 完整的低秩扰动进化")
    print("\n实现的 EGGROLL 功能:")
    print("✓ 低秩扰动生成 (A, B 矩阵)")
    print("✓ 种群适应度评估")
    print("✓ 高秩更新融合")
    print("✓ 按频率执行 EGGROLL 更新")


if __name__ == "__main__":
    demo_sel_eggroll_full()
