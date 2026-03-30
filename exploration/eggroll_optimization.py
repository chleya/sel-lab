# -*- coding: utf-8 -*-
"""
EGGROLL 参数优化与全面测试

重点：
1. 参数敏感性分析（秩数、种群大小、sigma、学习率）
2. 在标准基准数据集上测试
3. 与简单梯度下降对比
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time


@dataclass
class EGGROLLConfig:
    """EGGROLL 配置"""
    rank: int = 2              # 低秩秩数
    sigma: float = 0.1         # 扰动标准差
    population_size: int = 15  # 种群大小
    learning_rate: float = 0.02  # 学习率
    frequency: int = 5         # 更新频率


class EGGROLLModule:
    """纯 EGGROLL 模块（无 SEL 组件）"""
    
    def __init__(self, in_size: int, out_size: int, config: EGGROLLConfig, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.in_size = in_size
        self.out_size = out_size
        self.config = config
        
        # 主权重
        self.W = self.rng.normal(0.0, np.sqrt(2.0 / in_size), (in_size, out_size))
        
        # 统计
        self.update_count = 0
        self.fitness_history = []
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        return np.tanh(X @ self.W)
    
    def generate_perturbation(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成低秩扰动"""
        a, b = self.W.shape
        noise = self.rng.normal(0.0, 1.0, size=(a + b, self.config.rank))
        B = noise[:b]   # b × r
        A = noise[b:]   # a × r
        return A, B
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
        """评估扰动"""
        W_perturbed = self.W + self.config.sigma * (A @ B.T)
        y_pred = np.tanh(X @ W_perturbed)
        loss = np.mean((y_pred - y) ** 2)
        return 1.0 / (loss + 1e-8)
    
    def update(self, X: np.ndarray, y: np.ndarray) -> float:
        """执行 EGGROLL 更新"""
        cfg = self.config
        
        # 1. 生成扰动种群
        perturbations = []
        for _ in range(cfg.population_size):
            A, B = self.generate_perturbation()
            perturbations.append((A, B))
        
        # 2. 评估适应度
        fitness_scores = []
        for A, B in perturbations:
            fitness = self.evaluate(X, y, A, B)
            fitness_scores.append(fitness)
        
        # 3. 融合更新
        avg_update = np.zeros_like(self.W)
        total_fitness = sum(fitness_scores)
        
        if total_fitness > 0:
            for (A, B), fitness in zip(perturbations, fitness_scores):
                perturbation = cfg.sigma * (A @ B.T)
                weight = fitness / total_fitness
                avg_update += perturbation * weight
            
            self.W += cfg.learning_rate * avg_update
            self.W = np.clip(self.W, -3.0, 3.0)
        
        self.update_count += 1
        avg_fitness = np.mean(fitness_scores)
        self.fitness_history.append(avg_fitness)
        
        return avg_fitness
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算损失"""
        y_pred = self.forward(X)
        return float(np.mean((y_pred - y) ** 2))


class EGGROLLNetwork:
    """EGGROLL 网络（多模块集成）"""
    
    def __init__(self, input_size: int, output_size: int, n_modules: int = 3, config: EGGROLLConfig = None, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.config = config or EGGROLLConfig()
        self.modules = [
            EGGROLLModule(input_size, output_size, self.config, seed=seed+i)
            for i in range(n_modules)
        ]
        self.epoch = 0
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播（平均集成）"""
        outputs = [m.forward(X) for m in self.modules]
        return np.mean(outputs, axis=0)
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """训练步骤"""
        # 计算当前损失
        y_pred = self.forward(X)
        loss = float(np.mean((y_pred - y) ** 2))
        
        # 按频率执行 EGGROLL 更新
        avg_fitness = 0.0
        if self.epoch % self.config.frequency == 0:
            fitnesses = []
            for module in self.modules:
                fit = module.update(X, y)
                fitnesses.append(fit)
            avg_fitness = np.mean(fitnesses)
        
        self.epoch += 1
        return loss, avg_fitness


class SimpleSGD:
    """简单 SGD 对比基线"""
    
    def __init__(self, input_size: int, output_size: int, lr: float = 0.1, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0.0, np.sqrt(2.0 / input_size), (input_size, output_size))
        self.lr = lr
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.W)
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        # 前向
        y_pred = self.forward(X)
        
        # 反向传播（简化版）
        error = y - y_pred
        grad = X.T @ (error * (1 - y_pred**2))  # tanh 导数
        
        # 更新
        self.W += self.lr * grad
        
        return float(np.mean(error ** 2))


def create_datasets():
    """创建测试数据集"""
    datasets = {}
    rng = np.random.default_rng(42)
    
    # 1. XOR
    X_xor = rng.uniform(-1, 1, (400, 2))
    y_xor = np.zeros((400, 2))
    for i in range(400):
        label = int((X_xor[i, 0] > 0) != (X_xor[i, 1] > 0))
        y_xor[i, label] = 1
    X_xor += rng.normal(0, 0.1, X_xor.shape)
    datasets['XOR'] = (X_xor, y_xor)
    
    # 2. 螺旋
    n_samples = 500
    n_classes = 3
    X_spiral = np.zeros((n_samples * n_classes, 2))
    y_spiral = np.zeros((n_samples * n_classes, n_classes))
    for class_id in range(n_classes):
        ix = range(n_samples * class_id, n_samples * (class_id + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(class_id * 4, (class_id + 1) * 4, n_samples) + rng.normal(0, 0.2, n_samples)
        X_spiral[ix] = np.c_[r * np.sin(t * 2.0), r * np.cos(t * 2.0)]
        y_spiral[ix, class_id] = 1
    datasets['Spiral'] = (X_spiral, y_spiral)
    
    # 3. 回归
    X_reg = rng.normal(0, 1, (300, 10))
    y_reg = np.zeros((300, 1))
    y_reg[:, 0] = np.sin(X_reg[:, 0]) + np.cos(X_reg[:, 1]) + 0.5 * X_reg[:, 2]**2 + 0.1 * rng.normal(0, 1, 300)
    datasets['Regression'] = (X_reg, y_reg)
    
    return datasets


def grid_search_params(dataset_name: str, X: np.ndarray, y: np.ndarray, epochs: int = 100):
    """网格搜索最佳参数"""
    print(f"\n{'='*60}")
    print(f"参数搜索: {dataset_name}")
    print(f"{'='*60}")
    
    # 参数范围
    ranks = [1, 2, 4]
    sigmas = [0.05, 0.1, 0.2]
    populations = [10, 20, 30]
    lrs = [0.01, 0.02, 0.05]
    
    results = []
    
    for rank in ranks:
        for sigma in sigmas:
            for pop in populations:
                for lr in lrs:
                    config = EGGROLLConfig(
                        rank=rank, sigma=sigma,
                        population_size=pop, learning_rate=lr,
                        frequency=5
                    )
                    
                    network = EGGROLLNetwork(
                        X.shape[1], y.shape[1],
                        n_modules=3, config=config, seed=42
                    )
                    
                    # 训练
                    losses = []
                    for _ in range(epochs):
                        loss, _ = network.train_step(X, y)
                        losses.append(loss)
                    
                    final_loss = losses[-1]
                    results.append({
                        'rank': rank, 'sigma': sigma,
                        'pop': pop, 'lr': lr,
                        'final_loss': final_loss,
                        'config': config
                    })
                    
                    print(f"rank={rank}, sigma={sigma:.2f}, pop={pop:2d}, lr={lr:.2f} -> loss={final_loss:.6f}")
    
    # 排序找最佳
    results.sort(key=lambda x: x['final_loss'])
    best = results[0]
    
    print(f"\n✓ 最佳参数:")
    print(f"  rank={best['rank']}, sigma={best['sigma']}, pop={best['pop']}, lr={best['lr']}")
    print(f"  最终损失: {best['final_loss']:.6f}")
    
    return best['config'], results[:5]  # 返回最佳和前5


def compare_methods(datasets: Dict, epochs: int = 150):
    """对比 EGGROLL 和 SGD"""
    print("\n" + "="*70)
    print("方法对比: EGGROLL vs SGD")
    print("="*70)
    
    results = {}
    
    for name, (X, y) in datasets.items():
        print(f"\n{name}:")
        print("-"*60)
        
        # EGGROLL（使用默认配置）
        config = EGGROLLConfig(rank=2, sigma=0.1, population_size=15, learning_rate=0.02)
        egg_net = EGGROLLNetwork(X.shape[1], y.shape[1], n_modules=3, config=config)
        egg_losses = []
        start = time.time()
        for _ in range(epochs):
            loss, _ = egg_net.train_step(X, y)
            egg_losses.append(loss)
        egg_time = time.time() - start
        
        # SGD
        sgd = SimpleSGD(X.shape[1], y.shape[1], lr=0.1)
        sgd_losses = []
        start = time.time()
        for _ in range(epochs):
            loss = sgd.train_step(X, y)
            sgd_losses.append(loss)
        sgd_time = time.time() - start
        
        results[name] = {
            'eggroll': {'losses': egg_losses, 'time': egg_time, 'final': egg_losses[-1]},
            'sgd': {'losses': sgd_losses, 'time': sgd_time, 'final': sgd_losses[-1]}
        }
        
        print(f"  EGGROLL: 最终损失={egg_losses[-1]:.6f}, 时间={egg_time:.3f}s")
        print(f"  SGD:     最终损失={sgd_losses[-1]:.6f}, 时间={sgd_time:.3f}s")
        print(f"  改进:    {(sgd_losses[-1] - egg_losses[-1])/sgd_losses[-1]*100:+.1f}%")
    
    return results


def plot_comparison(results: Dict, save_path: str = "f:/sel-lab/exploration/eggroll_comparison.png"):
    """绘制对比图"""
    n_datasets = len(results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))
    
    if n_datasets == 1:
        axes = [axes]
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        ax.plot(data['eggroll']['losses'], label='EGGROLL', linewidth=2)
        ax.plot(data['sgd']['losses'], label='SGD', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name}\nEGGROLL: {data["eggroll"]["final"]:.4f} vs SGD: {data["sgd"]["final"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 对比图已保存: {save_path}")
    plt.show()


def main():
    """主函数"""
    print("\n" + "="*70)
    print("EGGROLL 参数优化与全面测试")
    print("="*70)
    
    # 创建数据集
    datasets = create_datasets()
    
    # 1. 参数搜索（只在 XOR 上做，节省时间）
    X_xor, y_xor = datasets['XOR']
    best_config, top5 = grid_search_params('XOR', X_xor, y_xor, epochs=100)
    
    # 2. 方法对比
    comparison = compare_methods(datasets, epochs=150)
    
    # 3. 可视化
    plot_comparison(comparison)
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()
