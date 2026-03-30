# -*- coding: utf-8 -*-
"""
高效 EGGROLL 实现

优化点：
1. 自适应种群大小
2. 并行评估
3. 早期停止
4. 多层网络支持
5. 分类任务优化
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time
import concurrent.futures


@dataclass
class EGGROLLConfig:
    """EGGROLL 配置"""
    rank: int = 2              # 低秩秩数
    sigma: float = 0.15        # 扰动标准差
    base_population: int = 10  # 基础种群大小
    max_population: int = 30   # 最大种群大小
    learning_rate: float = 0.03 # 学习率
    frequency: int = 5         # 更新频率
    patience: int = 10         # 早期停止耐心
    max_workers: int = 4       # 并行工作线程数


class EfficientEGGROLLModule:
    """高效 EGGROLL 模块"""
    
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
        self.loss_history = []
        
        # 早期停止
        self.best_loss = float('inf')
        self.stall_count = 0
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        logits = X @ self.W
        # 使用 softmax 作为激活函数（适合分类）
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
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
        
        # 分类任务：使用交叉熵或准确率
        if y.shape[1] > 1:  # 多分类
            loss = -np.mean(y * np.log(y_pred + 1e-8))
        else:  # 二分类
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1-y) * np.log(1-y_pred + 1e-8))
        
        return 1.0 / (loss + 1e-8)
    
    def _evaluate_single(self, args) -> float:
        """单个扰动评估（用于并行）"""
        X, y, A, B = args
        return self.evaluate(X, y, A, B)
    
    def update(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
        """执行 EGGROLL 更新
        
        Returns:
            (avg_fitness, early_stop) - 平均适应度和是否需要早期停止
        """
        cfg = self.config
        
        # 自适应种群大小
        current_loss = self.compute_loss(X, y)
        if current_loss > self.best_loss:
            # 损失增加，增加种群大小
            current_pop = min(cfg.base_population * 2, cfg.max_population)
        else:
            # 损失减少，保持较小种群
            current_pop = cfg.base_population
        
        # 1. 生成扰动种群
        perturbations = []
        for _ in range(current_pop):
            A, B = self.generate_perturbation()
            perturbations.append((A, B))
        
        # 2. 并行评估适应度
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            args = [(X, y, A, B) for A, B in perturbations]
            fitness_scores = list(executor.map(self._evaluate_single, args))
        
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
        
        # 4. 早期停止检查
        new_loss = self.compute_loss(X, y)
        self.loss_history.append(new_loss)
        
        if new_loss < self.best_loss:
            self.best_loss = new_loss
            self.stall_count = 0
        else:
            self.stall_count += 1
        
        early_stop = self.stall_count >= cfg.patience
        
        self.update_count += 1
        avg_fitness = np.mean(fitness_scores)
        self.fitness_history.append(avg_fitness)
        
        return avg_fitness, early_stop
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算损失"""
        y_pred = self.forward(X)
        
        if y.shape[1] > 1:
            # 多分类交叉熵
            loss = -np.mean(y * np.log(y_pred + 1e-8))
        else:
            # 二分类交叉熵
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1-y) * np.log(1-y_pred + 1e-8))
        
        return float(loss)


class EGGROLLLayer:
    """EGGROLL 层（用于多层网络）"""
    
    def __init__(self, in_size: int, out_size: int, config: EGGROLLConfig, seed: int = None):
        self.module = EfficientEGGROLLModule(in_size, out_size, config, seed)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.module.forward(X)
    
    def update(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
        return self.module.update(X, y)
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.module.compute_loss(X, y)


class MultiLayerEGGROLL:
    """多层 EGGROLL 网络"""
    
    def __init__(self, layer_sizes: List[int], config: EGGROLLConfig, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.config = config
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            layer = EGGROLLLayer(
                layer_sizes[i], layer_sizes[i+1], 
                config, seed=seed+i
            )
            self.layers.append(layer)
        
        self.epoch = 0
        self.early_stop = False
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        x = X
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算损失"""
        y_pred = self.forward(X)
        
        if y.shape[1] > 1:
            loss = -np.mean(y * np.log(y_pred + 1e-8))
        else:
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1-y) * np.log(1-y_pred + 1e-8))
        
        return float(loss)
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, bool]:
        """训练步骤（整体更新）"""
        if self.early_stop:
            return self.compute_loss(X, y), 0.0, True
        
        # 计算最终损失
        loss = self.compute_loss(X, y)
        
        # 按频率执行更新
        avg_fitness = 0.0
        if self.epoch % self.config.frequency == 0:
            # 对于多层网络，我们使用整体评估的方式
            # 保存原始权重
            original_weights = [layer.module.W.copy() for layer in self.layers]
            
            # 生成多个完整网络的扰动
            population_size = self.config.base_population
            perturbations = []
            fitness_scores = []
            
            for _ in range(population_size):
                # 为每个层生成扰动
                layer_perturbations = []
                for layer in self.layers:
                    A, B = layer.module.generate_perturbation()
                    layer_perturbations.append((A, B))
                
                # 应用扰动
                for layer, (A, B) in zip(self.layers, layer_perturbations):
                    layer.module.W += self.config.sigma * (A @ B.T)
                
                # 评估整个网络
                current_loss = self.compute_loss(X, y)
                fitness = 1.0 / (current_loss + 1e-8)
                fitness_scores.append(fitness)
                perturbations.append(layer_perturbations)
                
                # 恢复原始权重
                for layer, original_W in zip(self.layers, original_weights):
                    layer.module.W = original_W.copy()
            
            # 融合更新
            total_fitness = sum(fitness_scores)
            if total_fitness > 0:
                for layer_idx, layer in enumerate(self.layers):
                    avg_update = np.zeros_like(layer.module.W)
                    for i, (layer_perturbations, fitness) in enumerate(zip(perturbations, fitness_scores)):
                        A, B = layer_perturbations[layer_idx]
                        perturbation = self.config.sigma * (A @ B.T)
                        weight = fitness / total_fitness
                        avg_update += perturbation * weight
                    
                    layer.module.W += self.config.learning_rate * avg_update
                    layer.module.W = np.clip(layer.module.W, -3.0, 3.0)
            
            avg_fitness = np.mean(fitness_scores)
            
            # 早期停止检查
            new_loss = self.compute_loss(X, y)
            if new_loss < self.layers[0].module.best_loss:
                for layer in self.layers:
                    layer.module.best_loss = new_loss
                    layer.module.stall_count = 0
            else:
                for layer in self.layers:
                    layer.module.stall_count += 1
            
            self.early_stop = any(layer.module.stall_count >= self.config.patience for layer in self.layers)
        
        self.epoch += 1
        return loss, avg_fitness, self.early_stop


def create_classification_datasets():
    """创建分类测试数据集"""
    datasets = {}
    rng = np.random.default_rng(42)
    
    # 1. XOR
    X_xor = rng.uniform(-1, 1, (1000, 2))
    y_xor = np.zeros((1000, 2))
    for i in range(1000):
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
    
    # 3. 环形数据
    n_samples = 400
    X_ring = rng.normal(0, 1, (n_samples, 2))
    y_ring = np.zeros((n_samples, 2))
    for i in range(n_samples):
        r = np.sqrt(X_ring[i, 0]**2 + X_ring[i, 1]**2)
        label = int(r > 0.7)
        y_ring[i, label] = 1
    datasets['Ring'] = (X_ring, y_ring)
    
    return datasets

def test_efficient_eggroll():
    """测试高效 EGGROLL"""
    print("\n" + "="*70)
    print("高效 EGGROLL 测试")
    print("="*70)
    
    datasets = create_classification_datasets()
    config = EGGROLLConfig(
        rank=2, sigma=0.15, base_population=10,
        learning_rate=0.03, frequency=5
    )
    
    results = {}
    
    for name, (X, y) in datasets.items():
        print(f"\n{name}:")
        print("-"*60)
        
        # 单层 EGGROLL
        start = time.time()
        layer = EGGROLLLayer(X.shape[1], y.shape[1], config)
        losses = []
        fitnesses = []
        
        for epoch in range(200):
            loss = layer.compute_loss(X, y)
            losses.append(loss)
            
            if epoch % config.frequency == 0:
                fit, _ = layer.update(X, y)
                fitnesses.append(fit)
            
            if (epoch + 1) % 40 == 0:
                print(f"  Epoch {epoch+1:3d}: Loss = {loss:.6f}", end="")
                if epoch % config.frequency == 0:
                    print(f" | Fitness = {fit:.2f}")
                else:
                    print()
        
        time_taken = time.time() - start
        results[name] = {
            'losses': losses,
            'fitnesses': fitnesses,
            'time': time_taken,
            'final_loss': losses[-1]
        }
        
        print(f"  完成: 最终损失={losses[-1]:.6f}, 时间={time_taken:.3f}s")
    
    # 测试多层网络
    print(f"\n多层 EGGROLL 测试:")
    print("-"*60)
    
    # XOR 任务，2层网络
    X, y = datasets['XOR']
    layer_sizes = [2, 4, 2]  # 2输入 -> 4隐藏 -> 2输出
    multi_net = MultiLayerEGGROLL(layer_sizes, config)
    
    start = time.time()
    multi_losses = []
    for epoch in range(200):
        loss, fit, stop = multi_net.train_step(X, y)
        multi_losses.append(loss)
        
        if (epoch + 1) % 40 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss = {loss:.6f}", end="")
            if epoch % config.frequency == 0:
                print(f" | Fitness = {fit:.2f}")
            else:
                print()
        
        if stop:
            print(f"  早期停止 at epoch {epoch+1}")
            break
    
    multi_time = time.time() - start
    print(f"  多层网络: 最终损失={multi_losses[-1]:.6f}, 时间={multi_time:.3f}s")
    
    # 绘制结果
    plot_results(results, multi_losses)
    
    print("\n" + "="*70)
    print("高效 EGGROLL 测试完成！")
    print("="*70)


def plot_results(results: Dict, multi_losses: List):
    """绘制结果"""
    n_datasets = len(results)
    fig, axes = plt.subplots(2, n_datasets, figsize=(6*n_datasets, 10))
    
    for idx, (name, data) in enumerate(results.items()):
        # 损失曲线
        ax = axes[0, idx]
        ax.plot(data['losses'], label='EGGROLL', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name}\nFinal Loss: {data["final_loss"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 适应度曲线
        ax = axes[1, idx]
        if data['fitnesses']:
            epochs = [i * 5 for i in range(len(data['fitnesses']))]
            ax.plot(epochs, data['fitnesses'], label='Fitness', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Fitness')
            ax.set_title(f'{name} - Fitness')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 多层网络
    if multi_losses:
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(multi_losses, label='Multi-layer EGGROLL', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Multi-layer EGGROLL\nFinal Loss: {multi_losses[-1]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('f:/sel-lab/exploration/multi_layer_results.png', dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    plt.savefig('f:/sel-lab/exploration/efficient_eggroll_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ 结果图已保存")
    plt.show()


def main():
    """主函数"""
    test_efficient_eggroll()


if __name__ == "__main__":
    main()
