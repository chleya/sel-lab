# -*- coding: utf-8 -*-
"""
SEL + EGGROLL 复杂任务测试

测试更复杂的场景：
1. XOR 问题（非线性分类）
2. 高维回归问题
3. 螺旋数据分类
"""

import numpy as np
import matplotlib.pyplot as plt
from sel_eggroll_integration import (
    LowRankConfig,
    SELWithEGGROLL,
)


def create_xor_data(n_samples=400, noise=0.1, seed=42):
    """
    创建 XOR 问题数据
    XOR 是一个经典的非线性分类问题
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, 2))
    
    # XOR 标签: (x>0) XOR (y>0)
    y = np.zeros((n_samples, 2))
    for i in range(n_samples):
        label = int((X[i, 0] > 0) != (X[i, 1] > 0))
        y[i, label] = 1
    
    # 添加噪声
    X += rng.normal(0, noise, size=X.shape)
    
    return X, y


def create_spiral_data(n_samples=500, n_classes=3, seed=42):
    """
    创建螺旋数据（多分类非线性问题）
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros((n_samples * n_classes, n_classes))
    
    for class_id in range(n_classes):
        ix = range(n_samples * class_id, n_samples * (class_id + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(class_id * 4, (class_id + 1) * 4, n_samples) + rng.normal(0, 0.2, n_samples)
        X[ix] = np.c_[r * np.sin(t * 2.0), r * np.cos(t * 2.0)]
        y[ix, class_id] = 1
    
    return X, y


def create_regression_data(n_samples=300, input_dim=10, seed=42):
    """
    创建高维回归问题
    y = sin(x1) + cos(x2) + x3^2 + 噪声
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, input_dim))
    
    # 复杂的非线性关系
    y = np.zeros((n_samples, 1))
    y[:, 0] = (
        np.sin(X[:, 0]) + 
        np.cos(X[:, 1]) + 
        0.5 * X[:, 2]**2 +
        0.3 * X[:, 3] * X[:, 4] +
        0.1 * rng.normal(0, 1, n_samples)
    )
    
    return X, y


def run_experiment(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    use_eggroll: bool,
    use_sel_learning: bool,
    config: LowRankConfig,
    epochs: int = 150
):
    """运行单个实验"""
    network = SELWithEGGROLL(
        input_size=X.shape[1],
        output_size=y.shape[1],
        initial_modules=4,
        max_modules=6,
        low_rank_config=config,
        random_seed=42
    )
    
    loss_history = []
    eggroll_fitness_history = []
    
    print(f"\n{'='*60}")
    print(f"实验: {name}")
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        lr = 0.1 if use_sel_learning else 0.0
        avg_loss, eggroll_fitnesses = network.train_step(
            X, y, lr=lr, use_eggroll=use_eggroll
        )
        
        loss_history.append(avg_loss)
        
        if eggroll_fitnesses:
            avg_fitness = np.mean([np.mean(f) for f in eggroll_fitnesses])
            eggroll_fitness_history.append(avg_fitness)
        
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}", end="")
            if eggroll_fitnesses:
                print(f" | EGGROLL Fitness = {avg_fitness:.2f}")
            else:
                print()
    
    return {
        "name": name,
        "loss_history": loss_history,
        "eggroll_fitness_history": eggroll_fitness_history,
        "final_loss": loss_history[-1],
        "initial_loss": loss_history[0]
    }


def plot_results(results, task_name, save_path=None):
    """绘制实验结果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 损失曲线
    ax1 = axes[0]
    for res in results:
        ax1.plot(res["loss_history"], label=res["name"], linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{task_name} - Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 最终损失对比
    ax2 = axes[1]
    names = [res["name"] for res in results]
    final_losses = [res["final_loss"] for res in results]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax2.bar(names, final_losses, color=colors, alpha=0.7)
    ax2.set_ylabel("Final Loss")
    ax2.set_title(f"{task_name} - Final Loss Comparison")
    ax2.set_xticklabels(names, rotation=15, ha="right")
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # EGGROLL 适应度
    ax3 = axes[2]
    for res in results:
        if res["eggroll_fitness_history"]:
            freq = 5  # EGGROLL 频率
            epochs_with_eggroll = [i*freq for i in range(len(res["eggroll_fitness_history"]))]
            ax3.plot(epochs_with_eggroll, res["eggroll_fitness_history"], 
                    label=f"{res['name']}", 
                    linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("EGGROLL Fitness")
    ax3.set_title(f"{task_name} - EGGROLL Fitness")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ 结果图已保存到: {save_path}")
    plt.show()


def test_xor_problem():
    """测试 XOR 问题"""
    print("\n" + "="*70)
    print("任务 1: XOR 非线性分类")
    print("="*70)
    
    X, y = create_xor_data(n_samples=400, noise=0.1)
    config = LowRankConfig(rank=2, sigma=0.1, population_size=15, 
                          eggroll_lr=0.02, eggroll_frequency=5)
    
    results = []
    results.append(run_experiment("Pure SEL", X, y, False, True, config, epochs=150))
    results.append(run_experiment("Pure EGGROLL", X, y, True, False, config, epochs=150))
    results.append(run_experiment("SEL+EGGROLL", X, y, True, True, config, epochs=150))
    
    plot_results(results, "XOR Problem", "f:/sel-lab/exploration/xor_results.png")
    return results


def test_spiral_problem():
    """测试螺旋数据分类"""
    print("\n" + "="*70)
    print("任务 2: 螺旋数据多分类")
    print("="*70)
    
    X, y = create_spiral_data(n_samples=500, n_classes=3)
    config = LowRankConfig(rank=2, sigma=0.15, population_size=20, 
                          eggroll_lr=0.015, eggroll_frequency=5)
    
    results = []
    results.append(run_experiment("Pure SEL", X, y, False, True, config, epochs=200))
    results.append(run_experiment("Pure EGGROLL", X, y, True, False, config, epochs=200))
    results.append(run_experiment("SEL+EGGROLL", X, y, True, True, config, epochs=200))
    
    plot_results(results, "Spiral Classification", "f:/sel-lab/exploration/spiral_results.png")
    return results


def test_regression_problem():
    """测试高维回归"""
    print("\n" + "="*70)
    print("任务 3: 高维非线性回归")
    print("="*70)
    
    X, y = create_regression_data(n_samples=300, input_dim=10)
    config = LowRankConfig(rank=3, sigma=0.1, population_size=15, 
                          eggroll_lr=0.01, eggroll_frequency=5)
    
    results = []
    results.append(run_experiment("Pure SEL", X, y, False, True, config, epochs=150))
    results.append(run_experiment("Pure EGGROLL", X, y, True, False, config, epochs=150))
    results.append(run_experiment("SEL+EGGROLL", X, y, True, True, config, epochs=150))
    
    plot_results(results, "High-Dim Regression", "f:/sel-lab/exploration/regression_results.png")
    return results


def main():
    """主函数"""
    print("\n" + "="*70)
    print("SEL + EGGROLL 复杂任务测试")
    print("="*70)
    
    all_results = {}
    
    # 任务 1: XOR
    all_results["xor"] = test_xor_problem()
    
    # 任务 2: 螺旋数据
    all_results["spiral"] = test_spiral_problem()
    
    # 任务 3: 高维回归
    all_results["regression"] = test_regression_problem()
    
    # 总结合并
    print("\n" + "="*70)
    print("所有任务总结")
    print("="*70)
    
    for task_name, results in all_results.items():
        print(f"\n{task_name.upper()}:")
        for res in results:
            improvement = (res["initial_loss"] - res["final_loss"]) / res["initial_loss"] * 100
            print(f"  {res['name']:20s}: 初始={res['initial_loss']:.4f}, "
                  f"最终={res['final_loss']:.4f}, 改进={improvement:+.1f}%")
    
    print("\n" + "="*70)
    print("复杂任务测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()
