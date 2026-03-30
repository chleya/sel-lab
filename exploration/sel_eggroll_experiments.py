# -*- coding: utf-8 -*-
"""
SEL + EGGROLL 对比实验与可视化

这个文件提供了：
1. 纯 SEL  vs  纯 EGGROLL  vs  集成系统 的对比
2. 训练过程的可视化
3. 参数敏感性分析
"""

import numpy as np
import matplotlib.pyplot as plt
from sel_eggroll_integration import (
    LowRankConfig,
    SELWithEGGROLL,
    LowRankSELModule
)


def create_classification_data(n_samples=200, input_size=4, output_size=2, seed=42):
    """创建分类测试数据"""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 2.0, size=(n_samples, input_size))
    y = np.zeros((n_samples, output_size))
    
    # 简单的分类任务：基于前两个特征的符号
    for i in range(n_samples):
        if X[i, 0] + X[i, 1] > 0:
            y[i, 0] = 1
        else:
            y[i, 1] = 1
    
    return X, y


def run_experiment(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    use_eggroll: bool,
    use_sel_learning: bool,
    config: LowRankConfig,
    epochs: int = 100
):
    """运行单个实验"""
    network = SELWithEGGROLL(
        input_size=X.shape[1],
        output_size=y.shape[1],
        initial_modules=3,
        max_modules=5,
        low_rank_config=config,
        random_seed=42
    )
    
    loss_history = []
    eggroll_fitness_history = []
    
    print(f"\n{'='*60}")
    print(f"实验: {name}")
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
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}", end="")
            if eggroll_fitnesses:
                print(f" | EGGROLL Fitness = {avg_fitness:.2f}")
            else:
                print()
    
    return {
        "name": name,
        "loss_history": loss_history,
        "eggroll_fitness_history": eggroll_fitness_history,
        "final_loss": loss_history[-1]
    }


def plot_experiments(experiments, save_path="f:/sel-lab/exploration/results.png"):
    """绘制实验结果对比"""
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    for exp in experiments:
        plt.plot(exp["loss_history"], label=exp["name"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练损失对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 损失曲线（后半部分放大）
    plt.subplot(2, 2, 2)
    for exp in experiments:
        plt.plot(exp["loss_history"][50:], label=exp["name"], linewidth=2)
    plt.xlabel("Epoch (后半段)")
    plt.ylabel("Loss")
    plt.title("损失曲线放大（后50轮）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最终损失对比
    plt.subplot(2, 2, 3)
    names = [exp["name"] for exp in experiments]
    final_losses = [exp["final_loss"] for exp in experiments]
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
    bars = plt.bar(names, final_losses, color=colors, alpha=0.7)
    plt.ylabel("最终损失")
    plt.title("各方法最终损失对比")
    plt.xticks(rotation=45, ha="right")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3, axis='y')
    
    # EGGROLL 适应度（如果有）
    plt.subplot(2, 2, 4)
    for exp in experiments:
        if exp["eggroll_fitness_history"]:
            epochs_with_eggroll = [i*5 for i in range(len(exp["eggroll_fitness_history"]))]
            plt.plot(epochs_with_eggroll, exp["eggroll_fitness_history"], 
                    label=f"{exp['name']} (适应度)", 
                    linewidth=2, marker='o', markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("EGGROLL 适应度")
    plt.title("EGGROLL 适应度变化")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 结果图已保存到: {save_path}")
    plt.show()


def main():
    """主实验函数"""
    print("\n" + "="*70)
    print("SEL + EGGROLL 对比实验")
    print("="*70)
    
    # 创建数据
    X, y = create_classification_data(n_samples=200, seed=42)
    epochs = 100
    
    # 配置
    config_eggroll = LowRankConfig(
        rank=1,
        sigma=0.1,
        population_size=10,
        eggroll_lr=0.01,
        eggroll_frequency=5
    )
    
    # 实验1：纯 SEL（DFA 学习，无 EGGROLL）
    exp_sel = run_experiment(
        "纯 SEL (DFA)",
        X, y,
        use_eggroll=False,
        use_sel_learning=True,
        config=config_eggroll,
        epochs=epochs
    )
    
    # 实验2：纯 EGGROLL（无 SEL 学习）
    exp_eggroll = run_experiment(
        "纯 EGGROLL",
        X, y,
        use_eggroll=True,
        use_sel_learning=False,
        config=config_eggroll,
        epochs=epochs
    )
    
    # 实验3：完整集成（SEL + EGGROLL）
    exp_integrated = run_experiment(
        "集成系统 (SEL+EGGROLL)",
        X, y,
        use_eggroll=True,
        use_sel_learning=True,
        config=config_eggroll,
        epochs=epochs
    )
    
    experiments = [exp_sel, exp_eggroll, exp_integrated]
    
    # 总结
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)
    for exp in experiments:
        print(f"{exp['name']:30s}: 最终损失 = {exp['final_loss']:.6f}")
    
    # 可视化
    plot_experiments(experiments)
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)


if __name__ == "__main__":
    main()
