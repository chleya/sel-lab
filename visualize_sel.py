"""
SEL-Lab 可视化交互界面
功能：
1. 训练过程实时曲线
2. 网络结构可视化
3. 权重热力图
4. 交互式参数调整
5. 增量学习演示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from typing import List, Dict
import json
from dataclasses import dataclass
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class TrainingHistory:
    """训练历史"""
    epochs: List[int]
    train_acc: List[float]
    test_acc: List[float]
    tensions: List[List[float]]
    module_counts: List[int]


class SELVisualizer:
    """SEL 可视化器"""
    
    def __init__(self):
        self.history = None
        self.network = None
        self.fig = None
        self.axes = None
    
    def set_network(self, network):
        """设置网络"""
        self.network = network
    
    def set_history(self, history: TrainingHistory):
        """设置训练历史"""
        self.history = history
    
    def plot_training_curve(self, save_path: str = None):
        """绘制训练曲线"""
        if self.history is None:
            print("请先设置训练历史")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SEL Training Visualization', fontsize=14)
        
        # 1. 准确率曲线
        ax1 = axes[0, 0]
        ax1.plot(self.history.epochs, self.history.train_acc, 'b-', label='Train', linewidth=2)
        ax1.plot(self.history.epochs, self.history.test_acc, 'r--', label='Test', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # 2. 模块数量
        ax2 = axes[0, 1]
        ax2.plot(self.history.epochs, self.history.module_counts, 'g-', linewidth=2)
        ax2.fill_between(self.history.epochs, self.history.module_counts, alpha=0.3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Modules')
        ax2.set_title('Module Count Evolution')
        ax2.grid(True, alpha=0.3)
        
        # 3. 张力变化
        ax3 = axes[1, 0]
        for i, tension_series in enumerate(zip(*self.history.tensions)):
            ax3.plot(self.history.epochs[:len(tension_series)], tension_series, 
                    label=f'Module {i}', linewidth=1.5)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Local Tension')
        ax3.set_title('Tension Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 权重分布
        ax4 = axes[1, 1]
        all_weights = []
        for m in self.network.modules:
            all_weights.extend(m.weights.flatten())
        ax4.hist(all_weights, bins=50, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Weight Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Weight Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_path}")
        
        plt.show()
    
    def plot_network_structure(self, save_path: str = None):
        """绘制网络结构"""
        if self.network is None:
            print("请先设置网络")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 标题
        ax.text(2, 3.5, f'SEL Network Structure\n'
                       f'Modules: {len(self.network.modules)} | '
                       f'Params: {self.network.param_count}',
               ha='center', fontsize=14, fontweight='bold')
        
        # 绘制模块
        n_mods = len(self.network.modules)
        for i, m in enumerate(self.network.modules):
            x = i % 3
            y = 2 - (i // 3)
            
            # 模块框
            rect = plt.Rectangle((x*1.2+0.1, y*1.2+0.1), 1, 0.8, 
                                 fill=True, facecolor='lightblue', 
                                 edgecolor='navy', linewidth=2)
            ax.add_patch(rect)
            
            # 模块名
            ax.text(x*1.2+0.6, y*1.2+0.5, f'{m.name}\n'
                                         f'W: {m.weights.shape}\n'
                                         f'T: {m.local_tension:.2f}',
                   ha='center', va='center', fontsize=8)
            
            # 权重统计
            w_mean = np.mean(np.abs(m.weights))
            ax.text(x*1.2+0.6, y*1.2+0.15, f'|w|: {w_mean:.3f}',
                   ha='center', fontsize=7, color='gray')
        
        # 图例
        legend_text = "T = Local Tension\n|w| = Weight Magnitude"
        ax.text(2, 0, legend_text, ha='center', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title('Network Structure Visualization', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_path}")
        
        plt.show()
    
    def plot_weights_heatmap(self, save_path: str = None):
        """绘制权重热力图"""
        if self.network is None or not self.network.modules:
            print("网络为空")
            return
        
        n = len(self.network.modules)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        if n == 1:
            axes = [axes]
        
        fig.suptitle('Weight Matrices Heatmap', fontsize=14)
        
        for i, (ax, m) in enumerate(zip(axes, self.network.modules)):
            im = ax.imshow(m.weights, cmap='RdBu_r', aspect='auto',
                          vmin=-0.5, vmax=0.5)
            ax.set_title(f'{m.name}\n{m.weights.shape}')
            ax.set_xlabel('Output')
            ax.set_ylabel('Input')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_path}")
        
        plt.show()
    
    def interactive_training(self):
        """交互式训练界面"""
        print("\n" + "="*60)
        print("SEL 交互式训练")
        print("="*60)
        
        # 简单动画
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('SEL Interactive Training Demo', fontsize=14)
        
        # 初始化数据
        epochs = []
        accs = []
        mods = []
        
        line1, = ax1.plot([], [], 'b-', linewidth=2, label='Accuracy')
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        line2, = ax2.plot([], [], 'g-', linewidth=2)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 10)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Modules')
        ax2.grid(True, alpha=0.3)
        
        # 创建网络和数据
        from core.sel_core import SELNetwork, SELConfig, SELTrainer
        
        np.random.seed(42)
        X = np.random.randn(100, 4) * 2
        y = np.zeros((100, 2))
        for i in range(100):
            if X[i, 0] + X[i, 1] > 0:
                y[i, 0] = 1
            else:
                y[i, 1] = 1
        
        config = SELConfig(input_size=4, output_size=2, initial_modules=3, learning_rate=0.1)
        trainer = SELTrainer(config)
        
        def animate(frame):
            # 训练一步
            for i in range(10):
                trainer.network.forward_learning(X[i], y[i])
            
            # 更新数据
            epochs.append(frame * 5)
            correct = sum(1 for j in range(100) 
                         if trainer.network.predict(X[j]) == int(np.argmax(y[j])))
            acc = correct / 100
            accs.append(acc)
            mods.append(len(trainer.network.modules))
            
            # 更新图表
            line1.set_data(epochs, accs)
            line2.set_data(epochs, mods)
            ax1.set_xlim(0, max(100, len(epochs) * 5))
            ax2.set_xlim(0, max(100, len(epochs) * 5))
            
            if frame % 10 == 0:
                print(f"Epoch {frame*5}: Acc={acc:.1%}, Modules={len(trainer.network.modules)}")
            
            return line1, line2
        
        print("运行动画 (100 帧)...")
        ani = FuncAnimation(fig, animate, frames=20, interval=200, blit=True)
        plt.show()
        
        print("\n训练完成!")
    
    def compare_methods(self, save_path: str = None):
        """对比不同方法"""
        from core.sel_core import SELNetwork, SELConfig, SELTrainer
        
        print("\n对比 SEL 前向学习 vs 梯度下降...")
        
        # 数据
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
        
        # SEL (DFA 前向学习)
        np.random.seed(42)
        sel_config = SELConfig(input_size=4, output_size=2, learning_rate=0.1, epochs=50)
        sel_trainer = SELTrainer(sel_config)
        sel_trainer.train(X_train, y_train, X_test, y_test)
        sel_accs = [m.test_accuracy for m in sel_trainer.metrics]
        
        # 梯度下降（对比）- 简单BP网络
        np.random.seed(42)
        W1_bp = np.random.randn(4, 8) * 0.5
        W2_bp = np.random.randn(8, 2) * 0.5
        
        grad_accs = []
        for epoch in range(50):
            for i in range(len(X_train)):
                x = X_train[i]
                target = y_train[i]
                # 前向
                h = np.tanh(x @ W1_bp)
                out = h @ W2_bp
                err = out - target
                # 反向传播
                W2_bp -= 0.1 * np.outer(h, err)
                W1_bp -= 0.1 * np.outer(x, err @ W2_bp.T * (1 - h**2))
            
            # 计算准确率
            correct = 0
            for j in range(len(X_test)):
                h = np.tanh(X_test[j] @ W1_bp)
                out = h @ W2_bp
                if int(np.argmax(out)) == int(np.argmax(y_test[j])):
                    correct += 1
            grad_accs.append(correct / len(X_test))
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = list(range(50))
        
        ax.plot(epochs, sel_accs, 'b-', linewidth=2, label='SEL Forward-Only')
        ax.plot(epochs, grad_accs, 'r--', linewidth=2, label='Gradient Descent')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('SEL Forward Learning vs Gradient Descent')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_path}")
        
        plt.show()
        
        print(f"\n最终准确率:")
        print(f"  SEL 前向学习: {sel_accs[-1]:.1%}")
        print(f"  梯度下降: {grad_accs[-1]:.1%}")


def demo():
    """演示"""
    print("\n" + "="*60)
    print("SEL 可视化演示")
    print("="*60)
    
    viz = SELVisualizer()
    
    # 选择演示
    print("\n选择演示:")
    print("1. 训练曲线可视化")
    print("2. 网络结构可视化")
    print("3. 权重热力图")
    print("4. 交互式训练")
    print("5. 方法对比")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "4":
        viz.interactive_training()
    elif choice == "5":
        viz.compare_methods()
    else:
        # 创建简单演示数据
        from core.sel_core import SELNetwork, SELConfig
        
        np.random.seed(42)
        config = SELConfig(input_size=4, output_size=2)
        net = SELNetwork(config)
        for i in range(3):
            net.add_module(f"m{i}", seed=42+i)
        
        # 模拟训练历史
        history = TrainingHistory(
            epochs=list(range(50)),
            train_acc=[0.5 + 0.4 * (1 - np.exp(-i/20)) + np.random.uniform(-0.02, 0.02) for i in range(50)],
            test_acc=[0.48 + 0.42 * (1 - np.exp(-i/20)) + np.random.uniform(-0.02, 0.02) for i in range(50)],
            tensions=[[0.5 + 0.3 * np.exp(-i/15) for i in range(50)] for _ in range(3)],
            module_counts=[3] * 50
        )
        
        viz.set_network(net)
        viz.set_history(history)
        
        if choice == "1":
            viz.plot_training_curve()
        elif choice == "2":
            viz.plot_network_structure()
        elif choice == "3":
            viz.plot_weights_heatmap()
        else:
            print("无效选择")
    
    print("\n演示完成!")


if __name__ == "__main__":
    demo()
