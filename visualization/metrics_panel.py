# -*- coding: utf-8 -*-
"""
Metrics Panel - 指标监控面板
SEL 网络训练指标可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from .sel_visualizer import SELVisualizer, VisualizationConfig


class MetricsPanel(SELVisualizer):
    """
    指标监控面板
    
    功能：
    - 准确率曲线
    - 张力演化
    - 模块数量变化
    - 权重分布
    - 多面板组合
    """
    
    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)
        self.training_history: List[Dict[str, Any]] = []
    
    def add_training_history(self, history: List[Dict[str, Any]]):
        """
        添加训练历史
        
        Args:
            history: 训练历史列表
        """
        self.training_history = history
    
    def plot_accuracy_curve(self, ax: plt.Axes = None,
                          show_grid: bool = True) -> plt.Axes:
        """
        绘制准确率曲线
        
        Args:
            ax: matplotlib Axes
            show_grid: 是否显示网格
            
        Returns:
            Axes 实例
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.figures.append(fig)
        
        if not self.training_history:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return ax
        
        epochs = [m['epoch'] for m in self.training_history]
        train_accs = [m['train_accuracy'] for m in self.training_history]
        test_accs = [m['test_accuracy'] for m in self.training_history]
        
        ax.plot(epochs, train_accs, 'b-', linewidth=2, label='Train')
        ax.plot(epochs, test_accs, 'r--', linewidth=2, label='Test')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy Over Training', fontsize=14)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_tension_evolution(self, ax: plt.Axes = None,
                             show_grid: bool = True) -> plt.Axes:
        """
        绘制张力演化曲线
        
        Args:
            ax: matplotlib Axes
            show_grid: 是否显示网格
            
        Returns:
            Axes 实例
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.figures.append(fig)
        
        if not self.training_history:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return ax
        
        epochs = [m['epoch'] for m in self.training_history]
        tensions = [m['avg_tension'] for m in self.training_history]
        
        ax.plot(epochs, tensions, 'purple', linewidth=2)
        ax.fill_between(epochs, tensions, alpha=0.3, color='purple')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Average Tension', fontsize=12)
        ax.set_title('Tension Evolution', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_module_count(self, ax: plt.Axes = None,
                         show_grid: bool = True) -> plt.Axes:
        """
        绘制模块数量变化
        
        Args:
            ax: matplotlib Axes
            show_grid: 是否显示网格
            
        Returns:
            Axes 实例
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.figures.append(fig)
        
        if not self.training_history:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return ax
        
        epochs = [m['epoch'] for m in self.training_history]
        module_counts = [m['module_count'] for m in self.training_history]
        
        ax.step(epochs, module_counts, 'g-', linewidth=2, where='post')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Module Count', fontsize=12)
        ax.set_title('Module Count Evolution', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_weight_distribution(self, ax: plt.Axes = None,
                               show_grid: bool = True) -> plt.Axes:
        """
        绘制权重分布直方图
        
        Args:
            ax: matplotlib Axes
            show_grid: 是否显示网格
            
        Returns:
            Axes 实例
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.figures.append(fig)
        
        if not self.training_history:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return ax
        
        # 收集所有权重
        all_weights = []
        for m in self.training_history:
            weights_list = m.get('weights', [])
            for w in weights_list:
                all_weights.extend(w.flatten())
        
        if all_weights:
            ax.hist(all_weights, bins=50, edgecolor='black', 
                   alpha=0.7, color='steelblue')
        
        ax.set_xlabel('Weight Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Weight Distribution', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_structural_changes(self, ax: plt.Axes = None,
                              show_grid: bool = True) -> plt.Axes:
        """
        绘制结构变化事件
        
        Args:
            ax: matplotlib Axes
            show_grid: 是否显示网格
            
        Returns:
            Axes 实例
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.figures.append(fig)
        
        if not self.training_history:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return ax
        
        changes = [m['structural_changes'] for m in self.training_history]
        epochs = [m['epoch'] for m in self.training_history]
        
        ax.bar(epochs, changes, color='orange', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Structural Changes', fontsize=12)
        ax.set_title('Structural Changes Over Time', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def create_dashboard(self, figsize: tuple = (16, 12)) -> plt.Figure:
        """
        创建完整仪表板
        
        Args:
            figsize: 图形尺寸
            
        Returns:
            Figure 实例
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('SEL Training Dashboard', fontsize=16, fontweight='bold')
        self.figures.append(fig)
        
        # 1. 准确率曲线
        self.plot_accuracy_curve(ax=axes[0, 0])
        
        # 2. 张力演化
        self.plot_tension_evolution(ax=axes[0, 1])
        
        # 3. 模块数量
        self.plot_module_count(ax=axes[0, 2])
        
        # 4. 权重分布
        self.plot_weight_distribution(ax=axes[1, 0])
        
        # 5. 结构变化
        self.plot_structural_changes(ax=axes[1, 1])
        
        # 6. 统计摘要
        ax_summary = axes[1, 2]
        ax_summary.axis('off')
        
        if self.training_history:
            final = self.training_history[-1]
            best_idx = max(range(len(self.training_history)),
                          key=lambda i: self.training_history[i]['test_accuracy'])
            best = self.training_history[best_idx]
            
            summary_text = (
                f"Training Summary\n"
                f"{'='*25}\n"
                f"Final Train Acc: {final['train_accuracy']:.1%}\n"
                f"Final Test Acc: {final['test_accuracy']:.1%}\n"
                f"Best Test Acc: {best['test_accuracy']:.1%}\n"
                f"  (Epoch {best['epoch']})\n"
                f"Final Modules: {final['module_count']}\n"
                f"Total Changes: {sum(m['structural_changes'] for m in self.training_history)}"
            )
            ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                           fontsize=10, verticalalignment='top',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def save_dashboard(self, filename: str = 'training_dashboard',
                       format: str = 'png') -> str:
        """
        保存仪表板
        
        Args:
            filename: 文件名
            format: 格式
            
        Returns:
            保存路径
        """
        fig = self.create_dashboard()
        return self.save_figure(fig, filename, format)


# ============== 便捷函数 ==============

def create_metrics_dashboard(history: List[Dict],
                           output_dir: str = './visualization',
                           filename: str = 'training_dashboard') -> str:
    """
    便捷函数：创建指标仪表板
    
    Args:
        history: 训练历史
        output_dir: 输出目录
        filename: 文件名
        
    Returns:
        保存路径
    """
    config = VisualizationConfig(output_dir=output_dir)
    panel = MetricsPanel(config)
    panel.add_training_history(history)
    
    return panel.save_dashboard(filename)
