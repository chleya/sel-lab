# -*- coding: utf-8 -*-
"""
Experiment Comparator - 多实验对比
SEL 多次实验结果对比分析
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from .sel_visualizer import SELVisualizer, VisualizationConfig


class ExperimentComparator(SELVisualizer):
    """
    多实验对比器
    
    功能：
    - 多次实验准确率对比
    - 统计指标计算
    - 置信区间可视化
    - 箱线图对比
    """
    
    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)
        self.experiments: List[Dict] = []
    
    def add_experiment(self, name: str, history: List[Dict[str, Any]]):
        """
        添加实验结果
        
        Args:
            name: 实验名称
            history: 训练历史
        """
        self.experiments.append({
            'name': name,
            'history': history
        })
    
    def plot_comparison(self, ax: plt.Axes = None,
                       show_ci: bool = True,
                       show_grid: bool = True) -> plt.Axes:
        """
        绘制多实验对比
        
        Args:
            ax: matplotlib Axes
            show_ci: 是否显示置信区间
            show_grid: 是否显示网格
            
        Returns:
            Axes 实例
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
            self.figures.append(fig)
        
        if not self.experiments:
            ax.text(0.5, 0.5, 'No experiments', ha='center', va='center')
            return ax
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.experiments)))
        
        for i, exp in enumerate(self.experiments):
            history = exp['history']
            epochs = [m['epoch'] for m in history]
            test_accs = [m['test_accuracy'] for m in history]
            
            ax.plot(epochs, test_accs, 
                   color=colors[i], linewidth=2,
                   label=exp['name'], alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Experiment Comparison', fontsize=14)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_final_accuracies(self, ax: plt.Axes = None,
                             show_stats: bool = True) -> plt.Axes:
        """
        绘制最终准确率对比
        
        Args:
            ax: matplotlib Axes
            show_stats: 是否显示统计信息
            
        Returns:
            Axes 实例
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.figures.append(fig)
        
        if not self.experiments:
            ax.text(0.5, 0.5, 'No experiments', ha='center', va='center')
            return ax
        
        names = [exp['name'] for exp in self.experiments]
        final_accs = [
            exp['history'][-1]['test_accuracy'] 
            for exp in self.experiments
        ]
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        
        bars = ax.bar(names, final_accs, color=colors, 
                     edgecolor='black', alpha=0.8)
        
        # 添加数值标签
        for bar, acc in zip(bars, final_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{acc:.1%}', ha='center', va='bottom', fontsize=11)
        
        ax.set_ylabel('Final Test Accuracy', fontsize=12)
        ax.set_title('Final Test Accuracy Comparison', fontsize=14)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def plot_boxplot(self, ax: plt.Axes = None) -> plt.Axes:
        """
        绘制准确率箱线图
        
        Args:
            ax: matplotlib Axes
            
        Returns:
            Axes 实例
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.figures.append(fig)
        
        if not self.experiments:
            ax.text(0.5, 0.5, 'No experiments', ha='center', va='center')
            return ax
        
        data = []
        names = []
        
        for exp in self.experiments:
            history = exp['history']
            accs = [m['test_accuracy'] for m in history]
            data.append(accs)
            names.append(exp['name'])
        
        bp = ax.boxplot(data, labels=names, patch_artist=True)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Accuracy Distribution', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def plot_convergence(self, ax: plt.Axes = None) -> plt.Axes:
        """
        绘制收敛速度对比
        
        Args:
            ax: matplotlib Axes
            
        Returns:
            Axes 实例
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.figures.append(fig)
        
        if not self.experiments:
            ax.text(0.5, 0.5, 'No experiments', ha='center', va='center')
            return ax
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.experiments)))
        
        for i, exp in enumerate(self.experiments):
            history = exp['history']
            epochs = [m['epoch'] for m in history]
            
            # 计算达到不同阈值所需的 epoch
            thresholds = [0.5, 0.7, 0.8]
            for thresh in thresholds:
                for j, m in enumerate(history):
                    if m['test_accuracy'] >= thresh:
                        ax.scatter(j, thresh, color=colors[i], 
                                 marker='o', s=50, alpha=0.6)
                        break
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy Threshold', fontsize=12)
        ax.set_title('Convergence Milestones', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        计算统计指标
        
        Returns:
            统计结果字典
        """
        if not self.experiments:
            return {}
        
        stats = []
        
        for exp in self.experiments:
            history = exp['history']
            test_accs = [m['test_accuracy'] for m in history]
            train_accs = [m['train_accuracy'] for m in history]
            
            exp_stats = {
                'name': exp['name'],
                'final_test_acc': test_accs[-1],
                'max_test_acc': max(test_accs),
                'max_test_epoch': test_accs.index(max(test_accs)),
                'mean_test_acc': np.mean(test_accs),
                'std_test_acc': np.std(test_accs),
                'final_train_acc': train_accs[-1],
                'convergence_epoch': self._find_convergence_epoch(history)
            }
            stats.append(exp_stats)
        
        return stats
    
    def _find_convergence_epoch(self, history: List, 
                                threshold: float = 0.9) -> int:
        """
        找到收敛的 epoch
        
        Args:
            history: 训练历史
            threshold: 收敛阈值
            
        Returns:
            收敛 epoch
        """
        for i, m in enumerate(history):
            if m['test_accuracy'] >= threshold:
                return i
        return len(history) - 1
    
    def create_comparison_dashboard(self, figsize: tuple = (16, 10)) -> plt.Figure:
        """
        创建完整对比仪表板
        
        Args:
            figsize: 图形尺寸
            
        Returns:
            Figure 实例
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Experiment Comparison Dashboard', 
                    fontsize=16, fontweight='bold')
        self.figures.append(fig)
        
        # 1. 对比曲线
        self.plot_comparison(ax=axes[0, 0])
        
        # 2. 最终准确率
        self.plot_final_accuracies(ax=axes[0, 1])
        
        # 3. 箱线图
        self.plot_boxplot(ax=axes[1, 0])
        
        # 4. 统计摘要
        ax_summary = axes[1, 1]
        ax_summary.axis('off')
        
        stats = self.calculate_statistics()
        if stats:
            summary_text = "Statistics Summary\n" + "=" * 25 + "\n"
            for s in stats:
                summary_text += (
                    f"\n{s['name']}:\n"
                    f"  Final: {s['final_test_acc']:.1%}\n"
                    f"  Best: {s['max_test_acc']:.1%}\n"
                    f"  Mean: {s['mean_test_acc']:.1%} ± {s['std_test_acc']:.2%}\n"
                    f"  Conv: Epoch {s['convergence_epoch']}"
                )
            
            ax_summary.text(0.1, 0.95, summary_text, 
                           transform=ax_summary.transAxes,
                           fontsize=9, verticalalignment='top',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round', 
                                    facecolor='lightyellow', 
                                    alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def save_comparison(self, filename: str = 'experiment_comparison',
                       format: str = 'png') -> str:
        """
        保存对比仪表板
        
        Args:
            filename: 文件名
            format: 格式
            
        Returns:
            保存路径
        """
        fig = self.create_comparison_dashboard()
        return self.save_figure(fig, filename, format)


# ============== 便捷函数 ==============

def compare_experiments(experiments: Dict[str, List],
                       output_dir: str = './visualization',
                       filename: str = 'experiment_comparison') -> str:
    """
    便捷函数：对比多个实验
    
    Args:
        experiments: {name: history}
        output_dir: 输出目录
        filename: 文件名
        
    Returns:
        保存路径
    """
    config = VisualizationConfig(output_dir=output_dir)
    comparator = ExperimentComparator(config)
    
    for name, history in experiments.items():
        comparator.add_experiment(name, history)
    
    return comparator.save_comparison(filename)
