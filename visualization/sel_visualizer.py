# -*- coding: utf-8 -*-
"""
SEL Visualizer - 基类
SEL 网络可视化基类
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx


@dataclass
class VisualizationConfig:
    """可视化配置"""
    # 图像设置
    figure_size: tuple = (12, 8)
    dpi: int = 100
    colormap: str = 'viridis'
    
    # 动画设置
    frame_interval: int = 200  # ms
    save_fps: int = 10
    
    # 输出设置
    output_dir: str = './visualization_output'
    save_format: str = 'png'


class SELVisualizer:
    """
    SEL 网络可视化基类
    
    功能：
    - 数据采集和预处理
    - 基础绘图功能
    - 动画生成
    - 结果保存
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config or VisualizationConfig()
        self.figures: List[plt.Figure] = []
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def collect_metrics(self, network, epoch: int) -> Dict[str, Any]:
        """
        采集网络指标
        
        Args:
            network: SELNetwork 实例
            epoch: 当前轮次
            
        Returns:
            指标字典
        """
        metrics = {
            'epoch': epoch,
            'module_count': len(network.modules),
            'param_count': network.param_count,
            'tensions': [m.local_tension for m in network.modules],
            'weight_norms': [m.weight_norm for m in network.modules],
            'module_names': [m.name for m in network.modules]
        }
        
        # 收集权重矩阵
        weights = []
        for m in network.modules:
            weights.append(m.weights)
        metrics['weights'] = weights
        
        return metrics
    
    def collect_training_history(self, trainer) -> List[Dict[str, Any]]:
        """
        采集训练历史
        
        Args:
            trainer: SELTrainer 实例
            
        Returns:
            训练历史列表
        """
        history = []
        for metrics in trainer.metrics:
            history.append({
                'epoch': metrics.epoch,
                'train_accuracy': metrics.train_accuracy,
                'test_accuracy': metrics.test_accuracy,
                'avg_tension': metrics.avg_tension,
                'module_count': metrics.module_count,
                'structural_changes': metrics.structural_changes
            })
        return history
    
    def create_figure(self, figsize: tuple = None) -> plt.Figure:
        """
        创建新图形
        
        Args:
            figsize: 图形尺寸
            
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(
            figsize=figsize or self.config.figure_size,
            dpi=self.config.dpi
        )
        self.figures.append(fig)
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, 
                   format: str = None) -> str:
        """
        保存图形
        
        Args:
            fig: matplotlib Figure
            filename: 文件名
            format: 格式 (png, pdf, svg)
            
        Returns:
            保存路径
        """
        fmt = format or self.config.save_format
        filepath = os.path.join(
            self.config.output_dir,
            f"{filename}.{fmt}"
        )
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        return filepath
    
    def save_animation(self, anim: FuncAnimation, filename: str,
                      format: str = 'gif') -> str:
        """
        保存动画
        
        Args:
            anim: FuncAnimation 实例
            filename: 文件名
            format: 格式 (gif, mp4)
            
        Returns:
            保存路径
        """
        filepath = os.path.join(
            self.config.output_dir,
            f"{filename}.{format}"
        )
        
        if format == 'gif':
            writer = PillowWriter(fps=self.config.save_fps)
            anim.save(filepath, writer=writer)
        else:
            # MP4 需要 ffmpeg
            try:
                anim.save(filepath, writer='ffmpeg', 
                         fps=self.config.save_fps)
            except Exception as e:
                print(f"MP4 保存失败: {e}")
                # 回退到 GIF
                filepath = filepath.replace('.mp4', '.gif')
                writer = PillowWriter(fps=self.config.save_fps)
                anim.save(filepath, writer=writer)
        
        return filepath
    
    def save_json(self, data: Dict, filename: str) -> str:
        """
        保存 JSON 数据
        
        Args:
            data: 数据字典
            filename: 文件名
            
        Returns:
            保存路径
        """
        filepath = os.path.join(
            self.config.output_dir,
            f"{filename}.json"
        )
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath
    
    def close_all(self):
        """关闭所有图形"""
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
        return False
