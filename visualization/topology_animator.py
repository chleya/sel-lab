# -*- coding: utf-8 -*-
"""
Topology Animator - 拓扑演化动画
SEL 网络拓扑动态演化可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
from .sel_visualizer import SELVisualizer, VisualizationConfig


class TopologyAnimator(SELVisualizer):
    """
    拓扑演化动画器
    
    功能：
    - 网络拓扑结构可视化
    - 模块动态添加/移除动画
    - 连接权重动态展示
    - 交互式播放控制
    """
    
    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)
        self.history: List[Dict[str, Any]] = []
        self.animation: Optional[animation.FuncAnimation] = None
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        
        # 交互控件
        self.play_button: Optional[Button] = None
        self.slider: Optional[Slider] = None
        self.paused = False
    
    def add_history(self, metrics: Dict[str, Any]):
        """
        添加历史数据
        
        Args:
            metrics: 网络指标
        """
        self.history.append(metrics)
    
    def build_graph(self, metrics: Dict[str, Any]) -> nx.Graph:
        """
        构建 NetworkX 图
        
        Args:
            metrics: 网络指标
            
        Returns:
            NetworkX 图
        """
        G = nx.Graph()
        
        # 添加输入/输出节点
        G.add_node('input', pos=(0, 0), node_type='input')
        G.add_node('output', pos=(2, 0), node_type='output')
        
        # 添加模块节点
        tensions = metrics.get('tensions', [])
        weight_norms = metrics.get('weight_norms', [])
        names = metrics.get('module_names', [])
        
        n_modules = len(names)
        for i, (name, tension, w_norm) in enumerate(zip(names, tensions, weight_norms)):
            # 节点位置
            x = 0.5 + (i / max(n_modules - 1, 1)) * 1.0 if n_modules > 1 else 1.0
            y = tension * 2  # 张力影响高度
            
            G.add_node(
                name,
                pos=(x, y),
                node_type='module',
                tension=tension,
                weight_norm=w_norm
            )
            
            # 添加连接
            G.add_edge('input', name, weight=w_norm * 0.5)
            G.add_edge(name, 'output', weight=w_norm * 0.5)
        
        return G
    
    def get_node_colors(self, G: nx.Graph) -> List[str]:
        """
        获取节点颜色
        
        Args:
            G: NetworkX 图
            
        Returns:
            颜色列表
        """
        colors = []
        for node in G.nodes(data=True):
            node_type = node[1].get('node_type', 'module')
            if node_type == 'input':
                colors.append('lightgreen')
            elif node_type == 'output':
                colors.append('lightcoral')
            else:
                # 张力影响颜色
                tension = node[1].get('tension', 0.5)
                if tension > 0.7:
                    colors.append('red')
                elif tension > 0.4:
                    colors.append('orange')
                else:
                    colors.append('lightblue')
        return colors
    
    def get_edge_widths(self, G: nx.Graph) -> List[float]:
        """
        获取边宽度
        
        Args:
            G: NetworkX 图
            
        Returns:
            宽度列表
        """
        widths = []
        for edge in G.edges(data=True):
            weight = edge[2].get('weight', 0.5)
            widths.append(weight * 5)
        return widths
    
    def draw_frame(self, frame: int) -> List:
        """
        绘制动画帧
        
        Args:
            frame: 帧索引
            
        Returns:
            艺术家对象列表
        """
        if frame >= len(self.history):
            frame = len(self.history) - 1
        
        self.ax.clear()
        
        metrics = self.history[frame]
        G = self.build_graph(metrics)
        
        # 获取位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制节点
        node_colors = self.get_node_colors(G)
        node_sizes = [1000 if G.nodes[n].get('node_type') == 'module' else 1500 
                     for n in G.nodes()]
        
        nx.draw_networkx_nodes(
            G, pos, ax=self.ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )
        
        # 绘制边
        edge_widths = self.get_edge_widths(G)
        nx.draw_networkx_edges(
            G, pos, ax=self.ax,
            width=edge_widths,
            alpha=0.6,
            edge_color='gray'
        )
        
        # 绘制标签
        labels = {n: n for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels, ax=self.ax,
            font_size=8,
            font_weight='bold'
        )
        
        # 标题
        epoch = metrics.get('epoch', frame)
        module_count = len(metrics.get('module_names', []))
        self.ax.set_title(
            f'Topology Evolution - Epoch {epoch}\n'
            f'Modules: {module_count}',
            fontsize=12
        )
        
        self.ax.axis('off')
        
        return []
    
    def on_click(self, event):
        """播放/暂停点击处理"""
        self.paused = not self.paused
        if self.paused:
            self.animation.event_source.stop()
        else:
            self.animation.event_source.start()
    
    def on_slider_change(self, val):
        """滑块变化处理"""
        frame = int(val)
        self.draw_frame(frame)
        self.fig.canvas.draw_idle()
    
    def create_animation(self, figsize: tuple = (14, 8)) -> animation.FuncAnimation:
        """
        创建动画
        
        Args:
            figsize: 图形尺寸
            
        Returns:
            FuncAnimation 实例
        """
        if not self.history:
            raise ValueError("No history data. Call add_history() first.")
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.figures.append(self.fig)
        
        # 创建动画
        self.animation = animation.FuncAnimation(
            self.fig,
            self.draw_frame,
            frames=len(self.history),
            interval=self.config.frame_interval,
            blit=False,
            repeat=True
        )
        
        return self.animation
    
    def save_animation(self, filename: str, format: str = 'gif') -> str:
        """
        保存动画
        
        Args:
            filename: 文件名
            format: 格式
            
        Returns:
            保存路径
        """
        if not self.animation:
            self.create_animation()
        
        return super().save_animation(self.animation, filename, format)
    
    def show_interactive(self):
        """
        显示交互式动画
        
        包含播放/暂停按钮和进度滑块
        """
        if not self.animation:
            self.create_animation()
        
        # 添加按钮
        ax_play = plt.axes([0.8, 0.02, 0.1, 0.05])
        self.play_button = Button(ax_play, 'Play/Pause')
        self.play_button.on_clicked(self.on_click)
        
        # 添加滑块
        ax_slider = plt.axes([0.1, 0.02, 0.65, 0.03])
        self.slider = Slider(
            ax_slider, 'Epoch', 0, len(self.history) - 1,
            valinit=0, valstep=1
        )
        self.slider.on_changed(self.on_slider_change)
        
        plt.show()
    
    def create_static_snapshot(self, epoch: int, 
                               figsize: tuple = (12, 8)) -> plt.Figure:
        """
        创建静态快照
        
        Args:
            epoch: 指定的 epoch
            figsize: 图形尺寸
            
        Returns:
            Figure 实例
        """
        if epoch >= len(self.history):
            epoch = len(self.history) - 1
        
        fig, ax = plt.subplots(figsize=figsize)
        self.figures.append(fig)
        
        self.ax = ax
        self.draw_frame(epoch)
        
        return fig


# ============== 便捷函数 ==============

def animate_topology(network_history: List, 
                    output_dir: str = './visualization',
                    filename: str = 'topology_evolution') -> str:
    """
    便捷函数：动画化拓扑演化
    
    Args:
        network_history: 网络历史列表
        output_dir: 输出目录
        filename: 文件名
        
    Returns:
        保存路径
    """
    config = VisualizationConfig(output_dir=output_dir)
    animator = TopologyAnimator(config)
    
    # 添加所有历史
    for net, epoch in network_history:
        # 简化处理：假设是 metrics 字典
        if hasattr(net, 'modules'):
            # 是 SELNetwork 实例
            metrics = {
                'epoch': epoch,
                'module_count': len(net.modules),
                'tensions': [m.local_tension for m in net.modules],
                'weight_norms': [m.weight_norm for m in net.modules],
                'module_names': [m.name for m in net.modules],
                'weights': [m.weights for m in net.modules]
            }
        else:
            metrics = net
        
        animator.add_history(metrics)
    
    # 保存动画
    return animator.save_animation(filename)
