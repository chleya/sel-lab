# -*- coding: utf-8 -*-
"""
SEL-Lab Foundations Visualizer
基础研究可视化分析工具

可视化内容:
1. 信息论指标时间序列
2. 复杂系统涌现行为
3. 能量函数演化
4. 综合分析仪表板
"""

from __future__ import annotations

import numpy as np
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import resolve_results_path


class FoundationsVisualizer:
    """基础研究可视化器（纯文本/ASCII版本）"""
    
    def __init__(self, results_path: Optional[str] = None):
        self.results_path = results_path or resolve_results_path("integrated_foundations_results.json")
        self.results = self._load_results()
    
    def _load_results(self) -> Dict:
        """加载结果文件"""
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}
    
    def plot_ascii_timeseries(self, data: List[float], title: str, width: int = 60, height: int = 10) -> str:
        """创建ASCII时间序列图"""
        if not data:
            return f"{title}: No data"
        
        lines = []
        lines.append(f"\n{'='*width}")
        lines.append(f"{title:^60}")
        lines.append(f"{'='*width}")
        
        # 计算范围
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # 创建图表
        for row in range(height, 0, -1):
            threshold = min_val + (range_val * (row - 1) / height)
            line = f"{threshold:6.3f} |"
            
            for val in data[-width:]:  # 只显示最近的数据点
                if val >= threshold:
                    line += "*"
                else:
                    line += " "
            lines.append(line)
        
        lines.append(f"       {'-'*min(width, len(data))}")
        lines.append(f"       Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {np.mean(data):.4f}")
        
        return "\n".join(lines)
    
    def plot_ascii_histogram(self, data: List[float], title: str, bins: int = 10, width: int = 50) -> str:
        """创建ASCII直方图"""
        if not data:
            return f"{title}: No data"
        
        lines = []
        lines.append(f"\n{'='*width}")
        lines.append(f"{title:^50}")
        lines.append(f"{'='*width}")
        
        # 计算直方图
        hist, edges = np.histogram(data, bins=bins)
        max_count = max(hist) if max(hist) > 0 else 1
        
        # 绘制直方图
        for i in range(len(hist)):
            bar_length = int((hist[i] / max_count) * 40)
            bar = "█" * bar_length
            lines.append(f"{edges[i]:6.3f}-{edges[i+1]:6.3f} | {bar} ({hist[i]})")
        
        return "\n".join(lines)
    
    def visualize_info_theory(self) -> str:
        """可视化信息论指标"""
        history = self.results.get('history', [])
        if not history:
            return "No history data available"
        
        lines = []
        lines.append("\n" + "="*70)
        lines.append(" Information Theory Metrics ".center(70))
        lines.append("="*70)
        
        # 提取指标
        weight_entropy = [h['info_metrics']['weight_entropy'] for h in history]
        mutual_info = [h['info_metrics']['mutual_information'] for h in history]
        complexity = [h['info_metrics']['structure_complexity'] for h in history]
        
        # 绘制时间序列
        lines.append(self.plot_ascii_timeseries(weight_entropy, "Weight Entropy Evolution"))
        lines.append(self.plot_ascii_timeseries(mutual_info, "Mutual Information Evolution"))
        lines.append(self.plot_ascii_timeseries(complexity, "Structure Complexity Evolution"))
        
        # 统计摘要
        lines.append("\n" + "-"*70)
        lines.append(" Statistical Summary ".center(70))
        lines.append("-"*70)
        lines.append(f"Weight Entropy:  Mean={np.mean(weight_entropy):.4f}, Std={np.std(weight_entropy):.4f}")
        lines.append(f"Mutual Info:     Mean={np.mean(mutual_info):.4f}, Std={np.std(mutual_info):.4f}")
        lines.append(f"Complexity:      Mean={np.mean(complexity):.4f}, Std={np.std(complexity):.4f}")
        
        return "\n".join(lines)
    
    def visualize_complex_systems(self) -> str:
        """可视化复杂系统指标"""
        history = self.results.get('history', [])
        if not history:
            return "No history data available"
        
        lines = []
        lines.append("\n" + "="*70)
        lines.append(" Complex Systems Analysis ".center(70))
        lines.append("="*70)
        
        # 提取指标
        sync = [h['complex_metrics']['synchronization'] for h in history]
        
        # 绘制同步度
        lines.append(self.plot_ascii_timeseries(sync, "Synchronization Evolution"))
        
        # 涌现事件
        emergence_events = self.results.get('summary', {}).get('emergence_events', [])
        lines.append("\n" + "-"*70)
        lines.append(f" Emergence Events: {len(emergence_events)} ".center(70))
        lines.append("-"*70)
        
        if emergence_events:
            for event in emergence_events:
                lines.append(f"  Epoch {event['epoch']}: {event['type']}")
                lines.append(f"    Avg Tension: {event['avg_tension']:.4f}")
                lines.append(f"    Variance: {event['tension_variance']:.4f}")
        else:
            lines.append("  No emergence events detected")
        
        # 网络拓扑
        if history:
            final_topology = history[-1]['complex_metrics']['topology']
            lines.append("\n" + "-"*70)
            lines.append(" Final Network Topology ".center(70))
            lines.append("-"*70)
            for key, value in final_topology.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def visualize_energy(self) -> str:
        """可视化能量函数"""
        history = self.results.get('history', [])
        if not history:
            return "No history data available"
        
        lines = []
        lines.append("\n" + "="*70)
        lines.append(" Energy-Based Analysis ".center(70))
        lines.append("="*70)
        
        # 提取能量值
        energies = [h.get('energy', 0) for h in history if 'energy' in h]
        
        if energies:
            lines.append(self.plot_ascii_timeseries(energies, "System Energy Evolution"))
            lines.append(self.plot_ascii_histogram(energies, "Energy Distribution"))
            
            # 能量趋势
            lines.append("\n" + "-"*70)
            lines.append(" Energy Statistics ".center(70))
            lines.append("-"*70)
            lines.append(f"Initial Energy: {energies[0]:.4f}")
            lines.append(f"Final Energy: {energies[-1]:.4f}")
            lines.append(f"Energy Change: {energies[-1] - energies[0]:.4f}")
            lines.append(f"Mean Energy: {np.mean(energies):.4f}")
            lines.append(f"Min Energy: {min(energies):.4f}")
            lines.append(f"Max Energy: {max(energies):.4f}")
        else:
            lines.append("  No energy data available")
        
        return "\n".join(lines)
    
    def visualize_bayesian(self) -> str:
        """可视化贝叶斯优化"""
        history = self.results.get('history', [])
        if not history:
            return "No history data available"
        
        lines = []
        lines.append("\n" + "="*70)
        lines.append(" Bayesian Optimization ".center(70))
        lines.append("="*70)
        
        # 提取学习率建议
        suggested_lrs = [h['suggested_lr'] for h in history]
        
        lines.append(self.plot_ascii_timeseries(suggested_lrs, "Suggested Learning Rate"))
        lines.append(self.plot_ascii_histogram(suggested_lrs, "Learning Rate Distribution"))
        
        # 统计
        lines.append("\n" + "-"*70)
        lines.append(" Learning Rate Statistics ".center(70))
        lines.append("-"*70)
        lines.append(f"Mean LR: {np.mean(suggested_lrs):.4f}")
        lines.append(f"Std LR: {np.std(suggested_lrs):.4f}")
        lines.append(f"Min LR: {min(suggested_lrs):.4f}")
        lines.append(f"Max LR: {max(suggested_lrs):.4f}")
        
        return "\n".join(lines)
    
    def generate_full_report(self) -> str:
        """生成完整报告"""
        lines = []
        lines.append("\n" + "="*70)
        lines.append(" SEL-Lab Foundations Research Report ".center(70))
        lines.append("="*70)
        
        # 基本信息
        config = self.results.get('config', {})
        lines.append(f"\nEpochs: {config.get('epochs', 'N/A')}")
        lines.append(f"Module Count: {self.results.get('summary', {}).get('module_count', 'N/A')}")
        
        # 各个部分
        lines.append(self.visualize_info_theory())
        lines.append(self.visualize_complex_systems())
        lines.append(self.visualize_energy())
        lines.append(self.visualize_bayesian())
        
        lines.append("\n" + "="*70)
        lines.append(" End of Report ".center(70))
        lines.append("="*70)
        
        return "\n".join(lines)
    
    def save_report(self, output_path: Optional[str] = None):
        """保存报告到文件"""
        report = self.generate_full_report()
        
        if output_path is None:
            output_path = resolve_results_path("foundations_visualization_report.txt")
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {output_path}")
        return output_path
    
    def display_report(self):
        """显示报告"""
        print(self.generate_full_report())


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize SEL foundations research results')
    parser.add_argument('--input', type=str, help='Input results file path')
    parser.add_argument('--output', type=str, help='Output report file path')
    parser.add_argument('--display', action='store_true', help='Display report to console')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = FoundationsVisualizer(results_path=args.input)
    
    # 显示报告
    if args.display or not args.output:
        visualizer.display_report()
    
    # 保存报告
    if args.output:
        visualizer.save_report(args.output)
    else:
        visualizer.save_report()


if __name__ == "__main__":
    main()
