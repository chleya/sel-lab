#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEL-Lab Visualization Main Entry
SEL 可视化主入口
"""

import sys
import os

# 添加 sel-lab 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List
import numpy as np
from core.sel_core import SELConfig, SELTrainer, create_task
from visualization import (
    SELVisualizer, TopologyAnimator, 
    MetricsPanel, ExperimentComparator
)


def run_demo(mode: str = 'full'):
    """
    运行演示
    
    Args:
        mode: 'full', 'animation', 'metrics', 'comparison'
    """
    print("\n" + "=" * 60)
    print("SEL-Lab Visualization Demo")
    print("=" * 60)
    
    # 创建可视化器
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    if mode in ['full', 'animation']:
        print("\n[1/3] Training network for animation...")
        run_topology_demo(output_dir)
    
    if mode in ['full', 'metrics']:
        print("\n[2/3] Creating metrics dashboard...")
        run_metrics_demo(output_dir)
    
    if mode in ['full', 'comparison']:
        print("\n[3/3] Running experiment comparison...")
        run_comparison_demo(output_dir)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"Output: {output_dir}")
    print("=" * 60)


def run_topology_demo(output_dir: str):
    """拓扑动画演示"""
    config = SELConfig(
        input_size=4, output_size=2,
        initial_modules=3, epochs=50,
        tension_threshold=0.5
    )
    
    X, y = create_task("simple_classification")
    split = len(X) // 2
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    trainer = SELTrainer(config)
    result = trainer.train(X_train, y_train, X_test, y_test)
    
    # 创建动画器
    from visualization.sel_visualizer import VisualizationConfig
    viz_config = VisualizationConfig(
        output_dir=output_dir,
        frame_interval=300
    )
    animator = TopologyAnimator(viz_config)
    
    # 收集历史
    for metrics in trainer.metrics:
        animator.add_history(metrics)
    
    # 保存 GIF
    print("  Saving topology animation...")
    animator.create_animation()
    animator.save_animation('topology_evolution', format='gif')
    
    # 创建静态快照
    print("  Saving static snapshots...")
    for epoch in [0, 25, 49]:
        fig = animator.create_static_snapshot(epoch)
        animator.save_figure(fig, f'topology_epoch_{epoch}')
    
    animator.close_all()


def run_metrics_demo(output_dir: str):
    """指标面板演示"""
    config = SELConfig(
        input_size=4, output_size=2,
        initial_modules=3, epochs=100
    )
    
    X, y = create_task("simple_classification")
    split = len(X) // 2
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    trainer = SELTrainer(config)
    result = trainer.train(X_train, y_train, X_test, y_test)
    
    # 创建面板
    from visualization.sel_visualizer import VisualizationConfig
    viz_config = VisualizationConfig(output_dir=output_dir)
    panel = MetricsPanel(viz_config)
    panel.add_training_history(trainer.metrics)
    
    print("  Saving metrics dashboard...")
    panel.save_dashboard('training_dashboard')
    
    panel.close_all()


def run_comparison_demo(output_dir: str):
    """对比演示"""
    from visualization.sel_visualizer import VisualizationConfig
    viz_config = VisualizationConfig(output_dir=output_dir)
    comparator = ExperimentComparator(viz_config)
    
    print("  Running 3 experiments...")
    for i in range(3):
        config = SELConfig(
            input_size=4, output_size=2,
            initial_modules=3 + i, epochs=50
        )
        
        X, y = create_task("simple_classification")
        split = len(X) // 2
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        trainer = SELTrainer(config)
        trainer.train(X_train, y_train, X_test, y_test)
        
        comparator.add_experiment(f'Run {i+1}', trainer.metrics)
        print(f"    Run {i+1}: Test Acc={trainer.metrics[-1]['test_accuracy']:.1%}")
    
    print("  Saving comparison dashboard...")
    comparator.save_comparison('experiment_comparison')
    
    comparator.close_all()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SEL Visualization')
    parser.add_argument(
        '--mode', choices=['full', 'animation', 'metrics', 'comparison'],
        default='full', help='Demo mode'
    )
    
    args = parser.parse_args()
    run_demo(args.mode)


if __name__ == '__main__':
    main()
