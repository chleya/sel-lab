#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEL Visualization Example Run
SEL 可视化示例运行
"""

import sys
import os

# 确保路径正确
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.sel_core import SELConfig, SELTrainer, create_task
from visualization.sel_visualizer import VisualizationConfig
from visualization.topology_animator import TopologyAnimator
from visualization.metrics_panel import MetricsPanel
from visualization.experiment_comparator import ExperimentComparator


def example_1_basic_animation():
    """示例 1: 基础拓扑动画"""
    print("\n" + "=" * 50)
    print("Example 1: Topology Animation")
    print("=" * 50)
    
    # 配置
    config = SELConfig(
        input_size=4,
        output_size=2,
        initial_modules=3,
        learning_rate=0.1,
        epochs=50,
        tension_threshold=0.5
    )
    
    # 数据
    X, y = create_task("simple_classification")
    split = len(X) // 2
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # 训练
    trainer = SELTrainer(config)
    result = trainer.train(X_train, y_train, X_test, y_test)
    print(f"Final Test Accuracy: {result['final_test_accuracy']:.1%}")
    
    # 可视化
    viz_config = VisualizationConfig(
        output_dir='./results/visualization',
        frame_interval=300
    )
    
    animator = TopologyAnimator(viz_config)
    
    # 收集历史
    for metrics in trainer.metrics:
        animator.add_history(metrics)
    
    # 保存动画
    animator.create_animation()
    animator.save_animation('example_topology', format='gif')
    
    print("Saved: example_topology.gif")
    animator.close_all()


def example_2_metrics_dashboard():
    """示例 2: 指标仪表板"""
    print("\n" + "=" * 50)
    print("Example 2: Metrics Dashboard")
    print("=" * 50)
    
    config = SELConfig(
        input_size=4, output_size=2,
        initial_modules=3, epochs=100
    )
    
    X, y = create_task("simple_classification")
    split = len(X) // 2
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    trainer = SELTrainer(config)
    trainer.train(X_train, y_train, X_test, y_test)
    
    viz_config = VisualizationConfig(
        output_dir='./results/visualization'
    )
    
    panel = MetricsPanel(viz_config)
    panel.add_training_history(trainer.metrics)
    
    panel.save_dashboard('example_dashboard')
    print("Saved: example_dashboard.png")
    
    panel.close_all()


def example_3_experiment_comparison():
    """示例 3: 多实验对比"""
    print("\n" + "=" * 50)
    print("Example 3: Experiment Comparison")
    print("=" * 50)
    
    viz_config = VisualizationConfig(
        output_dir='./results/visualization'
    )
    
    comparator = ExperimentComparator(viz_config)
    
    for run in range(3):
        config = SELConfig(
            input_size=4, output_size=2,
            initial_modules=3,
            learning_rate=0.05 + run * 0.05,  # 不同的学习率
            epochs=50
        )
        
        X, y = create_task("simple_classification")
        split = len(X) // 2
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        trainer = SELTrainer(config)
        trainer.train(X_train, y_train, X_test, y_test)
        
        comparator.add_experiment(f'LR={config.learning_rate}', trainer.metrics)
        print(f"  LR={config.learning_rate}: {trainer.metrics[-1]['test_accuracy']:.1%}")
    
    comparator.save_comparison('example_comparison')
    print("Saved: example_comparison.png")
    
    comparator.close_all()


def example_4_interactive():
    """示例 4: 交互式可视化"""
    print("\n" + "=" * 50)
    print("Example 4: Interactive Visualization")
    print("=" * 50)
    print("Note: Interactive mode requires GUI environment")
    print("Showing static snapshot instead...")
    
    config = SELConfig(
        input_size=4, output_size=2,
        initial_modules=3, epochs=50
    )
    
    X, y = create_task("simple_classification")
    split = len(X) // 2
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    trainer = SELTrainer(config)
    trainer.train(X_train, y_train, X_test, y_test)
    
    viz_config = VisualizationConfig(
        output_dir='./results/visualization'
    )
    
    animator = TopologyAnimator(viz_config)
    for metrics in trainer.metrics:
        animator.add_history(metrics)
    
    # 创建静态快照
    fig = animator.create_static_snapshot(epoch=25)
    animator.save_figure(fig, 'example_snapshot_epoch25')
    print("Saved: example_snapshot_epoch25.png")
    
    animator.close_all()


def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("SEL Visualization Examples")
    print("=" * 50)
    
    examples = [
        ("Topology Animation", example_1_basic_animation),
        ("Metrics Dashboard", example_2_metrics_dashboard),
        ("Experiment Comparison", example_3_experiment_comparison),
        ("Interactive Demo", example_4_interactive),
    ]
    
    for i, (name, func) in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {name}")
        try:
            func()
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Check ./results/visualization/ for outputs")
    print("=" * 50)


if __name__ == '__main__':
    main()
