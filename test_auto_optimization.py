# -*- coding: utf-8 -*-
"""
测试自动结构优化机制
"""

import numpy as np
from core.sel_core import SELConfig, SELTrainer
from core.phase3_common import create_task

def test_auto_optimization():
    print("\n" + "=" * 60)
    print("Testing Automatic Structure Optimization")
    print("=" * 60)
    
    # 测试不同复杂度的任务
    task_info = {"name": "Medium", "task_id": 2, "task_suite": "default"}
    
    # 创建任务数据
    X, y = create_task(task_info['task_id'], task_suite=task_info['task_suite'])
    split = len(X) // 2
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # 配置 SEL 网络，使用较大的 max_modules 来测试剪枝功能
    config = SELConfig(
        input_size=X.shape[1],
        output_size=y.shape[1],
        initial_modules=6,  # 初始模块数超过最优范围
        max_modules=10,     # 允许增长但会被自动剪枝
        epochs=100
    )
    
    print(f"Testing {task_info['name']} task with initial modules=6")
    print(f"Expected: auto-pruning to 3-5 modules range")
    
    # 训练网络
    trainer = SELTrainer(config)
    result = trainer.train(X_train, y_train, X_test, y_test)
    
    # 分析结果
    final_accuracy = result['final_test_accuracy']
    final_modules = result['final_modules']
    evolution_log = result['evolution_log']
    
    print(f"\nResults:")
    print(f"Final Test Accuracy: {final_accuracy:.1%}")
    print(f"Final Modules: {final_modules}")
    print(f"Is in optimal range (3-5): {3 <= final_modules <= 5}")
    
    # 分析进化日志
    print(f"\nEvolution Summary:")
    pruning_events = [event for event in evolution_log if any('pruned' in change[1] for change in event['changes'])]
    cloning_events = [event for event in evolution_log if any('cloned' in change[1] for change in event['changes'])]
    
    print(f"Pruning events: {len(pruning_events)}")
    print(f"Cloning events: {len(cloning_events)}")
    
    if pruning_events:
        print("\nPruning details:")
        for event in pruning_events:
            print(f"  Epoch {event['epoch']}: {event['changes']}")
    
    return result

if __name__ == "__main__":
    test_auto_optimization()
