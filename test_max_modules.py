# -*- coding: utf-8 -*-
"""
测试 max_modules=5 在 Phase 3 任务上的表现
"""

import numpy as np
from core.sel_core import SELConfig, SELTrainer
from core.phase3_common import create_task

def test_max_modules():
    print("\n" + "=" * 60)
    print("Testing max_modules=5 on Phase 3 tasks")
    print("=" * 60)
    
    # 测试不同任务
    tasks = [0, 1, 2, 3, 4]  # Phase 3 任务
    results = []
    
    for task_id in tasks:
        print(f"\n--- Task {task_id} ---")
        
        # 创建任务数据
        X, y = create_task(task_id, task_suite="default")
        split = len(X) // 2
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        # 配置 SEL 网络，max_modules=5
        config = SELConfig(
            input_size=X.shape[1],
            output_size=y.shape[1],
            initial_modules=3,
            max_modules=5,  # 测试限制
            epochs=100
        )
        
        # 训练网络
        trainer = SELTrainer(config)
        result = trainer.train(X_train, y_train, X_test, y_test)
        
        # 记录结果
        final_accuracy = result['final_test_accuracy']
        final_modules = result['final_modules']
        print(f"  Test Accuracy: {final_accuracy:.1%}")
        print(f"  Final Modules: {final_modules}")
        
        results.append({
            'task_id': task_id,
            'accuracy': final_accuracy,
            'modules': final_modules
        })
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_modules = np.mean([r['modules'] for r in results])
    print(f"Average Test Accuracy: {avg_accuracy:.1%}")
    print(f"Average Final Modules: {avg_modules:.1f}")
    
    return results

if __name__ == "__main__":
    test_max_modules()
