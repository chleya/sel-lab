# -*- coding: utf-8 -*-
"""
验证不同任务复杂度下的最优模块数
"""

import numpy as np
from core.sel_core import SELConfig, SELTrainer
from core.phase3_common import create_task

def test_task_complexity():
    print("\n" + "=" * 60)
    print("Testing optimal module count across task complexities")
    print("=" * 60)
    
    # 定义不同复杂度的任务
    task_complexities = [
        {"name": "Simple", "task_id": 0, "task_suite": "default"},
        {"name": "Medium", "task_id": 2, "task_suite": "default"},
        {"name": "Complex", "task_id": 3, "task_suite": "default"},
        {"name": "Noisy", "task_id": 1, "task_suite": "noisy"},
    ]
    
    # 测试不同模块数量
    module_counts = [1, 3, 5, 7]
    
    results = []
    
    for task_info in task_complexities:
        print(f"\n=== {task_info['name']} Task ===")
        
        # 创建任务数据
        X, y = create_task(task_info['task_id'], task_suite=task_info['task_suite'])
        split = len(X) // 2
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        task_results = []
        
        for max_modules in module_counts:
            print(f"  Testing max_modules={max_modules}...")
            
            # 配置 SEL 网络
            config = SELConfig(
                input_size=X.shape[1],
                output_size=y.shape[1],
                initial_modules=3,
                max_modules=max_modules,
                epochs=100
            )
            
            # 训练网络
            trainer = SELTrainer(config)
            result = trainer.train(X_train, y_train, X_test, y_test)
            
            # 记录结果
            final_accuracy = result['final_test_accuracy']
            final_modules = result['final_modules']
            print(f"    Accuracy: {final_accuracy:.1%}, Final Modules: {final_modules}")
            
            task_results.append({
                'max_modules': max_modules,
                'final_modules': final_modules,
                'accuracy': final_accuracy
            })
        
        results.append({
            'task_name': task_info['name'],
            'task_id': task_info['task_id'],
            'task_suite': task_info['task_suite'],
            'results': task_results
        })
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("Summary by Task Complexity")
    print("=" * 60)
    
    for task_result in results:
        print(f"\n{task_result['task_name']} Task:")
        best_result = max(task_result['results'], key=lambda x: x['accuracy'])
        print(f"  Best: max_modules={best_result['max_modules']}, accuracy={best_result['accuracy']:.1%}")
        
        # 检查 3-5 模块范围的表现
        optimal_range_results = [r for r in task_result['results'] if 3 <= r['max_modules'] <= 5]
        if optimal_range_results:
            avg_optimal_accuracy = np.mean([r['accuracy'] for r in optimal_range_results])
            print(f"  3-5 modules avg accuracy: {avg_optimal_accuracy:.1%}")
        
        # 检查超出范围的表现
        high_range_results = [r for r in task_result['results'] if r['max_modules'] > 5]
        if high_range_results:
            avg_high_accuracy = np.mean([r['accuracy'] for r in high_range_results])
            print(f"  >5 modules avg accuracy: {avg_high_accuracy:.1%}")
    
    return results

if __name__ == "__main__":
    test_task_complexity()
