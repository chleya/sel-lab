# -*- coding: utf-8 -*-
"""
实际测试：结构-智能洞察的应用效果
"""

import numpy as np
from core.sel_core import SELConfig, SELTrainer
from core.phase3_common import create_task
from exploration.structure_insights import assess_task_complexity, analyze_structure_efficiency

def test_practical_application():
    print("\n" + "=" * 70)
    print("实际测试：结构-智能洞察的应用效果")
    print("=" * 70)
    
    # 测试不同复杂度的任务
    tasks = [
        {"name": "简单任务", "task_id": 0, "task_suite": "default"},
        {"name": "中等任务", "task_id": 2, "task_suite": "default"},
        {"name": "复杂任务", "task_id": 3, "task_suite": "default"},
        {"name": "噪声任务", "task_id": 1, "task_suite": "noisy"},
    ]
    
    all_results = []
    
    for task_info in tasks:
        print(f"\n=== {task_info['name']} ===")
        
        # 创建任务数据
        X, y = create_task(task_info['task_id'], task_suite=task_info['task_suite'])
        split = len(X) // 2
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        # 评估任务复杂度
        task_complexity = assess_task_complexity(X, y)
        print(f"任务复杂度: {task_complexity:.2f}")
        
        # 配置 SEL 网络，使用自动结构优化
        config = SELConfig(
            input_size=X.shape[1],
            output_size=y.shape[1],
            initial_modules=3,
            max_modules=10,  # 允许增长但会被自动剪枝
            epochs=100
        )
        
        # 训练网络
        print("训练网络（带自动结构优化）...")
        trainer = SELTrainer(config)
        result = trainer.train(X_train, y_train, X_test, y_test)
        
        # 分析结果
        final_accuracy = result['final_test_accuracy']
        final_modules = result['final_modules']
        evolution_log = result['evolution_log']
        
        # 分析结构效率
        efficiency = analyze_structure_efficiency(
            trainer.network, 
            performance=final_accuracy, 
            task_complexity=task_complexity
        )
        
        print(f"测试准确率: {final_accuracy:.1%}")
        print(f"最终模块数: {final_modules}")
        print(f"结构效率: {efficiency['efficiency']:.3f}")
        print(f"是否最优: {'✅' if efficiency['is_optimal'] else '❌'}")
        print(f"推荐: {efficiency['recommendation']}")
        
        # 分析进化日志
        pruning_events = [event for event in evolution_log if any('pruned' in change[1] for change in event['changes'])]
        cloning_events = [event for event in evolution_log if any('cloned' in change[1] for change in event['changes'])]
        
        print(f"剪枝事件: {len(pruning_events)}")
        print(f"克隆事件: {len(cloning_events)}")
        
        if pruning_events:
            print("剪枝详情:")
            for event in pruning_events[:2]:  # 只显示前两个剪枝事件
                print(f"  第 {event['epoch']} 轮: {event['changes']}")
        
        # 记录结果
        all_results.append({
            'task_name': task_info['name'],
            'task_complexity': task_complexity,
            'accuracy': final_accuracy,
            'final_modules': final_modules,
            'efficiency': efficiency['efficiency'],
            'is_optimal': efficiency['is_optimal'],
            'pruning_events': len(pruning_events),
            'cloning_events': len(cloning_events)
        })
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("综合测试结果汇总")
    print("=" * 70)
    
    print("| 任务类型 | 复杂度 | 准确率 | 模块数 | 结构效率 | 是否最优 | 剪枝事件 | 克隆事件 |")
    print("|---------|-------|-------|-------|---------|---------|---------|--------|")
    
    for result in all_results:
        optimal = "✅" if result['is_optimal'] else "❌"
        print(f"| {result['task_name']} | {result['task_complexity']:.2f} | {result['accuracy']:.1%} | {result['final_modules']} | {result['efficiency']:.3f} | {optimal} | {result['pruning_events']} | {result['cloning_events']} |")
    
    # 计算平均值
    avg_accuracy = np.mean([r['accuracy'] for r in all_results])
    avg_modules = np.mean([r['final_modules'] for r in all_results])
    avg_efficiency = np.mean([r['efficiency'] for r in all_results])
    optimal_count = sum(1 for r in all_results if r['is_optimal'])
    
    print("\n" + "=" * 70)
    print("平均值统计")
    print("=" * 70)
    print(f"平均准确率: {avg_accuracy:.1%}")
    print(f"平均模块数: {avg_modules:.1f}")
    print(f"平均结构效率: {avg_efficiency:.3f}")
    print(f"最优结构比例: {optimal_count}/{len(all_results)} ({optimal_count/len(all_results)*100:.0%})")
    
    return all_results

if __name__ == "__main__":
    test_practical_application()
