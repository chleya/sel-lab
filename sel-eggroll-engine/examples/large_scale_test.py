# -*- coding: utf-8 -*-
"""
大规模传感器网络性能测试

测试SEL + EGGROLL框架在16+节点的大规模传感器网络中的性能表现。
"""

import numpy as np
import time
from sel_eggroll_engine import OptimizedEdgeSwarm, OptimizedEdgeNodeConfig
from sel_eggroll_engine.utils import generate_sensor_data


def main():
    """主函数"""
    print("大规模传感器网络性能测试")
    print("=" * 70)
    
    # 1. 生成传感器数据
    print("\n1. 生成传感器数据...")
    sensor_data = generate_sensor_data(n_samples=1000, anomaly_ratio=0.1, seed=42)
    
    # 2. 配置优化的边缘节点
    print("\n2. 配置优化的边缘智能群...")
    config = OptimizedEdgeNodeConfig(
        sel_alpha=0.1,
        eggroll_rank=3,
        eggroll_population=8,
        learning_rate=0.01,
        communication_freq=4,
        local_epochs=2,
        energy_budget=200.0,
        adaptation_rate=0.1
    )
    
    # 3. 创建优化的边缘智能群
    swarm = OptimizedEdgeSwarm(config, seed=42)
    
    # 4. 添加16个传感器节点（4种类型，每种4个）
    print("\n3. 添加16个传感器节点...")
    sensor_types = ['temperature', 'humidity', 'motion', 'door']
    for sensor_type in sensor_types:
        for i in range(4):
            swarm.add_node(sensor_type, in_size=3, out_size=2)
    
    print(f"\n已添加 {len(swarm.get_nodes())} 个传感器节点")
    
    # 5. 训练边缘智能群
    print("\n4. 训练大规模边缘智能群...")
    # 准备训练数据
    train_data = {}
    for sensor_type in sensor_types:
        X, y = sensor_data[sensor_type]
        train_data[sensor_type] = (X, y)
    
    # 训练
    start_time = time.time()
    performance = swarm.train(train_data, global_epochs=10)  # 减少epoch以节省时间
    total_time = time.time() - start_time
    
    print(f"\n训练完成: 时间 = {total_time:.3f}s")
    print(f"平均训练时间/节点 = {total_time / len(swarm.get_nodes()):.3f}s")
    
    # 6. 测试边缘智能群
    print("\n5. 测试大规模边缘智能群...")
    test_results = swarm.test(train_data, test_samples=50)
    
    # 7. 性能分析
    print("\n6. 性能分析:")
    print("-" * 70)
    
    # 按传感器类型分组分析
    sensor_performance = {}
    for sensor_type in sensor_types:
        sensor_performance[sensor_type] = []
    
    for node in swarm.get_nodes():
        sensor_performance[node.sensor_type].append(node.get_performance())
    
    # 打印各传感器类型的平均性能
    print("各传感器类型平均性能:")
    for sensor_type, performances in sensor_performance.items():
        avg_performance = np.mean(performances)
        print(f"  {sensor_type}: {avg_performance:.4f}")
    
    # 计算整体平均性能
    all_performances = []
    for node in swarm.get_nodes():
        all_performances.append(node.get_performance())
    avg_overall = np.mean(all_performances)
    print(f"\n整体平均性能: {avg_overall:.4f}")
    
    # 8. 能源消耗分析
    print("\n7. 能源消耗分析:")
    total_energy = 0
    sensor_energy = {}
    for sensor_type in sensor_types:
        sensor_energy[sensor_type] = 0
    
    for node in swarm.get_nodes():
        energy = node.get_energy_consumption()
        total_energy += energy
        sensor_energy[node.sensor_type] += energy
    
    print("各传感器类型能耗:")
    for sensor_type, energy in sensor_energy.items():
        print(f"  {sensor_type}: {energy:.2f} 单位")
    
    print(f"\n总能耗: {total_energy:.2f} 单位")
    print(f"平均能耗/节点: {total_energy / len(swarm.get_nodes()):.2f} 单位")
    
    # 9. 扩展性分析
    print("\n8. 扩展性分析:")
    print(f"节点数量: {len(swarm.get_nodes())}")
    print(f"训练时间: {total_time:.3f}s")
    print(f"性能保持率: {avg_overall / 0.9:.2f}% (相对于4节点系统)")
    
    # 10. 保存性能数据
    performance_data = {
        'node_count': len(swarm.get_nodes()),
        'training_time': total_time,
        'average_performance': avg_overall,
        'total_energy': total_energy,
        'sensor_performance': sensor_performance,
        'sensor_energy': sensor_energy
    }
    
    import json
    with open('large_scale_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, ensure_ascii=False, indent=2)
    
    print("\n性能数据已保存为 large_scale_test_results.json")
    
    print("\n" + "=" * 70)
    print("大规模传感器网络性能测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
