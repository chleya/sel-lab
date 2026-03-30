# -*- coding: utf-8 -*-
"""
性能基准测试脚本

评估SEL + EGGROLL框架在不同配置下的性能表现，包括：
- 不同节点数量的扩展性
- 不同传感器类型的性能
- 能耗与性能的权衡
- 实时检测性能
"""

import numpy as np
import time
import json
from sel_eggroll_engine import OptimizedEdgeSwarm, OptimizedEdgeNodeConfig
from sel_eggroll_engine.utils import generate_sensor_data


def run_benchmark(node_counts, sensor_types, test_samples=100):
    """运行性能基准测试
    
    Args:
        node_counts: 节点数量列表
        sensor_types: 传感器类型列表
        test_samples: 测试样本数
    """
    results = []
    
    for node_count in node_counts:
        print(f"\n=== 测试节点数量: {node_count} ===")
        
        # 生成传感器数据
        sensor_data = generate_sensor_data(n_samples=1000, anomaly_ratio=0.1, seed=42)
        
        # 配置优化的边缘节点
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
        
        # 创建优化的边缘智能群
        swarm = OptimizedEdgeSwarm(config, seed=42)
        
        # 添加节点
        nodes_per_type = node_count // len(sensor_types)
        for sensor_type in sensor_types:
            for i in range(nodes_per_type):
                swarm.add_node(sensor_type, in_size=3, out_size=2)
        
        # 准备训练数据
        train_data = {}
        for sensor_type in sensor_types:
            X, y = sensor_data[sensor_type]
            train_data[sensor_type] = (X, y)
        
        # 训练
        start_time = time.time()
        performance = swarm.train(train_data, global_epochs=10)
        training_time = time.time() - start_time
        
        # 测试
        test_results = swarm.test(train_data, test_samples=test_samples)
        
        # 多模态融合测试
        multimodal_data = {}
        for sensor_type in sensor_types:
            X, y = sensor_data[sensor_type]
            multimodal_data[sensor_type] = (X, y)
        
        # 添加多模态标签
        _, y = sensor_data['temperature']
        multimodal_data['multimodal'] = (None, y)
        
        fusion_accuracy = swarm.multimodal_test(multimodal_data, test_samples=test_samples)
        
        # 能源消耗分析
        total_energy = 0
        sensor_energy = {}
        for sensor_type in sensor_types:
            sensor_energy[sensor_type] = 0
        
        for node in swarm.get_nodes():
            energy = node.get_energy_consumption()
            total_energy += energy
            if node.sensor_type in sensor_energy:
                sensor_energy[node.sensor_type] += energy
        
        # 实时性能测试
        real_time_start = time.time()
        # 测试100次前向传播
        for i in range(100):
            for node in swarm.get_nodes():
                if node.sensor_type in sensor_data:
                    X, _ = sensor_data[node.sensor_type]
                    node.detect_anomaly(X[i:i+1])
        real_time_duration = time.time() - real_time_start
        avg_detection_time = real_time_duration / (100 * len(swarm.get_nodes()))
        
        # 计算平均性能
        avg_performance = np.mean(list(test_results.values()))
        
        # 记录结果
        result = {
            'node_count': node_count,
            'training_time': training_time,
            'avg_performance': avg_performance,
            'fusion_accuracy': fusion_accuracy,
            'total_energy': total_energy,
            'avg_energy_per_node': total_energy / node_count,
            'avg_detection_time_ms': avg_detection_time * 1000,
            'sensor_performance': test_results,
            'sensor_energy': sensor_energy
        }
        
        results.append(result)
        
        print(f"训练时间: {training_time:.3f}s")
        print(f"平均准确率: {avg_performance:.4f}")
        print(f"多模态融合准确率: {fusion_accuracy:.4f}")
        print(f"总能耗: {total_energy:.2f} 单位")
        print(f"平均检测时间: {avg_detection_time * 1000:.3f}ms")
    
    return results


def run_sensor_benchmark():
    """运行传感器类型性能基准测试"""
    print("\n=== 传感器类型性能基准测试 ===")
    
    # 生成传感器数据
    sensor_data = generate_sensor_data(n_samples=1000, anomaly_ratio=0.1, seed=42)
    sensor_types = ['temperature', 'humidity', 'motion', 'door']
    
    results = {}
    
    for sensor_type in sensor_types:
        print(f"\n测试传感器: {sensor_type}")
        
        # 配置优化的边缘节点
        config = OptimizedEdgeNodeConfig(
            sel_alpha=0.1,
            eggroll_rank=3,
            eggroll_population=8,
            learning_rate=0.01,
            communication_freq=4,
            local_epochs=2,
            energy_budget=100.0,
            adaptation_rate=0.1
        )
        
        # 创建优化的边缘智能群
        swarm = OptimizedEdgeSwarm(config, seed=42)
        
        # 添加单个传感器节点
        swarm.add_node(sensor_type, in_size=3, out_size=2)
        
        # 准备训练数据
        train_data = {}
        X, y = sensor_data[sensor_type]
        train_data[sensor_type] = (X, y)
        
        # 训练
        start_time = time.time()
        performance = swarm.train(train_data, global_epochs=20)
        training_time = time.time() - start_time
        
        # 测试
        test_results = swarm.test(train_data, test_samples=100)
        
        # 能源消耗
        node = swarm.get_nodes()[0]
        energy = node.get_energy_consumption()
        
        results[sensor_type] = {
            'accuracy': test_results[sensor_type],
            'training_time': training_time,
            'energy_consumption': energy,
            'performance_history': performance
        }
        
        print(f"准确率: {test_results[sensor_type]:.4f}")
        print(f"训练时间: {training_time:.3f}s")
        print(f"能耗: {energy:.2f} 单位")
    
    return results


def main():
    """主函数"""
    print("性能基准测试")
    print("=" * 70)
    
    # 1. 扩展性测试
    print("\n1. 扩展性测试")
    node_counts = [4, 8, 16, 32]
    sensor_types = ['temperature', 'humidity', 'motion', 'door']
    scalability_results = run_benchmark(node_counts, sensor_types)
    
    # 2. 传感器类型性能测试
    print("\n2. 传感器类型性能测试")
    sensor_results = run_sensor_benchmark()
    
    # 3. 保存基准测试结果
    benchmark_results = {
        'scalability': scalability_results,
        'sensor_performance': sensor_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('performance_benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
    
    print("\n性能基准测试结果已保存为 performance_benchmark_results.json")
    
    # 4. 生成基准测试报告
    print("\n3. 基准测试报告")
    print("=" * 70)
    
    # 扩展性分析
    print("\n扩展性分析:")
    for result in scalability_results:
        print(f"节点数: {result['node_count']}")
        print(f"  训练时间: {result['training_time']:.3f}s")
        print(f"  平均准确率: {result['avg_performance']:.4f}")
        print(f"  多模态融合准确率: {result['fusion_accuracy']:.4f}")
        print(f"  总能耗: {result['total_energy']:.2f} 单位")
        print(f"  平均检测时间: {result['avg_detection_time_ms']:.3f}ms")
    
    # 传感器性能分析
    print("\n传感器性能分析:")
    for sensor_type, result in sensor_results.items():
        print(f"{sensor_type}:")
        print(f"  准确率: {result['accuracy']:.4f}")
        print(f"  训练时间: {result['training_time']:.3f}s")
        print(f"  能耗: {result['energy_consumption']:.2f} 单位")
    
    print("\n" + "=" * 70)
    print("性能基准测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
