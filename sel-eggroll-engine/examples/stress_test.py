# -*- coding: utf-8 -*-
"""
SEL + EGGROLL 框架极限测试

测试框架在大规模节点、实时性能、能源消耗等方面的极限性能。
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sel_eggroll_engine import EdgeSwarm, EdgeNodeConfig
from sel_eggroll_engine.utils import generate_sensor_data


def test_large_scale_nodes():
    """测试大规模节点性能"""
    print("\n=== 大规模节点测试 ===")
    
    # 测试不同节点数量
    node_counts = [4, 8, 16, 32, 64]
    training_times = []
    accuracies = []
    
    for node_count in node_counts:
        print(f"\n测试 {node_count} 个节点...")
        
        # 生成数据
        sensor_data = generate_sensor_data(n_samples=500, anomaly_ratio=0.1, seed=42)
        
        # 配置
        config = EdgeNodeConfig(
            sel_alpha=0.1,
            eggroll_rank=2,
            eggroll_population=4,
            learning_rate=0.01,
            communication_freq=2,
            local_epochs=1,
            energy_budget=50.0
        )
        
        # 创建边缘群
        swarm = EdgeSwarm(config, seed=42)
        
        # 添加节点
        sensor_types = ['temperature', 'humidity', 'motion', 'door'] * (node_count // 4 + 1)
        for i in range(node_count):
            sensor_type = sensor_types[i % len(sensor_types)]
            swarm.add_node(sensor_type, in_size=3, out_size=2)
        
        # 准备训练数据
        train_data = {}
        for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
            X, y = sensor_data[sensor_type]
            train_data[sensor_type] = (X, y)
        
        # 训练
        start_time = time.time()
        performance = swarm.train(train_data, global_epochs=10)
        training_time = time.time() - start_time
        
        # 测试
        test_results = swarm.test(train_data, test_samples=50)
        avg_accuracy = np.mean(list(test_results.values()))
        
        training_times.append(training_time)
        accuracies.append(avg_accuracy)
        
        print(f"  训练时间: {training_time:.2f}s")
        print(f"  平均准确率: {avg_accuracy:.4f}")
    
    # 绘制结果
    plt.figure(figsize=(12, 5))
    
    # 训练时间
    plt.subplot(1, 2, 1)
    plt.plot(node_counts, training_times, 'o-', linewidth=2)
    plt.xlabel('节点数量')
    plt.ylabel('训练时间 (s)')
    plt.title('节点数量 vs 训练时间')
    plt.grid(True, alpha=0.3)
    
    # 准确率
    plt.subplot(1, 2, 2)
    plt.plot(node_counts, accuracies, 'o-', linewidth=2, color='orange')
    plt.xlabel('节点数量')
    plt.ylabel('平均准确率')
    plt.title('节点数量 vs 准确率')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('large_scale_nodes_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ 大规模节点测试结果已保存为 large_scale_nodes_test.png")
    
    return node_counts, training_times, accuracies


def test_real_time_performance():
    """测试实时性能"""
    print("\n=== 实时性能测试 ===")
    
    # 生成数据
    sensor_data = generate_sensor_data(n_samples=1000, anomaly_ratio=0.1, seed=42)
    
    # 配置
    config = EdgeNodeConfig()
    swarm = EdgeSwarm(config, seed=42)
    
    # 添加节点
    swarm.add_node('temperature', in_size=3, out_size=2)
    swarm.add_node('humidity', in_size=3, out_size=2)
    swarm.add_node('motion', in_size=3, out_size=2)
    swarm.add_node('door', in_size=3, out_size=2)
    
    # 训练
    train_data = {}
    for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
        X, y = sensor_data[sensor_type]
        train_data[sensor_type] = (X, y)
    
    swarm.train(train_data, global_epochs=20)
    
    # 实时检测延迟测试
    test_samples = 100
    detection_times = []
    
    nodes = swarm.get_nodes()
    sensor_nodes = {node.sensor_type: node for node in nodes}
    
    for i in range(test_samples):
        start_time = time.time()
        
        # 模拟实时检测
        for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
            X, _ = sensor_data[sensor_type]
            data = X[i:i+1]
            node = sensor_nodes[sensor_type]
            node.detect_anomaly(data)
        
        detection_time = time.time() - start_time
        detection_times.append(detection_time)
    
    # 统计
    avg_time = np.mean(detection_times) * 1000  # 转换为毫秒
    max_time = np.max(detection_times) * 1000
    min_time = np.min(detection_times) * 1000
    std_time = np.std(detection_times) * 1000
    
    print(f"\n实时检测性能:")
    print(f"  平均检测时间: {avg_time:.2f} ms")
    print(f"  最大检测时间: {max_time:.2f} ms")
    print(f"  最小检测时间: {min_time:.2f} ms")
    print(f"  标准差: {std_time:.2f} ms")
    
    # 绘制延迟分布
    plt.figure(figsize=(10, 6))
    plt.hist(detection_times, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('检测时间 (s)')
    plt.ylabel('频率')
    plt.title('实时检测延迟分布')
    plt.grid(True, alpha=0.3)
    plt.savefig('real_time_performance_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ 实时性能测试结果已保存为 real_time_performance_test.png")
    
    return avg_time, max_time, min_time, std_time


def test_energy_consumption():
    """测试能源消耗"""
    print("\n=== 能源消耗测试 ===")
    
    # 测试不同负载下的能源消耗
    batch_sizes = [64, 128, 256, 512, 1024]
    energy_consumptions = []
    training_times = []
    
    for batch_size in batch_sizes:
        print(f"\n测试批次大小 {batch_size}...")
        
        # 生成数据
        sensor_data = generate_sensor_data(n_samples=batch_size, anomaly_ratio=0.1, seed=42)
        
        # 配置
        config = EdgeNodeConfig(
            energy_budget=100.0
        )
        swarm = EdgeSwarm(config, seed=42)
        
        # 添加节点
        swarm.add_node('temperature', in_size=3, out_size=2)
        swarm.add_node('humidity', in_size=3, out_size=2)
        swarm.add_node('motion', in_size=3, out_size=2)
        swarm.add_node('door', in_size=3, out_size=2)
        
        # 准备训练数据
        train_data = {}
        for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
            X, y = sensor_data[sensor_type]
            train_data[sensor_type] = (X, y)
        
        # 训练
        start_time = time.time()
        swarm.train(train_data, global_epochs=10)
        training_time = time.time() - start_time
        
        # 计算总能耗
        total_energy = sum(node.get_energy_consumption() for node in swarm.get_nodes())
        energy_per_sample = total_energy / batch_size
        
        energy_consumptions.append(energy_per_sample)
        training_times.append(training_time)
        
        print(f"  总能耗: {total_energy:.2f}")
        print(f"  每样本能耗: {energy_per_sample:.4f}")
        print(f"  训练时间: {training_time:.2f}s")
    
    # 绘制结果
    plt.figure(figsize=(12, 5))
    
    # 每样本能耗
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, energy_consumptions, 'o-', linewidth=2)
    plt.xlabel('批次大小')
    plt.ylabel('每样本能耗')
    plt.title('批次大小 vs 每样本能耗')
    plt.grid(True, alpha=0.3)
    
    # 训练时间
    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, training_times, 'o-', linewidth=2, color='orange')
    plt.xlabel('批次大小')
    plt.ylabel('训练时间 (s)')
    plt.title('批次大小 vs 训练时间')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_consumption_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ 能源消耗测试结果已保存为 energy_consumption_test.png")
    
    return batch_sizes, energy_consumptions, training_times


def test_anomaly_detection():
    """测试异常检测性能"""
    print("\n=== 异常检测性能测试 ===")
    
    # 测试不同异常率下的检测性能
    anomaly_ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    accuracies = []
    
    for anomaly_ratio in anomaly_ratios:
        print(f"\n测试异常率 {anomaly_ratio:.2f}...")
        
        # 生成数据
        sensor_data = generate_sensor_data(n_samples=1000, anomaly_ratio=anomaly_ratio, seed=42)
        
        # 配置
        config = EdgeNodeConfig()
        swarm = EdgeSwarm(config, seed=42)
        
        # 添加节点
        swarm.add_node('temperature', in_size=3, out_size=2)
        swarm.add_node('humidity', in_size=3, out_size=2)
        swarm.add_node('motion', in_size=3, out_size=2)
        swarm.add_node('door', in_size=3, out_size=2)
        
        # 准备训练数据
        train_data = {}
        for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
            X, y = sensor_data[sensor_type]
            train_data[sensor_type] = (X, y)
        
        # 训练
        swarm.train(train_data, global_epochs=20)
        
        # 测试
        test_results = swarm.test(train_data, test_samples=200)
        avg_accuracy = np.mean(list(test_results.values()))
        accuracies.append(avg_accuracy)
        
        print(f"  平均准确率: {avg_accuracy:.4f}")
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(anomaly_ratios, accuracies, 'o-', linewidth=2)
    plt.xlabel('异常率')
    plt.ylabel('平均准确率')
    plt.title('异常率 vs 检测准确率')
    plt.grid(True, alpha=0.3)
    plt.savefig('anomaly_detection_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ 异常检测性能测试结果已保存为 anomaly_detection_test.png")
    
    return anomaly_ratios, accuracies


def main():
    """主函数"""
    print("SEL + EGGROLL 框架极限测试")
    print("=" * 70)
    
    # 测试1: 大规模节点
    node_counts, training_times, accuracies = test_large_scale_nodes()
    
    # 测试2: 实时性能
    avg_time, max_time, min_time, std_time = test_real_time_performance()
    
    # 测试3: 能源消耗
    batch_sizes, energy_consumptions, training_times_energy = test_energy_consumption()
    
    # 测试4: 异常检测性能
    anomaly_ratios, detection_accuracies = test_anomaly_detection()
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结:")
    print("=" * 70)
    
    print("\n1. 大规模节点测试:")
    print(f"   最大节点数: {max(node_counts)}")
    print(f"   最大训练时间: {max(training_times):.2f}s")
    print(f"   最低准确率: {min(accuracies):.4f}")
    
    print("\n2. 实时性能测试:")
    print(f"   平均检测时间: {avg_time:.2f}ms")
    print(f"   最大检测时间: {max_time:.2f}ms")
    
    print("\n3. 能源消耗测试:")
    print(f"   最大批次大小: {max(batch_sizes)}")
    print(f"   最低每样本能耗: {min(energy_consumptions):.4f}")
    
    print("\n4. 异常检测性能测试:")
    print(f"   最高异常率: {max(anomaly_ratios):.2f}")
    print(f"   对应准确率: {detection_accuracies[-1]:.4f}")
    
    print("\n" + "=" * 70)
    print("极限测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
