# -*- coding: utf-8 -*-
"""
测试真实数据集

使用 SKAB 工业异常检测数据集或 UCI Occupancy Detection 数据集测试系统性能。
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sel_eggroll_engine import EdgeSwarm, EdgeNodeConfig
from sel_eggroll_engine import DatasetConfig, load_dataset


def test_skab_dataset():
    """测试 SKAB 工业异常检测数据集"""
    print("\n=== 测试 SKAB 工业异常检测数据集 ===")
    print("=" * 70)
    
    # 配置数据集
    data_config = DatasetConfig(
        name="skab",
        normalize=True,
        random_seed=42
    )
    
    # 加载数据集
    print("\n加载数据集...")
    sensor_data, sensor_labels = load_dataset(data_config)
    
    print("\n数据集信息:")
    for sensor_type, (X, y) in sensor_data.items():
        n_samples = X.shape[0]
        n_anomalies = np.sum(np.argmax(y, axis=1))
        print(f"  {sensor_type}: {n_samples} 样本, {n_anomalies} 异常")
    
    # 配置边缘智能群
    config = EdgeNodeConfig(
        sel_alpha=0.1,
        eggroll_rank=3,
        eggroll_population=8,
        learning_rate=0.01,
        communication_freq=4,
        local_epochs=2,
        energy_budget=100.0,
        adaptation_rate=0.1
    )
    
    # 创建边缘智能群
    swarm = EdgeSwarm(config, seed=42)
    
    # 添加传感器节点
    print("\n添加传感器节点...")
    for sensor_type in sensor_data.keys():
        swarm.add_node(sensor_type, in_size=3, out_size=2)
    
    # 训练
    print("\n开始训练...")
    train_data = {sensor_type: (X, y) for sensor_type, (X, y) in sensor_data.items()}
    performance = swarm.train(train_data, global_epochs=20)
    
    # 测试
    print("\n测试...")
    test_results = swarm.test(train_data, test_samples=100)
    
    # 性能总结
    print("\n性能总结:")
    print("-" * 70)
    
    total_accuracy = 0
    total_energy = 0
    
    for sensor_type, accuracy in test_results.items():
        print(f"{sensor_type}: 准确率={accuracy:.4f}")
        total_accuracy += accuracy
    
    for node in swarm.get_nodes():
        total_energy += node.get_energy_consumption()
    
    avg_accuracy = total_accuracy / len(test_results)
    print(f"\n平均准确率: {avg_accuracy:.4f}")
    print(f"总能耗: {total_energy:.2f} 单位")
    
    return avg_accuracy, total_energy


def test_occupancy_dataset():
    """测试 UCI Occupancy Detection 数据集"""
    print("\n=== 测试 UCI Occupancy Detection 数据集 ===")
    print("=" * 70)
    
    # 配置数据集
    data_config = DatasetConfig(
        name="occupancy",
        normalize=True,
        random_seed=42
    )
    
    # 加载数据集
    print("\n加载数据集...")
    sensor_data, sensor_labels = load_dataset(data_config)
    
    print("\n数据集信息:")
    for sensor_type, (X, y) in sensor_data.items():
        n_samples = X.shape[0]
        n_occupied = np.sum(np.argmax(y, axis=1))
        print(f"  {sensor_type}: {n_samples} 样本, {n_occupied} 占用")
    
    # 配置边缘智能群
    config = EdgeNodeConfig(
        sel_alpha=0.1,
        eggroll_rank=3,
        eggroll_population=8,
        learning_rate=0.01,
        communication_freq=4,
        local_epochs=2,
        energy_budget=100.0,
        adaptation_rate=0.1
    )
    
    # 创建边缘智能群
    swarm = EdgeSwarm(config, seed=42)
    
    # 添加传感器节点
    print("\n添加传感器节点...")
    for sensor_type in sensor_data.keys():
        swarm.add_node(sensor_type, in_size=3, out_size=2)
    
    # 训练
    print("\n开始训练...")
    train_data = {sensor_type: (X, y) for sensor_type, (X, y) in sensor_data.items()}
    performance = swarm.train(train_data, global_epochs=20)
    
    # 测试
    print("\n测试...")
    test_results = swarm.test(train_data, test_samples=100)
    
    # 性能总结
    print("\n性能总结:")
    print("-" * 70)
    
    total_accuracy = 0
    total_energy = 0
    
    for sensor_type, accuracy in test_results.items():
        print(f"{sensor_type}: 准确率={accuracy:.4f}")
        total_accuracy += accuracy
    
    for node in swarm.get_nodes():
        total_energy += node.get_energy_consumption()
    
    avg_accuracy = total_accuracy / len(test_results)
    print(f"\n平均准确率: {avg_accuracy:.4f}")
    print(f"总能耗: {total_energy:.2f} 单位")
    
    return avg_accuracy, total_energy


def main():
    """主函数"""
    print("=" * 70)
    print("测试真实数据集")
    print("=" * 70)
    
    # 测试两个数据集
    print("\n" + "=" * 70)
    print("1. SKAB 工业异常检测数据集")
    print("=" * 70)
    skab_accuracy, skab_energy = test_skab_dataset()
    
    print("\n" + "=" * 70)
    print("2. UCI Occupancy Detection 数据集")
    print("=" * 70)
    occupancy_accuracy, occupancy_energy = test_occupancy_dataset()
    
    # 总体总结
    print("\n" + "=" * 70)
    print("总体总结:")
    print("=" * 70)
    print(f"SKAB 数据集 - 平均准确率: {skab_accuracy:.4f}, 总能耗: {skab_energy:.2f}")
    print(f"Occupancy 数据集 - 平均准确率: {occupancy_accuracy:.4f}, 总能耗: {occupancy_energy:.2f}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

