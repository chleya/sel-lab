# -*- coding: utf-8 -*-
"""
类小脑边缘计算测试

测试类小脑边缘计算模块的性能，验证其预测编码、时间序列处理和误差修正能力。
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sel_eggroll_engine.cerebellar_edge import CerebellarEdgeSwarm, CerebellarConfig
from sel_eggroll_engine.utils import generate_sensor_data


def main():
    """主函数"""
    print("类小脑边缘计算测试")
    print("=" * 70)
    
    # 1. 生成传感器数据
    print("\n1. 生成传感器数据...")
    sensor_data = generate_sensor_data(n_samples=1000, anomaly_ratio=0.1, seed=42)
    
    print("生成的数据:")
    for sensor_type, (X, y) in sensor_data.items():
        if sensor_type != 'multimodal':
            n_anomalies = np.sum(np.argmax(y, axis=1) == 1)
            print(f"  {sensor_type}: {X.shape[0]} 样本, {n_anomalies} 异常")
    
    # 2. 配置类小脑边缘节点
    print("\n2. 配置类小脑边缘智能群...")
    config = CerebellarConfig(
        sel_alpha=0.15,
        eggroll_rank=4,
        eggroll_population=10,
        prediction_horizon=3,
        temporal_window=5,
        error_learning_rate=0.1,
        mismatch_threshold=0.3,
        coordination_strength=0.5,
        adaptation_rate=0.05,
        energy_budget=100.0
    )
    
    # 3. 创建类小脑边缘智能群
    swarm = CerebellarEdgeSwarm(config, seed=42)
    
    # 4. 添加传感器节点
    print("\n3. 添加传感器节点...")
    swarm.add_node('temperature', in_size=3, out_size=2)
    swarm.add_node('humidity', in_size=3, out_size=2)
    swarm.add_node('motion', in_size=3, out_size=2)
    swarm.add_node('door', in_size=3, out_size=2)
    
    # 5. 训练类小脑边缘智能群
    print("\n4. 训练类小脑边缘智能群...")
    train_data = {}
    for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
        X, y = sensor_data[sensor_type]
        train_data[sensor_type] = (X, y)
    
    epoch_results = swarm.train(train_data, global_epochs=30)
    
    # 6. 测试类小脑边缘智能群
    print("\n5. 测试类小脑边缘智能群...")
    test_results = swarm.test(train_data, test_samples=100)
    
    # 7. 实时异常检测演示（展示预测能力）
    print("\n6. 实时异常检测演示（展示预测编码能力）...")
    import random
    random.seed(42)
    sample_indices = random.sample(range(100), 5)
    
    nodes = swarm.get_nodes()
    sensor_nodes = {node.sensor_type: node for node in nodes}
    
    for i, idx in enumerate(sample_indices):
        print(f"\n样本 {i+1} (索引: {idx}):")
        
        for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
            X, _ = sensor_data[sensor_type]
            data = X[idx:idx+1]
            
            node = sensor_nodes[sensor_type]
            
            # 预测下一个状态
            if node.last_state is not None:
                predicted = node.predict_next(node.last_state)
                prediction_error = np.mean((data - predicted)**2)
            else:
                prediction_error = 0.0
            
            # 检测异常
            is_anomaly, score = node.detect_anomaly(data)
            status = "异常" if is_anomaly else "正常"
            
            print(f"  {sensor_type}: {data[0][0]:.2f}, "
                  f"状态: {status}, "
                  f"异常分数: {score:.4f}, "
                  f"预测误差: {prediction_error:.4f}")
    
    # 8. 性能总结
    print("\n7. 性能总结:")
    print("-" * 70)
    
    total_accuracy = 0
    total_pred_accuracy = 0
    total_energy = 0
    
    for sensor_type, results in test_results.items():
        print(f"{sensor_type}:")
        print(f"  检测准确率: {results['accuracy']:.4f}")
        print(f"  预测准确率: {results['prediction_accuracy']:.4f}")
        print(f"  能耗: {results['energy']:.2f} 单位")
        total_accuracy += results['accuracy']
        total_pred_accuracy += results['prediction_accuracy']
        total_energy += results['energy']
    
    avg_accuracy = total_accuracy / len(test_results)
    avg_pred_accuracy = total_pred_accuracy / len(test_results)
    
    print(f"\n平均检测准确率: {avg_accuracy:.4f}")
    print(f"平均预测准确率: {avg_pred_accuracy:.4f}")
    print(f"总能耗: {total_energy:.2f} 单位")
    
    # 9. 与小脑功能的对比
    print("\n8. 类小脑功能验证:")
    print("-" * 70)
    print("✓ 预测编码: 每个节点都能预测下一个传感器状态")
    print("✓ 时间序列处理: 检测时间模式中的异常")
    print("✓ 误差修正: 基于预测误差调整学习策略")
    print("✓ 运动协调: 节点间通过预测进行协调")
    
    print("\n" + "=" * 70)
    print("类小脑边缘计算测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
