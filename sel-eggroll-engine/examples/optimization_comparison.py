# -*- coding: utf-8 -*-
"""
优化效果对比测试

对比优化前后的性能差异，验证优化的实际效果。
"""

import numpy as np
import time
from sel_eggroll_engine import EdgeSwarm, EdgeNodeConfig
from sel_eggroll_engine import OptimizedEdgeSwarm, OptimizedEdgeNodeConfig
from sel_eggroll_engine.utils import generate_sensor_data


def test_original_swarm():
    """测试原始边缘智能群"""
    print("\n=== 测试原始边缘智能群 ===")
    
    # 生成传感器数据
    sensor_data = generate_sensor_data(n_samples=1000, anomaly_ratio=0.1, seed=42)
    
    # 配置边缘节点
    config = EdgeNodeConfig(
        sel_alpha=0.1,
        eggroll_rank=3,
        eggroll_population=8,
        learning_rate=0.01,
        communication_freq=4,
        local_epochs=2,
        energy_budget=100.0
    )
    
    # 创建边缘智能群
    swarm = EdgeSwarm(config, seed=42)
    
    # 添加传感器节点
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
    performance = swarm.train(train_data, global_epochs=30)
    training_time = time.time() - start_time
    
    # 测试
    test_results = swarm.test(train_data, test_samples=100)
    
    # 多模态融合测试（简化版）
    fusion_correct = 0
    test_samples = 100
    
    for i in range(test_samples):
        # 收集各传感器预测
        predictions = {}
        for node in swarm.get_nodes():
            if node.sensor_type in sensor_data:
                X, _ = sensor_data[node.sensor_type]
                is_anomaly, score = node.detect_anomaly(X[i:i+1])
                predictions[node.sensor_type] = (is_anomaly, score)
        
        # 简单融合：多数投票
        if predictions:
            anomaly_count = sum(1 for is_anomaly, _ in predictions.values() if is_anomaly)
            fused_anomaly = anomaly_count > len(predictions) / 2
            
            # 验证
            _, y = sensor_data['temperature']
            true_anomaly = np.argmax(y[i]) == 1
            if fused_anomaly == true_anomaly:
                fusion_correct += 1
    
    fusion_accuracy = fusion_correct / test_samples if test_samples > 0 else 0.0
    print(f"多模态融合准确率: {fusion_accuracy:.4f}")
    
    # 能源消耗分析
    total_energy = 0
    for node in swarm.get_nodes():
        total_energy += node.get_energy_consumption()
    
    # 实时性能测试
    real_time_start = time.time()
    for i in range(100):
        for node in swarm.get_nodes():
            if node.sensor_type in sensor_data:
                X, _ = sensor_data[node.sensor_type]
                node.detect_anomaly(X[i:i+1])
    real_time_duration = time.time() - real_time_start
    avg_detection_time = real_time_duration / (100 * len(swarm.get_nodes()))
    
    return {
        'training_time': training_time,
        'test_results': test_results,
        'fusion_accuracy': fusion_accuracy,
        'total_energy': total_energy,
        'avg_detection_time_ms': avg_detection_time * 1000
    }


def test_optimized_swarm():
    """测试优化的边缘智能群"""
    print("\n=== 测试优化的边缘智能群 ===")
    
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
        energy_budget=100.0,
        adaptation_rate=0.1
    )
    
    # 创建优化的边缘智能群
    swarm = OptimizedEdgeSwarm(config, seed=42)
    
    # 添加传感器节点
    swarm.add_node('temperature', in_size=3, out_size=2)
    swarm.add_node('humidity', in_size=3, out_size=2)
    swarm.add_node('motion', in_size=3, out_size=2)  # 运动传感器会使用特殊优化参数
    swarm.add_node('door', in_size=3, out_size=2)
    
    # 准备训练数据
    train_data = {}
    for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
        X, y = sensor_data[sensor_type]
        train_data[sensor_type] = (X, y)
    
    # 训练
    start_time = time.time()
    performance = swarm.train(train_data, global_epochs=30)
    training_time = time.time() - start_time
    
    # 测试
    test_results = swarm.test(train_data, test_samples=100)
    
    # 多模态融合测试（增强版）
    print(f"\n多模态融合测试 ({100} 样本)")
    print("-" * 70)
    
    fusion_correct = 0
    test_samples = 100
    
    for i in range(test_samples):
        # 收集各传感器预测
        predictions = {}
        for node in swarm.get_nodes():
            if node.sensor_type in sensor_data:
                X, _ = sensor_data[node.sensor_type]
                is_anomaly, score = node.detect_anomaly(X[i:i+1])
                predictions[node.sensor_type] = (is_anomaly, score)
        
        # 融合预测（带时间戳，模拟真实时间）
        if predictions:
            timestamp = time.time()  # 模拟当前时间
            fused_anomaly, _, _ = swarm.fusion_center.fuse_predictions(predictions, timestamp)
            
            # 验证
            _, y = sensor_data['temperature']
            true_anomaly = np.argmax(y[i]) == 1
            if fused_anomaly == true_anomaly:
                fusion_correct += 1
    
    fusion_accuracy = fusion_correct / test_samples if test_samples > 0 else 0.0
    print(f"多模态融合准确率: {fusion_accuracy:.4f}")
    
    # 能源消耗分析
    total_energy = 0
    for node in swarm.get_nodes():
        total_energy += node.get_energy_consumption()
    
    # 实时性能测试
    real_time_start = time.time()
    for i in range(100):
        for node in swarm.get_nodes():
            if node.sensor_type in sensor_data:
                X, _ = sensor_data[node.sensor_type]
                node.detect_anomaly(X[i:i+1])
    real_time_duration = time.time() - real_time_start
    avg_detection_time = real_time_duration / (100 * len(swarm.get_nodes()))
    
    return {
        'training_time': training_time,
        'test_results': test_results,
        'fusion_accuracy': fusion_accuracy,
        'total_energy': total_energy,
        'avg_detection_time_ms': avg_detection_time * 1000
    }


def main():
    """主函数"""
    print("优化效果对比测试")
    print("=" * 70)
    
    # 测试原始版本
    original_results = test_original_swarm()
    
    # 测试优化版本
    optimized_results = test_optimized_swarm()
    
    # 对比分析
    print("\n=== 优化效果对比 ===")
    print("=" * 70)
    
    # 训练时间
    print("\n1. 训练时间:")
    print(f"   原始版本: {original_results['training_time']:.3f}s")
    print(f"   优化版本: {optimized_results['training_time']:.3f}s")
    time_improvement = (original_results['training_time'] - optimized_results['training_time']) / original_results['training_time'] * 100
    print(f"   提升: {time_improvement:.2f}%")
    
    # 各传感器准确率
    print("\n2. 传感器准确率:")
    for sensor_type in original_results['test_results']:
        original_acc = original_results['test_results'][sensor_type]
        optimized_acc = optimized_results['test_results'][sensor_type]
        improvement = (optimized_acc - original_acc) * 100
        print(f"   {sensor_type}:")
        print(f"     原始: {original_acc:.4f}")
        print(f"     优化: {optimized_acc:.4f}")
        print(f"     提升: {improvement:.2f}%")
    
    # 多模态融合准确率
    print("\n3. 多模态融合准确率:")
    original_fusion = original_results['fusion_accuracy']
    optimized_fusion = optimized_results['fusion_accuracy']
    fusion_improvement = (optimized_fusion - original_fusion) * 100
    print(f"   原始版本: {original_fusion:.4f}")
    print(f"   优化版本: {optimized_fusion:.4f}")
    print(f"   提升: {fusion_improvement:.2f}%")
    
    # 能源消耗
    print("\n4. 能源消耗:")
    original_energy = original_results['total_energy']
    optimized_energy = optimized_results['total_energy']
    energy_improvement = (original_energy - optimized_energy) / original_energy * 100
    print(f"   原始版本: {original_energy:.2f} 单位")
    print(f"   优化版本: {optimized_energy:.2f} 单位")
    print(f"   降低: {energy_improvement:.2f}%")
    
    # 实时检测时间
    print("\n5. 实时检测时间:")
    original_time = original_results['avg_detection_time_ms']
    optimized_time = optimized_results['avg_detection_time_ms']
    time_improvement = (original_time - optimized_time) / original_time * 100
    print(f"   原始版本: {original_time:.3f}ms")
    print(f"   优化版本: {optimized_time:.3f}ms")
    print(f"   提升: {time_improvement:.2f}%")
    
    # 总结
    print("\n" + "=" * 70)
    print("优化效果总结:")
    print("-" * 70)
    print(f"• 训练时间: {'提升' if time_improvement > 0 else '下降'} {abs(time_improvement):.2f}%")
    print(f"• 多模态融合准确率: {'提升' if fusion_improvement > 0 else '下降'} {abs(fusion_improvement):.2f}%")
    print(f"• 能源消耗: {'降低' if energy_improvement > 0 else '增加'} {abs(energy_improvement):.2f}%")
    print(f"• 实时检测时间: {'提升' if time_improvement > 0 else '下降'} {abs(time_improvement):.2f}%")
    
    print("\n" + "=" * 70)
    print("优化效果对比测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
