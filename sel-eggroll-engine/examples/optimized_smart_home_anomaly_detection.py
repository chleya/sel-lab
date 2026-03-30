# -*- coding: utf-8 -*-
"""
优化的智能家居异常检测示例

演示如何使用优化的 SEL + EGGROLL 框架进行多模态传感器数据的异常检测，
特别针对运动传感器性能进行了优化。
"""

import numpy as np
from sel_eggroll_engine import OptimizedEdgeSwarm, OptimizedEdgeNodeConfig
from sel_eggroll_engine.utils import generate_sensor_data


def main():
    """主函数"""
    print("优化的智能家居异常检测示例")
    print("=" * 70)
    
    # 1. 生成传感器数据
    print("\n1. 生成传感器数据...")
    sensor_data = generate_sensor_data(n_samples=1000, anomaly_ratio=0.1, seed=42)
    
    print("生成的数据:")
    for sensor_type, (X, y) in sensor_data.items():
        if sensor_type != 'multimodal':
            n_anomalies = np.sum(np.argmax(y, axis=1) == 1)
            print(f"  {sensor_type}: {X.shape[0]} 样本, {n_anomalies} 异常")
    
    # 2. 配置优化的边缘节点
    print("\n2. 配置优化的边缘智能群...")
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
    
    # 3. 创建优化的边缘智能群
    swarm = OptimizedEdgeSwarm(config, seed=42)
    
    # 4. 添加传感器节点
    print("\n3. 添加传感器节点...")
    swarm.add_node('temperature', in_size=3, out_size=2)
    swarm.add_node('humidity', in_size=3, out_size=2)
    swarm.add_node('motion', in_size=3, out_size=2)  # 运动传感器会使用特殊优化参数
    swarm.add_node('door', in_size=3, out_size=2)
    
    # 5. 训练边缘智能群
    print("\n4. 训练优化的边缘智能群...")
    # 准备训练数据
    train_data = {}
    for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
        X, y = sensor_data[sensor_type]
        train_data[sensor_type] = (X, y)
    
    # 训练
    performance = swarm.train(train_data, global_epochs=30)
    
    # 6. 测试边缘智能群
    print("\n5. 测试优化的边缘智能群...")
    test_results = swarm.test(train_data, test_samples=100)
    
    # 7. 多模态融合测试
    print("\n6. 多模态融合测试...")
    # 准备多模态测试数据
    multimodal_data = {}
    for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
        X, y = sensor_data[sensor_type]
        multimodal_data[sensor_type] = (X, y)
    
    # 添加多模态标签
    _, y = sensor_data['temperature']
    multimodal_data['multimodal'] = (None, y)
    
    fusion_accuracy = swarm.multimodal_test(multimodal_data, test_samples=100)
    
    # 8. 实时异常检测演示
    print("\n7. 实时异常检测演示...")
    # 随机选择5个样本进行演示
    import random
    random.seed(42)
    sample_indices = random.sample(range(100), 5)
    
    nodes = swarm.get_nodes()
    sensor_nodes = {node.sensor_type: node for node in nodes}
    
    for i, idx in enumerate(sample_indices):
        print(f"\n样本 {i+1} (索引: {idx}):")
        
        # 收集各传感器数据
        sensor_readings = {}
        for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
            X, _ = sensor_data[sensor_type]
            data = X[idx:idx+1]
            sensor_readings[sensor_type] = data
            
            # 检测异常
            node = sensor_nodes[sensor_type]
            is_anomaly, score = node.detect_anomaly(data)
            status = "异常" if is_anomaly else "正常"
            print(f"  {sensor_type}: {data[0][0]:.2f}, 状态: {status}, 异常分数: {score:.4f}, 阈值: {node.anomaly_threshold:.2f}")
    
    # 9. 性能总结
    print("\n8. 性能总结:")
    print("-" * 70)
    print("各传感器准确率:")
    for sensor_type, accuracy in test_results.items():
        print(f"  {sensor_type}: {accuracy:.4f}")
    
    # 计算平均准确率
    avg_accuracy = np.mean(list(test_results.values()))
    print(f"\n平均准确率: {avg_accuracy:.4f}")
    print(f"多模态融合准确率: {fusion_accuracy:.4f}")
    
    # 10. 能源消耗分析
    print("\n9. 能源消耗分析:")
    total_energy = 0
    for node in nodes:
        energy = node.get_energy_consumption()
        total_energy += energy
        print(f"  {node.sensor_type}: {energy:.2f} 单位")
    print(f"总能耗: {total_energy:.2f} 单位")
    
    # 11. 保存模型（可选）
    print("\n10. 保存模型...")
    from sel_eggroll_engine.utils import save_model
    save_model(swarm, 'optimized_smart_home_swarm.pkl')
    print("模型已保存为 optimized_smart_home_swarm.pkl")
    
    print("\n" + "=" * 70)
    print("优化的智能家居异常检测示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
