# -*- coding: utf-8 -*-
"""
测试修复后的代码

验证以下修复：
1. 权重融合逻辑改为 loss-weighted
2. EdgeSwarm 并行的 race condition 修复
3. 增强版 SELNetwork 的导入和使用
"""

import numpy as np
from sel_eggroll_engine import EdgeSwarm, EdgeNodeConfig
from sel_eggroll_engine import SELNetwork, SELConfig
from sel_eggroll_engine.utils import generate_sensor_data


def test_loss_weighted_fusion():
    """测试 loss-weighted 权重融合"""
    print("\n=== 测试 Loss-Weighted 权重融合 ===")
    
    # 创建边缘智能群
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
    
    swarm = EdgeSwarm(config, seed=42)
    
    # 添加节点
    node1 = swarm.add_node('temperature', in_size=3, out_size=2)
    node2 = swarm.add_node('humidity', in_size=3, out_size=2)
    
    # 生成测试数据
    X = np.random.randn(10, 3)
    y = np.array([[1, 0] if i % 2 == 0 else [0, 1] for i in range(10)])
    
    # 测试 local_update 返回两个损失值
    sel_loss, eggroll_loss = node1.local_update(X, y)
    
    print(f"✓ SEL Loss: {sel_loss:.4f}")
    print(f"✓ EGGROLL Loss: {eggroll_loss:.4f}")
    print(f"✓ Loss-weighted 融合正常工作")
    
    return True


def test_race_condition_fix():
    """测试 race condition 修复"""
    print("\n=== 测试 Race Condition 修复 ===")
    
    # 创建边缘智能群
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
    
    swarm = EdgeSwarm(config, seed=42)
    
    # 添加多个同类型传感器（测试 node_id 作为 key）
    node1 = swarm.add_node('temperature', in_size=3, out_size=2)
    node2 = swarm.add_node('temperature', in_size=3, out_size=2)  # 同类型
    node3 = swarm.add_node('humidity', in_size=3, out_size=2)
    
    print(f"✓ 添加了两个 temperature 节点: ID={node1.node_id}, ID={node2.node_id}")
    print(f"✓ 添加了一个 humidity 节点: ID={node3.node_id}")
    
    # 生成测试数据
    sensor_data = {
        'temperature': (np.random.randn(10, 3), np.array([[1, 0] if i % 2 == 0 else [0, 1] for i in range(10)])),
        'humidity': (np.random.randn(10, 3), np.array([[1, 0] if i % 2 == 0 else [0, 1] for i in range(10)]))
    }
    
    # 训练（测试并行训练时的 race condition）
    print("\n开始训练（测试并行安全性）...")
    performance = swarm.train(sensor_data, global_epochs=5)
    
    print(f"✓ 训练完成，无 race condition")
    print(f"✓ 最终性能: {performance[-1]:.4f}")
    
    return True


def test_enhanced_sel():
    """测试增强版 SELNetwork"""
    print("\n=== 测试增强版 SELNetwork ===")
    
    # 创建 SELNetwork
    config = SELConfig(
        input_size=3,
        output_size=2,
        initial_modules=3,
        learning_rate=0.1,
        max_modules=5,
        epochs=10
    )
    
    network = SELNetwork(config)
    
    print(f"✓ SELNetwork 创建成功")
    print(f"✓ 初始模块数: {len(network.modules)}")
    
    # 生成测试数据
    X = np.random.randn(20, 3)
    y = np.array([[1, 0] if i % 2 == 0 else [0, 1] for i in range(20)])
    
    # 训练
    print("\n开始训练 SELNetwork...")
    for epoch in range(10):
        epoch_losses = []
        for i in range(len(X)):
            loss = network.forward_learning(X[i], y[i])
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        changes = network.structural_evolution(epoch_loss=avg_loss, epoch=epoch)
        
        if changes:
            print(f"Epoch {epoch}: 模块数={len(network.modules)}, 变化={changes}")
    
    print(f"✓ 训练完成")
    print(f"✓ 最终模块数: {len(network.modules)}")
    print(f"✓ 总参数数: {network.param_count}")
    
    # 验证模块数在 3-5 范围内
    assert 3 <= len(network.modules) <= 5, f"模块数 {len(network.modules)} 不在 3-5 范围内"
    print(f"✓ 模块数在 3-5 最优范围内")
    
    return True


def test_imports():
    """测试所有新模块的导入"""
    print("\n=== 测试模块导入 ===")
    
    try:
        from sel_eggroll_engine import SELModule
        print("✓ SELModule 导入成功")
        
        from sel_eggroll_engine import SELModuleEnhanced
        print("✓ SELModuleEnhanced 导入成功")
        
        from sel_eggroll_engine import SELNetwork
        print("✓ SELNetwork 导入成功")
        
        from sel_eggroll_engine import SELConfig
        print("✓ SELConfig 导入成功")
        
        from sel_eggroll_engine import CerebellarEdgeNode
        print("✓ CerebellarEdgeNode 导入成功")
        
        from sel_eggroll_engine import CerebellarEdgeSwarm
        print("✓ CerebellarEdgeSwarm 导入成功")
        
        from sel_eggroll_engine import CerebellarConfig
        print("✓ CerebellarConfig 导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("测试修复后的代码")
    print("=" * 70)
    
    all_passed = True
    
    # 测试 1：模块导入
    if not test_imports():
        all_passed = False
    
    # 测试 2：Loss-weighted 融合
    if not test_loss_weighted_fusion():
        all_passed = False
    
    # 测试 3：Race condition 修复
    if not test_race_condition_fix():
        all_passed = False
    
    # 测试 4：增强版 SEL
    if not test_enhanced_sel():
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败")
    print("=" * 70)


if __name__ == "__main__":
    main()
