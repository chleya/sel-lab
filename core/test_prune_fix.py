# -*- coding: utf-8 -*-
"""
快速验证剪枝 bug 修复效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.sel_core import SELConfig, SELNetwork


def test_prune_logic():
    """测试剪枝逻辑是否正确"""
    print("=" * 70)
    print("测试剪枝逻辑修复效果")
    print("=" * 70)
    
    # 创建配置
    config = SELConfig(
        input_size=10,
        output_size=2,
        initial_modules=6,  # 超过 5，触发剪枝
        max_modules=5,
        epochs=10,
        tension_threshold=0.5
    )
    
    # 创建网络
    network = SELNetwork(config)
    
    print(f"\n初始模块数: {len(network.modules)}")
    print(f"最大模块数: {config.max_modules}")
    
    # 生成测试数据
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, size=(100, 2))
    y = y / y.sum(axis=1, keepdims=True)
    
    # 训练几个 epoch，触发结构演化
    print("\n开始训练...")
    for epoch in range(20):
        loss = network.forward_learning(X, y)
        changes = network.structural_evolution(epoch_loss=loss, epoch=epoch)
        
        if changes:
            print(f"Epoch {epoch}: 模块数={len(network.modules)}, 变化={changes}")
            
            # 检查剪枝逻辑
            for change in changes:
                if "pruned" in str(change):
                    print(f"  ✓ 剪枝事件: {change}")
    
    print(f"\n最终模块数: {len(network.modules)}")
    
    # 验证模块数在合理范围内
    if len(network.modules) <= config.max_modules:
        print("✓ 模块数在最大限制内")
    else:
        print(f"✗ 模块数 {len(network.modules)} 超过最大限制 {config.max_modules}")
    
    # 检查张力分布
    tensions = [module.local_tension for module in network.modules]
    print(f"\n模块张力分布:")
    for i, tension in enumerate(tensions):
        print(f"  模块 {i}: 张力 = {tension:.4f}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_prune_logic()
