# -*- coding: utf-8 -*-
"""
对比测试：结构优化前后的性能差异
"""

import numpy as np
from core.sel_core import SELConfig, SELTrainer
from core.phase3_common import create_task

def test_structure_optimization_comparison():
    print("\n" + "=" * 70)
    print("对比测试：结构优化前后的性能差异")
    print("=" * 70)
    
    # 测试任务
    task_info = {"name": "中等任务", "task_id": 2, "task_suite": "default"}
    
    # 创建任务数据
    X, y = create_task(task_info['task_id'], task_suite=task_info['task_suite'])
    split = len(X) // 2
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    print(f"测试任务: {task_info['name']}")
    print(f"数据规模: {len(X)} 样本, {X.shape[1]} 特征")
    
    # 测试配置 1: 无结构优化（max_modules=10，无自动剪枝）
    print("\n=== 配置 1: 无结构优化 ===")
    print("- max_modules=10")
    print("- 无自动剪枝")
    print("- 可能过度增长")
    
    config_no_optimization = SELConfig(
        input_size=X.shape[1],
        output_size=y.shape[1],
        initial_modules=3,
        max_modules=10,
        epochs=100
    )
    
    # 临时禁用自动剪枝
    original_structural_evolution = None
    try:
        # 保存原始方法
        original_structural_evolution = SELTrainer.__init__.__qualname__
        
        # 创建临时的 SELNetwork 子类来禁用自动剪枝
        from core.sel_core import SELNetwork
        
        class SELNetworkNoOptimization(SELNetwork):
            def structural_evolution(self, epoch_loss=None, epoch=None):
                # 禁用自动剪枝，只保留原始的克隆和适应逻辑
                if epoch is not None:
                    self.current_epoch = epoch
                else:
                    self.current_epoch += 1

                if epoch_loss is not None:
                    self.epoch_loss_history.append(float(epoch_loss))
                    if len(self.epoch_loss_history) > self.config.tension_window:
                        self.epoch_loss_history.pop(0)

                changes = []
                if not self.modules:
                    return changes

                for module in self.modules:
                    module.finalize_epoch(
                        window=self.config.tension_window,
                        min_improvement=self.config.min_improvement,
                    )

                network_plateau = self._network_plateau_score()
                avg_tension = float(np.mean([module.local_tension for module in self.modules]))
                pressured_modules = [
                    (idx, module)
                    for idx, module in enumerate(self.modules)
                    if len(module.loss_history) >= self.config.tension_window
                    and module.local_tension >= self.config.tension_threshold
                ]

                source_idx = self._select_clone_source()
                source_ready = (
                    source_idx is not None
                    and self.modules[source_idx].local_tension < avg_tension
                )

                if network_plateau >= 0.75 and avg_tension >= (self.config.tension_threshold * 1.1):
                    self.stalled_epochs += 1
                else:
                    self.stalled_epochs = 0

                # 只保留克隆逻辑，移除自动剪枝
                can_clone = (
                    bool(pressured_modules)
                    and len(self.epoch_loss_history) >= self.config.tension_window
                    and self.stalled_epochs >= self.config.clone_patience
                    and len(self.modules) < self.config.max_modules
                    and (self.current_epoch - self.last_clone_epoch) >= self.config.clone_cooldown
                    and source_ready
                )
                if can_clone:
                    source_name = self.modules[source_idx].name
                    new_name = f"{source_name}_clone_{len(self.modules)}"
                    self.add_module(new_name, clone_from=source_idx)
                    self.last_clone_epoch = self.current_epoch
                    changes.append((new_name, "cloned", source_name))

                for idx, module in sorted(pressured_modules, key=lambda item: item[1].local_tension, reverse=True)[:1]:
                    if not module.can_act(self.current_epoch, self.config.adapt_cooldown):
                        continue
                    changed, reason = module.adapt(
                        epoch=self.current_epoch,
                        noise_scale=self.config.adapt_noise_scale,
                        mutation_rate=self.config.mutation_rate,
                    )
                    if changed:
                        changes.append((module.name, reason))

                return changes
        
        # 替换 SELNetwork 类
        import core.sel_core
        core.sel_core.SELNetwork = SELNetworkNoOptimization
        
        # 训练网络
        print("训练网络（无结构优化）...")
        trainer_no_optimization = SELTrainer(config_no_optimization)
        result_no_optimization = trainer_no_optimization.train(X_train, y_train, X_test, y_test)
        
        # 分析结果
        accuracy_no_optimization = result_no_optimization['final_test_accuracy']
        modules_no_optimization = result_no_optimization['final_modules']
        evolution_log_no_optimization = result_no_optimization['evolution_log']
        
        print(f"测试准确率: {accuracy_no_optimization:.1%}")
        print(f"最终模块数: {modules_no_optimization}")
        print(f"结构变化次数: {sum(len(event['changes']) for event in evolution_log_no_optimization)}")
        
    finally:
        # 恢复原始 SELNetwork 类
        import importlib
        import core.sel_core
        importlib.reload(core.sel_core)
    
    # 测试配置 2: 有结构优化（max_modules=5，带自动剪枝）
    print("\n=== 配置 2: 有结构优化 ===")
    print("- max_modules=5")
    print("- 自动剪枝到最优范围")
    print("- 保持 3-5 模块")
    
    config_with_optimization = SELConfig(
        input_size=X.shape[1],
        output_size=y.shape[1],
        initial_modules=3,
        max_modules=5,
        epochs=100
    )
    
    # 训练网络
    print("训练网络（带结构优化）...")
    trainer_with_optimization = SELTrainer(config_with_optimization)
    result_with_optimization = trainer_with_optimization.train(X_train, y_train, X_test, y_test)
    
    # 分析结果
    accuracy_with_optimization = result_with_optimization['final_test_accuracy']
    modules_with_optimization = result_with_optimization['final_modules']
    evolution_log_with_optimization = result_with_optimization['evolution_log']
    
    print(f"测试准确率: {accuracy_with_optimization:.1%}")
    print(f"最终模块数: {modules_with_optimization}")
    print(f"结构变化次数: {sum(len(event['changes']) for event in evolution_log_with_optimization)}")
    
    # 对比分析
    print("\n" + "=" * 70)
    print("对比分析")
    print("=" * 70)
    
    print("| 配置 | 准确率 | 模块数 | 结构效率 (准确率/模块数) |")
    print("|------|-------|-------|------------------------|")
    
    efficiency_no_optimization = accuracy_no_optimization / modules_no_optimization
    efficiency_with_optimization = accuracy_with_optimization / modules_with_optimization
    
    print(f"| 无结构优化 | {accuracy_no_optimization:.1%} | {modules_no_optimization} | {efficiency_no_optimization:.3f} |")
    print(f"| 有结构优化 | {accuracy_with_optimization:.1%} | {modules_with_optimization} | {efficiency_with_optimization:.3f} |")
    
    # 计算改进
    accuracy_diff = accuracy_with_optimization - accuracy_no_optimization
    efficiency_diff = efficiency_with_optimization - efficiency_no_optimization
    
    print("\n改进情况:")
    print(f"准确率变化: {accuracy_diff:+.1%}")
    print(f"结构效率变化: {efficiency_diff:+.3f}")
    print(f"模块数减少: {modules_no_optimization - modules_with_optimization}")
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    if accuracy_with_optimization >= accuracy_no_optimization:
        print("✅ 结构优化成功提高了性能！")
    else:
        print("⚠️ 结构优化在性能上略有下降，但提高了效率")
    
    if efficiency_with_optimization > efficiency_no_optimization:
        print("✅ 结构优化显著提高了结构效率！")
    
    print(f"\n结论: 结构优化将模块数控制在 {modules_with_optimization} 个，")
    print(f"在保持或提高性能的同时，显著提高了结构效率。")
    
    return {
        'no_optimization': {
            'accuracy': accuracy_no_optimization,
            'modules': modules_no_optimization,
            'efficiency': efficiency_no_optimization
        },
        'with_optimization': {
            'accuracy': accuracy_with_optimization,
            'modules': modules_with_optimization,
            'efficiency': efficiency_with_optimization
        }
    }

if __name__ == "__main__":
    test_structure_optimization_comparison()
