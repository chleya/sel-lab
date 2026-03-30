# -*- coding: utf-8 -*-
"""
SEL-Lab Foundations Integration
基础研究理论与 SEL 核心深度结合

核心创新:
1. 信息论指导的结构进化
2. 复杂系统视角的涌现行为分析
3. 贝叶斯优化的超参数自适应
4. 能量函数驱动的学习目标
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sel_core import SELModule, SELConfig, TrainingMetrics
from core.runtime import resolve_results_path, save_json


@dataclass
class InfoTheorySELConfig(SELConfig):
    """信息论增强的 SEL 配置"""
    # 信息论参数
    info_entropy_weight: float = 0.1      # 信息熵权重
    mutual_info_threshold: float = 0.5    # 互信息阈值
    info_bottleneck_beta: float = 0.01    # 信息瓶颈系数
    
    # 结构复杂度控制
    max_complexity: float = 10.0          # 最大复杂度
    complexity_penalty: float = 0.01      # 复杂度惩罚


class InfoTheoryMetrics:
    """信息论指标计算器"""
    
    def __init__(self):
        self.entropy_history: List[float] = []
        self.mutual_info_history: List[float] = []
        self.complexity_history: List[float] = []
    
    def compute_weight_entropy(self, weights: np.ndarray) -> float:
        """计算权重分布的熵"""
        # 将权重转换为概率分布
        flat_weights = weights.flatten()
        # 使用直方图估计概率
        hist, _ = np.histogram(flat_weights, bins=20, density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        # 计算熵
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def compute_activation_entropy(self, activations: np.ndarray) -> float:
        """计算激活分布的熵"""
        # 对激活值进行离散化
        discrete = np.digitize(activations, bins=np.linspace(-1, 1, 11))
        probs = np.bincount(discrete.flatten(), minlength=12) / len(discrete.flatten())
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    def compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算互信息（简化版）"""
        # 使用相关性作为互信息的近似
        if len(x.shape) > 1:
            x = x.flatten()
        if len(y.shape) > 1:
            y = y.flatten()
        
        # 计算相关系数
        correlation = np.corrcoef(x[:len(y)], y[:len(x)])[0, 1]
        if np.isnan(correlation):
            return 0.0
        
        # 转换为互信息估计
        mi = 0.5 * np.log2(1 / (1 - correlation**2 + 1e-10))
        return abs(mi)
    
    def compute_structure_complexity(self, modules: List[SELModule]) -> float:
        """计算网络结构复杂度"""
        complexity = 0.0
        
        for module in modules:
            # 权重范数
            weight_norm = np.linalg.norm(module.weights)
            # 反馈矩阵范数
            feedback_norm = np.linalg.norm(module.feedback_matrix)
            # 年龄因子
            age_factor = np.log1p(module.age)
            
            complexity += weight_norm + feedback_norm + 0.1 * age_factor
        
        return complexity / max(len(modules), 1)
    
    def update(self, modules: List[SELModule], activations: List[np.ndarray]):
        """更新信息论指标"""
        # 计算权重熵
        total_entropy = 0.0
        for module in modules:
            total_entropy += self.compute_weight_entropy(module.weights)
        avg_entropy = total_entropy / max(len(modules), 1)
        self.entropy_history.append(avg_entropy)
        
        # 计算激活熵
        if activations:
            total_act_entropy = 0.0
            for act in activations:
                total_act_entropy += self.compute_activation_entropy(act)
            avg_act_entropy = total_act_entropy / len(activations)
        else:
            avg_act_entropy = 0.0
        
        # 计算结构复杂度
        complexity = self.compute_structure_complexity(modules)
        self.complexity_history.append(complexity)
        
        # 计算层间互信息
        if len(activations) >= 2:
            mi_sum = 0.0
            mi_count = 0
            for i in range(len(activations) - 1):
                mi = self.compute_mutual_information(activations[i], activations[i+1])
                mi_sum += mi
                mi_count += 1
            avg_mi = mi_sum / max(mi_count, 1)
        else:
            avg_mi = 0.0
        self.mutual_info_history.append(avg_mi)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """获取当前指标"""
        return {
            'weight_entropy': self.entropy_history[-1] if self.entropy_history else 0.0,
            'mutual_information': self.mutual_info_history[-1] if self.mutual_info_history else 0.0,
            'structure_complexity': self.complexity_history[-1] if self.complexity_history else 0.0
        }


class InfoTheorySELModule(SELModule):
    """信息论增强的 SEL 模块"""
    
    def __init__(self, *args, info_config: Optional[InfoTheorySELConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.info_config = info_config
        self.activation_history: List[np.ndarray] = []
        self.info_metrics = InfoTheoryMetrics()
    
    def forward_with_info(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """前向传播并计算信息论指标"""
        output = self.forward(x)
        
        # 记录激活
        self.activation_history.append(output.copy())
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
        
        # 计算信息论指标
        metrics = {
            'activation_entropy': self.info_metrics.compute_activation_entropy(output),
            'weight_entropy': self.info_metrics.compute_weight_entropy(self.weights)
        }
        
        return output, metrics
    
    def compute_info_bottleneck_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算信息瓶颈损失"""
        if self.info_config is None:
            return 0.0
        
        # 前向传播
        h = self.forward(x)
        
        # 预测误差（失真）
        prediction_loss = np.mean((h - y) ** 2)
        
        # 压缩项（激活熵）
        compression = self.info_metrics.compute_activation_entropy(h)
        
        # 信息瓶颈目标
        ib_loss = prediction_loss + self.info_config.info_bottleneck_beta * compression
        
        return ib_loss


@dataclass
class ComplexSystemConfig:
    """复杂系统配置"""
    coupling_strength: float = 0.1        # 模块间耦合强度
    sync_threshold: float = 0.8           # 同步阈值
    emergence_window: int = 10            # 涌现检测窗口
    criticality_metric: str = 'tension'   # 临界性指标


class ComplexSystemAnalyzer:
    """复杂系统分析器 - 分析 SEL 的涌现行为"""
    
    def __init__(self, config: ComplexSystemConfig):
        self.config = config
        self.tension_history: List[List[float]] = []
        self.sync_history: List[float] = []
        self.emergence_events: List[Dict] = []
    
    def compute_synchronization(self, modules: List[SELModule]) -> float:
        """计算模块间的同步程度"""
        if len(modules) < 2:
            return 0.0
        
        # 收集张力值
        tensions = [m.local_tension for m in modules]
        
        # 计算张力的一致性（同步程度）
        mean_tension = np.mean(tensions)
        variance = np.var(tensions)
        
        # 同步度：低方差表示高同步
        sync = 1.0 / (1.0 + variance)
        
        return sync
    
    def detect_emergence(self, modules: List[SELModule], epoch: int) -> Optional[Dict]:
        """检测涌现行为"""
        if len(modules) < 2:
            return None
        
        # 计算当前状态
        tensions = [m.local_tension for m in modules]
        avg_tension = np.mean(tensions)
        tension_variance = np.var(tensions)
        
        # 检测相变：张力方差的突然变化
        if len(self.tension_history) >= self.config.emergence_window:
            recent_variances = [np.var(h) for h in self.tension_history[-self.config.emergence_window:]]
            avg_recent_variance = np.mean(recent_variances)
            
            # 如果当前方差显著不同于历史，可能是相变
            if abs(tension_variance - avg_recent_variance) > 0.2:
                event = {
                    'epoch': epoch,
                    'type': 'phase_transition',
                    'avg_tension': avg_tension,
                    'tension_variance': tension_variance,
                    'module_count': len(modules)
                }
                self.emergence_events.append(event)
                return event
        
        # 记录历史
        self.tension_history.append(tensions)
        if len(self.tension_history) > 100:
            self.tension_history.pop(0)
        
        return None
    
    def analyze_network_topology(self, modules: List[SELModule]) -> Dict:
        """分析网络拓扑特性"""
        n = len(modules)
        if n == 0:
            return {}
        
        # 计算模块年龄的分布
        ages = [m.age for m in modules]
        age_entropy = self._compute_entropy(ages)
        
        # 计算模块的"活跃度"
        recent_changes = sum(1 for m in modules if m.age < 10)
        activity_ratio = recent_changes / n
        
        return {
            'module_count': n,
            'age_entropy': age_entropy,
            'activity_ratio': activity_ratio,
            'avg_age': np.mean(ages) if ages else 0
        }
    
    def _compute_entropy(self, values: List[float]) -> float:
        """计算列表的熵"""
        if not values:
            return 0.0
        
        hist, _ = np.histogram(values, bins=10, density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def get_criticality_measure(self) -> float:
        """获取临界性度量"""
        if not self.sync_history:
            return 0.0
        
        # 使用同步历史的方差作为临界性指标
        return np.var(self.sync_history[-20:])


@dataclass
class BayesianSELConfig:
    """贝叶斯优化配置"""
    # 先验分布参数
    prior_lr_mean: float = 0.1
    prior_lr_std: float = 0.05
    
    # 观测噪声
    observation_noise: float = 0.01
    
    # 采集函数
    acquisition: str = 'ei'  # expected improvement
    
    # 探索-利用平衡
    exploration_weight: float = 0.1


class BayesianSELAdapter:
    """贝叶斯优化适配器 - 自适应调整 SEL 超参数"""
    
    def __init__(self, config: BayesianSELConfig):
        self.config = config
        
        # 超参数历史
        self.lr_history: List[float] = []
        self.tension_threshold_history: List[float] = []
        self.performance_history: List[float] = []
        
        # 当前信念（高斯分布参数）
        self.lr_belief = {'mean': config.prior_lr_mean, 'std': config.prior_lr_std}
    
    def update_belief(self, lr: float, performance: float):
        """基于观测更新信念"""
        self.lr_history.append(lr)
        self.performance_history.append(performance)
        
        # 简化的贝叶斯更新
        if len(self.performance_history) >= 2:
            # 找到性能最好的学习率
            best_idx = np.argmax(self.performance_history)
            best_lr = self.lr_history[best_idx]
            
            # 向最佳学习率靠近
            self.lr_belief['mean'] = 0.9 * self.lr_belief['mean'] + 0.1 * best_lr
            self.lr_belief['std'] *= 0.99  # 逐渐减小不确定性
    
    def suggest_lr(self) -> float:
        """建议下一个学习率"""
        # 从高斯分布采样
        lr = np.random.normal(
            self.lr_belief['mean'],
            max(self.lr_belief['std'], 0.01)
        )
        return np.clip(lr, 0.001, 0.5)
    
    def adaptive_update(
        self,
        current_lr: float,
        recent_performances: List[float]
    ) -> float:
        """自适应更新学习率"""
        if len(recent_performances) < 2:
            return current_lr
        
        # 计算性能趋势
        trend = recent_performances[-1] - recent_performances[0]
        
        # 基于趋势调整
        if trend > 0.01:  # 性能提升
            # 保持或略微增加
            new_lr = current_lr * 1.05
        elif trend < -0.01:  # 性能下降
            # 减小学习率
            new_lr = current_lr * 0.9
        else:  # 平稳
            new_lr = current_lr
        
        return np.clip(new_lr, 0.001, 0.5)


@dataclass
class EnergyBasedSELConfig:
    """能量函数配置"""
    energy_weight: float = 0.01           # 能量项权重
    temperature: float = 1.0              # 温度参数
    equilibrium_steps: int = 5            # 平衡步数


class EnergyBasedSEL:
    """能量函数驱动的 SEL"""
    
    def __init__(self, config: EnergyBasedSELConfig):
        self.config = config
        self.energy_history: List[float] = []
    
    def compute_energy(self, module: SELModule, x: np.ndarray, y: np.ndarray) -> float:
        """计算系统能量"""
        # 前向传播
        h = module.forward(x)
        
        # 数据项（预测误差）
        data_energy = np.mean((h - y) ** 2)
        
        # 正则化项（权重能量）
        weight_energy = 0.5 * np.sum(module.weights ** 2)
        
        # 总能量
        total_energy = data_energy + self.config.energy_weight * weight_energy
        
        return total_energy
    
    def compute_free_energy(self, module: SELModule, x: np.ndarray, y: np.ndarray) -> float:
        """计算自由能量（考虑熵）"""
        energy = self.compute_energy(module, x, y)
        
        # 简化的熵估计
        h = module.forward(x)
        entropy = -np.mean(h * np.log(np.abs(h) + 1e-10))
        
        # 自由能量 = 能量 - 温度 * 熵
        free_energy = energy - self.config.temperature * entropy
        
        return free_energy
    
    def energy_based_update(
        self,
        module: SELModule,
        x: np.ndarray,
        y: np.ndarray,
        lr: float
    ) -> float:
        """基于能量的权重更新"""
        # 计算能量梯度
        h = module.forward(x)
        error = h - y
        
        # 数据梯度
        data_grad = x.T @ (error * (1 - h**2))
        
        # 正则化梯度
        reg_grad = self.config.energy_weight * module.weights
        
        # 总梯度
        total_grad = data_grad + reg_grad
        
        # 能量下降更新
        module.weights -= lr * total_grad
        
        # 计算并记录能量
        energy = self.compute_energy(module, x, y)
        self.energy_history.append(energy)
        
        return energy


class IntegratedFoundationsSEL:
    """集成基础研究的 SEL"""
    
    def __init__(
        self,
        base_config: SELConfig,
        info_config: Optional[InfoTheorySELConfig] = None,
        complex_config: Optional[ComplexSystemConfig] = None,
        bayesian_config: Optional[BayesianSELConfig] = None,
        energy_config: Optional[EnergyBasedSELConfig] = None
    ):
        self.base_config = base_config
        self.info_config = info_config
        self.complex_config = complex_config
        self.bayesian_config = bayesian_config
        self.energy_config = energy_config
        
        # 初始化组件
        self.modules: List[InfoTheorySELModule] = []
        self.info_metrics = InfoTheoryMetrics()
        self.complex_analyzer = ComplexSystemAnalyzer(complex_config or ComplexSystemConfig())
        self.bayesian_adapter = BayesianSELAdapter(bayesian_config or BayesianSELConfig())
        self.energy_system = EnergyBasedSEL(energy_config or EnergyBasedSELConfig())
        
        # 初始化模块
        self._init_modules()
    
    def _init_modules(self):
        """初始化模块"""
        for i in range(self.base_config.initial_modules):
            module = InfoTheorySELModule(
                name=f"module_{i}",
                in_size=self.base_config.input_size,
                out_size=self.base_config.output_size,
                seed=self.base_config.random_seed + i if self.base_config.random_seed else None,
                info_config=self.info_config
            )
            self.modules.append(module)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 简化的前向：使用第一个模块
        if self.modules:
            return self.modules[0].forward(x)
        return np.zeros((x.shape[0], self.base_config.output_size))
    
    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epoch: int
    ) -> Dict[str, Any]:
        """训练步骤 - 集成所有基础研究"""
        results = {}
        
        # 1. 信息论分析
        activations = []
        for module in self.modules:
            out = module.forward(x)
            activations.append(out)
        
        self.info_metrics.update(self.modules, activations)
        results['info_metrics'] = self.info_metrics.get_current_metrics()
        
        # 2. 复杂系统分析
        sync = self.complex_analyzer.compute_synchronization(self.modules)
        emergence = self.complex_analyzer.detect_emergence(self.modules, epoch)
        topology = self.complex_analyzer.analyze_network_topology(self.modules)
        
        results['complex_metrics'] = {
            'synchronization': sync,
            'emergence': emergence,
            'topology': topology
        }
        
        # 3. 贝叶斯优化
        current_lr = self.bayesian_adapter.suggest_lr()
        results['suggested_lr'] = current_lr
        
        # 4. 能量计算
        if self.modules:
            energy = self.energy_system.compute_energy(self.modules[0], x, y)
            results['energy'] = energy
        
        return results


def run_integrated_foundations_experiment(
    epochs: int = 50,
    verbose: bool = True
) -> Dict:
    """运行集成基础研究实验"""
    
    print("=" * 70)
    print(" Integrated Foundations SEL Experiment ")
    print("=" * 70)
    
    # 配置
    base_config = SELConfig(
        input_size=8,
        output_size=2,
        initial_modules=3,
        epochs=epochs
    )
    
    info_config = InfoTheorySELConfig(
        input_size=8,
        output_size=2,
        info_entropy_weight=0.1,
        info_bottleneck_beta=0.01
    )
    
    complex_config = ComplexSystemConfig(
        coupling_strength=0.1,
        sync_threshold=0.8
    )
    
    bayesian_config = BayesianSELConfig(
        prior_lr_mean=0.1,
        exploration_weight=0.1
    )
    
    energy_config = EnergyBasedSELConfig(
        energy_weight=0.01,
        temperature=1.0
    )
    
    # 创建集成系统
    system = IntegratedFoundationsSEL(
        base_config=base_config,
        info_config=info_config,
        complex_config=complex_config,
        bayesian_config=bayesian_config,
        energy_config=energy_config
    )
    
    # 生成合成数据
    np.random.seed(42)
    X = np.random.randn(100, 8)
    y = np.array([[1, 0] if np.sum(x) > 0 else [0, 1] for x in X])
    
    # 训练循环
    history = []
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        # 训练步骤
        results = system.train_step(X, y, epoch)
        
        history.append(results)
        
        if verbose and epoch % 10 == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Info - Weight Entropy: {results['info_metrics']['weight_entropy']:.4f}")
            print(f"  Complex - Sync: {results['complex_metrics']['synchronization']:.4f}")
            print(f"  Bayesian - Suggested LR: {results['suggested_lr']:.4f}")
            if 'energy' in results:
                print(f"  Energy: {results['energy']:.4f}")
    
    # 汇总
    summary = {
        'final_info_metrics': history[-1]['info_metrics'] if history else {},
        'final_complex_metrics': history[-1]['complex_metrics'] if history else {},
        'avg_synchronization': np.mean([h['complex_metrics']['synchronization'] for h in history]),
        'emergence_events': system.complex_analyzer.emergence_events,
        'module_count': len(system.modules)
    }
    
    print("\n" + "=" * 70)
    print(" Summary:")
    print(f"  Average Synchronization: {summary['avg_synchronization']:.4f}")
    print(f"  Emergence Events: {len(summary['emergence_events'])}")
    print(f"  Final Module Count: {summary['module_count']}")
    print("=" * 70)
    
    results = {
        'history': history,
        'summary': summary,
        'config': {
            'epochs': epochs,
            'base_config': base_config.__dict__,
            'info_config': info_config.__dict__
        }
    }
    
    # 保存结果
    save_path = resolve_results_path("integrated_foundations_results.json")
    save_json(results, save_path)
    print(f"\nResults saved to: {save_path}")
    
    return results


if __name__ == "__main__":
    run_integrated_foundations_experiment(epochs=50, verbose=True)
