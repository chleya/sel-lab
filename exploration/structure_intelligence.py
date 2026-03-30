# -*- coding: utf-8 -*-
"""
SEL-Lab Structure-Intelligence Research
结构即智能 - 探索网络结构与智能能力的深层关系

核心命题: 有什么结构就有什么智能能力

研究方向:
1. 结构复杂度与任务能力的关系
2. 结构拓扑与计算能力的关系
3. 结构动态与适应智能的关系
4. 结构涌现与创造性智能的关系
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path
import sys
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sel_core import SELModule, SELConfig
from core.runtime import resolve_results_path, save_json


@dataclass
class StructureIntelligenceConfig:
    """结构-智能研究配置"""
    # 结构复杂度范围
    min_modules: int = 1
    max_modules: int = 20
    module_step: int = 2
    
    # 连接密度范围
    min_density: float = 0.1
    max_density: float = 1.0
    density_step: float = 0.2
    
    # 任务复杂度范围
    task_complexities: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    
    # 评估指标
    n_trials: int = 5
    epochs_per_trial: int = 50


class StructureMetrics:
    """结构度量 - 量化网络结构的各个方面"""
    
    @staticmethod
    def compute_connectivity_matrix(modules: List[SELModule]) -> np.ndarray:
        """计算模块间的连接矩阵"""
        n = len(modules)
        if n == 0:
            return np.array([])
        
        # 简化的连接度量：基于权重相似性
        connectivity = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 计算权重向量的相关性
                    w_i = modules[i].weights.flatten()
                    w_j = modules[j].weights.flatten()
                    
                    # 归一化
                    if np.linalg.norm(w_i) > 0 and np.linalg.norm(w_j) > 0:
                        corr = np.corrcoef(w_i[:100], w_j[:100])[0, 1] if len(w_i) >= 100 else 0
                        connectivity[i, j] = abs(corr)
        
        return connectivity
    
    @staticmethod
    def compute_graph_complexity(connectivity: np.ndarray) -> Dict[str, float]:
        """计算图复杂度指标"""
        if connectivity.size == 0:
            return {'density': 0, 'clustering': 0, 'path_length': 0}
        
        n = connectivity.shape[0]
        
        # 连接密度
        density = np.sum(connectivity > 0.5) / (n * (n - 1)) if n > 1 else 0
        
        # 聚类系数（简化版）
        clustering = 0
        for i in range(n):
            neighbors = np.where(connectivity[i] > 0.5)[0]
            if len(neighbors) > 1:
                # 计算邻居间的连接
                neighbor_connections = 0
                for j in neighbors:
                    for k in neighbors:
                        if j < k and connectivity[j, k] > 0.5:
                            neighbor_connections += 1
                possible = len(neighbors) * (len(neighbors) - 1) / 2
                if possible > 0:
                    clustering += neighbor_connections / possible
        clustering = clustering / n if n > 0 else 0
        
        # 平均路径长度（简化版）
        path_length = 0
        if n > 1:
            # 使用连接矩阵的逆作为距离估计
            distances = 1 / (connectivity + 1e-10)
            np.fill_diagonal(distances, 0)
            path_length = np.mean(distances[distances > 0])
        
        return {
            'density': density,
            'clustering': clustering,
            'path_length': path_length
        }
    
    @staticmethod
    def compute_information_capacity(modules: List[SELModule]) -> float:
        """计算信息容量 - 网络能存储多少信息"""
        total_capacity = 0
        
        for module in modules:
            # 基于权重矩阵的秩估计容量
            weights = module.weights
            try:
                # 使用奇异值分解估计有效维度
                u, s, vh = np.linalg.svd(weights)
                # 有效秩（显著奇异值的数量）
                effective_rank = np.sum(s > 0.01 * s[0])
                total_capacity += effective_rank
            except:
                total_capacity += weights.shape[1]
        
        return total_capacity
    
    @staticmethod
    def compute_computational_depth(modules: List[SELModule]) -> float:
        """计算计算深度 - 网络的计算能力"""
        if not modules:
            return 0
        
        # 基于模块年龄和复杂度的计算深度
        depths = []
        for module in modules:
            # 年龄代表经验
            age_factor = np.log1p(module.age)
            # 权重复杂度
            weight_norm = np.linalg.norm(module.weights)
            # 反馈矩阵复杂度
            feedback_norm = np.linalg.norm(module.feedback_matrix)
            
            depth = age_factor * (weight_norm + feedback_norm)
            depths.append(depth)
        
        return np.mean(depths)


class IntelligenceMetrics:
    """智能度量 - 量化网络的智能能力"""
    
    @staticmethod
    def evaluate_learning_efficiency(
        module: SELModule,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20
    ) -> Dict[str, float]:
        """评估学习效率"""
        initial_loss = None
        final_loss = None
        losses = []
        
        for epoch in range(epochs):
            # 简化的训练
            h = module.forward(X)
            loss = np.mean((h - y) ** 2)
            losses.append(loss)
            
            if epoch == 0:
                initial_loss = loss
            
            # 更新
            error = h - y
            grad = X.T @ (error * (1 - h**2))
            module.weights -= 0.1 * grad
        
        final_loss = losses[-1]
        
        # 计算学习效率指标
        improvement = (initial_loss - final_loss) / (initial_loss + 1e-10)
        convergence_speed = next((i for i, l in enumerate(losses) if l < 0.1 * initial_loss), epochs) / epochs
        stability = 1.0 / (1.0 + np.std(losses[-5:]))
        
        return {
            'improvement': improvement,
            'convergence_speed': convergence_speed,
            'stability': stability,
            'final_loss': final_loss
        }
    
    @staticmethod
    def evaluate_generalization(
        module: SELModule,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> float:
        """评估泛化能力"""
        # 在训练集上训练
        for _ in range(30):
            h = module.forward(X_train)
            error = h - y_train
            grad = X_train.T @ (error * (1 - h**2))
            module.weights -= 0.1 * grad
        
        # 在测试集上评估
        h_test = module.forward(X_test)
        test_loss = np.mean((h_test - y_test) ** 2)
        
        # 归一化到 0-1 范围（越低越好）
        generalization = 1.0 / (1.0 + test_loss)
        
        return generalization
    
    @staticmethod
    def evaluate_adaptability(
        module: SELModule,
        tasks: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """评估适应能力 - 快速学习新任务的能力"""
        adaptation_scores = []
        
        for X, y in tasks:
            # 保存当前权重
            original_weights = module.weights.copy()
            
            # 快速适应新任务
            initial_loss = np.mean((module.forward(X) - y) ** 2)
            
            for _ in range(10):  # 快速适应
                h = module.forward(X)
                error = h - y
                grad = X.T @ (error * (1 - h**2))
                module.weights -= 0.2 * grad
            
            final_loss = np.mean((module.forward(X) - y) ** 2)
            
            # 恢复权重
            module.weights = original_weights
            
            # 计算适应效率
            if initial_loss > 0:
                adaptation = (initial_loss - final_loss) / initial_loss
                adaptation_scores.append(adaptation)
        
        return np.mean(adaptation_scores) if adaptation_scores else 0
    
    @staticmethod
    def evaluate_creativity(
        module: SELModule,
        X: np.ndarray,
        n_perturbations: int = 10
    ) -> float:
        """评估创造性 - 产生新颖输出的能力"""
        original_output = module.forward(X)
        
        novelties = []
        for _ in range(n_perturbations):
            # 扰动输入
            X_perturbed = X + np.random.randn(*X.shape) * 0.1
            perturbed_output = module.forward(X_perturbed)
            
            # 计算新颖性（与原始输出的差异）
            novelty = np.mean(np.abs(perturbed_output - original_output))
            novelties.append(novelty)
        
        return np.mean(novelties)


class StructureIntelligenceExperiment:
    """结构-智能关系实验"""
    
    def __init__(self, config: StructureIntelligenceConfig):
        self.config = config
        self.structure_metrics = StructureMetrics()
        self.intelligence_metrics = IntelligenceMetrics()
        self.results: List[Dict] = []
    
    def create_network_with_structure(
        self,
        n_modules: int,
        connectivity_density: float,
        input_size: int,
        output_size: int
    ) -> List[SELModule]:
        """创建具有特定结构的网络"""
        modules = []
        
        for i in range(n_modules):
            module = SELModule(
                name=f"module_{i}",
                in_size=input_size,
                out_size=output_size,
                seed=42 + i
            )
            modules.append(module)
        
        # 根据密度调整模块间的连接（通过权重初始化模拟）
        for i in range(n_modules):
            for j in range(n_modules):
                if i != j and np.random.rand() < connectivity_density:
                    # 模拟连接：使权重更相似
                    modules[i].weights = 0.7 * modules[i].weights + 0.3 * modules[j].weights
        
        return modules
    
    def run_single_experiment(
        self,
        n_modules: int,
        connectivity_density: float,
        task_complexity: int
    ) -> Dict:
        """运行单组实验"""
        # 创建任务
        np.random.seed(42)
        X = np.random.randn(100, task_complexity)
        y = np.array([[1, 0] if np.sum(x) > 0 else [0, 1] for x in X])
        
        X_test = np.random.randn(50, task_complexity)
        y_test = np.array([[1, 0] if np.sum(x) > 0 else [0, 1] for x in X_test])
        
        # 创建网络
        modules = self.create_network_with_structure(
            n_modules, connectivity_density, task_complexity, 2
        )
        
        # 度量结构
        connectivity = self.structure_metrics.compute_connectivity_matrix(modules)
        graph_complexity = self.structure_metrics.compute_graph_complexity(connectivity)
        info_capacity = self.structure_metrics.compute_information_capacity(modules)
        comp_depth = self.structure_metrics.compute_computational_depth(modules)
        
        structure_score = (
            graph_complexity['density'] +
            graph_complexity['clustering'] +
            info_capacity / 10 +
            comp_depth / 10
        ) / 4
        
        # 度量智能（使用第一个模块作为代表）
        if modules:
            # 学习效率
            learning_eff = self.intelligence_metrics.evaluate_learning_efficiency(
                modules[0], X, y, epochs=20
            )
            
            # 泛化能力
            generalization = self.intelligence_metrics.evaluate_generalization(
                modules[0], X, y, X_test, y_test
            )
            
            # 适应能力
            tasks = [
                (np.random.randn(30, task_complexity), 
                 np.array([[1, 0] if np.sum(x) > 0 else [0, 1] for x in np.random.randn(30, task_complexity)]))
                for _ in range(3)
            ]
            adaptability = self.intelligence_metrics.evaluate_adaptability(modules[0], tasks)
            
            # 创造性
            creativity = self.intelligence_metrics.evaluate_creativity(modules[0], X[:10])
            
            intelligence_score = (
                learning_eff['improvement'] +
                generalization +
                adaptability +
                creativity
            ) / 4
        else:
            intelligence_score = 0
            learning_eff = {}
            generalization = 0
            adaptability = 0
            creativity = 0
        
        return {
            'n_modules': n_modules,
            'connectivity_density': connectivity_density,
            'task_complexity': task_complexity,
            'structure': {
                'density': graph_complexity['density'],
                'clustering': graph_complexity['clustering'],
                'path_length': graph_complexity['path_length'],
                'info_capacity': info_capacity,
                'computational_depth': comp_depth,
                'structure_score': structure_score
            },
            'intelligence': {
                'learning_improvement': learning_eff.get('improvement', 0),
                'convergence_speed': learning_eff.get('convergence_speed', 0),
                'stability': learning_eff.get('stability', 0),
                'generalization': generalization,
                'adaptability': adaptability,
                'creativity': creativity,
                'intelligence_score': intelligence_score
            }
        }
    
    def run_full_experiment(self, verbose: bool = True) -> Dict:
        """运行完整实验"""
        print("=" * 70)
        print(" Structure-Intelligence Relationship Experiment ")
        print("=" * 70)
        print(f"\n探索参数范围:")
        print(f"  模块数: {self.config.min_modules} - {self.config.max_modules}")
        print(f"  连接密度: {self.config.min_density} - {self.config.max_density}")
        print(f"  任务复杂度: {self.config.task_complexities}")
        print(f"  每组实验重复: {self.config.n_trials} 次")
        
        all_results = []
        
        # 遍历参数空间
        for n_modules in range(
            self.config.min_modules,
            self.config.max_modules + 1,
            self.config.module_step
        ):
            for density in np.arange(
                self.config.min_density,
                self.config.max_density + self.config.density_step,
                self.config.density_step
            ):
                for task_complexity in self.config.task_complexities:
                    if verbose:
                        print(f"\n测试: modules={n_modules}, density={density:.2f}, task={task_complexity}")
                    
                    # 多次试验取平均
                    trial_results = []
                    for trial in range(self.config.n_trials):
                        result = self.run_single_experiment(
                            n_modules, density, task_complexity
                        )
                        trial_results.append(result)
                    
                    # 平均结果
                    avg_result = {
                        'n_modules': n_modules,
                        'connectivity_density': density,
                        'task_complexity': task_complexity,
                        'structure': {
                            'structure_score': np.mean([r['structure']['structure_score'] for r in trial_results]),
                            'density': np.mean([r['structure']['density'] for r in trial_results]),
                            'info_capacity': np.mean([r['structure']['info_capacity'] for r in trial_results])
                        },
                        'intelligence': {
                            'intelligence_score': np.mean([r['intelligence']['intelligence_score'] for r in trial_results]),
                            'learning_improvement': np.mean([r['intelligence']['learning_improvement'] for r in trial_results]),
                            'generalization': np.mean([r['intelligence']['generalization'] for r in trial_results]),
                            'adaptability': np.mean([r['intelligence']['adaptability'] for r in trial_results]),
                            'creativity': np.mean([r['intelligence']['creativity'] for r in trial_results])
                        }
                    }
                    
                    all_results.append(avg_result)
                    
                    if verbose:
                        print(f"  结构得分: {avg_result['structure']['structure_score']:.4f}")
                        print(f"  智能得分: {avg_result['intelligence']['intelligence_score']:.4f}")
        
        # 分析结构-智能关系
        structure_scores = [r['structure']['structure_score'] for r in all_results]
        intelligence_scores = [r['intelligence']['intelligence_score'] for r in all_results]
        
        # 计算相关性
        if len(structure_scores) > 1:
            correlation = np.corrcoef(structure_scores, intelligence_scores)[0, 1]
        else:
            correlation = 0
        
        # 找到最优配置
        best_idx = np.argmax(intelligence_scores)
        best_config = all_results[best_idx]
        
        summary = {
            'correlation': correlation,
            'best_config': best_config,
            'n_experiments': len(all_results),
            'avg_structure_score': np.mean(structure_scores),
            'avg_intelligence_score': np.mean(intelligence_scores),
            'structure_range': (min(structure_scores), max(structure_scores)),
            'intelligence_range': (min(intelligence_scores), max(intelligence_scores))
        }
        
        print("\n" + "=" * 70)
        print(" Experiment Summary ")
        print("=" * 70)
        print(f"总实验数: {summary['n_experiments']}")
        print(f"结构-智能相关性: {correlation:.4f}")
        print(f"\n最优配置:")
        print(f"  模块数: {best_config['n_modules']}")
        print(f"  连接密度: {best_config['connectivity_density']:.2f}")
        print(f"  任务复杂度: {best_config['task_complexity']}")
        print(f"  结构得分: {best_config['structure']['structure_score']:.4f}")
        print(f"  智能得分: {best_config['intelligence']['intelligence_score']:.4f}")
        print(f"\n平均结构得分: {summary['avg_structure_score']:.4f}")
        print(f"平均智能得分: {summary['avg_intelligence_score']:.4f}")
        print("=" * 70)
        
        return {
            'results': all_results,
            'summary': summary,
            'config': self.config.__dict__
        }


def run_structure_intelligence_experiment():
    """运行结构-智能实验"""
    config = StructureIntelligenceConfig(
        min_modules=1,
        max_modules=10,
        module_step=2,
        min_density=0.2,
        max_density=0.8,
        density_step=0.3,
        task_complexities=[4, 8],
        n_trials=3,
        epochs_per_trial=30
    )
    
    experiment = StructureIntelligenceExperiment(config)
    results = experiment.run_full_experiment(verbose=True)
    
    # 保存结果
    save_path = resolve_results_path("structure_intelligence_results.json")
    save_json(results, save_path)
    print(f"\nResults saved to: {save_path}")
    
    return results


if __name__ == "__main__":
    run_structure_intelligence_experiment()
