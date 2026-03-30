# -*- coding: utf-8 -*-
"""
边缘计算模块

实现边缘设备的管理和协同学习。
"""

import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from .sel import SELModule, SELConfig
from .eggroll import EGGROLLOptimizer, EGGROLLConfig


@dataclass
class EdgeNodeConfig:
    """边缘节点配置"""
    sel_alpha: float = 0.1
    eggroll_rank: int = 3
    eggroll_population: int = 8
    learning_rate: float = 0.01
    communication_freq: int = 4
    local_epochs: int = 2
    energy_budget: float = 100.0
    adaptation_rate: float = 0.1


class EdgeNode:
    """边缘节点"""
    
    def __init__(self, node_id: int, sensor_type: str, 
                 in_size: int, out_size: int, 
                 config: Optional[EdgeNodeConfig] = None, 
                 seed: Optional[int] = None):
        """初始化边缘节点
        
        Args:
            node_id: 节点ID
            sensor_type: 传感器类型
            in_size: 输入维度
            out_size: 输出维度
            config: 节点配置
            seed: 随机种子
        """
        self.node_id = node_id
        self.sensor_type = sensor_type
        self.rng = np.random.default_rng(seed)
        self.in_size = in_size
        self.out_size = out_size
        self.config = config or EdgeNodeConfig()
        
        # 初始化模型
        sel_config = SELConfig(alpha=self.config.sel_alpha)
        self.sel_module = SELModule(in_size, out_size, sel_config, seed)
        
        eggroll_config = EGGROLLConfig(
            rank=self.config.eggroll_rank,
            population_size=self.config.eggroll_population,
            learning_rate=self.config.learning_rate
        )
        self.eggroll_optimizer = EGGROLLOptimizer(in_size, out_size, eggroll_config, seed)
        
        # 统计信息
        self.local_losses: List[float] = []
        self.accuracy_history: List[float] = []
        self.performance_score: float = 0.0
        self.energy_consumption: float = 0.0
        
        # 异常检测阈值
        self.anomaly_threshold: float = 0.8
        
        # 锁
        self.lock = threading.Lock()
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播
        
        Args:
            X: 输入数据
            
        Returns:
            输出数据
        """
        return self.sel_module.forward(X)
    
    def local_update(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """本地更新
        
        Args:
            X: 输入数据
            y: 真实标签
            
        Returns:
            (sel_loss, eggroll_loss)：SEL 损失和 EGGROLL 损失
        """
        # 计算能耗
        energy_cost = len(X) * self.config.eggroll_population * 0.0001
        self.energy_consumption += energy_cost
        
        # 检查能源预算
        if self.energy_consumption > self.config.energy_budget:
            return float('inf'), float('inf')
        
        # 1. SEL 更新
        sel_loss = self.sel_module.update(X, y)
        
        # 2. EGGROLL 优化
        eggroll_loss = self.eggroll_optimizer.step(X, y)
        
        # 同步权重（基于 loss 的加权融合）
        sel_weights = self.sel_module.get_weights()
        eggroll_weights = self.eggroll_optimizer.get_weights()
        
        # Loss-weighted 融合：损失小的优化器权重更高
        w_sel = 1.0 / (sel_loss + 1e-6)
        w_egg = 1.0 / (eggroll_loss + 1e-6)
        alpha = w_egg / (w_sel + w_egg)  # EGGROLL 权重占比
        
        # 融合权重
        fused_weights = (1 - alpha) * sel_weights + alpha * eggroll_weights
        
        # 更新权重
        with self.lock:
            self.sel_module.set_weights(fused_weights)
            self.eggroll_optimizer.set_weights(fused_weights)
        
        # 计算准确率
        accuracy = self.compute_accuracy(X, y)
        
        # 记录历史
        self.local_losses.append(eggroll_loss)
        self.accuracy_history.append(accuracy)
        
        # 更新性能分数
        if len(self.accuracy_history) > 0:
            self.performance_score = self.accuracy_history[-1]
        
        return sel_loss, eggroll_loss
    
    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率
        
        Args:
            X: 输入数据
            y: 真实标签
            
        Returns:
            准确率
        """
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y, axis=1)
        return float(np.mean(predictions == labels))
    
    def detect_anomaly(self, sensor_data: np.ndarray) -> Tuple[bool, float]:
        """检测异常
        
        Args:
            sensor_data: 传感器数据
            
        Returns:
            (is_anomaly, score)：是否异常、异常分数
        """
        y_pred = self.forward(sensor_data)
        anomaly_score = y_pred[0, 1]
        is_anomaly = anomaly_score > self.anomaly_threshold
        return is_anomaly, anomaly_score
    
    def get_model(self) -> np.ndarray:
        """获取模型权重
        
        Returns:
            权重矩阵
        """
        with self.lock:
            return self.sel_module.get_weights()
    
    def update_model(self, new_weights: np.ndarray, performance_weight: float = 1.0):
        """更新模型
        
        Args:
            new_weights: 新的权重
            performance_weight: 性能权重
        """
        with self.lock:
            # 基于性能的权重融合
            weight = performance_weight / (performance_weight + self.performance_score + 1e-8)
            current_weights = self.sel_module.get_weights()
            fused_weights = (1 - weight) * current_weights + weight * new_weights
            
            # 更新权重
            self.sel_module.set_weights(fused_weights)
            self.eggroll_optimizer.set_weights(fused_weights)
    
    def adapt_parameters(self):
        """自适应调整参数"""
        if len(self.accuracy_history) > 10:
            recent_improvement = self.accuracy_history[-1] - self.accuracy_history[-10]
            
            # 调整学习率
            if recent_improvement > 0:
                self.config.learning_rate *= (1 + self.config.adaptation_rate)
                self.sel_module.config.alpha *= (1 + self.config.adaptation_rate * 0.5)
            else:
                self.config.learning_rate *= (1 - self.config.adaptation_rate)
                self.sel_module.config.alpha *= (1 - self.config.adaptation_rate * 0.5)
            
            # 限制学习率范围
            self.config.learning_rate = max(0.001, min(0.1, self.config.learning_rate))
            self.sel_module.config.alpha = max(0.001, min(0.2, self.sel_module.config.alpha))
            
            # 调整异常检测阈值
            if self.performance_score > 0.8:
                self.anomaly_threshold = max(0.7, self.anomaly_threshold - 0.05)
            elif self.performance_score < 0.6:
                self.anomaly_threshold = min(0.9, self.anomaly_threshold + 0.05)
    
    def get_performance(self) -> float:
        """获取性能分数
        
        Returns:
            性能分数
        """
        return self.performance_score
    
    def get_energy_consumption(self) -> float:
        """获取能源消耗
        
        Returns:
            能源消耗
        """
        return self.energy_consumption


class EdgeSwarm:
    """边缘智能群"""
    
    def __init__(self, config: Optional[EdgeNodeConfig] = None, seed: Optional[int] = 42):
        """初始化边缘智能群
        
        Args:
            config: 节点配置
            seed: 随机种子
        """
        self.rng = np.random.default_rng(seed)
        self.config = config or EdgeNodeConfig()
        self.nodes: List[EdgeNode] = []
        self.global_epoch = 0
        self.swarm_performance: List[float] = []
    
    def add_node(self, sensor_type: str, in_size: int, out_size: int) -> EdgeNode:
        """添加边缘节点
        
        Args:
            sensor_type: 传感器类型
            in_size: 输入维度
            out_size: 输出维度
            
        Returns:
            边缘节点实例
        """
        node_id = len(self.nodes)
        node = EdgeNode(
            node_id=node_id,
            sensor_type=sensor_type,
            in_size=in_size,
            out_size=out_size,
            config=self.config,
            seed=self.rng.integers(0, 10000)
        )
        self.nodes.append(node)
        print(f"✓ 添加 {sensor_type} 传感器节点 (ID: {node_id})")
        return node
    
    def select_best_nodes(self, top_k: int = 2) -> List[EdgeNode]:
        """选择表现最好的节点
        
        Args:
            top_k: 选择数量
            
        Returns:
            表现最好的节点列表
        """
        nodes_with_performance = [(node, node.get_performance()) for node in self.nodes]
        nodes_with_performance.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in nodes_with_performance[:top_k]]
    
    def communicate(self):
        """节点间通信"""
        # 选择表现最好的节点
        best_nodes = self.select_best_nodes(top_k=2)
        
        # 收集最好的模型
        best_models = {node.sensor_type: node.get_model() for node in best_nodes}
        best_performances = {node.sensor_type: node.get_performance() for node in best_nodes}
        
        # 向同类传感器节点广播
        for node in self.nodes:
            if node.sensor_type in best_models:
                model = best_models[node.sensor_type]
                performance = best_performances[node.sensor_type]
                node.update_model(model, performance)
    
    def train(self, sensor_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
              global_epochs: int = 30) -> List[float]:
        """训练边缘智能群
        
        Args:
            sensor_data: 各传感器的数据和标签
            global_epochs: 全局迭代次数
            
        Returns:
            性能历史
        """
        print(f"\n开始训练边缘智能群 ({len(self.nodes)} 节点)")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(global_epochs):
            # 本地训练
            local_threads = []
            local_losses = {}
            local_losses_lock = threading.Lock()  # 保护 local_losses 的锁
            
            def train_node(node):
                if node.sensor_type in sensor_data:
                    X, y = sensor_data[node.sensor_type]
                    losses = []
                    for _ in range(self.config.local_epochs):
                        sel_loss, eggroll_loss = node.local_update(X, y)
                        losses.append(eggroll_loss)
                    # 使用 node_id 作为 key，避免同一 sensor_type 的冲突
                    with local_losses_lock:
                        local_losses[node.node_id] = np.mean(losses)
                    node.adapt_parameters()
            
            # 并行训练
            for node in self.nodes:
                thread = threading.Thread(target=train_node, args=(node,))
                local_threads.append(thread)
                thread.start()
            
            # 等待完成
            for thread in local_threads:
                thread.join()
            
            # 智能通信
            if epoch % self.config.communication_freq == 0:
                self.communicate()
            
            # 计算性能
            avg_performance = np.mean([node.get_performance() for node in self.nodes])
            self.swarm_performance.append(avg_performance)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"全局 Epoch {epoch+1:2d}: 平均准确率 = {avg_performance:.4f}")
                for node in self.nodes:
                    print(f"  {node.sensor_type}: 准确率={node.get_performance():.4f}, "
                          f"阈值={node.anomaly_threshold:.2f}, "
                          f"能耗={node.get_energy_consumption():.2f}")
            
            self.global_epoch += 1
        
        total_time = time.time() - start_time
        print(f"\n训练完成: 时间 = {total_time:.3f}s")
        
        return self.swarm_performance
    
    def test(self, sensor_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
             test_samples: int = 100) -> Dict[str, float]:
        """测试边缘智能群
        
        Args:
            sensor_data: 各传感器的测试数据和标签
            test_samples: 测试样本数
            
        Returns:
            测试结果
        """
        print(f"\n测试边缘智能群 ({test_samples} 样本)")
        print("-" * 70)
        
        results = {}
        
        for node in self.nodes:
            if node.sensor_type in sensor_data:
                X, y = sensor_data[node.sensor_type]
                test_X = X[:test_samples]
                test_y = y[:test_samples]
                
                correct = 0
                for i in range(test_samples):
                    is_anomaly, _ = node.detect_anomaly(test_X[i:i+1])
                    true_anomaly = np.argmax(test_y[i]) == 1
                    if is_anomaly == true_anomaly:
                        correct += 1
                
                accuracy = correct / test_samples
                results[node.sensor_type] = accuracy
                print(f"{node.sensor_type}: 准确率={accuracy:.4f}")
        
        return results
    
    def get_nodes(self) -> List[EdgeNode]:
        """获取所有节点
        
        Returns:
            节点列表
        """
        return self.nodes
    
    def get_performance_history(self) -> List[float]:
        """获取性能历史
        
        Returns:
            性能历史
        """
        return self.swarm_performance
