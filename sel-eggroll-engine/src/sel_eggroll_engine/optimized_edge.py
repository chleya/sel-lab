# -*- coding: utf-8 -*-
"""
优化的边缘计算模块

包含运动传感器优化、多模态融合改进、模型压缩和自适应参数调整。
"""

import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from .sel import SELModule, SELConfig
from .eggroll import EGGROLLOptimizer, EGGROLLConfig
from .fusion import FusionCenter


@dataclass
class OptimizedEdgeNodeConfig:
    """优化的边缘节点配置"""
    sel_alpha: float = 0.1
    eggroll_rank: int = 3
    eggroll_population: int = 8
    learning_rate: float = 0.01
    communication_freq: int = 4
    local_epochs: int = 2
    energy_budget: float = 100.0
    adaptation_rate: float = 0.1
    sensor_specific_params: Dict[str, Dict] = None  # 传感器特定参数


class OptimizedEdgeNode:
    """优化的边缘节点"""
    
    def __init__(self, node_id: int, sensor_type: str, 
                 in_size: int, out_size: int, 
                 config: Optional[OptimizedEdgeNodeConfig] = None, 
                 seed: Optional[int] = None):
        """初始化优化的边缘节点
        
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
        self.config = config or OptimizedEdgeNodeConfig()
        
        # 传感器特定参数
        self.sensor_params = self._get_sensor_specific_params()
        
        # 初始化模型
        sel_config = SELConfig(
            alpha=self.sensor_params.get('sel_alpha', self.config.sel_alpha),
            activation=self.sensor_params.get('activation', 'softmax')
        )
        self.sel_module = SELModule(in_size, out_size, sel_config, seed)
        
        eggroll_config = EGGROLLConfig(
            rank=self.sensor_params.get('eggroll_rank', self.config.eggroll_rank),
            population_size=self.sensor_params.get('eggroll_population', self.config.eggroll_population),
            learning_rate=self.sensor_params.get('learning_rate', self.config.learning_rate)
        )
        self.eggroll_optimizer = EGGROLLOptimizer(in_size, out_size, eggroll_config, seed)
        
        # 统计信息
        self.local_losses: List[float] = []
        self.accuracy_history: List[float] = []
        self.performance_score: float = 0.0
        self.energy_consumption: float = 0.0
        
        # 异常检测阈值（传感器特定）
        self.anomaly_threshold: float = self.sensor_params.get('anomaly_threshold', 0.8)
        
        # 能耗管理
        self.energy_efficiency: float = 1.0  # 能耗效率因子
        self.active_time: float = 0.0  # 活动时间
        self.sleep_probability: float = 0.0  # 休眠概率
        
        # 锁
        self.lock = threading.Lock()
    
    def _get_sensor_specific_params(self) -> Dict:
        """获取传感器特定参数"""
        default_params = {
            'temperature': {
                'sel_alpha': 0.1,
                'anomaly_threshold': 0.75,
                'activation': 'softmax'
            },
            'humidity': {
                'sel_alpha': 0.1,
                'anomaly_threshold': 0.75,
                'activation': 'softmax'
            },
            'motion': {
                'sel_alpha': 0.2,  # 进一步提高学习率
                'anomaly_threshold': 0.6,  # 进一步降低阈值
                'activation': 'sigmoid',  # 保持sigmoid激活
                'eggroll_rank': 5,  # 更高的秩以捕捉复杂模式
                'eggroll_population': 12,  # 更大的种群以提高搜索能力
                'fusion_weight': 0.6  # 增加SEL权重，提高学习速度
            },
            'door': {
                'sel_alpha': 0.1,
                'anomaly_threshold': 0.8,
                'activation': 'softmax'
            }
        }
        
        # 合并配置中的传感器特定参数
        if self.config.sensor_specific_params:
            for sensor_type, params in self.config.sensor_specific_params.items():
                if sensor_type == self.sensor_type:
                    default_params[sensor_type].update(params)
        
        return default_params.get(self.sensor_type, default_params['temperature'])
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        return self.sel_module.forward(X)
    
    def local_update(self, X: np.ndarray, y: np.ndarray) -> float:
        """本地更新"""
        # 检查是否应该休眠
        if self.rng.random() < self.sleep_probability:
            # 休眠模式，返回当前损失
            if self.local_losses:
                return self.local_losses[-1]
            return float('inf')
        
        # 计算能耗（考虑能耗效率因子）
        energy_cost = len(X) * self.eggroll_optimizer.config.population_size * 0.0001 / self.energy_efficiency
        self.energy_consumption += energy_cost
        
        # 检查能源预算
        if self.energy_consumption > self.config.energy_budget:
            # 触发节能模式
            self.enter_power_save_mode()
            return float('inf')
        
        # 1. SEL 更新
        sel_loss = self.sel_module.update(X, y)
        
        # 2. EGGROLL 优化
        eggroll_loss = self.eggroll_optimizer.step(X, y)
        
        # 同步权重
        sel_weights = self.sel_module.get_weights()
        eggroll_weights = self.eggroll_optimizer.get_weights()
        
        # 权重融合（传感器特定的融合权重）
        fusion_weight = self.sensor_params.get('fusion_weight', 0.5)
        fused_weights = fusion_weight * sel_weights + (1 - fusion_weight) * eggroll_weights
        
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
        
        # 更新活动时间
        self.active_time += 1
        
        # 动态调整能耗效率
        self.adjust_energy_efficiency()
        
        return eggroll_loss
    
    def enter_power_save_mode(self):
        """进入节能模式"""
        # 降低学习率
        self.config.learning_rate *= 0.5
        self.sel_module.config.alpha *= 0.5
        
        # 减少EGGROLL种群大小
        if hasattr(self.eggroll_optimizer.config, 'population_size'):
            self.eggroll_optimizer.config.population_size = max(4, self.eggroll_optimizer.config.population_size // 2)
        
        # 增加休眠概率
        self.sleep_probability = min(0.7, self.sleep_probability + 0.2)
        
        # 提高能耗效率
        self.energy_efficiency = min(2.0, self.energy_efficiency + 0.3)
    
    def adjust_energy_efficiency(self):
        """动态调整能耗效率"""
        if len(self.accuracy_history) > 5:
            # 计算最近的性能趋势
            recent_accuracy = self.accuracy_history[-5:]
            trend = recent_accuracy[-1] - recent_accuracy[0]
            
            if trend > 0.05:
                # 性能提升，可适当增加能耗
                self.energy_efficiency = max(0.8, self.energy_efficiency - 0.1)
                self.sleep_probability = max(0.0, self.sleep_probability - 0.1)
            elif trend < -0.05:
                # 性能下降，需要节能
                self.energy_efficiency = min(1.5, self.energy_efficiency + 0.2)
                self.sleep_probability = min(0.5, self.sleep_probability + 0.1)
            elif self.performance_score > 0.9:
                # 性能良好，可以节能
                self.energy_efficiency = min(1.8, self.energy_efficiency + 0.1)
                self.sleep_probability = min(0.4, self.sleep_probability + 0.05)
    
    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y, axis=1)
        return float(np.mean(predictions == labels))
    
    def detect_anomaly(self, sensor_data: np.ndarray) -> Tuple[bool, float]:
        """检测异常"""
        y_pred = self.forward(sensor_data)
        anomaly_score = y_pred[0, 1]
        is_anomaly = anomaly_score > self.anomaly_threshold
        return is_anomaly, anomaly_score
    
    def get_model(self) -> np.ndarray:
        """获取模型权重"""
        with self.lock:
            return self.sel_module.get_weights()
    
    def update_model(self, new_weights: np.ndarray, performance_weight: float = 1.0):
        """更新模型"""
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
            
            # 调整异常检测阈值（传感器特定的调整策略）
            if self.performance_score > 0.8:
                self.anomaly_threshold = max(0.5, self.anomaly_threshold - 0.05)
            elif self.performance_score < 0.6:
                self.anomaly_threshold = min(0.95, self.anomaly_threshold + 0.05)
    
    def get_performance(self) -> float:
        """获取性能分数"""
        return self.performance_score
    
    def get_energy_consumption(self) -> float:
        """获取能源消耗"""
        return self.energy_consumption


class OptimizedEdgeSwarm:
    """优化的边缘智能群"""
    
    def __init__(self, config: Optional[OptimizedEdgeNodeConfig] = None, seed: Optional[int] = 42):
        """初始化优化的边缘智能群"""
        self.rng = np.random.default_rng(seed)
        self.config = config or OptimizedEdgeNodeConfig()
        self.nodes: List[OptimizedEdgeNode] = []
        self.global_epoch = 0
        self.swarm_performance: List[float] = []
        
        # 高级融合中心
        self.fusion_center = FusionCenter()
    
    def add_node(self, sensor_type: str, in_size: int, out_size: int) -> OptimizedEdgeNode:
        """添加边缘节点"""
        node_id = len(self.nodes)
        node = OptimizedEdgeNode(
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
    
    def select_best_nodes(self, top_k: int = 2) -> List[OptimizedEdgeNode]:
        """选择表现最好的节点"""
        nodes_with_performance = [(node, node.get_performance()) for node in self.nodes]
        nodes_with_performance.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in nodes_with_performance[:top_k]]
    
    def communicate(self):
        """智能通信"""
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
        """训练边缘智能群"""
        print(f"\n开始训练优化边缘智能群 ({len(self.nodes)} 节点)")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(global_epochs):
            # 本地训练
            local_threads = []
            local_losses = {}
            
            def train_node(node):
                if node.sensor_type in sensor_data:
                    X, y = sensor_data[node.sensor_type]
                    losses = []
                    for _ in range(self.config.local_epochs):
                        loss = node.local_update(X, y)
                        losses.append(loss)
                    local_losses[node.sensor_type] = np.mean(losses)
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
        """测试边缘智能群"""
        print(f"\n测试优化边缘智能群 ({test_samples} 样本)")
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
                # 更新融合中心的传感器性能
                self.fusion_center.update_performance(node.sensor_type, accuracy)
                print(f"{node.sensor_type}: 准确率={accuracy:.4f}")
        
        return results
    
    def multimodal_test(self, sensor_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                       test_samples: int = 100) -> float:
        """多模态融合测试"""
        print(f"\n多模态融合测试 ({test_samples} 样本)")
        print("-" * 70)
        
        correct = 0
        
        for i in range(test_samples):
            # 收集各传感器预测
            predictions = {}
            for node in self.nodes:
                if node.sensor_type in sensor_data:
                    X, _ = sensor_data[node.sensor_type]
                    is_anomaly, score = node.detect_anomaly(X[i:i+1])
                    predictions[node.sensor_type] = (is_anomaly, score)
            
            # 融合预测（带时间戳，模拟真实时间）
            if predictions:
                timestamp = time.time()  # 模拟当前时间
                fused_anomaly, _, _ = self.fusion_center.fuse_predictions(predictions, timestamp)
                
                # 验证
                if 'multimodal' in sensor_data:
                    _, y = sensor_data['multimodal']
                    true_anomaly = np.argmax(y[i]) == 1
                    if fused_anomaly == true_anomaly:
                        correct += 1
        
        accuracy = correct / test_samples if test_samples > 0 else 0.0
        print(f"多模态融合准确率: {accuracy:.4f}")
        return accuracy
    
    def get_nodes(self) -> List[OptimizedEdgeNode]:
        """获取所有节点"""
        return self.nodes
    
    def get_performance_history(self) -> List[float]:
        """获取性能历史"""
        return self.swarm_performance
    
    def get_fusion_center(self) -> FusionCenter:
        """获取融合中心"""
        return self.fusion_center
