# -*- coding: utf-8 -*-
"""
类小脑边缘计算模块

模拟小脑的核心功能：
- 预测编码：预测下一个状态并与实际比较
- 时间序列处理：处理连续的时间信号
- 误差修正：类似小脑的监督学习机制
- 运动协调：多传感器协调与同步
"""

import numpy as np
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable

from .sel import SELModule, SELConfig
from .eggroll import EGGROLLOptimizer, EGGROLLConfig


@dataclass
class CerebellarConfig:
    """类小脑配置"""
    # SEL配置
    sel_alpha: float = 0.15
    
    # EGGROLL配置
    eggroll_rank: int = 4
    eggroll_population: int = 10
    
    # 预测编码配置
    prediction_horizon: int = 3  # 预测未来多少步
    temporal_window: int = 5     # 时间窗口大小
    
    # 误差修正配置
    error_learning_rate: float = 0.1
    mismatch_threshold: float = 0.3
    
    # 运动协调配置
    coordination_strength: float = 0.5
    adaptation_rate: float = 0.05
    
    # 能耗配置
    energy_budget: float = 100.0


class PredictiveCoder:
    """预测编码器 - 模拟小脑的前向模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 使用更小的初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, input_size) * 0.01
        
        # 历史记录
        self.history = deque(maxlen=10)
        self.error_history = deque(maxlen=20)
        
    def predict(self, current_state: np.ndarray) -> np.ndarray:
        """预测下一个状态"""
        # 输入归一化
        current_normalized = current_state / (np.linalg.norm(current_state) + 1e-8)
        
        # 前向传播
        hidden = np.tanh(current_normalized @ self.W1)
        prediction = hidden @ self.W2
        
        # 输出限制
        prediction = np.clip(prediction, -10, 10)
        
        return prediction
    
    def update(self, current_state: np.ndarray, actual_next: np.ndarray, 
               learning_rate: float = 0.001):
        """基于预测误差更新权重"""
        # 输入归一化
        current_normalized = current_state / (np.linalg.norm(current_state) + 1e-8)
        actual_normalized = actual_next / (np.linalg.norm(actual_next) + 1e-8)
        
        prediction = self.predict(current_state)
        error = actual_normalized - prediction
        
        # 记录误差
        error_magnitude = np.mean(error**2)
        self.error_history.append(error_magnitude)
        
        # 如果误差太大，减小学习率
        if len(self.error_history) > 5:
            recent_error = np.mean(list(self.error_history)[-5:])
            if recent_error > 1.0:
                learning_rate *= 0.5
        
        # 简单的梯度下降更新
        hidden = np.tanh(current_normalized @ self.W1)
        
        # 更新W2（带梯度裁剪）
        dW2 = hidden.T @ error
        dW2 = np.clip(dW2, -1, 1)  # 梯度裁剪
        self.W2 += learning_rate * dW2
        self.W2 = np.clip(self.W2, -1, 1)  # 权重限制
        
        # 更新W1（带梯度裁剪）
        dW1 = current_normalized.T @ (error @ self.W2.T * (1 - hidden**2))
        dW1 = np.clip(dW1, -1, 1)  # 梯度裁剪
        self.W1 += learning_rate * dW1
        self.W1 = np.clip(self.W1, -1, 1)  # 权重限制
        
        return error_magnitude
    
    def get_prediction_error(self, current_state: np.ndarray, 
                            actual_next: np.ndarray) -> float:
        """计算预测误差"""
        prediction = self.predict(current_state)
        # 归一化后计算误差
        actual_normalized = actual_next / (np.linalg.norm(actual_next) + 1e-8)
        error = np.mean((actual_normalized - prediction)**2)
        return min(error, 10.0)  # 限制误差范围


class TemporalIntegrator:
    """时间整合器 - 处理时间序列信息"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
    def add(self, data: np.ndarray, timestamp: Optional[float] = None):
        """添加时间序列数据"""
        self.buffer.append(data.copy())
        self.timestamps.append(timestamp or time.time())
    
    def get_temporal_pattern(self) -> np.ndarray:
        """获取时间模式特征"""
        if len(self.buffer) < 2:
            return np.zeros(self.buffer[0].shape if self.buffer else (1,))
        
        # 计算时间导数（变化率）
        patterns = []
        for i in range(1, len(self.buffer)):
            dt = self.timestamps[i] - self.timestamps[i-1]
            if dt > 0:
                derivative = (self.buffer[i] - self.buffer[i-1]) / dt
                patterns.append(derivative)
        
        if patterns:
            return np.mean(patterns, axis=0)
        return np.zeros(self.buffer[0].shape)
    
    def detect_anomaly_in_pattern(self) -> Tuple[bool, float]:
        """检测时间模式中的异常"""
        if len(self.buffer) < 3:
            return False, 0.0
        
        # 计算变化的一致性
        changes = []
        for i in range(1, len(self.buffer)):
            change = np.linalg.norm(self.buffer[i] - self.buffer[i-1])
            changes.append(change)
        
        if len(changes) < 2:
            return False, 0.0
        
        # 检测突变
        recent_change = changes[-1]
        avg_change = np.mean(changes[:-1])
        
        if avg_change > 0:
            anomaly_score = recent_change / (avg_change + 1e-8)
            is_anomaly = anomaly_score > 2.0  # 突变阈值
            return is_anomaly, min(anomaly_score / 3.0, 1.0)
        
        return False, 0.0


class ErrorCorrector:
    """误差修正器 - 类似小脑的监督学习"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.error_history = deque(maxlen=20)
        self.correction_weights = {}
        
    def compute_error_signal(self, predicted: np.ndarray, 
                            actual: np.ndarray) -> np.ndarray:
        """计算误差信号"""
        error = actual - predicted
        self.error_history.append(np.mean(error**2))
        return error
    
    def adapt_correction(self, sensor_type: str, error: np.ndarray):
        """自适应修正"""
        if sensor_type not in self.correction_weights:
            self.correction_weights[sensor_type] = 1.0
        
        # 基于误差历史调整修正权重
        if len(self.error_history) > 5:
            recent_error = np.mean(list(self.error_history)[-5:])
            old_error = np.mean(list(self.error_history)[:5])
            
            if recent_error < old_error:
                # 误差在减小，增加权重
                self.correction_weights[sensor_type] *= (1 + self.learning_rate)
            else:
                # 误差在增加，减小权重
                self.correction_weights[sensor_type] *= (1 - self.learning_rate)
            
            # 限制权重范围
            self.correction_weights[sensor_type] = np.clip(
                self.correction_weights[sensor_type], 0.1, 2.0
            )
    
    def get_correction_weight(self, sensor_type: str) -> float:
        """获取修正权重"""
        return self.correction_weights.get(sensor_type, 1.0)


class CerebellarEdgeNode:
    """类小脑边缘节点"""
    
    def __init__(self, node_id: int, sensor_type: str, 
                 in_size: int, out_size: int,
                 config: Optional[CerebellarConfig] = None,
                 seed: Optional[int] = None):
        self.node_id = node_id
        self.sensor_type = sensor_type
        self.rng = np.random.default_rng(seed)
        self.in_size = in_size
        self.out_size = out_size
        self.config = config or CerebellarConfig()
        
        # 基础模型
        sel_config = SELConfig(
            alpha=self.config.sel_alpha,
            activation='sigmoid'
        )
        self.sel_module = SELModule(in_size, out_size, sel_config, seed)
        
        eggroll_config = EGGROLLConfig(
            rank=self.config.eggroll_rank,
            population_size=self.config.eggroll_population,
            learning_rate=0.01
        )
        self.eggroll_optimizer = EGGROLLOptimizer(in_size, out_size, eggroll_config, seed)
        
        # 类小脑组件
        self.predictive_coder = PredictiveCoder(in_size)
        self.temporal_integrator = TemporalIntegrator(self.config.temporal_window)
        self.error_corrector = ErrorCorrector(self.config.error_learning_rate)
        
        # 状态记录
        self.last_state = None
        self.prediction_error = 0.0
        self.anomaly_score = 0.0
        
        # 能耗管理
        self.energy_consumption = 0.0
        self.lock = threading.Lock()
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        return self.sel_module.forward(X)
    
    def predict_next(self, current_state: np.ndarray) -> np.ndarray:
        """预测下一个状态"""
        return self.predictive_coder.predict(current_state)
    
    def cerebellar_update(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """类小脑更新 - 核心学习循环"""
        results = {}
        
        # 1. 预测编码
        if self.last_state is not None:
            # 基于上一个状态预测当前状态
            predicted = self.predict_next(self.last_state)
            actual = X
            
            # 计算预测误差
            self.prediction_error = self.predictive_coder.get_prediction_error(
                self.last_state, actual
            )
            
            # 更新预测器
            pred_loss = self.predictive_coder.update(
                self.last_state, actual, 
                learning_rate=self.config.error_learning_rate
            )
            results['prediction_loss'] = pred_loss
            
            # 误差修正
            error_signal = self.error_corrector.compute_error_signal(predicted, actual)
            self.error_corrector.adapt_correction(self.sensor_type, error_signal)
        
        # 2. 时间整合
        self.temporal_integrator.add(X)
        temporal_pattern = self.temporal_integrator.get_temporal_pattern()
        
        # 检测时间异常
        is_temporal_anomaly, temporal_score = self.temporal_integrator.detect_anomaly_in_pattern()
        results['temporal_anomaly'] = temporal_score
        
        # 3. SEL学习
        sel_loss = self.sel_module.update(X, y)
        results['sel_loss'] = sel_loss
        
        # 4. EGGROLL优化
        eggroll_loss = self.eggroll_optimizer.step(X, y)
        results['eggroll_loss'] = eggroll_loss
        
        # 5. 权重同步（考虑预测误差）
        correction_weight = self.error_corrector.get_correction_weight(self.sensor_type)
        
        with self.lock:
            sel_weights = self.sel_module.get_weights()
            eggroll_weights = self.eggroll_optimizer.get_weights()
            
            # 基于预测误差的自适应融合
            if self.prediction_error > self.config.mismatch_threshold:
                # 预测误差大，更信任EGGROLL的探索
                fusion_weight = 0.3 * correction_weight
            else:
                # 预测准确，更信任SEL的学习
                fusion_weight = 0.7 * correction_weight
            
            fused_weights = fusion_weight * sel_weights + (1 - fusion_weight) * eggroll_weights
            self.sel_module.set_weights(fused_weights)
            self.eggroll_optimizer.set_weights(fused_weights)
        
        # 6. 能耗计算
        energy_cost = len(X) * self.config.eggroll_population * 0.0001
        self.energy_consumption += energy_cost
        
        # 更新状态
        self.last_state = X.copy()
        
        return results
    
    def detect_anomaly(self, sensor_data: np.ndarray) -> Tuple[bool, float]:
        """异常检测 - 综合考虑预测误差和时间模式"""
        # 基础预测
        y_pred = self.forward(sensor_data)
        base_score = y_pred[0, 1]
        
        # 预测误差贡献（使用归一化误差）
        if self.last_state is not None:
            prediction_error = self.predictive_coder.get_prediction_error(self.last_state, sensor_data)
            # 将误差转换为0-1范围的贡献
            error_contribution = min(prediction_error / 5.0, 0.3)  # 最多贡献30%
        else:
            error_contribution = 0.0
        
        # 时间异常贡献
        self.temporal_integrator.add(sensor_data)
        _, temporal_score = self.temporal_integrator.detect_anomaly_in_pattern()
        temporal_contribution = temporal_score * 0.2  # 最多贡献20%
        
        # 综合异常分数
        self.anomaly_score = base_score * 0.5 + error_contribution + temporal_contribution
        self.anomaly_score = np.clip(self.anomaly_score, 0, 1)
        
        # 自适应阈值（基于预测误差调整）
        adaptive_threshold = 0.7 - min(self.prediction_error * 0.1, 0.2)
        adaptive_threshold = np.clip(adaptive_threshold, 0.5, 0.8)
        
        is_anomaly = self.anomaly_score > adaptive_threshold
        
        return is_anomaly, self.anomaly_score
    
    def coordinate_with_neighbors(self, neighbor_predictions: Dict[str, np.ndarray]):
        """与邻居节点协调 - 类似小脑的运动协调"""
        if not neighbor_predictions:
            return
        
        # 计算与邻居的协调性
        coordination_signals = []
        for neighbor_type, neighbor_pred in neighbor_predictions.items():
            if neighbor_pred.shape == self.last_state.shape:
                # 计算相关性
                correlation = np.corrcoef(
                    self.last_state.flatten(), 
                    neighbor_pred.flatten()
                )[0, 1]
                coordination_signals.append(correlation)
        
        if coordination_signals:
            avg_coordination = np.mean(coordination_signals)
            
            # 调整学习率基于协调性
            if avg_coordination > 0.5:
                # 高度协调，可以加快学习
                self.config.sel_alpha *= (1 + self.config.adaptation_rate)
            else:
                # 不协调，需要更谨慎
                self.config.sel_alpha *= (1 - self.config.adaptation_rate)
            
            self.config.sel_alpha = np.clip(self.config.sel_alpha, 0.01, 0.3)
    
    def get_energy_consumption(self) -> float:
        """获取能源消耗"""
        return self.energy_consumption
    
    def get_prediction_accuracy(self) -> float:
        """获取预测准确率"""
        # 使用预测编码器的误差历史
        if len(self.predictive_coder.error_history) > 0:
            recent_errors = list(self.predictive_coder.error_history)[-10:]
            avg_error = np.mean(recent_errors)
            # 将误差转换为准确率（误差越小，准确率越高）
            accuracy = max(0, 1.0 - min(avg_error, 1.0))
            return accuracy
        return 0.5  # 默认中等准确率


class CerebellarEdgeSwarm:
    """类小脑边缘智能群"""
    
    def __init__(self, config: Optional[CerebellarConfig] = None, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)
        self.config = config or CerebellarConfig()
        self.nodes: List[CerebellarEdgeNode] = []
        self.global_epoch = 0
        
    def add_node(self, sensor_type: str, in_size: int, out_size: int) -> CerebellarEdgeNode:
        """添加类小脑边缘节点"""
        node_id = len(self.nodes)
        node = CerebellarEdgeNode(
            node_id=node_id,
            sensor_type=sensor_type,
            in_size=in_size,
            out_size=out_size,
            config=self.config,
            seed=self.rng.integers(0, 10000)
        )
        self.nodes.append(node)
        print(f"✓ 添加类小脑 {sensor_type} 传感器节点 (ID: {node_id})")
        return node
    
    def train(self, sensor_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
              global_epochs: int = 30) -> List[Dict]:
        """训练类小脑边缘智能群"""
        print(f"\n开始训练类小脑边缘智能群 ({len(self.nodes)} 节点)")
        print("=" * 70)
        
        start_time = time.time()
        epoch_results = []
        
        for epoch in range(global_epochs):
            epoch_metrics = {
                'prediction_losses': [],
                'temporal_anomalies': [],
                'sel_losses': [],
                'eggroll_losses': []
            }
            
            # 并行训练
            threads = []
            for node in self.nodes:
                if node.sensor_type in sensor_data:
                    X, y = sensor_data[node.sensor_type]
                    
                    def train_node(n=node, data_x=X, data_y=y):
                        results = n.cerebellar_update(data_x, data_y)
                        epoch_metrics['prediction_losses'].append(results.get('prediction_loss', 0))
                        epoch_metrics['temporal_anomalies'].append(results.get('temporal_anomaly', 0))
                        epoch_metrics['sel_losses'].append(results.get('sel_loss', 0))
                        epoch_metrics['eggroll_losses'].append(results.get('eggroll_loss', 0))
                    
                    thread = threading.Thread(target=train_node)
                    threads.append(thread)
                    thread.start()
            
            # 等待完成
            for thread in threads:
                thread.join()
            
            # 节点间协调
            if epoch % 5 == 0:
                self._coordinate_nodes()
            
            # 记录结果
            epoch_summary = {
                'epoch': epoch + 1,
                'avg_prediction_loss': np.mean(epoch_metrics['prediction_losses']) if epoch_metrics['prediction_losses'] else 0,
                'avg_temporal_anomaly': np.mean(epoch_metrics['temporal_anomalies']) if epoch_metrics['temporal_anomalies'] else 0,
                'avg_sel_loss': np.mean(epoch_metrics['sel_losses']) if epoch_metrics['sel_losses'] else 0,
                'avg_eggroll_loss': np.mean(epoch_metrics['eggroll_losses']) if epoch_metrics['eggroll_losses'] else 0
            }
            epoch_results.append(epoch_summary)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"全局 Epoch {epoch+1:2d}: "
                      f"预测损失={epoch_summary['avg_prediction_loss']:.4f}, "
                      f"时间异常={epoch_summary['avg_temporal_anomaly']:.4f}, "
                      f"SEL损失={epoch_summary['avg_sel_loss']:.4f}")
            
            self.global_epoch += 1
        
        total_time = time.time() - start_time
        print(f"\n训练完成: 时间 = {total_time:.3f}s")
        
        return epoch_results
    
    def _coordinate_nodes(self):
        """节点间协调"""
        # 收集各节点的预测
        predictions = {}
        for node in self.nodes:
            if node.last_state is not None:
                pred = node.predict_next(node.last_state)
                predictions[node.sensor_type] = pred
        
        # 每个节点与邻居协调
        for node in self.nodes:
            neighbor_preds = {k: v for k, v in predictions.items() 
                            if k != node.sensor_type}
            node.coordinate_with_neighbors(neighbor_preds)
    
    def test(self, sensor_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
             test_samples: int = 100) -> Dict[str, float]:
        """测试类小脑边缘智能群"""
        print(f"\n测试类小脑边缘智能群 ({test_samples} 样本)")
        print("-" * 70)
        
        results = {}
        
        for node in self.nodes:
            if node.sensor_type in sensor_data:
                X, y = sensor_data[node.sensor_type]
                test_X = X[:test_samples]
                test_y = y[:test_samples]
                
                correct = 0
                prediction_accuracies = []
                
                for i in range(test_samples):
                    # 异常检测
                    is_anomaly, score = node.detect_anomaly(test_X[i:i+1])
                    true_anomaly = np.argmax(test_y[i]) == 1
                    
                    if is_anomaly == true_anomaly:
                        correct += 1
                    
                    # 记录预测准确率
                    pred_acc = node.get_prediction_accuracy()
                    prediction_accuracies.append(pred_acc)
                
                accuracy = correct / test_samples
                avg_pred_acc = np.mean(prediction_accuracies)
                
                results[node.sensor_type] = {
                    'accuracy': accuracy,
                    'prediction_accuracy': avg_pred_acc,
                    'energy': node.get_energy_consumption()
                }
                
                print(f"{node.sensor_type}: 检测准确率={accuracy:.4f}, "
                      f"预测准确率={avg_pred_acc:.4f}, 能耗={node.get_energy_consumption():.2f}")
        
        return results
    
    def get_nodes(self) -> List[CerebellarEdgeNode]:
        """获取所有节点"""
        return self.nodes
