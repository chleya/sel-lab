# -*- coding: utf-8 -*-
"""
多模态融合模块

实现不同传感器数据的智能融合。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class FusionConfig:
    """融合配置"""
    confidence_threshold: float = 0.7  # 异常检测阈值
    fusion_weight: float = 0.3  # 融合权重
    history_alpha: float = 0.7  # 历史性能平滑系数


class FusionCenter:
    """融合中心"""
    
    def __init__(self, config: Optional[FusionConfig] = None):
        """初始化融合中心
        
        Args:
            config: 融合配置
        """
        self.config = config or FusionConfig()
        self.historical_performance: Dict[str, float] = {}
        self.sensor_weights: Dict[str, float] = {
            'temperature': 0.3,
            'humidity': 0.2,
            'motion': 0.25,
            'door': 0.25
        }
        self.recent_predictions: Dict[str, List[Tuple[bool, float]]] = {}  # 最近的预测历史
        self.sensor_correlations: Dict[str, Dict[str, float]] = {}  # 传感器间相关性
        self.time_context: Dict[str, float] = {}  # 时间上下文信息
    
    def fuse_predictions(self, predictions: Dict[str, Tuple[bool, float]], timestamp: Optional[float] = None) -> Tuple[bool, float, Dict[str, float]]:
        """融合多模态预测
        
        Args:
            predictions: 各传感器的预测结果，格式为 {sensor_type: (is_anomaly, score)}
            timestamp: 时间戳（可选）
            
        Returns:
            (is_anomaly, fused_score, confidences)：融合结果、融合分数、各传感器置信度
        """
        # 更新最近的预测历史
        self._update_prediction_history(predictions, timestamp)
        
        # 计算传感器间相关性
        self._update_sensor_correlations(predictions)
        
        total_weight = 0.0
        weighted_score = 0.0
        confidences = {}
        contextual_weights = {}
        
        for sensor_type, (is_anomaly, score) in predictions.items():
            # 1. 基础权重
            weight = self.sensor_weights.get(sensor_type, 0.25)
            
            # 2. 基于历史性能调整权重
            if sensor_type in self.historical_performance:
                performance = self.historical_performance[sensor_type]
                weight *= (0.5 + performance)
            
            # 3. 基于时间上下文调整权重
            time_weight = self._get_time_context_weight(sensor_type, timestamp)
            weight *= time_weight
            
            # 4. 基于预测历史的一致性调整
            consistency_weight = self._get_consistency_weight(sensor_type)
            weight *= consistency_weight
            
            # 5. 基于传感器相关性调整
            correlation_weight = self._get_correlation_weight(sensor_type, predictions)
            weight *= correlation_weight
            
            # 6. 基于置信度的权重调整
            confidence = score if is_anomaly else 1.0 - score
            confidences[sensor_type] = confidence
            
            # 7. 综合调整后的权重
            adjusted_weight = weight * confidence
            contextual_weights[sensor_type] = weight
            total_weight += adjusted_weight
            weighted_score += adjusted_weight * score
        
        # 计算融合分数
        if total_weight > 0:
            fused_score = weighted_score / total_weight
        else:
            fused_score = 0.0
        
        # 8. 自适应阈值确定
        adaptive_threshold = self._calculate_adaptive_threshold(predictions, confidences)
        
        # 确定最终异常状态
        is_anomaly = fused_score > adaptive_threshold
        
        return is_anomaly, fused_score, confidences
    
    def _update_prediction_history(self, predictions: Dict[str, Tuple[bool, float]], timestamp: Optional[float] = None):
        """更新预测历史"""
        for sensor_type, (is_anomaly, score) in predictions.items():
            if sensor_type not in self.recent_predictions:
                self.recent_predictions[sensor_type] = []
            
            # 添加最新预测，保持最近10个
            self.recent_predictions[sensor_type].append((is_anomaly, score))
            if len(self.recent_predictions[sensor_type]) > 10:
                self.recent_predictions[sensor_type] = self.recent_predictions[sensor_type][-10:]
        
        # 更新时间上下文
        if timestamp:
            self.time_context['last_update'] = timestamp
            self.time_context['hour_of_day'] = timestamp % 24
    
    def _update_sensor_correlations(self, predictions: Dict[str, Tuple[bool, float]]):
        """更新传感器间相关性"""
        sensor_types = list(predictions.keys())
        for i, sensor1 in enumerate(sensor_types):
            if sensor1 not in self.sensor_correlations:
                self.sensor_correlations[sensor1] = {}
            
            for sensor2 in sensor_types[i+1:]:
                if sensor2 not in self.sensor_correlations[sensor1]:
                    self.sensor_correlations[sensor1][sensor2] = 0.5
                
                # 计算当前相关性
                is_anomaly1 = predictions[sensor1][0]
                is_anomaly2 = predictions[sensor2][0]
                
                if is_anomaly1 == is_anomaly2:
                    # 增加相关性
                    self.sensor_correlations[sensor1][sensor2] = min(1.0, self.sensor_correlations[sensor1][sensor2] + 0.1)
                else:
                    # 减少相关性
                    self.sensor_correlations[sensor1][sensor2] = max(0.0, self.sensor_correlations[sensor1][sensor2] - 0.05)
    
    def _get_time_context_weight(self, sensor_type: str, timestamp: Optional[float] = None) -> float:
        """获取时间上下文权重"""
        if not timestamp:
            return 1.0
        
        hour = timestamp % 24
        
        # 不同传感器在不同时间的重要性不同
        time_weights = {
            'motion': {range(6, 22): 1.2, range(22, 24): 1.5, range(0, 6): 1.8},  # 夜间运动更重要
            'door': {range(8, 20): 1.0, range(20, 24): 1.3, range(0, 8): 1.5},    # 夜间开门更重要
            'temperature': {range(10, 18): 0.8, range(18, 24): 1.0, range(0, 10): 1.2},  # 夜间温度异常更重要
            'humidity': {range(0, 24): 1.0}  # 湿度全天候重要性一致
        }
        
        sensor_time_weights = time_weights.get(sensor_type, {range(0, 24): 1.0})
        
        for hour_range, weight in sensor_time_weights.items():
            if hour in hour_range:
                return weight
        
        return 1.0
    
    def _get_consistency_weight(self, sensor_type: str) -> float:
        """获取一致性权重"""
        if sensor_type not in self.recent_predictions or len(self.recent_predictions[sensor_type]) < 3:
            return 1.0
        
        # 计算最近预测的一致性
        recent_preds = self.recent_predictions[sensor_type][-3:]
        anomalies = [pred[0] for pred in recent_preds]
        
        # 如果最近预测一致，增加权重
        if all(anomalies) or not any(anomalies):
            return 1.2
        else:
            # 预测不一致，减少权重
            return 0.8
    
    def _get_correlation_weight(self, sensor_type: str, predictions: Dict[str, Tuple[bool, float]]) -> float:
        """获取相关性权重"""
        if sensor_type not in self.sensor_correlations:
            return 1.0
        
        correlation_sum = 0.0
        count = 0
        
        for other_sensor, correlation in self.sensor_correlations[sensor_type].items():
            if other_sensor in predictions:
                correlation_sum += correlation
                count += 1
        
        if count > 0:
            avg_correlation = correlation_sum / count
            # 相关性越高，权重越大
            return 0.8 + (avg_correlation * 0.4)
        
        return 1.0
    
    def _calculate_adaptive_threshold(self, predictions: Dict[str, Tuple[bool, float]], confidences: Dict[str, float]) -> float:
        """计算自适应阈值"""
        # 基础阈值
        base_threshold = self.config.confidence_threshold
        
        # 基于整体置信度调整
        if confidences:
            avg_confidence = sum(confidences.values()) / len(confidences)
            # 置信度越高，阈值越低
            threshold_adjustment = 1.0 - (avg_confidence - 0.5) * 0.3
            base_threshold *= threshold_adjustment
        
        # 基于异常传感器数量调整
        anomaly_count = sum(1 for is_anomaly, _ in predictions.values() if is_anomaly)
        total_sensors = len(predictions)
        
        if total_sensors > 0:
            anomaly_ratio = anomaly_count / total_sensors
            # 异常传感器越多，阈值越低
            if anomaly_ratio > 0.5:
                base_threshold *= 0.9
            elif anomaly_ratio > 0.75:
                base_threshold *= 0.8
        
        # 限制阈值范围
        return max(0.5, min(0.9, base_threshold))
    
    def update_performance(self, sensor_type: str, accuracy: float):
        """更新传感器历史性能
        
        Args:
            sensor_type: 传感器类型
            accuracy: 准确率
        """
        if sensor_type not in self.historical_performance:
            self.historical_performance[sensor_type] = accuracy
        else:
            # 指数移动平均
            self.historical_performance[sensor_type] = \
                self.config.history_alpha * self.historical_performance[sensor_type] + \
                (1 - self.config.history_alpha) * accuracy
    
    def get_sensor_weights(self) -> Dict[str, float]:
        """获取传感器权重
        
        Returns:
            传感器权重字典
        """
        return self.sensor_weights.copy()
    
    def set_sensor_weights(self, weights: Dict[str, float]):
        """设置传感器权重
        
        Args:
            weights: 新的传感器权重
        """
        self.sensor_weights.update(weights)


class MultimodalFusion:
    """多模态融合器"""
    
    def __init__(self, config: Optional[FusionConfig] = None):
        """初始化多模态融合器
        
        Args:
            config: 融合配置
        """
        self.fusion_center = FusionCenter(config)
        self.sensor_models: Dict[str, object] = {}
    
    def register_sensor(self, sensor_type: str, model):
        """注册传感器模型
        
        Args:
            sensor_type: 传感器类型
            model: 传感器模型（需要有 predict 方法）
        """
        self.sensor_models[sensor_type] = model
    
    def predict(self, sensor_data: Dict[str, np.ndarray], timestamp: Optional[float] = None) -> Dict:
        """多模态预测
        
        Args:
            sensor_data: 各传感器的数据，格式为 {sensor_type: data}
            timestamp: 时间戳（可选）
            
        Returns:
            融合预测结果
        """
        # 收集各传感器预测
        predictions = {}
        for sensor_type, data in sensor_data.items():
            if sensor_type in self.sensor_models:
                model = self.sensor_models[sensor_type]
                # 假设模型返回 (is_anomaly, score)
                is_anomaly, score = model.detect_anomaly(data)
                predictions[sensor_type] = (is_anomaly, score)
        
        # 融合预测（带时间戳）
        is_anomaly, fused_score, confidences = self.fusion_center.fuse_predictions(predictions, timestamp)
        
        return {
            'is_anomaly': is_anomaly,
            'fused_score': fused_score,
            'confidences': confidences,
            'sensor_predictions': predictions,
            'timestamp': timestamp
        }
    
    def evaluate(self, sensor_data: Dict[str, np.ndarray], true_labels: Dict[str, np.ndarray]) -> Dict[str, float]:
        """评估融合性能
        
        Args:
            sensor_data: 各传感器的数据
            true_labels: 真实标签
            
        Returns:
            性能指标
        """
        # 评估各传感器
        sensor_accuracies = {}
        for sensor_type, data in sensor_data.items():
            if sensor_type in self.sensor_models and sensor_type in true_labels:
                model = self.sensor_models[sensor_type]
                y_true = true_labels[sensor_type]
                
                # 计算准确率
                correct = 0
                total = len(y_true)
                for i in range(total):
                    is_anomaly, _ = model.detect_anomaly(data[i:i+1])
                    true_anomaly = np.argmax(y_true[i]) == 1
                    if is_anomaly == true_anomaly:
                        correct += 1
                
                accuracy = correct / total if total > 0 else 0.0
                sensor_accuracies[sensor_type] = accuracy
                # 更新历史性能
                self.fusion_center.update_performance(sensor_type, accuracy)
        
        # 评估融合性能
        fusion_correct = 0
        fusion_total = 0
        
        if 'multimodal' in true_labels:
            y_true = true_labels['multimodal']
            fusion_total = len(y_true)
            
            for i in range(fusion_total):
                # 构建当前样本的传感器数据
                sample_data = {}
                for sensor_type, data in sensor_data.items():
                    sample_data[sensor_type] = data[i:i+1]
                
                # 融合预测
                result = self.predict(sample_data)
                is_anomaly = result['is_anomaly']
                true_anomaly = np.argmax(y_true[i]) == 1
                
                if is_anomaly == true_anomaly:
                    fusion_correct += 1
        
        fusion_accuracy = fusion_correct / fusion_total if fusion_total > 0 else 0.0
        
        return {
            'fusion_accuracy': fusion_accuracy,
            'sensor_accuracies': sensor_accuracies
        }
    
    def get_fusion_center(self) -> FusionCenter:
        """获取融合中心
        
        Returns:
            融合中心实例
        """
        return self.fusion_center
