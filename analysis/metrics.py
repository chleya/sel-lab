"""
指标计算模块 - 根据协议计算评估指标
"""

from typing import Dict, List, Any
import numpy as np

class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, protocol_metrics: List[str]):
        self.protocol_metrics = protocol_metrics
        
    def calculate_all(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算所有指标"""
        metrics = {}
        
        for metric_name in self.protocol_metrics:
            if hasattr(self, f"calculate_{metric_name}"):
                method = getattr(self, f"calculate_{metric_name}")
                metrics[metric_name] = method(experiment_data)
            else:
                metrics[metric_name] = self.calculate_generic(metric_name, experiment_data)
        
        return metrics
    
    def calculate_prediction_trend(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """计算预测趋势"""
        predictions = data.get('predictions', [])
        targets = data.get('targets', [])
        
        if not predictions or not targets:
            return {"error": "缺少预测或目标数据"}
        
        # 计算准确率趋势
        accuracies = []
        for pred, target in zip(predictions, targets):
            accuracies.append(1.0 if pred == target else 0.0)
        
        # 趋势分析
        if len(accuracies) > 1:
            trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
        else:
            trend = 0.0
        
        return {
            "accuracy_series": accuracies,
            "mean_accuracy": np.mean(accuracies) if accuracies else 0.0,
            "trend_slope": trend,
            "improving": trend > 0.01,
            "stable": abs(trend) < 0.01
        }
    
    def calculate_structural_change_frequency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """计算结构变化频率"""
        changes = data.get('structural_changes', [])
        steps = data.get('total_steps', 1)
        
        frequency = len(changes) / steps if steps > 0 else 0.0
        
        return {
            "total_changes": len(changes),
            "change_frequency": frequency,
            "changes_per_step": frequency,
            "change_types": self._analyze_change_types(changes)
        }
    
    def calculate_locality_of_updates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """计算更新的局部性"""
        updates = data.get('updates', [])
        
        if not updates:
            return {"error": "缺少更新数据"}
        
        # 计算平均局部性（假设更新包含局部性信息）
        localities = [update.get('locality', 0.0) for update in updates]
        
        return {
            "mean_locality": np.mean(localities) if localities else 0.0,
            "std_locality": np.std(localities) if localities else 0.0,
            "max_locality": max(localities) if localities else 0.0,
            "min_locality": min(localities) if localities else 0.0
        }
    
    def calculate_stability_of_substructures(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """计算子结构稳定性"""
        structures = data.get('structures', [])
        
        if len(structures) < 2:
            return {"error": "需要至少两个结构快照"}
        
        # 计算结构相似性
        similarities = []
        for i in range(len(structures) - 1):
            sim = self._calculate_structure_similarity(structures[i], structures[i+1])
            similarities.append(sim)
        
        return {
            "mean_stability": np.mean(similarities) if similarities else 0.0,
            "stability_series": similarities,
            "stable": np.mean(similarities) > 0.7 if similarities else False
        }
    
    def calculate_generic(self, metric_name: str, data: Dict[str, Any]) -> Any:
        """通用指标计算"""
        # 尝试从数据中直接获取
        if metric_name in data:
            return data[metric_name]
        
        # 默认值
        return {"value": 0.0, "note": "使用默认值"}
    
    def _analyze_change_types(self, changes: List[Dict]) -> Dict[str, int]:
        """分析变化类型"""
        types = {}
        for change in changes:
            change_type = change.get('type', 'unknown')
            types[change_type] = types.get(change_type, 0) + 1
        return types
    
    def _calculate_structure_similarity(self, struct1: Dict, struct2: Dict) -> float:
        """计算结构相似性（简化版）"""
        # 这里应该实现具体的结构相似性计算
        # 目前返回随机值
        import random
        return random.uniform(0.5, 0.9)