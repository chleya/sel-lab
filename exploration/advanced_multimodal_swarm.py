# -*- coding: utf-8 -*-
"""
高级多模态边缘智能群 - SEL + EGGROLL 双演化框架

特点：
1. SEL 结构演化 + EGGROLL 参数优化
2. 多模态数据融合与协同学习
3. 自适应学习率与参数调整
4. 智能通信与模型共享
5. 实时异常检测与预测
"""

import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class AdvancedConfig:
    """高级配置"""
    rank: int = 3
    sigma: float = 0.1
    population_size: int = 8
    learning_rate: float = 0.01
    communication_freq: int = 3
    local_epochs: int = 3
    energy_budget: float = 80.0
    adaptation_rate: float = 0.1
    fusion_weight: float = 0.3
    sel_alpha: float = 0.1  # SEL 学习率
    sel_beta: float = 0.9   # SEL 动量


class SELModule:
    """SEL 结构演化模块"""
    
    def __init__(self, in_size: int, out_size: int, config: AdvancedConfig, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.in_size = in_size
        self.out_size = out_size
        self.config = config
        
        # 权重矩阵
        self.W = self.rng.normal(0.0, np.sqrt(2.0 / in_size), (in_size, out_size))
        
        # SEL 相关参数
        self.activations = []
        self.gradients = []
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        logits = X @ self.W
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 保存激活值用于 SEL
        self.activations.append(output)
        return output
    
    def sel_update(self, X: np.ndarray, y: np.ndarray):
        """SEL 结构演化更新"""
        # 前向传播
        y_pred = self.forward(X)
        
        # 计算损失
        loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
        
        # SEL 直接反馈对齐
        error = y - y_pred
        
        # 计算权重更新
        if len(self.activations) > 0:
            activation = self.activations[-1]
            # 直接反馈对齐：输入梯度 * 输出误差
            update = X.T @ error * self.config.sel_alpha
            self.W += update
        
        # 清空激活值
        self.activations = []
        
        return loss


class AdvancedSensorNode:
    """高级传感器节点"""
    
    def __init__(self, node_id: int, sensor_type: str, 
                 in_size: int, out_size: int, 
                 config: AdvancedConfig, seed: int = None):
        self.node_id = node_id
        self.sensor_type = sensor_type
        self.rng = np.random.default_rng(seed)
        self.in_size = in_size
        self.out_size = out_size
        self.config = config
        
        # 模型
        self.sel_module = SELModule(in_size, out_size, config, seed)
        self.W = self.sel_module.W
        
        # 统计
        self.local_losses = []
        self.accuracy_history = []
        self.performance_score = 0.0
        self.energy_consumption = 0.0
        
        # 异常检测阈值
        self.anomaly_threshold = 0.8
        
        # 锁
        self.lock = threading.Lock()
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        return self.sel_module.forward(X)
    
    def generate_perturbation(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成低秩扰动"""
        a, b = self.W.shape
        noise = self.rng.normal(0.0, 1.0, size=(a + b, self.config.rank))
        B = noise[:b]
        A = noise[b:]
        return A, B
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
        """评估扰动"""
        W_perturbed = self.W + self.config.sigma * (A @ B.T)
        
        # 保存原始权重
        original_W = self.W.copy()
        self.W = W_perturbed
        
        # 评估
        y_pred = self.forward(X)
        loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
        
        # 恢复原始权重
        self.W = original_W
        self.sel_module.W = original_W
        
        return 1.0 / (loss + 1e-8)
    
    def local_update(self, X: np.ndarray, y: np.ndarray) -> float:
        """本地双演化更新"""
        cfg = self.config
        
        # 计算能耗
        energy_cost = len(X) * cfg.population_size * 0.0001
        self.energy_consumption += energy_cost
        
        # 检查能源预算
        if self.energy_consumption > cfg.energy_budget:
            return float('inf')
        
        # 1. SEL 结构演化更新
        sel_loss = self.sel_module.sel_update(X, y)
        
        # 2. EGGROLL 参数优化
        # 生成扰动种群
        perturbations = []
        for _ in range(cfg.population_size):
            A, B = self.generate_perturbation()
            perturbations.append((A, B))
        
        # 评估适应度
        fitness_scores = []
        for A, B in perturbations:
            fitness = self.evaluate(X, y, A, B)
            fitness_scores.append(fitness)
        
        # 融合更新
        avg_update = np.zeros_like(self.W)
        total_fitness = sum(fitness_scores)
        
        if total_fitness > 0:
            for (A, B), fitness in zip(perturbations, fitness_scores):
                perturbation = cfg.sigma * (A @ B.T)
                weight = fitness / total_fitness
                avg_update += perturbation * weight
            
            with self.lock:
                self.W += cfg.learning_rate * avg_update
                self.W = np.clip(self.W, -3.0, 3.0)
                self.sel_module.W = self.W
        
        loss = self.compute_loss(X, y)
        acc = self.compute_accuracy(X, y)
        
        self.local_losses.append(loss)
        self.accuracy_history.append(acc)
        
        # 更新性能分数
        if len(self.accuracy_history) > 0:
            self.performance_score = self.accuracy_history[-1]
        
        return loss
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算损失"""
        y_pred = self.forward(X)
        loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
        return float(loss)
    
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
        """获取当前模型"""
        with self.lock:
            return self.W.copy()
    
    def update_model(self, new_W: np.ndarray, performance_weight: float = 1.0):
        """更新模型"""
        with self.lock:
            weight = performance_weight / (performance_weight + self.performance_score + 1e-8)
            self.W = (1 - weight) * self.W + weight * new_W
            self.sel_module.W = self.W
    
    def adapt_parameters(self):
        """自适应调整参数"""
        if len(self.accuracy_history) > 10:
            recent_improvement = self.accuracy_history[-1] - self.accuracy_history[-10]
            
            # 调整学习率
            if recent_improvement > 0:
                self.config.learning_rate *= (1 + self.config.adaptation_rate)
                self.config.sel_alpha *= (1 + self.config.adaptation_rate * 0.5)
            else:
                self.config.learning_rate *= (1 - self.config.adaptation_rate)
                self.config.sel_alpha *= (1 - self.config.adaptation_rate * 0.5)
            
            # 限制学习率范围
            self.config.learning_rate = max(0.001, min(0.1, self.config.learning_rate))
            self.config.sel_alpha = max(0.001, min(0.2, self.config.sel_alpha))
            
            # 调整异常检测阈值
            if self.performance_score > 0.8:
                self.anomaly_threshold = max(0.7, self.anomaly_threshold - 0.05)
            elif self.performance_score < 0.6:
                self.anomaly_threshold = min(0.9, self.anomaly_threshold + 0.05)


class AdvancedFusionCenter:
    """高级融合中心"""
    
    def __init__(self, config: AdvancedConfig, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.config = config
        
        # 融合模型
        self.fusion_weights = {}
        self.confidence_threshold = 0.7
        self.historical_performance = {}
    
    def fuse_predictions(self, predictions: Dict[str, Tuple[bool, float]]) -> Tuple[bool, float, Dict[str, float]]:
        """融合多模态预测"""
        # 计算加权融合
        total_weight = 0.0
        weighted_score = 0.0
        confidences = {}
        
        for sensor_type, (is_anomaly, score) in predictions.items():
            # 基于传感器类型的权重
            sensor_weights = {
                'temperature': 0.3,
                'humidity': 0.2,
                'motion': 0.25,
                'door': 0.25
            }
            weight = sensor_weights.get(sensor_type, 0.25)
            
            # 基于历史性能调整权重
            if sensor_type in self.historical_performance:
                performance = self.historical_performance[sensor_type]
                weight *= (0.5 + performance)
            
            # 基于置信度的权重调整
            confidence = score if is_anomaly else 1.0 - score
            confidences[sensor_type] = confidence
            
            adjusted_weight = weight * confidence
            total_weight += adjusted_weight
            weighted_score += adjusted_weight * score
        
        if total_weight > 0:
            fused_score = weighted_score / total_weight
        else:
            fused_score = 0.0
        
        # 确定最终异常状态
        is_anomaly = fused_score > self.confidence_threshold
        
        return is_anomaly, fused_score, confidences
    
    def update_performance(self, sensor_type: str, accuracy: float):
        """更新历史性能"""
        if sensor_type not in self.historical_performance:
            self.historical_performance[sensor_type] = accuracy
        else:
            # 指数移动平均
            self.historical_performance[sensor_type] = \
                0.7 * self.historical_performance[sensor_type] + 0.3 * accuracy


class AdvancedMultimodalSwarm:
    """高级多模态边缘智能群"""
    
    def __init__(self, config: AdvancedConfig, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.config = config
        
        # 创建传感器节点
        self.nodes = []
        self.sensor_types = ['temperature', 'humidity', 'motion', 'door']
        
        # 创建各个传感器节点
        for sensor_type in self.sensor_types:
            if sensor_type == 'temperature':
                self.add_node(sensor_type, in_size=3, out_size=2)
            elif sensor_type == 'humidity':
                self.add_node(sensor_type, in_size=3, out_size=2)
            elif sensor_type == 'motion':
                self.add_node(sensor_type, in_size=3, out_size=2)
            elif sensor_type == 'door':
                self.add_node(sensor_type, in_size=3, out_size=2)
        
        # 创建融合中心
        self.fusion_center = AdvancedFusionCenter(config, seed)
        
        # 全局统计
        self.global_epoch = 0
        self.swarm_performance = []
        self.fusion_accuracy = []
    
    def add_node(self, sensor_type: str, in_size: int, out_size: int):
        """添加传感器节点"""
        node_id = len(self.nodes)
        node = AdvancedSensorNode(
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
    
    def select_best_nodes(self, top_k: int = 2) -> List[AdvancedSensorNode]:
        """选择表现最好的节点"""
        nodes_with_performance = [(node, node.performance_score) for node in self.nodes]
        nodes_with_performance.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in nodes_with_performance[:top_k]]
    
    def communicate(self):
        """节点间智能通信"""
        # 选择表现最好的 2 个节点
        best_nodes = self.select_best_nodes(top_k=2)
        
        # 收集最好的模型
        best_models = {node.sensor_type: node.get_model() for node in best_nodes}
        best_performances = {node.sensor_type: node.performance_score for node in best_nodes}
        
        # 向同类传感器节点广播
        for node in self.nodes:
            if node.sensor_type in best_models:
                model = best_models[node.sensor_type]
                performance = best_performances[node.sensor_type]
                node.update_model(model, performance)
    
    def generate_multimodal_data(self, n_samples: int = 1000, anomaly_ratio: float = 0.1):
        """生成多模态传感器数据"""
        data = {}
        
        # 温度数据
        temp_data = []
        temp_labels = []
        base_temp = 22.0
        for i in range(n_samples):
            if self.rng.random() > anomaly_ratio:
                temp = base_temp + self.rng.normal(0, 1.0)
                temp_data.append([temp, i % 24, base_temp])
                temp_labels.append([1, 0])
                base_temp = temp
            else:
                temp = base_temp + self.rng.normal(5, 2.0)
                temp_data.append([temp, i % 24, base_temp])
                temp_labels.append([0, 1])
                base_temp = temp
        data['temperature'] = (np.array(temp_data), np.array(temp_labels))
        
        # 湿度数据
        hum_data = []
        hum_labels = []
        base_hum = 45.0
        for i in range(n_samples):
            if self.rng.random() > anomaly_ratio:
                hum = base_hum + self.rng.normal(0, 3.0)
                hum_data.append([hum, i % 24, base_hum])
                hum_labels.append([1, 0])
                base_hum = hum
            else:
                hum = base_hum + self.rng.normal(20, 5.0)
                hum_data.append([hum, i % 24, base_hum])
                hum_labels.append([0, 1])
                base_hum = hum
        data['humidity'] = (np.array(hum_data), np.array(hum_labels))
        
        # 运动数据
        motion_data = []
        motion_labels = []
        base_motion = 0.0
        for i in range(n_samples):
            if self.rng.random() > anomaly_ratio:
                motion = base_motion + self.rng.normal(0, 0.1)
                motion_data.append([motion, i % 24, base_motion])
                motion_labels.append([1, 0])
                base_motion = motion
            else:
                motion = base_motion + self.rng.normal(1.0, 0.3)
                motion_data.append([motion, i % 24, base_motion])
                motion_labels.append([0, 1])
                base_motion = motion
        data['motion'] = (np.array(motion_data), np.array(motion_labels))
        
        # 门数据
        door_data = []
        door_labels = []
        base_door = 0.0
        for i in range(n_samples):
            if self.rng.random() > anomaly_ratio:
                door = base_door
                door_data.append([door, i % 24, base_door])
                door_labels.append([1, 0])
                if self.rng.random() < 0.05:
                    base_door = 1.0 - base_door
            else:
                door = 1.0 - base_door
                door_data.append([door, i % 24, base_door])
                door_labels.append([0, 1])
                base_door = door
        data['door'] = (np.array(door_data), np.array(door_labels))
        
        # 多模态标签（综合异常）
        multimodal_labels = []
        for i in range(n_samples):
            is_anomaly = False
            for sensor_type in self.sensor_types:
                _, y = data[sensor_type]
                if np.argmax(y[i]) == 1:
                    is_anomaly = True
                    break
            multimodal_labels.append([0, 1] if is_anomaly else [1, 0])
        data['multimodal'] = (np.array(multimodal_labels), np.array(multimodal_labels))
        
        return data
    
    def train(self, sensor_data: Dict, global_epochs: int = 30):
        """训练高级多模态边缘智能群"""
        print(f"\n开始训练高级多模态边缘智能群 ({len(self.nodes)} 节点)")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(global_epochs):
            # 本地训练
            local_threads = []
            local_losses = {}
            
            def train_node(node):
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
            avg_accuracy = np.mean([node.performance_score for node in self.nodes])
            self.swarm_performance.append(avg_accuracy)
            
            # 测试融合性能
            if epoch % 5 == 0:
                fusion_acc = self.test_fusion(sensor_data, test_samples=100)
                self.fusion_accuracy.append(fusion_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"全局 Epoch {epoch+1:2d}: 平均准确率 = {avg_accuracy:.4f}")
                for node in self.nodes:
                    print(f"  {node.sensor_type}: 准确率={node.performance_score:.4f}, "
                          f"阈值={node.anomaly_threshold:.2f}, "
                          f"能耗={node.energy_consumption:.2f}")
            
            self.global_epoch += 1
        
        total_time = time.time() - start_time
        print(f"\n训练完成: 时间 = {total_time:.3f}s")
        
        return self.swarm_performance
    
    def test_fusion(self, sensor_data: Dict, test_samples: int = 100) -> float:
        """测试融合性能"""
        correct = 0
        
        for i in range(test_samples):
            # 收集各传感器预测
            predictions = {}
            for node in self.nodes:
                X, _ = sensor_data[node.sensor_type]
                is_anomaly, score = node.detect_anomaly(X[i:i+1])
                predictions[node.sensor_type] = (is_anomaly, score)
            
            # 融合预测
            fused_anomaly, _, _ = self.fusion_center.fuse_predictions(predictions)
            
            # 验证
            _, y = sensor_data['multimodal']
            true_anomaly = np.argmax(y[i]) == 1
            
            if fused_anomaly == true_anomaly:
                correct += 1
        
        return correct / test_samples
    
    def test_real_time(self, sensor_data: Dict, test_samples: int = 100):
        """实时测试"""
        print(f"\n实时多模态异常检测测试 ({test_samples} 样本)")
        print("-"*70)
        
        results = {}
        fusion_results = []
        
        for i in range(test_samples):
            # 收集各传感器预测
            predictions = {}
            for node in self.nodes:
                X, _ = sensor_data[node.sensor_type]
                is_anomaly, score = node.detect_anomaly(X[i:i+1])
                predictions[node.sensor_type] = (is_anomaly, score)
            
            # 融合预测
            fused_anomaly, fused_score, confidences = self.fusion_center.fuse_predictions(predictions)
            
            # 验证
            _, y = sensor_data['multimodal']
            true_anomaly = np.argmax(y[i]) == 1
            
            fusion_results.append({
                'fused_anomaly': fused_anomaly,
                'fused_score': fused_score,
                'true_anomaly': true_anomaly,
                'correct': fused_anomaly == true_anomaly,
                'confidences': confidences
            })
        
        # 计算结果
        correct = sum(1 for r in fusion_results if r['correct'])
        fusion_accuracy = correct / test_samples
        
        print(f"多模态融合准确率: {fusion_accuracy:.4f}")
        print(f"正确预测: {correct}/{test_samples}")
        
        # 各传感器性能
        for node in self.nodes:
            X, y = sensor_data[node.sensor_type]
            test_X = X[:test_samples]
            test_y = y[:test_samples]
            
            correct = 0
            for j in range(test_samples):
                is_anomaly, _ = node.detect_anomaly(test_X[j:j+1])
                true_anomaly = np.argmax(test_y[j]) == 1
                if is_anomaly == true_anomaly:
                    correct += 1
            
            accuracy = correct / test_samples
            results[node.sensor_type] = accuracy
            self.fusion_center.update_performance(node.sensor_type, accuracy)
            print(f"{node.sensor_type}: 准确率={accuracy:.4f}")
        
        return {'fusion_accuracy': fusion_accuracy, 'sensor_accuracies': results}


def test_advanced_swarm():
    """测试高级多模态边缘智能群"""
    print("\n" + "="*70)
    print("高级多模态边缘智能群测试")
    print("="*70)
    
    # 配置
    config = AdvancedConfig(
        rank=3,
        sigma=0.1,
        population_size=8,
        learning_rate=0.01,
        communication_freq=3,
        local_epochs=3,
        energy_budget=80.0,
        sel_alpha=0.1
    )
    
    # 创建边缘群
    swarm = AdvancedMultimodalSwarm(config)
    
    # 生成传感器数据
    sensor_data = swarm.generate_multimodal_data(n_samples=1000, anomaly_ratio=0.1)
    print(f"\n生成传感器数据:")
    for sensor_type, (X, y) in sensor_data.items():
        if sensor_type != 'multimodal':
            n_anomalies = np.sum(np.argmax(y, axis=1) == 1)
            print(f"  {sensor_type}: {X.shape[0]} 样本, {n_anomalies} 异常")
    
    # 训练
    performance = swarm.train(sensor_data, global_epochs=30)
    
    # 实时测试
    test_results = swarm.test_real_time(sensor_data, test_samples=100)
    
    # 绘制结果
    plot_results(performance, swarm.fusion_accuracy, test_results)
    
    print("\n" + "="*70)
    print("高级多模态边缘智能群测试完成！")
    print("="*70)


def plot_results(performance: List[float], fusion_accuracy: List[float], test_results: Dict):
    """绘制结果"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练性能
    ax = axes[0, 0]
    ax.plot(performance, linewidth=2, label='单传感器平均')
    ax.set_xlabel('全局 Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('训练性能')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.legend()
    
    # 融合性能
    ax = axes[0, 1]
    ax.plot(range(0, len(fusion_accuracy)*5, 5), fusion_accuracy, linewidth=2, color='orange', label='多模态融合')
    ax.set_xlabel('全局 Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('融合性能')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.legend()
    
    # 传感器准确率
    ax = axes[1, 0]
    sensor_types = list(test_results['sensor_accuracies'].keys())
    accuracies = [test_results['sensor_accuracies'][st] for st in sensor_types]
    ax.bar(sensor_types, accuracies, alpha=0.7)
    ax.axhline(y=test_results['fusion_accuracy'], color='red', linestyle='--', label='融合准确率')
    ax.set_ylabel('准确率')
    ax.set_title('各传感器准确率')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # 融合vs单传感器
    ax = axes[1, 1]
    categories = ['融合准确率'] + sensor_types
    values = [test_results['fusion_accuracy']] + accuracies
    ax.bar(categories, values, alpha=0.7)
    ax.set_ylabel('准确率')
    ax.set_title('融合 vs 单传感器')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('f:/sel-lab/exploration/advanced_multimodal_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ 结果图已保存: f:/sel-lab/exploration/advanced_multimodal_results.png")
    plt.show()


if __name__ == "__main__":
    test_advanced_swarm()
