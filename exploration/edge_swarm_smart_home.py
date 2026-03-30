# -*- coding: utf-8 -*-
"""
边缘演化智能群 - 智能家居异常检测

真实场景应用：
- 多个传感器节点监控家庭环境
- 检测异常情况（温度异常、湿度异常、入侵检测等）
- 分布式学习，无需云端依赖
- 实时响应，低延迟

特点：
1. 真实传感器数据模拟
2. 多任务异常检测
3. 自适应阈值调整
4. 节点间协作与信息共享
5. 能源效率优化
"""

import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class SensorNodeConfig:
    """传感器节点配置"""
    rank: int = 2
    sigma: float = 0.1
    population_size: int = 6  # 边缘设备资源受限
    learning_rate: float = 0.015
    communication_freq: int = 4
    local_epochs: int = 2
    energy_budget: float = 100.0  # 能源预算
    adaptation_rate: float = 0.1


class SensorNode:
    """智能家居传感器节点"""
    
    def __init__(self, node_id: int, sensor_type: str, 
                 in_size: int, out_size: int, 
                 config: SensorNodeConfig, seed: int = None):
        self.node_id = node_id
        self.sensor_type = sensor_type  # 传感器类型：'temperature', 'humidity', 'motion', 'door'
        self.rng = np.random.default_rng(seed)
        self.in_size = in_size
        self.out_size = out_size
        self.config = config
        
        # 本地模型
        self.W = self.rng.normal(0.0, np.sqrt(2.0 / in_size), (in_size, out_size))
        
        # 统计
        self.local_losses = []
        self.accuracy_history = []
        self.performance_score = 0.0
        self.energy_consumption = 0.0
        
        # 异常检测阈值
        self.anomaly_threshold = 0.8  # 初始阈值
        
        # 锁
        self.lock = threading.Lock()
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        logits = X @ self.W
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
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
        logits = X @ W_perturbed
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
        return 1.0 / (loss + 1e-8)
    
    def local_update(self, X: np.ndarray, y: np.ndarray) -> float:
        """本地 EGGROLL 更新"""
        cfg = self.config
        
        # 计算能耗
        energy_cost = len(X) * cfg.population_size * 0.01
        self.energy_consumption += energy_cost
        
        # 检查能源预算
        if self.energy_consumption > cfg.energy_budget:
            return float('inf')  # 能源耗尽
        
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
        
        loss = self.compute_loss(X, y)
        acc = self.compute_accuracy(X, y)
        
        self.local_losses.append(loss)
        self.accuracy_history.append(acc)
        
        # 更新性能分数
        if len(self.accuracy_history) > 5:
            recent_acc = np.mean(self.accuracy_history[-5:])
            self.performance_score = recent_acc
        
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
        anomaly_score = y_pred[0, 1]  # 异常类的概率
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
    
    def adapt_parameters(self):
        """自适应调整参数"""
        if len(self.accuracy_history) > 10:
            recent_improvement = self.accuracy_history[-1] - self.accuracy_history[-10]
            
            # 调整学习率
            if recent_improvement > 0:
                self.config.learning_rate *= (1 + self.config.adaptation_rate)
            else:
                self.config.learning_rate *= (1 - self.config.adaptation_rate)
            
            # 限制学习率范围
            self.config.learning_rate = max(0.001, min(0.1, self.config.learning_rate))
            
            # 调整异常检测阈值
            if self.performance_score > 0.8:
                self.anomaly_threshold = max(0.7, self.anomaly_threshold - 0.05)
            elif self.performance_score < 0.6:
                self.anomaly_threshold = min(0.9, self.anomaly_threshold + 0.05)


class SmartHomeEdgeSwarm:
    """智能家居边缘智能群"""
    
    def __init__(self, config: SensorNodeConfig, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.config = config
        
        # 创建传感器节点
        self.nodes = []
        
        # 温度传感器 (输入: 温度, 时间, 前一时刻温度)
        self.add_node('temperature', in_size=3, out_size=2)
        
        # 湿度传感器 (输入: 湿度, 时间, 前一时刻湿度)
        self.add_node('humidity', in_size=3, out_size=2)
        
        # 运动传感器 (输入: 运动强度, 时间, 前一时刻运动)
        self.add_node('motion', in_size=3, out_size=2)
        
        # 门传感器 (输入: 开关状态, 时间, 前一时刻状态)
        self.add_node('door', in_size=3, out_size=2)
        
        # 全局统计
        self.global_epoch = 0
        self.swarm_performance = []
    
    def add_node(self, sensor_type: str, in_size: int, out_size: int):
        """添加传感器节点"""
        node_id = len(self.nodes)
        node = SensorNode(
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
    
    def select_best_nodes(self, top_k: int = 2) -> List[SensorNode]:
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
    
    def generate_sensor_data(self, n_samples: int = 1000, anomaly_ratio: float = 0.1):
        """生成传感器数据"""
        data = {}
        
        # 温度数据
        temp_data = []
        temp_labels = []
        base_temp = 22.0
        for i in range(n_samples):
            # 正常温度波动
            if self.rng.random() > anomaly_ratio:
                temp = base_temp + self.rng.normal(0, 1.0)
                temp_data.append([temp, i % 24, base_temp])
                temp_labels.append([1, 0])  # 正常
                base_temp = temp
            else:
                # 异常温度
                temp = base_temp + self.rng.normal(5, 2.0)
                temp_data.append([temp, i % 24, base_temp])
                temp_labels.append([0, 1])  # 异常
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
                if self.rng.random() < 0.05:  # 正常开关
                    base_door = 1.0 - base_door
            else:
                door = 1.0 - base_door
                door_data.append([door, i % 24, base_door])
                door_labels.append([0, 1])
                base_door = door
        data['door'] = (np.array(door_data), np.array(door_labels))
        
        return data
    
    def train(self, sensor_data: Dict, global_epochs: int = 30):
        """训练边缘智能群"""
        print(f"\n开始训练智能家居边缘智能群 ({len(self.nodes)} 节点)")
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
    
    def test_real_time(self, sensor_data: Dict, test_samples: int = 50):
        """实时测试"""
        print(f"\n实时异常检测测试 ({test_samples} 样本)")
        print("-"*70)
        
        results = {}
        
        for node in self.nodes:
            X, y = sensor_data[node.sensor_type]
            test_X = X[:test_samples]
            test_y = y[:test_samples]
            
            correct = 0
            anomalies_detected = 0
            false_alarms = 0
            
            for i in range(test_samples):
                is_anomaly, score = node.detect_anomaly(test_X[i:i+1])
                true_anomaly = np.argmax(test_y[i]) == 1
                
                if is_anomaly == true_anomaly:
                    correct += 1
                elif is_anomaly and not true_anomaly:
                    false_alarms += 1
                elif not is_anomaly and true_anomaly:
                    pass  # 漏检
                
                if is_anomaly:
                    anomalies_detected += 1
            
            accuracy = correct / test_samples
            results[node.sensor_type] = {
                'accuracy': accuracy,
                'anomalies_detected': anomalies_detected,
                'false_alarms': false_alarms
            }
            
            print(f"{node.sensor_type}:")
            print(f"  准确率: {accuracy:.4f}")
            print(f"  检测到异常: {anomalies_detected}")
            print(f"  误报: {false_alarms}")
        
        return results


def test_smart_home_swarm():
    """测试智能家居边缘智能群"""
    print("\n" + "="*70)
    print("智能家居边缘智能群测试")
    print("="*70)
    
    # 配置
    config = SensorNodeConfig(
        rank=2,
        sigma=0.1,
        population_size=6,
        learning_rate=0.015,
        communication_freq=4,
        local_epochs=2,
        energy_budget=50.0
    )
    
    # 创建边缘群
    swarm = SmartHomeEdgeSwarm(config)
    
    # 生成传感器数据
    sensor_data = swarm.generate_sensor_data(n_samples=1000, anomaly_ratio=0.1)
    print(f"\n生成传感器数据:")
    for sensor_type, (X, y) in sensor_data.items():
        n_anomalies = np.sum(np.argmax(y, axis=1) == 1)
        print(f"  {sensor_type}: {X.shape[0]} 样本, {n_anomalies} 异常")
    
    # 训练
    performance = swarm.train(sensor_data, global_epochs=30)
    
    # 实时测试
    test_results = swarm.test_real_time(sensor_data, test_samples=100)
    
    # 绘制结果
    plot_results(performance, test_results)
    
    print("\n" + "="*70)
    print("智能家居边缘智能群测试完成！")
    print("="*70)


def plot_results(performance: List[float], test_results: Dict):
    """绘制结果"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练性能
    ax = axes[0, 0]
    ax.plot(performance, linewidth=2)
    ax.set_xlabel('全局 Epoch')
    ax.set_ylabel('平均准确率')
    ax.set_title('训练性能')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 测试结果
    ax = axes[0, 1]
    sensor_types = list(test_results.keys())
    accuracies = [test_results[st]['accuracy'] for st in sensor_types]
    ax.bar(sensor_types, accuracies, alpha=0.7)
    ax.set_ylabel('准确率')
    ax.set_title('传感器异常检测准确率')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # 误报率
    ax = axes[1, 0]
    false_alarms = [test_results[st]['false_alarms'] for st in sensor_types]
    ax.bar(sensor_types, false_alarms, alpha=0.7, color='orange')
    ax.set_ylabel('误报次数')
    ax.set_title('误报情况')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 异常检测
    ax = axes[1, 1]
    anomalies = [test_results[st]['anomalies_detected'] for st in sensor_types]
    ax.bar(sensor_types, anomalies, alpha=0.7, color='green')
    ax.set_ylabel('检测到的异常数')
    ax.set_title('异常检测情况')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('f:/sel-lab/exploration/smart_home_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ 结果图已保存: f:/sel-lab/exploration/smart_home_results.png")
    plt.show()


if __name__ == "__main__":
    test_smart_home_swarm()
