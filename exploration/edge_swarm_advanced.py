# -*- coding: utf-8 -*-
"""
高级边缘演化智能群 (Advanced Edge Evolutionary Swarm)

改进点：
1. 智能通信：只共享表现最好的模型
2. 改进融合：基于表现的加权融合
3. 自适应参数：根据节点性能调整
4. 星型拓扑：有中心协调节点
5. 分布式评估：每个节点评估不同数据子集
6. 动态节点加入/退出
"""

import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class AdvancedEdgeNodeConfig:
    """高级边缘节点配置"""
    rank: int = 2
    sigma: float = 0.1
    population_size: int = 8
    learning_rate: float = 0.02
    communication_freq: int = 5
    local_epochs: int = 3
    adaptation_rate: float = 0.1  # 参数自适应率


class AdvancedEdgeNode:
    """高级边缘节点"""
    
    def __init__(self, node_id: int, in_size: int, out_size: int, 
                 config: AdvancedEdgeNodeConfig, seed: int = None):
        self.node_id = node_id
        self.rng = np.random.default_rng(seed)
        self.in_size = in_size
        self.out_size = out_size
        self.config = config
        
        # 本地模型
        self.W = self.rng.normal(0.0, np.sqrt(2.0 / in_size), (in_size, out_size))
        
        # 统计
        self.local_losses = []
        self.accuracy_history = []
        self.performance_score = 0.0  # 节点性能分数
        
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
    
    def get_model(self) -> np.ndarray:
        """获取当前模型"""
        with self.lock:
            return self.W.copy()
    
    def update_model(self, new_W: np.ndarray, performance_weight: float = 1.0):
        """更新模型（基于性能加权）"""
        with self.lock:
            # 基于性能的加权融合
            weight = performance_weight / (performance_weight + self.performance_score + 1e-8)
            self.W = (1 - weight) * self.W + weight * new_W
    
    def adapt_parameters(self):
        """自适应调整参数"""
        if len(self.accuracy_history) > 10:
            recent_improvement = self.accuracy_history[-1] - self.accuracy_history[-10]
            
            # 根据性能调整学习率
            if recent_improvement > 0:
                self.config.learning_rate *= (1 + self.config.adaptation_rate)
            else:
                self.config.learning_rate *= (1 - self.config.adaptation_rate)
            
            # 限制学习率范围
            self.config.learning_rate = max(0.001, min(0.1, self.config.learning_rate))


class AdvancedEdgeSwarm:
    """高级边缘演化智能群"""
    
    def __init__(self, n_nodes: int, in_size: int, out_size: int, 
                 config: AdvancedEdgeNodeConfig, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_nodes = n_nodes
        self.in_size = in_size
        self.out_size = out_size
        self.config = config
        
        # 创建边缘节点
        self.nodes = []
        for i in range(n_nodes):
            node = AdvancedEdgeNode(i, in_size, out_size, config, seed=seed+i)
            self.nodes.append(node)
        
        # 中心协调节点（索引 0）
        self.coordinator_id = 0
        
        # 全局统计
        self.global_epoch = 0
        self.swarm_losses = []
        self.swarm_accuracies = []
    
    def select_best_nodes(self, top_k: int = 2) -> List[AdvancedEdgeNode]:
        """选择表现最好的节点"""
        nodes_with_performance = [(node, node.performance_score) for node in self.nodes]
        nodes_with_performance.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in nodes_with_performance[:top_k]]
    
    def communicate(self):
        """智能通信：只共享最好的模型"""
        # 选择表现最好的 2 个节点
        best_nodes = self.select_best_nodes(top_k=2)
        
        # 中心节点收集最好的模型
        best_models = [node.get_model() for node in best_nodes]
        best_performances = [node.performance_score for node in best_nodes]
        
        # 中心节点融合最好的模型
        if best_models:
            fused_model = np.zeros_like(best_models[0])
            total_performance = sum(best_performances) + 1e-8
            
            for model, performance in zip(best_models, best_performances):
                weight = performance / total_performance
                fused_model += weight * model
            
            # 向所有节点广播融合后的模型
            for node in self.nodes:
                node.update_model(fused_model, total_performance)
    
    def distribute_data(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """将数据分配给各个节点"""
        n_samples = X.shape[0]
        samples_per_node = n_samples // self.n_nodes
        
        data_chunks = []
        for i in range(self.n_nodes):
            start = i * samples_per_node
            end = (i + 1) * samples_per_node if i < self.n_nodes - 1 else n_samples
            data_chunks.append((X[start:end], y[start:end]))
        
        return data_chunks
    
    def train(self, X: np.ndarray, y: np.ndarray, global_epochs: int = 50):
        """训练整个群体"""
        print(f"\n开始训练高级边缘演化智能群 ({self.n_nodes} 节点)")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(global_epochs):
            # 分配数据
            data_chunks = self.distribute_data(X, y)
            
            # 本地训练
            local_threads = []
            local_losses = [0.0] * self.n_nodes
            
            def train_node(node_id):
                node = self.nodes[node_id]
                node_X, node_y = data_chunks[node_id]
                for _ in range(self.config.local_epochs):
                    loss = node.local_update(node_X, node_y)
                local_losses[node_id] = node.compute_loss(node_X, node_y)
                # 自适应参数
                node.adapt_parameters()
            
            # 并行本地训练
            for i in range(self.n_nodes):
                thread = threading.Thread(target=train_node, args=(i,))
                local_threads.append(thread)
                thread.start()
            
            # 等待所有节点完成
            for thread in local_threads:
                thread.join()
            
            # 智能通信
            if epoch % self.config.communication_freq == 0:
                self.communicate()
            
            # 计算群体性能
            avg_loss = np.mean(local_losses)
            avg_acc = np.mean([node.compute_accuracy(X, y) for node in self.nodes])
            
            self.swarm_losses.append(avg_loss)
            self.swarm_accuracies.append(avg_acc)
            
            if (epoch + 1) % 10 == 0:
                best_node = self.select_best_nodes(top_k=1)[0]
                print(f"全局 Epoch {epoch+1:2d}: 群体平均损失 = {avg_loss:.4f}, "
                      f"准确率 = {avg_acc:.4f}, 最佳节点准确率 = {best_node.performance_score:.4f}")
            
            self.global_epoch += 1
        
        total_time = time.time() - start_time
        print(f"\n训练完成: 时间 = {total_time:.3f}s")
        
        return self.swarm_losses, self.swarm_accuracies
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """评估群体性能"""
        # 集成所有节点的预测
        predictions = []
        for node in self.nodes:
            y_pred = node.forward(X)
            predictions.append(y_pred)
        
        # 基于性能的加权集成
        performances = [node.performance_score for node in self.nodes]
        total_performance = sum(performances) + 1e-8
        weights = [p / total_performance for p in performances]
        
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_pred += weight * pred
        
        final_pred = np.argmax(weighted_pred, axis=1)
        labels = np.argmax(y, axis=1)
        
        return float(np.mean(final_pred == labels))
    
    def add_node(self, node_id: int = None):
        """动态添加节点"""
        new_id = node_id if node_id is not None else len(self.nodes)
        new_node = AdvancedEdgeNode(new_id, self.in_size, self.out_size, 
                                   self.config, seed=self.rng.integers(0, 10000))
        self.nodes.append(new_node)
        self.n_nodes += 1
        print(f"✓ 添加新节点: {new_id}")
        return new_node
    
    def remove_node(self, node_id: int):
        """动态移除节点"""
        if 0 <= node_id < len(self.nodes):
            removed_node = self.nodes.pop(node_id)
            self.n_nodes -= 1
            # 重新分配节点 ID
            for i, node in enumerate(self.nodes):
                node.node_id = i
            print(f"✓ 移除节点: {removed_node.node_id}")
            return removed_node
        return None


def create_synthetic_data(n_samples=1000, in_size=10, out_size=2):
    """创建合成数据"""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (n_samples, in_size))
    y = np.zeros((n_samples, out_size))
    
    # 简单的分类任务
    for i in range(n_samples):
        if np.sum(X[i, :3]) > 0:
            y[i, 0] = 1
        else:
            y[i, 1] = 1
    
    return X, y

def test_advanced_edge_swarm():
    """测试高级边缘演化智能群"""
    print("\n" + "="*70)
    print("高级边缘演化智能群测试")
    print("="*70)
    
    # 创建数据
    X, y = create_synthetic_data(n_samples=1000, in_size=10, out_size=2)
    print(f"数据: {X.shape[0]} 样本, 输入维度={X.shape[1]}, 输出维度={y.shape[1]}")
    
    # 配置
    config = AdvancedEdgeNodeConfig(
        rank=2,
        sigma=0.1,
        population_size=8,
        learning_rate=0.02,
        communication_freq=3,  # 更频繁的通信
        local_epochs=3,
        adaptation_rate=0.1
    )
    
    # 测试不同节点数量
    node_counts = [2, 4, 6]
    results = {}
    
    for n_nodes in node_counts:
        print(f"\n测试 {n_nodes} 个边缘节点:")
        print("-"*60)
        
        swarm = AdvancedEdgeSwarm(n_nodes, 10, 2, config, seed=42)
        
        # 训练
        losses, accuracies = swarm.train(X, y, global_epochs=50)
        
        # 评估
        accuracy = swarm.evaluate(X, y)
        print(f"  最终准确率: {accuracy:.4f}")
        
        results[n_nodes] = {
            'losses': losses,
            'accuracies': accuracies,
            'accuracy': accuracy,
            'nodes': n_nodes
        }
    
    # 与单机 EGGROLL 对比
    print(f"\n与单机 EGGROLL 对比:")
    print("-"*60)
    
    single_node = AdvancedEdgeNode(0, 10, 2, config, seed=42)
    single_losses = []
    single_accuracies = []
    
    start_time = time.time()
    for epoch in range(50):
        for _ in range(config.local_epochs):
            single_node.local_update(X, y)
        loss = single_node.compute_loss(X, y)
        acc = single_node.compute_accuracy(X, y)
        single_losses.append(loss)
        single_accuracies.append(acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}: Loss = {loss:.4f}, Acc = {acc:.4f}")
    
    single_time = time.time() - start_time
    single_acc = single_node.compute_accuracy(X, y)
    print(f"  单机 EGGROLL: 准确率={single_acc:.4f}, 时间={single_time:.3f}s")
    
    results['single'] = {
        'losses': single_losses,
        'accuracies': single_accuracies,
        'accuracy': single_acc,
        'nodes': 1
    }
    
    # 测试动态节点管理
    print(f"\n测试动态节点管理:")
    print("-"*60)
    
    swarm = AdvancedEdgeSwarm(2, 10, 2, config, seed=42)
    print(f"  初始节点数: {swarm.n_nodes}")
    
    # 添加节点
    swarm.add_node()
    swarm.add_node()
    print(f"  添加后节点数: {swarm.n_nodes}")
    
    # 训练
    losses, accuracies = swarm.train(X, y, global_epochs=20)
    acc = swarm.evaluate(X, y)
    print(f"  训练后准确率: {acc:.4f}")
    
    # 移除节点
    swarm.remove_node(1)
    print(f"  移除后节点数: {swarm.n_nodes}")
    
    # 继续训练
    losses2, accuracies2 = swarm.train(X, y, global_epochs=20)
    acc2 = swarm.evaluate(X, y)
    print(f"  继续训练后准确率: {acc2:.4f}")
    
    # 绘制结果
    plot_results(results)
    
    print("\n" + "="*70)
    print("高级边缘演化智能群测试完成！")
    print("="*70)


def plot_results(results: Dict):
    """绘制结果"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 损失曲线
    ax = axes[0]
    for key, data in results.items():
        if key == 'single':
            label = f"单机 EGGROLL"
        else:
            label = f"{key} 节点群"
        
        ax.plot(data['losses'], label=label, linewidth=2)
    
    ax.set_xlabel('全局 Epoch')
    ax.set_ylabel('损失')
    ax.set_title('高级边缘演化智能群 vs 单机 EGGROLL')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 准确率曲线
    ax = axes[1]
    for key, data in results.items():
        if key == 'single':
            label = f"单机 EGGROLL"
        else:
            label = f"{key} 节点群"
        
        ax.plot(data['accuracies'], label=label, linewidth=2)
    
    ax.set_xlabel('全局 Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('准确率对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('f:/sel-lab/exploration/advanced_edge_swarm_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ 结果图已保存: f:/sel-lab/exploration/advanced_edge_swarm_results.png")
    plt.show()


if __name__ == "__main__":
    test_advanced_edge_swarm()
