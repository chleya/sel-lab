# -*- coding: utf-8 -*-
"""
边缘演化智能群 (Edge Evolutionary Intelligence Swarm)

概念：
- 多个边缘节点组成智能群体
- 每个节点运行 EGGROLL 优化器
- 节点间通过通信共享最优解
- 分布式演化，资源受限环境

特点：
1. 边缘计算：在资源受限设备上运行
2. 群体智能：节点间协同工作
3. 演化算法：通过进化优化
4. 分布式计算：并行处理
"""

import numpy as np
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import threading


@dataclass
class EdgeNodeConfig:
    """边缘节点配置"""
    rank: int = 2              # 低秩秩数
    sigma: float = 0.1         # 扰动标准差
    population_size: int = 8   # 种群大小（边缘设备资源受限）
    learning_rate: float = 0.02 # 学习率
    communication_freq: int = 10 # 通信频率
    local_epochs: int = 5      # 本地训练轮数


class EdgeNode:
    """边缘节点"""
    
    def __init__(self, node_id: int, in_size: int, out_size: int, 
                 config: EdgeNodeConfig, seed: int = None):
        self.node_id = node_id
        self.rng = np.random.default_rng(seed)
        self.in_size = in_size
        self.out_size = out_size
        self.config = config
        
        # 本地模型
        self.W = self.rng.normal(0.0, np.sqrt(2.0 / in_size), (in_size, out_size))
        
        # 统计
        self.local_losses = []
        self.global_losses = []
        self.communication_count = 0
        
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
        self.local_losses.append(loss)
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
    
    def update_model(self, new_W: np.ndarray):
        """更新模型（从其他节点）"""
        with self.lock:
            # 模型融合：加权平均
            self.W = 0.7 * self.W + 0.3 * new_W
            self.communication_count += 1


class EdgeSwarm:
    """边缘演化智能群"""
    
    def __init__(self, n_nodes: int, in_size: int, out_size: int, 
                 config: EdgeNodeConfig, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_nodes = n_nodes
        self.in_size = in_size
        self.out_size = out_size
        self.config = config
        
        # 创建边缘节点
        self.nodes = []
        for i in range(n_nodes):
            node = EdgeNode(i, in_size, out_size, config, seed=seed+i)
            self.nodes.append(node)
        
        # 通信网络
        self.communication_network = self._build_communication_network()
        
        # 全局统计
        self.global_epoch = 0
        self.swarm_losses = []
    
    def _build_communication_network(self) -> Dict[int, List[int]]:
        """构建通信网络（环形拓扑）"""
        network = {}
        for i in range(self.n_nodes):
            # 每个节点与左右邻居通信
            left = (i - 1) % self.n_nodes
            right = (i + 1) % self.n_nodes
            network[i] = [left, right]
        return network
    
    def communicate(self):
        """节点间通信"""
        # 每个节点向邻居发送模型
        for node_id, neighbors in self.communication_network.items():
            current_node = self.nodes[node_id]
            current_model = current_node.get_model()
            
            # 向邻居发送模型
            for neighbor_id in neighbors:
                neighbor_node = self.nodes[neighbor_id]
                neighbor_node.update_model(current_model)
    
    def train(self, X: np.ndarray, y: np.ndarray, global_epochs: int = 50):
        """训练整个群体"""
        print(f"\n开始训练边缘演化智能群 ({self.n_nodes} 节点)")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(global_epochs):
            # 本地训练
            local_threads = []
            local_losses = [0.0] * self.n_nodes
            
            def train_node(node_id):
                node = self.nodes[node_id]
                for _ in range(self.config.local_epochs):
                    loss = node.local_update(X, y)
                local_losses[node_id] = node.compute_loss(X, y)
            
            # 并行本地训练
            for i in range(self.n_nodes):
                thread = threading.Thread(target=train_node, args=(i,))
                local_threads.append(thread)
                thread.start()
            
            # 等待所有节点完成
            for thread in local_threads:
                thread.join()
            
            # 通信（按频率）
            if epoch % self.config.communication_freq == 0:
                self.communicate()
            
            # 计算群体平均损失
            avg_loss = np.mean(local_losses)
            self.swarm_losses.append(avg_loss)
            
            # 计算群体准确率
            avg_acc = np.mean([node.compute_accuracy(X, y) for node in self.nodes])
            
            if (epoch + 1) % 10 == 0:
                print(f"全局 Epoch {epoch+1:2d}: 群体平均损失 = {avg_loss:.4f}, 准确率 = {avg_acc:.4f}")
            
            self.global_epoch += 1
        
        total_time = time.time() - start_time
        print(f"\n训练完成: 时间 = {total_time:.3f}s")
        
        return self.swarm_losses
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """评估群体性能"""
        # 集成所有节点的预测
        predictions = []
        for node in self.nodes:
            y_pred = node.forward(X)
            predictions.append(y_pred)
        
        # 多数投票
        avg_pred = np.mean(predictions, axis=0)
        final_pred = np.argmax(avg_pred, axis=1)
        labels = np.argmax(y, axis=1)
        
        return float(np.mean(final_pred == labels))
    
    def get_best_node(self) -> EdgeNode:
        """获取表现最好的节点"""
        best_node = None
        best_acc = -1
        
        for node in self.nodes:
            acc = node.compute_accuracy(self.test_X, self.test_y)
            if acc > best_acc:
                best_acc = acc
                best_node = node
        
        return best_node


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

def test_edge_swarm():
    """测试边缘演化智能群"""
    print("\n" + "="*70)
    print("边缘演化智能群测试")
    print("="*70)
    
    # 创建数据
    X, y = create_synthetic_data(n_samples=1000, in_size=10, out_size=2)
    print(f"数据: {X.shape[0]} 样本, 输入维度={X.shape[1]}, 输出维度={y.shape[1]}")
    
    # 配置
    config = EdgeNodeConfig(
        rank=2,
        sigma=0.1,
        population_size=8,  # 边缘设备资源受限
        learning_rate=0.02,
        communication_freq=5,
        local_epochs=3
    )
    
    # 测试不同节点数量
    node_counts = [2, 4, 8]
    results = {}
    
    for n_nodes in node_counts:
        print(f"\n测试 {n_nodes} 个边缘节点:")
        print("-"*60)
        
        swarm = EdgeSwarm(n_nodes, 10, 2, config, seed=42)
        swarm.test_X = X  # 保存测试数据
        swarm.test_y = y
        
        # 训练
        losses = swarm.train(X, y, global_epochs=50)
        
        # 评估
        accuracy = swarm.evaluate(X, y)
        print(f"  最终准确率: {accuracy:.4f}")
        
        results[n_nodes] = {
            'losses': losses,
            'accuracy': accuracy,
            'nodes': n_nodes
        }
    
    # 与单机 EGGROLL 对比
    print(f"\n与单机 EGGROLL 对比:")
    print("-"*60)
    
    single_node = EdgeNode(0, 10, 2, config, seed=42)
    single_losses = []
    
    start_time = time.time()
    for epoch in range(50):
        for _ in range(config.local_epochs):
            single_node.local_update(X, y)
        loss = single_node.compute_loss(X, y)
        single_losses.append(loss)
        
        if (epoch + 1) % 10 == 0:
            acc = single_node.compute_accuracy(X, y)
            print(f"  Epoch {epoch+1:2d}: Loss = {loss:.4f}, Acc = {acc:.4f}")
    
    single_time = time.time() - start_time
    single_acc = single_node.compute_accuracy(X, y)
    print(f"  单机 EGGROLL: 准确率={single_acc:.4f}, 时间={single_time:.3f}s")
    
    results['single'] = {
        'losses': single_losses,
        'accuracy': single_acc,
        'nodes': 1
    }
    
    # 绘制结果
    plot_results(results)
    
    print("\n" + "="*70)
    print("边缘演化智能群测试完成！")
    print("="*70)


def plot_results(results: Dict):
    """绘制结果"""
    plt.figure(figsize=(12, 6))
    
    for key, data in results.items():
        if key == 'single':
            label = f"单机 EGGROLL"
        else:
            label = f"{key} 节点群"
        
        plt.plot(data['losses'], label=label, linewidth=2)
    
    plt.xlabel('全局 Epoch')
    plt.ylabel('损失')
    plt.title('边缘演化智能群 vs 单机 EGGROLL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('f:/sel-lab/exploration/edge_swarm_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ 结果图已保存: f:/sel-lab/exploration/edge_swarm_results.png")
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_edge_swarm()
