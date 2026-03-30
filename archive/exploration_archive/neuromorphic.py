# -*- coding: utf-8 -*-
"""
SEL-Lab Neuromorphic Exploration
神经形态计算与脉冲神经网络结合

探索方向:
1. 将 SEL 的结构进化与脉冲神经网络 (SNN) 结合
2. 事件驱动的学习和进化
3. 时间编码和时序模式学习
4. 低功耗硬件适配
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import resolve_results_path, save_json


@dataclass
class SNNConfig:
    """脉冲神经网络配置"""
    input_size: int = 16
    hidden_size: int = 32
    output_size: int = 2
    time_steps: int = 20          # 模拟时间步长
    tau_mem: float = 20.0         # 膜时间常数 (ms)
    tau_syn: float = 5.0          # 突触时间常数 (ms)
    v_thresh: float = 1.0         # 发放阈值
    v_reset: float = 0.0          # 重置电位
    dt: float = 1.0               # 时间步长 (ms)
    learning_rate: float = 0.01
    max_neurons: int = 64         # 最大神经元数（进化用）


class LIFNeuron:
    """Leaky Integrate-and-Fire 神经元"""
    
    def __init__(self, config: SNNConfig, neuron_id: int):
        self.config = config
        self.neuron_id = neuron_id
        self.v = config.v_reset  # 膜电位
        self.i_syn = 0.0         # 突触电流
        self.spike_history: List[int] = []
        self.v_history: List[float] = []
        
        # 可学习参数
        self.tau_adapt = config.tau_mem * 0.5  # 自适应时间常数
        self.adaptation = 0.0                   # 自适应电流
        
    def reset(self):
        """重置神经元状态"""
        self.v = self.config.v_reset
        self.i_syn = 0.0
        self.adaptation = 0.0
        self.spike_history.clear()
        self.v_history.clear()
    
    def step(self, input_current: float) -> int:
        """模拟一个时间步，返回是否发放脉冲"""
        cfg = self.config
        
        # 更新突触电流（指数衰减）
        self.i_syn = self.i_syn * np.exp(-cfg.dt / cfg.tau_syn) + input_current
        
        # 更新膜电位（LIF 模型）
        dv = (-(self.v - cfg.v_reset) + self.i_syn - self.adaptation) * (cfg.dt / cfg.tau_mem)
        self.v += dv
        
        # 检查发放
        spike = 1 if self.v >= cfg.v_thresh else 0
        
        if spike:
            self.v = cfg.v_reset
            self.adaptation += 0.1  # 增加自适应电流
        
        # 自适应电流衰减
        self.adaptation *= np.exp(-cfg.dt / self.tau_adapt)
        
        # 记录历史
        self.spike_history.append(spike)
        self.v_history.append(self.v)
        
        return spike


class SpikingLayer:
    """脉冲神经网络层，支持结构进化"""
    
    def __init__(self, config: SNNConfig, layer_id: str, in_size: int, out_size: int):
        self.config = config
        self.layer_id = layer_id
        self.in_size = in_size
        self.out_size = out_size
        
        # 神经元群体
        self.neurons: List[LIFNeuron] = []
        self.add_neurons(out_size)
        
        # 突触权重 (前向连接)
        self.weights = np.random.randn(in_size, out_size) * 0.1
        
        # 反馈矩阵（用于 DFA）
        self.feedback = np.random.randn(out_size, out_size) * 0.05
        
        # 结构进化状态
        self.tension = 0.5
        self.age = 0
        self.activity_trace = np.zeros(out_size)
        
    def add_neurons(self, count: int):
        """添加新神经元（结构进化）"""
        start_id = len(self.neurons)
        for i in range(count):
            if len(self.neurons) < self.config.max_neurons:
                self.neurons.append(LIFNeuron(self.config, start_id + i))
        self.out_size = len(self.neurons)
    
    def clone_neuron(self, source_idx: int, perturbation: float = 0.05):
        """克隆现有神经元并添加扰动"""
        if len(self.neurons) >= self.config.max_neurons:
            return -1
        
        new_id = len(self.neurons)
        new_neuron = LIFNeuron(self.config, new_id)
        
        # 继承源神经元的参数特征
        source = self.neurons[source_idx]
        new_neuron.tau_adapt = source.tau_adapt * (1 + np.random.randn() * perturbation)
        
        self.neurons.append(new_neuron)
        
        # 扩展权重矩阵
        new_col = self.weights[:, source_idx:source_idx+1] + np.random.randn(self.in_size, 1) * perturbation
        self.weights = np.hstack([self.weights, new_col])
        
        new_fb = self.feedback[:, source_idx:source_idx+1] + np.random.randn(self.out_size, 1) * perturbation * 0.5
        self.feedback = np.hstack([self.feedback, new_col])
        
        self.out_size = len(self.neurons)
        return new_id
    
    def forward(self, spike_input: np.ndarray, time_steps: Optional[int] = None) -> np.ndarray:
        """前向传播，返回脉冲计数或发放率"""
        t_steps = time_steps or self.config.time_steps
        
        # 重置所有神经元
        for neuron in self.neurons:
            neuron.reset()
        
        # 模拟时间步
        spike_counts = np.zeros(self.out_size)
        
        for t in range(t_steps):
            # 计算输入电流
            input_current = spike_input @ self.weights
            
            # 更新每个神经元
            for i, neuron in enumerate(self.neurons):
                spike = neuron.step(input_current[i])
                spike_counts[i] += spike
        
        # 更新活动迹
        self.activity_trace = 0.9 * self.activity_trace + 0.1 * (spike_counts / t_steps)
        
        return spike_counts / t_steps  # 返回发放率
    
    def compute_tension(self) -> float:
        """计算层的张力（用于结构进化决策）"""
        # 基于活动迹的变异系数
        if np.mean(self.activity_trace) > 0:
            cv = np.std(self.activity_trace) / (np.mean(self.activity_trace) + 1e-8)
        else:
            cv = 0
        
        # 基于脉冲同步性的张力
        sync_measure = 0
        if len(self.neurons) > 1:
            spike_correlations = []
            for i in range(len(self.neurons)):
                for j in range(i+1, len(self.neurons)):
                    if len(self.neurons[i].spike_history) > 0 and len(self.neurons[j].spike_history) > 0:
                        corr = np.corrcoef(
                            self.neurons[i].spike_history, 
                            self.neurons[j].spike_history
                        )[0, 1]
                        spike_correlations.append(abs(corr))
            if spike_correlations:
                sync_measure = np.mean(spike_correlations)
        
        self.tension = 0.5 * cv + 0.3 * (1 - sync_measure) + 0.2 * (self.age / 1000)
        return self.tension


class EvolvingSNN:
    """可进化的脉冲神经网络"""
    
    def __init__(self, config: SNNConfig):
        self.config = config
        self.layers: List[SpikingLayer] = []
        
        # 初始结构
        self.layers.append(SpikingLayer(config, "input", config.input_size, config.hidden_size))
        self.layers.append(SpikingLayer(config, "output", config.hidden_size, config.output_size))
        
        self.epoch_count = 0
        self.performance_history: List[float] = []
        
    def forward(self, x: np.ndarray, time_steps: Optional[int] = None) -> np.ndarray:
        """前向传播"""
        # 将输入转换为脉冲序列（速率编码）
        spike_input = self._rate_encode(x, time_steps or self.config.time_steps)
        
        # 逐层传播
        for layer in self.layers:
            spike_input = layer.forward(spike_input, time_steps)
        
        return spike_input
    
    def _rate_encode(self, x: np.ndarray, time_steps: int) -> np.ndarray:
        """将连续值编码为脉冲发放率"""
        # 简单的泊松编码
        rates = np.clip(np.abs(x), 0, 1)
        return rates
    
    def evolve_structure(self):
        """基于张力进行结构进化"""
        for layer in self.layers:
            tension = layer.compute_tension()
            
            # 高张力 -> 添加神经元
            if tension > 0.7 and len(layer.neurons) < self.config.max_neurons:
                if layer.neurons:
                    # 克隆最活跃的神经元
                    best_idx = np.argmax(layer.activity_trace[:len(layer.neurons)])
                    layer.clone_neuron(best_idx)
            
            # 低张力且年龄大 -> 考虑剪枝（可选）
            elif tension < 0.3 and layer.age > 100:
                pass  # 剪枝策略可在此实现
            
            layer.age += 1
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """单步训练（使用 DFA 的脉冲版本）"""
        # 简化的训练：基于奖励的突触可塑性
        predictions = []
        
        for i in range(len(X)):
            out = self.forward(X[i])
            predictions.append(out)
        
        predictions = np.array(predictions)
        targets = np.array(y)
        
        # 计算误差
        error = np.mean((predictions - targets) ** 2)
        
        # 更新权重（简化的 STDP 风格）
        for layer in self.layers:
            # 基于活动迹的权重调整
            for i in range(layer.weights.shape[0]):
                for j in range(layer.weights.shape[1]):
                    if j < len(layer.neurons):
                        # 简单的可塑性规则
                        pre_activity = np.random.rand()  # 简化：应该是实际的前突触活动
                        post_activity = layer.activity_trace[j]
                        
                        # STDP 风格更新
                        dw = self.config.learning_rate * (pre_activity * post_activity - 0.01 * layer.weights[i, j])
                        layer.weights[i, j] += dw
        
        return error
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """评估准确率"""
        correct = 0
        for i in range(len(X)):
            out = self.forward(X[i])
            pred = np.argmax(out)
            true = np.argmax(y[i])
            if pred == true:
                correct += 1
        return correct / len(X)


def run_neuromorphic_exploration(
    runs: int = 3,
    epochs: int = 50,
    verbose: bool = True
) -> Dict:
    """运行神经形态探索实验"""
    
    print("=" * 60)
    print("Neuromorphic SEL Exploration")
    print("=" * 60)
    
    results = {
        "runs": [],
        "summary": {}
    }
    
    for run in range(runs):
        print(f"\n--- Run {run+1}/{runs} ---")
        
        config = SNNConfig(
            input_size=4,
            hidden_size=8,
            output_size=2,
            time_steps=20,
            max_neurons=16
        )
        
        model = EvolvingSNN(config)
        
        # 生成简单合成数据
        np.random.seed(42 + run)
        X_train = np.random.randn(100, 4)
        y_train = np.array([[1, 0] if np.sum(x) > 0 else [0, 1] for x in X_train])
        X_test = np.random.randn(50, 4)
        y_test = np.array([[1, 0] if np.sum(x) > 0 else [0, 1] for x in X_test])
        
        train_errors = []
        test_accuracies = []
        neuron_counts = []
        
        for epoch in range(epochs):
            # 训练
            error = model.train_step(X_train, y_train)
            train_errors.append(error)
            
            # 评估
            acc = model.evaluate(X_test, y_test)
            test_accuracies.append(acc)
            
            # 记录结构
            total_neurons = sum(len(layer.neurons) for layer in model.layers)
            neuron_counts.append(total_neurons)
            
            # 进化
            if epoch % 10 == 0:
                model.evolve_structure()
            
            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch}: error={error:.4f}, acc={acc:.2%}, neurons={total_neurons}")
        
        run_result = {
            "final_error": train_errors[-1],
            "final_accuracy": test_accuracies[-1],
            "max_accuracy": max(test_accuracies),
            "final_neurons": neuron_counts[-1],
            "train_errors": train_errors,
            "test_accuracies": test_accuracies,
            "neuron_counts": neuron_counts
        }
        results["runs"].append(run_result)
        
        print(f"  Final: acc={test_accuracies[-1]:.2%}, neurons={neuron_counts[-1]}")
    
    # 汇总
    results["summary"] = {
        "mean_final_accuracy": np.mean([r["final_accuracy"] for r in results["runs"]]),
        "std_final_accuracy": np.std([r["final_accuracy"] for r in results["runs"]]),
        "mean_max_accuracy": np.mean([r["max_accuracy"] for r in results["runs"]]),
        "mean_final_neurons": np.mean([r["final_neurons"] for r in results["runs"]])
    }
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Mean final accuracy: {results['summary']['mean_final_accuracy']:.2%} ± {results['summary']['std_final_accuracy']:.2%}")
    print(f"  Mean max accuracy: {results['summary']['mean_max_accuracy']:.2%}")
    print(f"  Mean final neurons: {results['summary']['mean_final_neurons']:.1f}")
    print("=" * 60)
    
    # 保存结果
    save_path = resolve_results_path("neuromorphic_exploration_results.json")
    save_json(results, save_path)
    print(f"\nResults saved to: {save_path}")
    
    return results


if __name__ == "__main__":
    run_neuromorphic_exploration(runs=3, epochs=50)
