# -*- coding: utf-8 -*-
"""
SEL (Structural Evolution Learning) 增强版模块

整合 Phase 3 研究成果的完整实现，包含：
- 张力驱动的结构演化 (tension-driven structural evolution)
- 模块克隆与剪枝 (clone/prune)
- 直接反馈对齐 (DFA)
- 自动结构优化 (auto-optimization to 3-5 modules)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


@dataclass
class SELConfig:
    """SEL 配置"""
    input_size: int = 3
    output_size: int = 2
    initial_modules: int = 3
    learning_rate: float = 0.1
    momentum: float = 0.9
    tension_threshold: float = 0.7
    max_modules: int = 5
    mutation_rate: float = 0.2
    epochs: int = 100
    random_seed: Optional[int] = 42
    tension_window: int = 8
    min_improvement: float = 1e-4
    clone_perturbation: float = 0.05
    adapt_noise_scale: float = 0.03
    clone_cooldown: int = 5
    adapt_cooldown: int = 2
    min_clone_source_age: int = 10
    clone_patience: int = 4
    activation: str = "tanh"


class SELModule:
    """SEL 模块 - 带张力感知和结构演化能力"""
    
    def __init__(
        self,
        name: str,
        in_size: int,
        out_size: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        clone_from: Optional["SELModule"] = None,
        clone_perturbation: float = 0.05,
    ):
        self.rng = rng or np.random.default_rng(seed)
        self.name = name
        self.in_size = in_size
        self.out_size = out_size
        
        if clone_from is None:
            self.weights = self.rng.normal(0.0, np.sqrt(2.0 / in_size), size=(in_size, out_size))
            self.feedback_matrix = self.rng.normal(0.0, 0.1, size=(out_size, out_size))
            self.velocity = np.zeros((in_size, out_size))
            self.local_tension = 0.0
        else:
            self.weights = clone_from.weights.copy()
            self.feedback_matrix = clone_from.feedback_matrix.copy()
            self.velocity = clone_from.velocity.copy() * 0.5
            self.local_tension = clone_from.local_tension * 0.7
            self.weights += self.rng.normal(0.0, clone_perturbation, size=self.weights.shape)
            self.feedback_matrix += self.rng.normal(
                0.0, clone_perturbation, size=self.feedback_matrix.shape
            )
        
        self.error_history: List[float] = []
        self.loss_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self.improvement_history: List[float] = []
        self.age = 0
        self.last_action_epoch = -10**9
        self._epoch_loss_sum = 0.0
        self._epoch_steps = 0
        self._epoch_grad_norms: List[float] = []
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        x = np.atleast_2d(x)
        return np.tanh(x @ self.weights)
    
    def project_feedback(self, output_error: np.ndarray) -> np.ndarray:
        """通过 DFA 投影输出误差"""
        output_error = np.atleast_2d(output_error)
        return (self.feedback_matrix.T @ output_error.T).T
    
    def _update_tension(self, loss: float, grad_norm: float, window: int, min_improvement: float) -> None:
        """更新张力指标"""
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        if len(self.loss_history) > 1:
            self.improvement_history.append(self.loss_history[-2] - self.loss_history[-1])
        
        if len(self.loss_history) > window:
            self.loss_history.pop(0)
        if len(self.grad_norm_history) > window:
            self.grad_norm_history.pop(0)
        if len(self.improvement_history) > window - 1:
            self.improvement_history.pop(0)
        
        recent_improvement = float(np.mean(self.improvement_history)) if self.improvement_history else 0.0
        plateau = max(0.0, min_improvement - recent_improvement) / max(min_improvement, 1e-8)
        plateau = float(np.clip(plateau, 0.0, 1.5))
        
        grad_mean = float(np.mean(self.grad_norm_history)) if self.grad_norm_history else 0.0
        grad_std = float(np.std(self.grad_norm_history)) if self.grad_norm_history else 0.0
        volatility = float(np.tanh(grad_std / (grad_mean + 1e-8)))
        
        residual = float(np.tanh(np.mean(self.loss_history)))
        self.local_tension = float(0.45 * plateau + 0.35 * volatility + 0.20 * residual)
    
    def finalize_epoch(self, window: int, min_improvement: float) -> None:
        """从 epoch 级别摘要更新张力"""
        if self._epoch_steps == 0:
            return
        
        avg_loss = self._epoch_loss_sum / self._epoch_steps
        avg_grad_norm = float(np.mean(self._epoch_grad_norms)) if self._epoch_grad_norms else 0.0
        self._update_tension(avg_loss, avg_grad_norm, window, min_improvement)
        self._epoch_loss_sum = 0.0
        self._epoch_steps = 0
        self._epoch_grad_norms.clear()
    
    def forward_learning(
        self,
        x: np.ndarray,
        output_error: np.ndarray,
        lr: float = 0.1,
        momentum: float = 0.9,
        tension_window: int = 8,
        min_improvement: float = 1e-3,
    ) -> float:
        """仅前向学习（使用 DFA）"""
        x = np.atleast_2d(x)
        output_error = np.atleast_2d(output_error)
        
        fb_error = self.project_feedback(output_error)
        grad = x.T @ fb_error
        self.velocity = momentum * self.velocity + lr * grad
        self.weights += self.velocity
        self.weights = np.clip(self.weights, -2.0, 2.0)
        
        loss = float(np.mean(output_error ** 2))
        grad_norm = float(np.linalg.norm(grad))
        self.error_history.append(float(np.mean(np.abs(output_error))))
        if len(self.error_history) > tension_window:
            self.error_history.pop(0)
        
        self._epoch_loss_sum += loss
        self._epoch_steps += 1
        self._epoch_grad_norms.append(grad_norm)
        self.age += 1
        return loss
    
    def can_act(self, epoch: int, cooldown: int) -> bool:
        """检查是否可以执行结构动作"""
        return len(self.loss_history) >= 2 and (epoch - self.last_action_epoch) >= cooldown
    
    def clone(self, name: str, clone_perturbation: float = 0.05) -> "SELModule":
        """克隆模块"""
        return SELModule(
            name=name,
            in_size=self.in_size,
            out_size=self.out_size,
            rng=self.rng,
            clone_from=self,
            clone_perturbation=clone_perturbation,
        )
    
    def adapt(
        self,
        epoch: int,
        noise_scale: float = 0.03,
        mutation_rate: float = 0.2,
    ) -> Tuple[bool, str]:
        """适应权重"""
        if not self.grad_norm_history:
            return False, "stable"
        
        mask = self.rng.random(self.weights.shape) < mutation_rate
        perturbation = self.rng.normal(0.0, noise_scale, size=self.weights.shape)
        self.weights[mask] += perturbation[mask]
        self.weights = np.clip(self.weights, -2.0, 2.0)
        self.local_tension *= 0.65
        self.last_action_epoch = epoch
        return True, "adapted"
    
    @property
    def plateau_score(self) -> float:
        """平台期分数"""
        if not self.improvement_history:
            return 0.0
        return max(0.0, -float(np.mean(self.improvement_history)))
    
    @property
    def param_count(self) -> int:
        """参数数量"""
        return self.weights.size + self.feedback_matrix.size
    
    def get_weights(self) -> np.ndarray:
        """获取权重"""
        return self.weights.copy()
    
    def set_weights(self, weights: np.ndarray):
        """设置权重"""
        self.weights = weights.copy()
    
    def __repr__(self) -> str:
        return f"SELModule({self.name}, tension={self.local_tension:.3f}, params={self.param_count})"


class SELNetwork:
    """模块化 SEL 网络 - 带张力驱动的结构演化"""
    
    def __init__(self, config: SELConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.modules: List[SELModule] = []
        self.epoch_loss_history: List[float] = []
        self.current_epoch = -1
        self.last_clone_epoch = -10**9
        self.stalled_epochs = 0
        
        for i in range(config.initial_modules):
            self.add_module(f"m{i}")
    
    def add_module(
        self,
        name: str,
        clone_from: Optional[int | SELModule] = None,
        seed: Optional[int] = None,
    ) -> SELModule:
        """添加模块，可选从现有模块克隆"""
        source_module = None
        if isinstance(clone_from, int) and 0 <= clone_from < len(self.modules):
            source_module = self.modules[clone_from]
        elif isinstance(clone_from, SELModule):
            source_module = clone_from
        
        if source_module is None:
            module = SELModule(
                name,
                self.config.input_size,
                self.config.output_size,
                rng=self.rng if seed is None else np.random.default_rng(seed),
            )
        else:
            module = source_module.clone(name, clone_perturbation=self.config.clone_perturbation)
            module.local_tension = source_module.local_tension
        
        self.modules.append(module)
        return module
    
    def remove_module(self, index: int) -> Optional[SELModule]:
        """移除模块，同时保持至少一个存活"""
        if len(self.modules) > 1 and -len(self.modules) <= index < len(self.modules):
            return self.modules.pop(index)
        return None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        x = np.atleast_2d(x)
        outputs = [module.forward(x) for module in self.modules]
        return np.mean(outputs, axis=0)
    
    def forward_learning(self, x: np.ndarray, target: np.ndarray) -> float:
        """前向学习"""
        output = self.forward(x)
        target = np.atleast_2d(target)
        error = target - output
        
        losses = []
        for module in self.modules:
            losses.append(
                module.forward_learning(
                    x,
                    error,
                    lr=self.config.learning_rate,
                    momentum=self.config.momentum,
                    tension_window=self.config.tension_window,
                    min_improvement=self.config.min_improvement,
                )
            )
        
        return float(np.mean(losses))
    
    def predict(self, x: np.ndarray) -> int:
        """预测"""
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """准确率"""
        correct = sum(
            1 for i in range(len(X)) if self.predict(X[i]) == int(np.argmax(y[i]))
        )
        return correct / len(X)
    
    def _network_plateau_score(self) -> float:
        """网络平台期分数"""
        if len(self.epoch_loss_history) < self.config.tension_window:
            return 0.0
        recent = self.epoch_loss_history[-self.config.tension_window :]
        improvements = [recent[i] - recent[i + 1] for i in range(len(recent) - 1)]
        if not improvements:
            return 0.0
        mean_improvement = float(np.mean(improvements))
        return max(0.0, self.config.min_improvement - mean_improvement) / max(
            self.config.min_improvement, 1e-8
        )
    
    def _select_clone_source(self) -> Optional[int]:
        """选择克隆源"""
        eligible = [
            (idx, module)
            for idx, module in enumerate(self.modules)
            if module.age >= self.config.min_clone_source_age
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda item: item[1].local_tension)[0]
    
    def structural_evolution(self, epoch_loss: Optional[float] = None, epoch: Optional[int] = None) -> List[Tuple]:
        """基于张力触发 CLONE 或 ADAPT"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if epoch_loss is not None:
            self.epoch_loss_history.append(float(epoch_loss))
            if len(self.epoch_loss_history) > self.config.tension_window:
                self.epoch_loss_history.pop(0)
        
        changes: List[Tuple] = []
        if not self.modules:
            return changes
        
        for module in self.modules:
            module.finalize_epoch(
                window=self.config.tension_window,
                min_improvement=self.config.min_improvement,
            )
        
        network_plateau = self._network_plateau_score()
        avg_tension = float(np.mean([module.local_tension for module in self.modules]))
        pressured_modules = [
            (idx, module)
            for idx, module in enumerate(self.modules)
            if len(module.loss_history) >= self.config.tension_window
            and module.local_tension >= self.config.tension_threshold
        ]
        
        source_idx = self._select_clone_source()
        source_ready = (
            source_idx is not None
            and self.modules[source_idx].local_tension < avg_tension
        )
        
        if network_plateau >= 0.75 and avg_tension >= (self.config.tension_threshold * 1.1):
            self.stalled_epochs += 1
        else:
            self.stalled_epochs = 0
        
        # 自动结构优化：保持 3-5 模块范围
        if len(self.modules) < 3 and network_plateau >= 0.5:
            if source_ready:
                source_name = self.modules[source_idx].name
                new_name = f"{source_name}_clone_{len(self.modules)}"
                self.add_module(new_name, clone_from=source_idx)
                self.last_clone_epoch = self.current_epoch
                changes.append((new_name, "cloned for optimal structure", source_name))
        elif len(self.modules) > 5:
            while len(self.modules) > 5:
                # 选择张力最低的模块进行剪枝（张力低 = 模块已停滞）
                # 张力高 = 模块正在努力学习，应保留
                if pressured_modules:
                    prune_idx = min(pressured_modules, key=lambda item: item[1].local_tension)[0]
                else:
                    # 如果没有压力模块，选择张力最低的模块
                    prune_idx = min(range(len(self.modules)), 
                                   key=lambda i: self.modules[i].local_tension)
                
                removed_module = self.remove_module(prune_idx)
                if removed_module:
                    changes.append((removed_module.name, "pruned for optimal structure"))
                
                pressured_modules = [
                    (idx, module)
                    for idx, module in enumerate(self.modules)
                    if len(module.loss_history) >= self.config.tension_window
                    and module.local_tension >= self.config.tension_threshold
                ]
        
        can_clone = (
            bool(pressured_modules)
            and len(self.epoch_loss_history) >= self.config.tension_window
            and self.stalled_epochs >= self.config.clone_patience
            and len(self.modules) < self.config.max_modules
            and (self.current_epoch - self.last_clone_epoch) >= self.config.clone_cooldown
            and source_ready
        )
        if can_clone:
            source_name = self.modules[source_idx].name
            new_name = f"{source_name}_clone_{len(self.modules)}"
            self.add_module(new_name, clone_from=source_idx)
            self.last_clone_epoch = self.current_epoch
            changes.append((new_name, "cloned", source_name))
        
        for idx, module in sorted(pressured_modules, key=lambda item: item[1].local_tension, reverse=True)[:1]:
            if not module.can_act(self.current_epoch, self.config.adapt_cooldown):
                continue
            changed, reason = module.adapt(
                epoch=self.current_epoch,
                noise_scale=self.config.adapt_noise_scale,
                mutation_rate=self.config.mutation_rate,
            )
            if changed:
                changes.append((module.name, reason))
        
        return changes
    
    def get_weights(self) -> np.ndarray:
        """获取所有模块的平均权重"""
        if not self.modules:
            return np.array([])
        weights = [module.get_weights() for module in self.modules]
        return np.mean(weights, axis=0)
    
    def set_weights(self, weights: np.ndarray):
        """设置所有模块的权重"""
        for module in self.modules:
            module.set_weights(weights)
    
    @property
    def param_count(self) -> int:
        """参数总数"""
        return sum(module.param_count for module in self.modules)
    
    def __repr__(self) -> str:
        return f"SELNetwork(modules={len(self.modules)}, params={self.param_count})"
