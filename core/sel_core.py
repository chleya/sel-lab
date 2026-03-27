# -*- coding: utf-8 -*-
"""
SEL Core Framework
Structural Evolution Learning - Core Implementation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SELConfig:
    """Configuration for the SEL core."""

    input_size: int
    output_size: int
    initial_modules: int = 3
    learning_rate: float = 0.1
    momentum: float = 0.9
    tension_threshold: float = 0.7
    max_modules: int = 10
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


@dataclass
class TrainingMetrics:
    """Per-epoch training metrics."""

    epoch: int
    train_accuracy: float
    test_accuracy: float
    avg_tension: float
    module_count: int
    structural_changes: int = 0
    avg_loss: float = 0.0


class SELModule:
    """Single SEL module with DFA learning and tension-aware adaptation."""

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
        """Forward pass."""
        x = np.atleast_2d(x)
        return np.tanh(x @ self.weights)

    def project_feedback(self, output_error: np.ndarray) -> np.ndarray:
        """Project output error through DFA as B^T @ error."""
        output_error = np.atleast_2d(output_error)
        return (self.feedback_matrix.T @ output_error.T).T

    def _update_tension(self, loss: float, grad_norm: float, window: int, min_improvement: float) -> None:
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
        """Update tension from epoch-level summaries instead of sample noise."""
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
        """Forward-only learning with DFA."""
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
        return len(self.loss_history) >= 2 and (epoch - self.last_action_epoch) >= cooldown

    def clone(self, name: str, clone_perturbation: float = 0.05) -> "SELModule":
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
        """Adapt weights only when tension indicates stagnation or unstable gradients."""
        if not self.grad_norm_history:
            return False, "stable"

        mask = self.rng.random(self.weights.shape) < mutation_rate
        perturbation = self.rng.normal(0.0, noise_scale, size=self.weights.shape)
        self.weights[mask] += perturbation[mask]
        self.weights = np.clip(self.weights, -2.0, 2.0)
        self.local_tension *= 0.65
        self.last_action_epoch = epoch
        return True, "adapted"

    def structural_adaptation(
        self,
        threshold: float = 0.7,
        epoch: int = 0,
        noise_scale: float = 0.03,
        mutation_rate: float = 0.2,
    ) -> Tuple[bool, str]:
        """Backward-compatible wrapper for tension-driven adaptation."""
        if self.local_tension < threshold:
            return False, "stable"
        return self.adapt(epoch=epoch, noise_scale=noise_scale, mutation_rate=mutation_rate)

    def mutate_weights(self, rate: float = 0.2, magnitude: float = 0.1) -> None:
        """Backward-compatible local mutation helper."""
        mask = self.rng.random(self.weights.shape) < rate
        perturbation = self.rng.normal(0.0, magnitude, size=self.weights.shape)
        self.weights[mask] += perturbation[mask]
        self.weights = np.clip(self.weights, -2.0, 2.0)

    @property
    def plateau_score(self) -> float:
        if not self.improvement_history:
            return 0.0
        return max(0.0, -float(np.mean(self.improvement_history)))

    @property
    def param_count(self) -> int:
        return self.weights.size + self.feedback_matrix.size

    @property
    def weight_norm(self) -> float:
        return float(np.mean(np.abs(self.weights)))

    def serialize(self) -> Dict:
        return {
            "name": self.name,
            "weights": self.weights.tolist(),
            "feedback_matrix": self.feedback_matrix.tolist(),
            "velocity": self.velocity.tolist(),
            "local_tension": self.local_tension,
            "age": self.age,
        }

    def load_state(self, data: Dict) -> None:
        self.name = data.get("name", self.name)
        self.weights = np.array(data["weights"], dtype=float)
        self.feedback_matrix = np.array(data["feedback_matrix"], dtype=float)
        self.velocity = np.array(data.get("velocity", np.zeros_like(self.weights)), dtype=float)
        self.local_tension = float(data.get("local_tension", 0.0))
        self.age = int(data.get("age", 0))

    def __repr__(self) -> str:
        return f"SELModule({self.name}, tension={self.local_tension:.3f}, params={self.param_count})"


class SELNetwork:
    """Modular SEL network with tension-driven structural evolution."""

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
        """Add a module, optionally cloning from an existing one."""
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
        """Remove a module while keeping at least one alive."""
        if len(self.modules) > 1 and -len(self.modules) <= index < len(self.modules):
            return self.modules.pop(index)
        return None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        outputs = [module.forward(x) for module in self.modules]
        return np.mean(outputs, axis=0)

    def forward_learning(self, x: np.ndarray, target: np.ndarray) -> float:
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
        return int(np.argmax(self.forward(x)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(
            1 for i in range(len(X)) if self.predict(X[i]) == int(np.argmax(y[i]))
        )
        return correct / len(X)

    def _network_plateau_score(self) -> float:
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
        eligible = [
            (idx, module)
            for idx, module in enumerate(self.modules)
            if module.age >= self.config.min_clone_source_age
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda item: item[1].local_tension)[0]

    def structural_evolution(self, epoch_loss: Optional[float] = None, epoch: Optional[int] = None) -> List[Tuple]:
        """Trigger CLONE or ADAPT based on tension rather than random probability."""
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

    @property
    def param_count(self) -> int:
        return sum(module.param_count for module in self.modules)

    def serialize(self) -> Dict:
        return {
            "modules": [module.serialize() for module in self.modules],
            "epoch_loss_history": self.epoch_loss_history,
            "current_epoch": self.current_epoch,
            "last_clone_epoch": self.last_clone_epoch,
            "stalled_epochs": self.stalled_epochs,
        }

    def snapshot(self, epoch: Optional[int] = None) -> Dict:
        return {
            "epoch": self.current_epoch if epoch is None else epoch,
            "module_count": len(self.modules),
            "module_names": [module.name for module in self.modules],
            "tensions": [module.local_tension for module in self.modules],
            "weight_norms": [module.weight_norm for module in self.modules],
            "weights": [module.weights.copy() for module in self.modules],
        }

    def deserialize(self, data: Dict) -> None:
        self.modules = []
        for index, module_data in enumerate(data.get("modules", [])):
            module = SELModule(
                name=module_data.get("name", f"m{index}"),
                in_size=self.config.input_size,
                out_size=self.config.output_size,
                rng=self.rng,
            )
            module.load_state(module_data)
            self.modules.append(module)

        if not self.modules:
            self.add_module("m0")

        self.epoch_loss_history = [float(v) for v in data.get("epoch_loss_history", [])]
        self.current_epoch = int(data.get("current_epoch", self.current_epoch))
        self.last_clone_epoch = int(data.get("last_clone_epoch", self.last_clone_epoch))
        self.stalled_epochs = int(data.get("stalled_epochs", 0))

    def __repr__(self) -> str:
        return f"SELNetwork(modules={len(self.modules)}, params={self.param_count})"


class SELTrainer:
    """Trainer wrapper for SEL experiments."""

    def __init__(self, config: SELConfig):
        self.config = config
        self.network: Optional[SELNetwork] = None
        self.metrics: List[TrainingMetrics] = []
        self.evolution_log: List[Dict] = []
        self.topology_history: List[Dict] = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
    ) -> Dict:
        self.network = SELNetwork(self.config)
        self.metrics = []
        self.evolution_log = []
        self.topology_history = []

        for epoch in range(self.config.epochs):
            indices = self.network.rng.permutation(len(X_train))
            epoch_losses = []

            for i in indices:
                epoch_losses.append(self.network.forward_learning(X_train[i], y_train[i]))

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            changes = self.network.structural_evolution(epoch_loss=avg_loss, epoch=epoch)

            train_acc = self.network.accuracy(X_train, y_train)
            test_acc = 0.0
            if X_test is not None and y_test is not None:
                test_acc = self.network.accuracy(X_test, y_test)

            avg_tension = float(np.mean([module.local_tension for module in self.network.modules]))
            if changes:
                self.evolution_log.append(
                    {
                        "epoch": epoch,
                        "avg_loss": avg_loss,
                        "avg_tension": avg_tension,
                        "module_count": len(self.network.modules),
                        "changes": changes,
                    }
                )
            self.metrics.append(
                TrainingMetrics(
                    epoch=epoch,
                    train_accuracy=train_acc,
                    test_accuracy=test_acc,
                    avg_tension=avg_tension,
                    module_count=len(self.network.modules),
                    structural_changes=len(changes),
                    avg_loss=avg_loss,
                )
            )
            self.topology_history.append(self.network.snapshot(epoch=epoch))

        return {
            "final_train_accuracy": self.metrics[-1].train_accuracy,
            "final_test_accuracy": self.metrics[-1].test_accuracy,
            "final_modules": self.metrics[-1].module_count,
            "total_structural_changes": sum(metric.structural_changes for metric in self.metrics),
            "network": self.network,
            "evolution_log": self.evolution_log,
            "topology_history": self.topology_history,
        }

    def get_results(self) -> Dict:
        if not self.metrics:
            return {}

        best_epoch = max(range(len(self.metrics)), key=lambda i: self.metrics[i].test_accuracy)
        best = self.metrics[best_epoch]
        return {
            "best_test_accuracy": best.test_accuracy,
            "best_epoch": best.epoch,
            "final_train_accuracy": self.metrics[-1].train_accuracy,
            "final_test_accuracy": self.metrics[-1].test_accuracy,
            "final_modules": self.metrics[-1].module_count,
            "total_changes": sum(metric.structural_changes for metric in self.metrics),
            "avg_tension": float(np.mean([metric.avg_tension for metric in self.metrics])),
            "avg_loss": float(np.mean([metric.avg_loss for metric in self.metrics])),
        }


def create_task(task_type: str = "simple_classification") -> Tuple[np.ndarray, np.ndarray]:
    """Create a small benchmark task."""
    rng = np.random.default_rng(42)

    if task_type == "simple_classification":
        X = rng.normal(0.0, 2.0, size=(200, 4))
        y = np.zeros((200, 2))
        for i in range(200):
            if X[i, 0] + X[i, 1] > 0:
                y[i, 0] = 1
            else:
                y[i, 1] = 1
        return X, y

    if task_type == "xor":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
        return X, y

    raise ValueError(f"Unknown task type: {task_type}")


def run_experiment(
    config: SELConfig = None,
    task_type: str = "simple_classification",
) -> Dict:
    """Convenience experiment runner."""
    if config is None:
        config = SELConfig(input_size=4, output_size=2)

    X, y = create_task(task_type)
    split = len(X) // 2
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    trainer = SELTrainer(config)
    return trainer.train(X_train, y_train, X_test, y_test)


def main() -> None:
    """Demo entrypoint."""
    print("\n" + "=" * 60)
    print("SEL Framework Demo")
    print("=" * 60)

    config = SELConfig(
        input_size=4,
        output_size=2,
        initial_modules=3,
        learning_rate=0.1,
        epochs=100,
    )
    result = run_experiment(config)

    print("\nResults:")
    print(f"  Train Accuracy: {result['final_train_accuracy']:.1%}")
    print(f"  Test Accuracy: {result['final_test_accuracy']:.1%}")
    print(f"  Final Modules: {result['final_modules']}")
    print(f"  Structural Changes: {result['total_structural_changes']}")
    print("\n" + "=" * 60)
    print("Status: SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()
