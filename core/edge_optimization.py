# -*- coding: utf-8 -*-
"""
SEL-Lab Edge Optimization
Optimize for edge deployment
"""

import numpy as np
from dataclasses import dataclass
import json


@dataclass
class EdgeConfig:
    input_size: int = 16
    hidden_size: int = 8
    output_size: int = 2
    learning_rate: float = 0.05
    max_units: int = 4
    runs: int = 3
    epochs: int = 50


class EdgeDFA:
    def __init__(self, config, seed=None):
        if seed:
            np.random.seed(seed)
        self.config = config
        scale1 = np.sqrt(2.0 / config.input_size) * 0.5
        scale2 = np.sqrt(2.0 / config.hidden_size) * 0.5
        self.W1 = np.random.randn(config.input_size, config.hidden_size) * scale1
        self.W2 = np.random.randn(config.hidden_size, config.output_size) * scale2
        self.feedback = np.random.randn(config.output_size, config.hidden_size) * 0.05
    
    def forward(self, x):
        h = np.tanh(x @ self.W1)
        return h @ self.W2
    
    def learn(self, x, target):
        h = np.tanh(x @ self.W1)
        out = h @ self.W2
        error = target - out
        fb_error = error @ self.feedback
        self.W2 += self.config.learning_rate * np.outer(h, error)
        self.W1 += self.config.learning_rate * np.outer(x, fb_error)
        np.clip(self.W1, -2, 2, out=self.W1)
        np.clip(self.W2, -2, 2, out=self.W2)
        return np.mean(error ** 2)
    
    def predict(self, x):
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == int(np.argmax(y[i])):
                correct += 1
        return correct / len(X)
    
    def get_memory_bytes(self):
        return self.W1.nbytes + self.W2.nbytes + self.feedback.nbytes


class EdgeEvolving:
    def __init__(self, config, seed=None):
        if seed:
            np.random.seed(seed)
        self.config = config
        self.units = []
        self.add_unit()
    
    def add_unit(self, clone_from=-1):
        scale1 = np.sqrt(2.0 / self.config.input_size) * 0.5
        scale2 = np.sqrt(2.0 / self.config.hidden_size) * 0.5
        
        if clone_from >= 0 and clone_from < len(self.units):
            source = self.units[clone_from]
            new_unit = {
                'W1': source['W1'] + np.random.randn(self.config.input_size, self.config.hidden_size) * 0.05,
                'W2': source['W2'] + np.random.randn(self.config.hidden_size, self.config.output_size) * 0.05,
                'feedback': source['feedback'] + np.random.randn(self.config.output_size, self.config.hidden_size) * 0.02,
                'tension': source['tension'],
                'age': 0,
                'active': True
            }
        else:
            new_unit = {
                'W1': np.random.randn(self.config.input_size, self.config.hidden_size) * scale1,
                'W2': np.random.randn(self.config.hidden_size, self.config.output_size) * scale2,
                'feedback': np.random.randn(self.config.output_size, self.config.hidden_size) * 0.05,
                'tension': 0.5,
                'age': 0,
                'active': True
            }
        
        self.units.append(new_unit)
        return len(self.units) - 1
    
    def forward(self, x):
        outputs = []
        for u in self.units:
            if u['active']:
                h = np.tanh(x @ u['W1'])
                out = h @ u['W2']
                outputs.append(out)
        return np.mean(outputs, axis=0) if outputs else np.zeros(self.config.output_size)
    
    def learn(self, x, target):
        out = self.forward(x)
        error = target - out
        
        for u in self.units:
            if u['active']:
                h = np.tanh(x @ u['W1'])
                fb_error = error @ u['feedback']
                lr = self.config.learning_rate * np.exp(-u['age'] * 0.01)
                u['W2'] += lr * np.outer(h, error)
                u['W1'] += lr * np.outer(x, fb_error)
                np.clip(u['W1'], -2, 2, out=u['W1'])
                np.clip(u['W2'], -2, 2, out=u['W2'])
                u['age'] += 1
                u['tension'] = 0.9 * u['tension'] + 0.1 * np.mean(error ** 2)
        
        return np.mean(error ** 2)
    
    def evolve(self):
        changes = []
        active = [u for u in self.units if u['active']]
        if not active:
            return changes
        
        avg_tension = np.mean([u['tension'] for u in active])
        
        if avg_tension > 0.2 and len(self.units) < self.config.max_units:
            best_idx = np.argmin([u['tension'] for u in active])
            best_unit_idx = self.units.index(active[best_idx])
            self.add_unit(clone_from=best_unit_idx)
            changes.append('add')
        
        return changes
    
    def predict(self, x):
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == int(np.argmax(y[i])):
                correct += 1
        return correct / len(X)
    
    def get_memory_bytes(self):
        total = 0
        for u in self.units:
            total += u['W1'].nbytes + u['W2'].nbytes + u['feedback'].nbytes
        return total


def create_edge_task():
    np.random.seed(42)
    X = np.random.randn(200, 16)
    y = np.zeros((200, 2))
    for i in range(200):
        if X[i, 0] + X[i, 1] + 0.5 * X[i, 2] > 0:
            y[i, 0] = 1
        else:
            y[i, 1] = 1
    return X, y


def run_edge_optimization():
    print("\n" + "=" * 60)
    print("Edge Optimization Benchmark")
    print("=" * 60)
    
    config = EdgeConfig()
    X, y = create_edge_task()
    X_train, y_train = X[:100], y[:100]
    X_test, y_test = X[100:], y[100:]
    
    print(f"\nTask: {config.input_size} features -> {config.output_size} classes")
    print(f"Constraint: Max {config.max_units} units")
    
    results = []
    
    for run in range(config.runs):
        print(f"\n--- Run {run + 1}/{config.runs} ---")
        
        fixed = EdgeDFA(config, seed=run * 100 + 42)
        evolving = EdgeEvolving(config, seed=run * 100 + 42)
        
        fixed_accs, evolving_accs = [], []
        
        for epoch in range(config.epochs):
            for i in range(len(X_train)):
                fixed.learn(X_train[i], y_train[i])
                evolving.learn(X_train[i], y_train[i])
            
            if (epoch + 1) % 10 == 0:
                evolving.evolve()
            
            f_acc = fixed.accuracy(X_test, y_test)
            e_acc = evolving.accuracy(X_test, y_test)
            fixed_accs.append(f_acc)
            evolving_accs.append(e_acc)
            
            if (epoch + 1) % 25 == 0:
                units = sum(1 for u in evolving.units if u['active'])
                print(f"Epoch {epoch+1}: Fixed={f_acc:.0%}, Evolving={e_acc:.0%}, Units={units}")
        
        results.append({
            'fixed_final': fixed_accs[-1],
            'evolving_final': evolving_accs[-1],
            'fixed_auc': np.mean(fixed_accs),
            'evolving_auc': np.mean(evolving_accs)
        })
        
        print(f"Final: Fixed={fixed_accs[-1]:.0%}, Evolving={evolving_accs[-1]:.0%}")
    
    # Summary
    f_final = np.mean([r['fixed_final'] for r in results])
    e_final = np.mean([r['evolving_final'] for r in results])
    
    # Memory comparison
    fixed_mem = EdgeDFA(config).get_memory_bytes()
    evolving_mem = EdgeEvolving(config).get_memory_bytes()
    
    print(f"\n" + "=" * 60)
    print("Edge Optimization Results")
    print("=" * 60)
    
    print(f"\nAccuracy:")
    print(f"  Fixed (DFA):   {f_final:.1%}")
    print(f"  Evolving:      {e_final:.1%}")
    
    print(f"\nMemory Footprint:")
    print(f"  Fixed:   {fixed_mem} bytes ({fixed_mem/1024:.2f} KB)")
    print(f"  Evolving: {evolving_mem} bytes ({evolving_mem/1024:.2f} KB)")
    
    advantage = e_final - f_final
    memory_ratio = evolving_mem / fixed_mem
    
    print(f"\nEvolution Advantage: {advantage:+.1%}")
    print(f"Memory Ratio: {memory_ratio:.2f}x")
    
    # Compression vs full model
    full_model = 784 * 128 * 4 + 128 * 10 * 4 + 10 * 128 * 4
    compression = full_model / fixed_mem
    
    print(f"\nCompression vs Full MNIST:")
    print(f"  Full: {full_model:,} bytes ({full_model/1024:.0f} KB)")
    print(f"  Edge: {fixed_mem} bytes ({fixed_mem/1024:.2f} KB)")
    print(f"  Compression: {compression:.0f}x smaller")
    
    success = advantage > 0
    print(f"\n{'[SUCCESS] Edge optimization effective!' if success else '[NEUTRAL] Mixed results'}")
    
    with open("F:/skill/sel-lab/results/edge_optimization_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'input_size': config.input_size,
                'hidden_size': config.hidden_size,
                'max_units': config.max_units
            },
            'fixed_final': float(f_final),
            'evolving_final': float(e_final),
            'advantage': float(advantage),
            'memory_fixed': float(fixed_mem),
            'memory_evolving': float(evolving_mem),
            'memory_ratio': float(memory_ratio),
            'compression_ratio': float(compression),
            'decision': 'success' if success else 'neutral'
        }, f, indent=2)
    
    print(f"\nResults saved: results/edge_optimization_results.json")


if __name__ == "__main__":
    run_edge_optimization()
