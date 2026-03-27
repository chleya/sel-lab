# -*- coding: utf-8 -*-
"""
SEL-Lab Phase 3 - Incremental Learning & Task Transfer
Test: Can evolving networks learn new tasks without forgetting old ones?
"""

import numpy as np
from dataclasses import dataclass
import json


@dataclass
class Phase3Config:
    input_size: int = 4
    output_size: int = 2
    hidden_size: int = 8
    learning_rate: float = 0.03
    runs: int = 5
    epochs_per_task: int = 50


class FixedNetwork:
    """Fixed structure - baseline"""
    def __init__(self, config, seed=None):
        if seed:
            np.random.seed(seed)
        self.W1 = np.random.randn(config.input_size, config.hidden_size) * 0.5
        self.W2 = np.random.randn(config.hidden_size, config.output_size) * 0.5
        self.feedback = np.random.randn(config.output_size, config.hidden_size) * 0.5
    
    def forward(self, x):
        return np.tanh(x @ self.W1) @ self.W2
    
    def learn(self, x, target):
        h = np.tanh(x @ self.W1)
        out = h @ self.W2
        error = target - out
        fb_error = error @ self.feedback
        self.W2 += 0.01 * np.outer(h, error)
        self.W1 += 0.01 * np.outer(x, fb_error)
        return np.mean(error ** 2)
    
    def predict(self, x):
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == int(np.argmax(y[i])):
                correct += 1
        return correct / len(X)


class EvolvingNetwork:
    """Evolving network with knowledge reuse for incremental learning"""
    
    def __init__(self, config, seed=None):
        if seed:
            np.random.seed(seed)
        self.config = config
        self.units = []
        self.task_count = 0
        self.add_unit()  # Start with 1 unit
    
    def add_unit(self, clone_from=-1):
        if clone_from >= 0 and clone_from < len(self.units):
            source = self.units[clone_from]
            new_unit = {
                'W1': source['W1'].copy() + np.random.randn(*source['W1'].shape) * 0.1,
                'W2': source['W2'].copy() + np.random.randn(*source['W2'].shape) * 0.1,
                'feedback': source['feedback'].copy() + np.random.randn(*source['feedback'].shape) * 0.1,
                'active': True,
                'tension': source['tension'],
                'age': 0
            }
        else:
            new_unit = {
                'W1': np.random.randn(self.config.input_size, self.config.hidden_size) * 0.5,
                'W2': np.random.randn(self.config.hidden_size, self.config.output_size) * 0.5,
                'feedback': np.random.randn(self.config.output_size, self.config.hidden_size) * 0.5,
                'active': True,
                'tension': 0.5,
                'age': 0
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
        if outputs:
            return np.mean(outputs, axis=0)
        return np.zeros(self.config.output_size)
    
    def learn(self, x, target):
        out = self.forward(x)
        error = target - out
        
        for u in self.units:
            if u['active']:
                h = np.tanh(x @ u['W1'])
                fb_error = error @ u['feedback']
                lr = 0.01 * (1.0 / (1.0 + u['age'] * 0.1))
                u['W2'] += lr * np.outer(h, error)
                u['W1'] += lr * np.outer(x, fb_error)
                u['age'] += 1
                u['tension'] = 0.9 * u['tension'] + 0.1 * np.mean(error ** 2)
        
        return np.mean(error ** 2)
    
    def evolve(self):
        changes = []
        active_units = [u for u in self.units if u['active']]
        
        if not active_units:
            return changes
        
        avg_tension = np.mean([u['tension'] for u in active_units])
        
        # Add unit when learning is hard
        if avg_tension > 0.15 and len(self.units) < 12:
            best_idx = np.argmin([u['tension'] for u in active_units])
            best_unit_idx = self.units.index(active_units[best_idx])
            self.add_unit(clone_from=best_unit_idx)
            changes.append(f'add(clone_from={best_unit_idx})')
        
        return changes
    
    def predict(self, x):
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == int(np.argmax(y[i])):
                correct += 1
        return correct / len(X)
    
    def reset_tension(self):
        """Reset tension after task switch"""
        for u in self.units:
            u['tension'] = 0.3  # Medium tension for new task


def create_task(task_id, seed=42):
    """Create different tasks"""
    np.random.seed(seed + task_id)
    
    if task_id == 0:
        # Task 0: Simple linear boundary
        X = np.random.randn(100, 4) * 2
        y = np.zeros((100, 2))
        for i in range(100):
            if X[i, 0] + X[i, 1] > 0:
                y[i, 0] = 1
            else:
                y[i, 1] = 1
    
    elif task_id == 1:
        # Task 1: XOR-like
        X = np.random.randn(100, 4) * 2
        y = np.zeros((100, 2))
        for i in range(100):
            if X[i, 0] * X[i, 1] > 0:
                y[i, 0] = 1
            else:
                y[i, 1] = 1
    
    elif task_id == 2:
        # Task 2: Non-linear
        X = np.random.randn(100, 4) * 2
        y = np.zeros((100, 2))
        for i in range(100):
            if np.sin(X[i, 0]) + np.cos(X[i, 1]) > 0:
                y[i, 0] = 1
            else:
                y[i, 1] = 1
    
    else:
        # Task 3: More complex
        X = np.random.randn(100, 4) * 2
        y = np.zeros((100, 2))
        for i in range(100):
            if X[i, 0] ** 2 + X[i, 1] ** 2 > 2:
                y[i, 0] = 1
            else:
                y[i, 1] = 1
    
    return X, y


def run_phase3():
    """Run incremental learning experiment"""
    print("\n" + "=" * 60)
    print("Phase 3: Incremental Learning & Task Transfer")
    print("=" * 60)
    
    config = Phase3Config()
    num_tasks = 4
    
    results = []
    
    for run in range(config.runs):
        print(f"\n--- Run {run + 1}/{config.runs} ---")
        
        fixed = FixedNetwork(config, seed=run * 100 + 42)
        evolving = EvolvingNetwork(config, seed=run * 100 + 42)
        
        task_results = []
        
        for task_id in range(num_tasks):
            X, y = create_task(task_id)
            
            print(f"\n  Task {task_id}: ", end="")
            
            # Train on new task
            for epoch in range(config.epochs_per_task):
                for i in range(len(X)):
                    fixed.learn(X[i], y[i])
                    evolving.learn(X[i], y[i])
                
                if (epoch + 1) % 10 == 0:
                    evolving.evolve()
            
            # Evaluate on ALL tasks (including old ones)
            all_accuracies = {}
            for prev_task in range(task_id + 1):
                X_test, y_test = create_task(prev_task)
                f_acc = fixed.accuracy(X_test, y_test)
                e_acc = evolving.accuracy(X_test, y_test)
                all_accuracies[f'task_{prev_task}'] = {'fixed': f_acc, 'evolving': e_acc}
            
            # Average accuracy across all seen tasks
            avg_fixed = np.mean([all_accuracies[f'task_{t}']['fixed'] for t in range(task_id + 1)])
            avg_evolving = np.mean([all_accuracies[f'task_{t}']['evolving'] for t in range(task_id + 1)])
            
            current_fixed = all_accuracies[f'task_{task_id}']['fixed']
            current_evolving = all_accuracies[f'task_{task_id}']['evolving']
            
            print(f"Fixed={current_fixed:.0%}, Evolving={current_evolving:.0%}, Avg={avg_evolving:.0%}")
            
            task_results.append({
                'task_id': task_id,
                'current_fixed': current_fixed,
                'current_evolving': current_evolving,
                'avg_fixed': avg_fixed,
                'avg_evolving': avg_evolving,
                'unit_count': sum(1 for u in evolving.units if u['active'])
            })
            
            # Reset tension for new task
            evolving.reset_tension()
        
        results.append(task_results)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("Phase 3 Results Summary")
    print("=" * 60)
    
    # Final performance (after all tasks)
    final_avg_fixed = np.mean([r[-1]['avg_fixed'] for r in results])
    final_avg_evolving = np.mean([r[-1]['avg_evolving'] for r in results])
    
    # Forgetting: difference between first task performance and final
    first_task_fixed = np.mean([r[0]['current_fixed'] for r in results])
    first_task_evolving = np.mean([r[0]['current_evolving'] for r in results])
    
    final_task_fixed = np.mean([r[-1]['current_fixed'] for r in results])
    final_task_evolving = np.mean([r[-1]['current_evolving'] for r in results])
    
    # Forgetting metric (lower is better - less forgetting)
    forgetting_fixed = first_task_fixed - final_avg_fixed
    forgetting_evolving = first_task_evolving - final_avg_evolving
    
    print(f"\nFinal Average Accuracy (all tasks):")
    print(f"  Fixed:   {final_avg_fixed:.1%}")
    print(f"  Evolving: {final_avg_evolving:.1%}")
    
    print(f"\nForgetting (first_task - avg_all):")
    print(f"  Fixed:   {forgetting_fixed:+.1%}")
    print(f"  Evolving: {forgetting_evolving:+.1%}")
    
    advantage = final_avg_evolving - final_avg_fixed
    transfer_success = final_avg_evolving > final_avg_fixed
    less_forgetting = forgetting_evolving > forgetting_fixed  # Actually more forgetting (negative is better)
    
    print(f"\nIncremental Learning Advantage: {advantage:+.1%}")
    
    success = transfer_success and abs(forgetting_evolving) < abs(forgetting_fixed)
    print(f"\n{'[SUCCESS] Evolution helps incremental learning!' if success else '[NEUTRAL] Mixed results'}")
    
    # Save results
    with open("F:/skill/sel-lab/results/phase3_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            'num_tasks': num_tasks,
            'final_avg_fixed': float(final_avg_fixed),
            'final_avg_evolving': float(final_avg_evolving),
            'forgetting_fixed': float(forgetting_fixed),
            'forgetting_evolving': float(forgetting_evolving),
            'advantage': float(advantage),
            'decision': 'success' if success else 'neutral'
        }, f, indent=2)
    
    print(f"\nResults saved: results/phase3_results.json")


if __name__ == "__main__":
    run_phase3()
