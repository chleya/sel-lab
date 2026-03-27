# -*- coding: utf-8 -*-
"""
SEL-Lab Phase 2c - Improved Evolution with Knowledge Reuse
New units clone existing knowledge instead of random init
"""

import numpy as np
from dataclasses import dataclass
import json


@dataclass
class Phase2Config:
    input_size: int = 4
    output_size: int = 3
    hidden_size: int = 8
    learning_rate: float = 0.03
    runs: int = 5
    epochs: int = 100


class FixedNetwork:
    """Fixed structure baseline"""
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


class EvolvingNetworkWithKnowledge:
    """Evolving network that reuses knowledge when adding units"""
    
    def __init__(self, config, seed=None):
        if seed:
            np.random.seed(seed)
        self.config = config
        self.units = []
        self.add_unit()  # Start with 1 unit
    
    def add_unit(self, clone_from=-1):
        """Add a new unit - optionally clone from existing unit"""
        if clone_from >= 0 and clone_from < len(self.units):
            # Clone existing unit's knowledge
            source = self.units[clone_from]
            new_unit = {
                'W1': source['W1'].copy() + np.random.randn(*source['W1'].shape) * 0.1,  # Small perturbation
                'W2': source['W2'].copy() + np.random.randn(*source['W2'].shape) * 0.1,
                'feedback': source['feedback'].copy() + np.random.randn(*source['feedback'].shape) * 0.1,
                'active': True,
                'tension': source['tension'],  # Inherit tension
                'age': 0
            }
        else:
            # Random initialization (first unit)
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
                
                # Experienced units learn faster, new units learn slower
                lr = 0.01 * (1.0 / (1.0 + u['age'] * 0.1))
                
                u['W2'] += lr * np.outer(h, error)
                u['W1'] += lr * np.outer(x, fb_error)
                u['age'] += 1
                u['tension'] = 0.9 * u['tension'] + 0.1 * np.mean(error ** 2)
        
        return np.mean(error ** 2)
    
    def evolve(self):
        """Smart evolution - clone from best unit when adding"""
        changes = []
        active_units = [u for u in self.units if u['active']]
        
        if not active_units:
            return changes
        
        # Find unit with lowest tension (best performer)
        best_idx = np.argmin([u['tension'] for u in active_units])
        
        avg_tension = np.mean([u['tension'] for u in active_units])
        
        # High tension -> add unit (clone from best)
        if avg_tension > 0.15 and len(self.units) < 10:
            best_unit_idx = self.units.index(active_units[best_idx])
            new_idx = self.add_unit(clone_from=best_unit_idx)
            changes.append(f'add(clone_from={best_unit_idx})')
        
        # Very low tension + many units -> remove worst
        elif avg_tension < 0.02 and len(active_units) > 3:
            worst_idx = np.argmax([u['tension'] for u in active_units])
            self.units[worst_idx]['active'] = False
            changes.append(f'remove(worst={worst_idx})')
        
        return changes
    
    def predict(self, x):
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == int(np.argmax(y[i])):
                correct += 1
        return correct / len(X)


def create_hard_task():
    """Hard 3-class XOR-like problem"""
    np.random.seed(42)
    X = np.random.randn(200, 4) * 2
    y = np.zeros((200, 3))
    
    for i in range(200):
        score = np.sin(X[i, 0] * 2) + np.cos(X[i, 1] * 2) + X[i, 2] * X[i, 3]
        
        if score > 1.0:
            y[i, 0] = 1
        elif score < -1.0:
            y[i, 1] = 1
        else:
            y[i, 2] = 1
    
    return X, y


def run_phase2_improved():
    print("\n" + "=" * 60)
    print("Phase 2c: Improved Evolution with Knowledge Reuse")
    print("New units clone from best existing unit")
    print("=" * 60)
    
    X, y = create_hard_task()
    X_train, y_train = X[:100], y[:100]
    X_test, y_test = X[100:], y[100:]
    
    config = Phase2Config()
    results = []
    
    for run in range(config.runs):
        print(f"\n--- Run {run + 1}/{config.runs} ---")
        
        fixed = FixedNetwork(config, seed=run * 100 + 42)
        evolving = EvolvingNetworkWithKnowledge(config, seed=run * 100 + 42)
        
        fixed_accs, evolving_accs = [], []
        
        for epoch in range(config.epochs):
            for i in range(len(X_train)):
                fixed.learn(X_train[i], y_train[i])
                evolving.learn(X_train[i], y_train[i])
            
            if (epoch + 1) % 10 == 0:
                changes = evolving.evolve()
            
            f_acc = fixed.accuracy(X_test, y_test)
            e_acc = evolving.accuracy(X_test, y_test)
            fixed_accs.append(f_acc)
            evolving_accs.append(e_acc)
            
            if (epoch + 1) % 20 == 0:
                active = sum(1 for u in evolving.units if u['active'])
                print(f"Epoch {epoch+1}: Fixed={f_acc:.1%}, Evolving={e_acc:.1%}, Units={active}")
        
        results.append({
            'fixed_final': fixed_accs[-1],
            'evolving_final': evolving_accs[-1],
            'fixed_auc': np.mean(fixed_accs),
            'evolving_auc': np.mean(evolving_accs)
        })
        print(f"Final: Fixed={fixed_accs[-1]:.1%}, Evolving={evolving_accs[-1]:.1%}")
    
    # Summary
    f_final = np.mean([r['fixed_final'] for r in results])
    e_final = np.mean([r['evolving_final'] for r in results])
    f_auc = np.mean([r['fixed_auc'] for r in results])
    e_auc = np.mean([r['evolving_auc'] for r in results])
    
    print(f"\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Final Accuracy: Fixed={f_final:.1%}, Evolving={e_final:.1%}")
    print(f"AUC (Avg): Fixed={f_auc:.1%}, Evolving={e_auc:.1%}")
    
    advantage = e_final - f_final
    print(f"Evolution Advantage: {advantage:+.1%}")
    
    success = advantage > 0.02
    print(f"\n{'[SUCCESS] Evolution provides advantage!' if success else '[NEUTRAL] No clear advantage'}")
    
    with open("F:/skill/sel-lab/results/phase2_improved_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            'method': 'knowledge_reuse',
            'fixed_final': float(f_final),
            'evolving_final': float(e_final),
            'fixed_auc': float(f_auc),
            'evolving_auc': float(e_auc),
            'advantage': float(advantage),
            'decision': 'success' if success else 'neutral'
        }, f, indent=2)
    
    print(f"\nResults saved: results/phase2_improved_results.json")


if __name__ == "__main__":
    run_phase2_improved()
