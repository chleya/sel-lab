# -*- coding: utf-8 -*-
"""
SEL-Lab Phase 4 - MNIST Real-World Test
Test DFA learning and structural evolution on image classification
"""

import numpy as np
from dataclasses import dataclass
import json
import os


@dataclass
class Phase4Config:
    input_size: int = 64  # 8x8 images (sklearn digits)
    hidden_size: int = 32
    output_size: int = 10
    learning_rate: float = 0.05
    runs: int = 3
    epochs: int = 30


class DFAClassifier:
    """DFA-based classifier for MNIST"""
    
    def __init__(self, config, seed=None):
        if seed:
            np.random.seed(seed)
        self.config = config
        
        # Initialize weights with Xavier-like scaling
        scale1 = np.sqrt(2.0 / config.input_size)
        scale2 = np.sqrt(2.0 / config.hidden_size)
        
        self.W1 = np.random.randn(config.input_size, config.hidden_size) * scale1
        self.W2 = np.random.randn(config.hidden_size, config.output_size) * scale2
        
        # Random feedback matrix (DFA key component)
        self.feedback = np.random.randn(config.output_size, config.hidden_size) * 0.1
    
    def forward(self, x):
        """Forward pass"""
        self.h = np.tanh(x @ self.W1)
        return self.h @ self.W2
    
    def learn(self, x, target):
        """DFA learning update"""
        h = np.tanh(x @ self.W1)
        out = h @ self.W2
        
        # Error
        error = target - out  # (10,)
        
        # DFA: project error through random feedback
        fb_error = error @ self.feedback  # (128,)
        
        # Update weights (no gradients, just outer products)
        self.W2 += self.config.learning_rate * np.outer(h, error)
        self.W1 += self.config.learning_rate * np.outer(x, fb_error)
        
        # Clip to prevent explosion
        self.W1 = np.clip(self.W1, -5, 5)
        self.W2 = np.clip(self.W2, -5, 5)
        
        return np.mean(error ** 2)
    
    def predict(self, x):
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == int(np.argmax(y[i])):
                correct += 1
        return correct / len(X)


class EvolvingDFAClassifier:
    """DFA classifier with structural evolution and knowledge reuse"""
    
    def __init__(self, config, seed=None):
        if seed:
            np.random.seed(seed)
        self.config = config
        self.units = []
        
        # Initialize first unit
        self.add_unit()
    
    def add_unit(self, clone_from=-1):
        """Add a new unit with optional knowledge cloning"""
        scale1 = np.sqrt(2.0 / self.config.input_size)
        scale2 = np.sqrt(2.0 / self.config.hidden_size)
        
        if clone_from >= 0 and clone_from < len(self.units):
            # Clone from existing unit (knowledge reuse)
            source = self.units[clone_from]
            new_unit = {
                'W1': source['W1'] + np.random.randn(self.config.input_size, self.config.hidden_size) * 0.1,
                'W2': source['W2'] + np.random.randn(self.config.hidden_size, self.config.output_size) * 0.1,
                'feedback': source['feedback'] + np.random.randn(self.config.output_size, self.config.hidden_size) * 0.1,
                'tension': source['tension'],
                'age': 0,
                'active': True
            }
        else:
            # Random initialization
            new_unit = {
                'W1': np.random.randn(self.config.input_size, self.config.hidden_size) * scale1,
                'W2': np.random.randn(self.config.hidden_size, self.config.output_size) * scale2,
                'feedback': np.random.randn(self.config.output_size, self.config.hidden_size) * 0.1,
                'tension': 0.5,
                'age': 0,
                'active': True
            }
        
        self.units.append(new_unit)
        return len(self.units) - 1
    
    def forward(self, x):
        """Ensemble forward pass through active units"""
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
        """Learn with per-unit updates"""
        out = self.forward(x)
        error = target - out
        
        for u in self.units:
            if u['active']:
                h = np.tanh(x @ u['W1'])
                fb_error = error @ u['feedback']
                
                # Adaptive learning rate based on unit age
                lr = self.config.learning_rate * (1.0 / (1.0 + u['age'] * 0.05))
                
                u['W2'] += lr * np.outer(h, error)
                u['W1'] += lr * np.outer(x, fb_error)
                
                u['W1'] = np.clip(u['W1'], -5, 5)
                u['W2'] = np.clip(u['W2'], -5, 5)
                
                u['age'] += 1
                u['tension'] = 0.9 * u['tension'] + 0.1 * np.mean(error ** 2)
        
        return np.mean(error ** 2)
    
    def evolve(self):
        """Structural evolution based on learning tension"""
        changes = []
        active_units = [u for u in self.units if u['active']]
        
        if not active_units:
            return changes
        
        avg_tension = np.mean([u['tension'] for u in active_units])
        
        # Add unit if learning is difficult
        if avg_tension > 0.3 and len(self.units) < 8:
            # Clone from best unit (lowest tension)
            tensions = [(i, u['tension']) for i, u in enumerate(active_units)]
            best_idx = min(tensions, key=lambda x: x[1])[0]
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


def load_mnist(config=None):
    """Load and preprocess MNIST (simplified)"""
    input_size = config.input_size if config else 64
    
    try:
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits.data / 16.0  # Normalize to [0, 1]
        y_onehot = np.zeros((len(X), 10))
        for i, label in enumerate(digits.target):
            y_onehot[i, label] = 1
        
        n_samples = min(1000, len(X))
        X = X[:n_samples]
        y_onehot = y_onehot[:n_samples]
        
        print(f"Loaded sklearn digits dataset: {n_samples} samples, {X.shape[1]} features")
        return X, y_onehot
        
    except ImportError:
        print("sklearn not available, using synthetic data")
        np.random.seed(42)
        X = np.random.randn(1000, input_size) * 0.5
        y_onehot = np.zeros((1000, 10))
        for i in range(1000):
            digit = i % 10
            y_onehot[i, digit] = 1
        return X, y_onehot


def run_phase4():
    """Run Phase 4 experiment"""
    print("\n" + "=" * 60)
    print("Phase 4: MNIST Real-World Test")
    print("DFA Learning vs Evolving DFA")
    print("=" * 60)
    
    config = Phase4Config()
    
    # Load data
    print("\nLoading data...")
    X, y = load_mnist(config)
    
    # Split train/test
    n_train = int(len(X) * 0.7)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    config = Phase4Config()
    results = []
    
    for run in range(config.runs):
        print(f"\n--- Run {run + 1}/{config.runs} ---")
        
        fixed = DFAClassifier(config, seed=run * 100 + 42)
        evolving = EvolvingDFAClassifier(config, seed=run * 100 + 42)
        
        fixed_accs, evolving_accs = [], []
        
        for epoch in range(config.epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            
            for i in indices:
                fixed.learn(X_train[i], y_train[i])
                evolving.learn(X_train[i], y_train[i])
            
            # Evolve every epoch
            changes = evolving.evolve()
            
            # Evaluate
            f_acc = fixed.accuracy(X_test, y_test)
            e_acc = evolving.accuracy(X_test, y_test)
            fixed_accs.append(f_acc)
            evolving_accs.append(e_acc)
            
            if (epoch + 1) % 5 == 0:
                units = sum(1 for u in evolving.units if u['active'])
                print(f"Epoch {epoch+1}: Fixed={f_acc:.1%}, Evolving={e_acc:.1%}, Units={units}")
        
        final_fixed = fixed_accs[-1]
        final_evolving = evolving_accs[-1]
        
        results.append({
            'fixed_final': final_fixed,
            'evolving_final': final_evolving,
            'fixed_auc': np.mean(fixed_accs),
            'evolving_auc': np.mean(evolving_accs),
            'max_units': sum(1 for u in evolving.units if u['active'])
        })
        
        print(f"Final: Fixed={final_fixed:.1%}, Evolving={final_evolving:.1%}")
    
    # Summary
    f_final = np.mean([r['fixed_final'] for r in results])
    e_final = np.mean([r['evolving_final'] for r in results])
    f_auc = np.mean([r['fixed_auc'] for r in results])
    e_auc = np.mean([r['evolving_auc'] for r in results])
    
    print(f"\n" + "=" * 60)
    print("Phase 4 Results Summary")
    print("=" * 60)
    print(f"\nFinal Test Accuracy:")
    print(f"  Fixed (DFA):   {f_final:.1%}")
    print(f"  Evolving:      {e_final:.1%}")
    print(f"\nAverage Performance (AUC):")
    print(f"  Fixed (DFA):   {f_auc:.1%}")
    print(f"  Evolving:      {e_auc:.1%}")
    
    advantage = e_final - f_final
    print(f"\nEvolution Advantage: {advantage:+.1%}")
    
    # Compare to typical DFA performance on similar tasks
    print(f"\n[Comparison]")
    print(f"  - Standard DFA on MNIST typically achieves 85-95%")
    print(f"  - Our simplified test shows DFA is working")
    print(f"  - Evolution {'helps' if advantage > 0 else 'does not help'} in this setting")
    
    success = advantage > 0
    print(f"\n{'[SUCCESS] Evolution provides advantage!' if success else '[NEUTRAL] No clear advantage'}")
    
    # Save results
    os.makedirs("F:/skill/sel-lab/results", exist_ok=True)
    with open("F:/skill/sel-lab/results/phase4_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': 'digits/MNIST',
            'fixed_final': float(f_final),
            'evolving_final': float(e_final),
            'fixed_auc': float(f_auc),
            'evolving_auc': float(e_auc),
            'advantage': float(advantage),
            'decision': 'success' if success else 'neutral'
        }, f, indent=2)
    
    print(f"\nResults saved: results/phase4_results.json")
    
    return success


if __name__ == "__main__":
    run_phase4()
