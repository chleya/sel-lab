# -*- coding: utf-8 -*-
"""
SEL-Lab Phase 4 - Real MNIST Test
Test DFA learning and structural evolution on real MNIST data
"""

import numpy as np
from dataclasses import dataclass
import json


@dataclass
class Phase4Config:
    input_size: int = 784  # 28x28 images
    hidden_size: int = 128
    output_size: int = 10  # 10 digits
    learning_rate: float = 0.01
    runs: int = 3
    epochs: int = 15  # Fewer epochs for real MNIST


class DFAClassifier:
    """DFA-based classifier for MNIST"""
    
    def __init__(self, config, seed=None):
        if seed:
            np.random.seed(seed)
        self.config = config
        
        scale1 = np.sqrt(2.0 / config.input_size)
        scale2 = np.sqrt(2.0 / config.hidden_size)
        
        self.W1 = np.random.randn(config.input_size, config.hidden_size) * scale1 * 0.5
        self.W2 = np.random.randn(config.hidden_size, config.output_size) * scale2 * 0.5
        self.feedback = np.random.randn(config.output_size, config.hidden_size) * 0.1
    
    def forward(self, x):
        self.h = np.tanh(x @ self.W1)
        return self.h @ self.W2
    
    def learn(self, x, target):
        h = np.tanh(x @ self.W1)
        out = h @ self.W2
        error = target - out
        fb_error = error @ self.feedback
        self.W2 += self.config.learning_rate * np.outer(h, error)
        self.W1 += self.config.learning_rate * np.outer(x, fb_error)
        self.W1 = np.clip(self.W1, -3, 3)
        self.W2 = np.clip(self.W2, -3, 3)
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
    """DFA classifier with structural evolution"""
    
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
                'W1': source['W1'] + np.random.randn(self.config.input_size, self.config.hidden_size) * 0.1,
                'W2': source['W2'] + np.random.randn(self.config.hidden_size, self.config.output_size) * 0.1,
                'feedback': source['feedback'] + np.random.randn(self.config.output_size, self.config.hidden_size) * 0.1,
                'tension': source['tension'],
                'age': 0,
                'active': True
            }
        else:
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
                lr = self.config.learning_rate * (1.0 / (1.0 + u['age'] * 0.02))
                u['W2'] += lr * np.outer(h, error)
                u['W1'] += lr * np.outer(x, fb_error)
                u['W1'] = np.clip(u['W1'], -3, 3)
                u['W2'] = np.clip(u['W2'], -3, 3)
                u['age'] += 1
                u['tension'] = 0.9 * u['tension'] + 0.1 * np.mean(error ** 2)
        
        return np.mean(error ** 2)
    
    def evolve(self):
        changes = []
        active_units = [u for u in self.units if u['active']]
        
        if not active_units:
            return changes
        
        avg_tension = np.mean([u['tension'] for u in active_units])
        
        if avg_tension > 0.3 and len(self.units) < 6:
            tensions = [(i, u['tension']) for i, u in enumerate(active_units)]
            best_idx = min(tensions, key=lambda x: x[1])[0]
            best_unit_idx = self.units.index(active_units[best_idx])
            self.add_unit(clone_from=best_unit_idx)
            changes.append(f'add')
        
        return changes
    
    def predict(self, x):
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == int(np.argmax(y[i])):
                correct += 1
        return correct / len(X)


def load_real_mnist():
    """Load real MNIST dataset"""
    try:
        from sklearn.datasets import fetch_openml
        
        print("Loading MNIST from OpenML...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32) / 255.0  # Normalize to [0, 1]
        y = mnist.target.astype(int)
        
        # One-hot encode
        y_onehot = np.zeros((len(y), 10))
        for i, label in enumerate(y):
            y_onehot[i, label] = 1
        
        print(f"Loaded MNIST: {len(X)} samples, {X.shape[1]} features")
        return X, y_onehot
        
    except Exception as e:
        print(f"OpenML failed: {e}")
        print("Falling back to smaller subset...")
        
        # Fallback: Use smaller subset for faster testing
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits.data / 16.0
        y_onehot = np.zeros((len(digits.target), 10))
        for i, label in enumerate(digits.target):
            y_onehot[i, label] = 1
        
        print(f"Loaded digits subset: {len(X)} samples")
        return X, y_onehot


def run_phase4_real():
    """Run Phase 4 with real MNIST data"""
    print("\n" + "=" * 60)
    print("Phase 4: Real MNIST Test")
    print("DFA Learning vs Evolving DFA")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X, y = load_real_mnist()
    
    # Split train/test (standard: 60K/10K)
    n_train = 10000  # Use subset for faster testing
    indices = np.random.permutation(len(X))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + 2000]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
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
            perm = np.random.permutation(len(X_train))
            
            for i in perm:
                fixed.learn(X_train[i], y_train[i])
                evolving.learn(X_train[i], y_train[i])
            
            # Evolve
            changes = evolving.evolve()
            
            # Evaluate
            f_acc = fixed.accuracy(X_test, y_test)
            e_acc = evolving.accuracy(X_test, y_test)
            fixed_accs.append(f_acc)
            evolving_accs.append(e_acc)
            
            if (epoch + 1) % 5 == 0:
                units = sum(1 for u in evolving.units if u['active'])
                print(f"Epoch {epoch+1}: Fixed={f_acc:.1%}, Evolving={e_acc:.1%}, Units={units}")
        
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
    print("Phase 4 Real MNIST Results")
    print("=" * 60)
    
    print(f"\nFinal Test Accuracy:")
    print(f"  Fixed (DFA):   {f_final:.1%}")
    print(f"  Evolving:      {e_final:.1%}")
    
    print(f"\nAverage Performance (AUC):")
    print(f"  Fixed (DFA):   {f_auc:.1%}")
    print(f"  Evolving:      {e_auc:.1%}")
    
    advantage = e_final - f_final
    print(f"\nEvolution Advantage: {advantage:+.1%}")
    
    # Comparison with expected DFA performance
    print(f"\n[Comparison]")
    print(f"  - Standard DFA on MNIST typically: 85-95%")
    print(f"  - Our result: {f_final:.1%}")
    print(f"  - Note: Limited epochs ({config.epochs})")
    
    success = advantage > 0
    print(f"\n{'[SUCCESS] Evolution provides advantage!' if success else '[NEUTRAL] No clear advantage'}")
    
    # Save results
    with open("F:/skill/sel-lab/results/phase4_real_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': 'MNIST',
            'n_train': n_train,
            'n_test': len(X_test),
            'fixed_final': float(f_final),
            'evolving_final': float(e_final),
            'fixed_auc': float(f_auc),
            'evolving_auc': float(e_auc),
            'advantage': float(advantage),
            'decision': 'success' if success else 'neutral'
        }, f, indent=2)
    
    print(f"\nResults saved: results/phase4_real_results.json")


if __name__ == "__main__":
    run_phase4_real()
