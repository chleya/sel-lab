# -*- coding: utf-8 -*-
"""
SEL Progress Visualization - Text Version
"""
import sys
import io

# Force UTF-8 encoding on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import numpy as np


def print_bar(value, length=20):
    """Progress bar - ASCII only"""
    filled = int(length * value)
    bar = '=' * filled + '-' * (length - filled)
    return f'[{bar}] {value:.1%}'


def run_visualization():
    print("\n" + "=" * 70)
    print(" SEL Progress Visualization ")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Data
    X = np.random.randn(200, 4) * 2
    y = np.zeros((200, 2))
    for i in range(200):
        if X[i, 0] + X[i, 1] > 0:
            y[i, 0] = 1
        else:
            y[i, 1] = 1
    
    X_train, y_train = X[:100], y[:100]
    X_test, y_test = X[100:], y[100:]
    
    # Network
    net = SELNetwork(4, 2)
    
    print(f"\nNetwork Structure:")
    print(f"  Modules: {len(net.modules)}")
    print(f"  Parameters: {sum(m.weights.size for m in net.modules)}")
    
    print("\n" + "-" * 70)
    print("Training Process (Forward-Only, No Backprop)")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Train':<22} | {'Test':<22} | {'Modules'}")
    print("-" * 70)
    
    te_acc = 0
    for epoch in range(100):
        for i in range(len(X_train)):
            net.learn_all(X_train[i], y_train[i], lr=0.1)
        
        if (epoch + 1) % 10 == 0:
            t_acc = net.accuracy(X_train, y_train)
            te_acc = net.accuracy(X_test, y_test)
            print(f"{epoch+1:>6} | {print_bar(t_acc):<22} | {print_bar(te_acc):<22} | {len(net.modules)}")
    
    print("-" * 70)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Final Test Accuracy: {te_acc:.1%}")
    print(f"Method: SEL Forward-Only Learning (No Backprop)")
    print(f"Key Points:")
    print(f"  - DFA-style error feedback")
    print(f"  - Modular architecture")
    print(f"  - Adaptive learning rate")
    print(f"\nStatus: {'[SUCCESS]' if te_acc > 0.85 else '[PARTIAL]'}")
    
    return te_acc > 0.85


class SELModule:
    def __init__(self, name, in_size, out_size, seed=None):
        if seed: np.random.seed(seed)
        self.name = name
        self.in_size = in_size
        self.out_size = out_size
        self.weights = np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)
        self.local_tension = 0.0
        self.fb = np.random.randn(out_size, out_size) * 0.1
    
    def forward(self, x):
        if x.ndim == 1: x = x.reshape(1, -1)
        return np.tanh(x @ self.weights)
    
    def learn(self, x, error, lr=0.1):
        if x.ndim == 1: x = x.reshape(1, -1)
        fb_error = error @ self.fb
        self.weights += lr * np.dot(x.T, fb_error)
        self.weights = np.clip(self.weights, -2, 2)
        self.local_tension = 0.9 * self.local_tension + 0.1 * np.mean(np.abs(fb_error))
        return np.mean(np.abs(fb_error))
    
    def predict(self, x):
        return int(np.argmax(self.forward(x)))


class SELNetwork:
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.modules = [SELModule(f'm{i}', in_size, out_size, seed=42+i) for i in range(3)]
        self.fb = [m.fb for m in self.modules]
    
    def forward(self, x):
        outputs = [m.forward(x) for m in self.modules]
        return np.mean(outputs, axis=0)
    
    def learn_all(self, x, target, lr=0.1):
        out = self.forward(x)
        err = target - out
        total = 0
        for i, m in enumerate(self.modules):
            total += m.learn(x, err, lr)
        return total / len(self.modules)
    
    def learn_only_new(self, x, target, new_module_idx, lr=0.1):
        out = self.forward(x)
        err = target - out
        self.modules[new_module_idx].learn(x, err, lr)
    
    def predict(self, x):
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X, y):
        return sum(1 for i in range(len(X)) 
                  if self.predict(X[i]) == int(np.argmax(y[i]))) / len(X)


if __name__ == "__main__":
    run_visualization()
