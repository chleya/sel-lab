# SEL-Lab 阶段 D: 规模扩展测试
# 目标: 测试SEL方法在更大规模问题上的表现

import numpy as np
from sklearn.datasets import fetch_openml
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Phase D: Scale Testing")
print("=" * 60)

# Load larger dataset
print("\nLoading larger dataset...")
try:
    # Try MNIST full
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32) / 255.0
    y_labels = mnist.target.astype(int)
    dataset = "MNIST"
    print(f"Loaded MNIST: {len(X)} samples, {X.shape[1]} features")
except Exception as e:
    print(f"MNIST failed: {e}")
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data / 16.0
    y_labels = d.target.astype(int)
    dataset = "digits"
    print(f"Fallback to digits: {len(X)} samples")

# One-hot encode
y = np.zeros((len(y_labels), 10))
for i, label in enumerate(y_labels):
    y[i, label] = 1

# Split (larger test set)
np.random.seed(42)
indices = np.random.permutation(len(X))
n_train = 5000
n_test = 2000
train_idx = indices[:n_train]
test_idx = indices[n_train:n_train + n_test]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train: {n_train}, Test: {n_test}")

# DFA Model (larger)
input_size = X_train.shape[1]
hidden_sizes = [32, 64, 128]  # Multiple sizes to test
output_size = 10

results = {}

for hidden_size in hidden_sizes:
    print(f"\n--- Testing {input_size}->{hidden_size}->{output_size} ---")
    
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size) * 0.5
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size) * 0.5
    feedback = np.random.randn(output_size, hidden_size) * 0.1
    
    # Training
    epochs, lr = 15, 0.01
    
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        for i in perm:
            h = np.tanh(X_train[i] @ W1)
            out = h @ W2
            error = y_train[i] - out
            fb = error @ feedback
            
            W1 = W1 + lr * np.outer(X_train[i], fb)
            W2 = W2 + lr * np.outer(h, error)
            
            W1 = np.clip(W1, -2, 2)
            W2 = np.clip(W2, -2, 2)
    
    # Evaluation
    correct = 0
    for j in range(len(X_test)):
        h = np.tanh(X_test[j] @ W1)
        out = h @ W2
        pred = int(np.argmax(out))
        true = int(np.argmax(y_test[j]))
        if pred == true:
            correct += 1
    
    acc = correct / len(X_test)
    results[hidden_size] = {
        'accuracy': acc,
        'params': input_size * hidden_size + hidden_size * output_size
    }
    print(f"Hidden size {hidden_size}: {acc:.1%}")

# Analysis
print(f"\n{'='*60}")
print("Scale Testing Results")
print(f"{'='*60}")

print(f"\nDataset: {dataset}")
print(f"Train samples: {n_train}")
print(f"Test samples: {n_test}")

print(f"\nModel Scaling:")
for hs, res in results.items():
    params = res['params']
    acc = res['accuracy']
    print(f"  {hs} hidden: {acc:.1%} ({params:,} params)")

# Best model
best_hs = max(results.keys(), key=lambda k: results[k]['accuracy'])
best_acc = results[best_hs]['accuracy']

print(f"\nBest: {best_hs} hidden units -> {best_acc:.1%}")

# Comparison with small model
small_acc = results[32]['accuracy']
large_acc = results[128]['accuracy']
improvement = large_acc - small_acc

print(f"\nScale improvement (32->128 units): {improvement:+.1%}")

# Save
output = {
    'test': 'Scale Testing',
    'dataset': dataset,
    'n_train': n_train,
    'n_test': n_test,
    'results': {str(k): v for k, v in results.items()},
    'best_hidden_size': best_hs,
    'best_accuracy': best_acc,
    'scale_improvement': float(improvement)
}

with open('F:/skill/sel-lab/results/scale_testing_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved!")
