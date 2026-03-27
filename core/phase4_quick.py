# Simplified real MNIST test
import numpy as np
from sklearn.datasets import load_digits
import json

from runtime import resolve_exploratory_results_path

print("=== Phase 4 Real MNIST Test ===")

# Load real data
digits = load_digits()
X = digits.data / 16.0
y = np.zeros((len(digits.target), 10))
for i, label in enumerate(digits.target):
    y[i, label] = 1

print(f"Dataset: {len(X)} samples, {X.shape[1]} features")

# Split
np.random.seed(42)
indices = np.random.permutation(len(X))
n_train = int(len(X) * 0.7)
train_idx, test_idx = indices[:n_train], indices[n_train:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# DFA model
np.random.seed(42)
input_size, hidden_size, output_size = 64, 16, 10
W1 = np.random.randn(input_size, hidden_size) * 0.5
W2 = np.random.randn(hidden_size, output_size) * 0.5
feedback = np.random.randn(output_size, hidden_size) * 0.1

def forward(x):
    h = np.tanh(x @ W1)
    return h @ W2

def learn(x, target):
    h = np.tanh(x @ W1)
    out = h @ W2
    error = target - out
    fb = error @ feedback
    W1_new = W1 + 0.01 * np.outer(x, fb)
    W2_new = W2 + 0.01 * np.outer(h, error)
    return W1_new, W2_new, np.mean(error ** 2)

def accuracy(X, y):
    correct = sum(1 for i in range(len(X)) if int(np.argmax(forward(X[i]))) == int(np.argmax(y[i])))
    return correct / len(X)

# Train
print("\nTraining...")
for epoch in range(50):
    perm = np.random.permutation(len(X_train))
    for i in perm:
        new_W1, new_W2, _ = learn(X_train[i], y_train[i])
        W1, W2 = new_W1, new_W2
    
    if (epoch + 1) % 10 == 0:
        acc = accuracy(X_test, y_test)
        print(f"Epoch {epoch+1}: Accuracy = {acc:.1%}")

# Final
final_acc = accuracy(X_test, y_test)
print(f"\n{'='*50}")
print("Phase 4 Real Data Results")
print(f"{'='*50}")
print(f"Dataset: sklearn-digits (real data)")
print(f"Samples: {len(X)}, Features: {X.shape[1]}")
print(f"Final Accuracy: {final_acc:.1%}")

# Save
result = {
    'dataset': 'sklearn-digits',
    'samples': len(X),
    'features': X.shape[1],
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'final_accuracy': float(final_acc),
    'decision': 'success' if final_acc > 0.8 else 'neutral'
}

target = resolve_exploratory_results_path("phase4_real_results.json")
with target.open('w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved: {target}")
