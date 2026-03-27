# Full MNIST Test with Stable DFA
import numpy as np
from sklearn.datasets import fetch_openml
import json
import warnings

from runtime import resolve_exploratory_results_path
warnings.filterwarnings('ignore')

print("=" * 60)
print("Full MNIST Test - Stable Version")
print("=" * 60)

# Load MNIST
print("\nLoading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data.astype(np.float32) / 255.0
y_labels = mnist.target.astype(int)

y = np.zeros((len(y_labels), 10))
for i, label in enumerate(y_labels):
    y[i, label] = 1

print(f"Loaded: {len(X)} samples")

# Split
np.random.seed(42)
indices = np.random.permutation(len(X))
n_train = 3000
n_test = 1000
train_idx = indices[:n_train]
test_idx = indices[n_train:n_train + n_test]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train: {n_train}, Test: {n_test}")

# Model
input_size, hidden_size, output_size = 784, 64, 10
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
W2 = np.random.randn(hidden_size, output_size) * 0.01
feedback = np.random.randn(output_size, hidden_size) * 0.01


def train_and_test():
    global W1, W2
    
    # Training with smaller lr and clipping
    epochs, lr = 10, 0.005
    
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        for i in perm:
            h = np.tanh(X_train[i] @ W1)
            out = h @ W2
            error = y_train[i] - out
            fb = error @ feedback
            
            W1 = W1 + lr * np.outer(X_train[i], fb)
            W2 = W2 + lr * np.outer(h, error)
            
            # Clip for stability
            W1 = np.clip(W1, -1, 1)
            W2 = np.clip(W2, -1, 1)
        
        # Evaluate
        acc = test_accuracy(W1, W2)
        print(f"Epoch {epoch+1}/{epochs}: {acc:.1%}")
    
    return test_accuracy(W1, W2)


def test_accuracy(w1, w2):
    correct = 0
    for i in range(len(X_test)):
        h = np.tanh(X_test[i] @ w1)
        out = h @ w2
        pred = int(np.argmax(out))
        true = int(np.argmax(y_test[i]))
        if pred == true:
            correct += 1
    return correct / len(X_test)


print("\nTraining...")
final_acc = train_and_test()

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Final Accuracy: {final_acc:.1%}")

# Save
result = {
    'test': 'MNIST Stable',
    'n_train': n_train,
    'n_test': n_test,
    'model': f'{input_size}->{hidden_size}->{output_size}',
    'epochs': 10,
    'final_accuracy': float(final_acc),
    'lr': 0.005,
    'clip': (-1, 1)
}

target = resolve_exploratory_results_path("phase4_mnist_full_results.json")
with target.open('w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved: {target}")
