# Learning Rate Schedule Test (B.2)
import numpy as np
from sklearn.datasets import load_digits
import json

from runtime import resolve_exploratory_results_path

print("=" * 60)
print("Learning Rate Schedule Test (B.2)")
print("=" * 60)

# Load data
digits = load_digits()
X = digits.data / 16.0
y = np.zeros((len(digits.target), 10))
for i, label in enumerate(digits.target):
    y[i, label] = 1

print(f"Dataset: {len(X)} samples")

# Split
np.random.seed(42)
indices = np.random.permutation(len(X))
n_train = int(len(X) * 0.7)
train_idx, test_idx = indices[:n_train], indices[n_train:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Model
input_size, hidden_size, output_size = 64, 16, 10


def run_experiment(schedule_type):
    """Run experiment with different lr schedules"""
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.5
    W2 = np.random.randn(hidden_size, output_size) * 0.5
    feedback = np.random.randn(output_size, hidden_size) * 0.1
    
    epochs = 50
    results = []
    
    for epoch in range(epochs):
        # LR schedule
        if schedule_type == 'constant':
            lr = 0.01
        elif schedule_type == 'warmup':
            lr = 0.01 * min(1.0, (epoch + 1) / 5)
        elif schedule_type == 'cosine':
            lr = 0.01 * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        elif schedule_type == 'decay':
            lr = 0.01 * np.exp(-epoch * 0.05)
        
        perm = np.random.permutation(len(X_train))
        for i in perm:
            h = np.tanh(X_train[i] @ W1)
            out = h @ W2
            error = y_train[i] - out
            fb = error @ feedback
            
            W1 = W1 + lr * np.outer(X_train[i], fb)
            W2 = W2 + lr * np.outer(h, error)
            
            np.clip(W1, -2, 2, out=W1)
            np.clip(W2, -2, 2, out=W2)
        
        # Evaluate
        acc = accuracy(X_test, y_test, W1, W2)
        results.append(acc)
    
    return results[-1], np.mean(results[-5:]), results


def accuracy(X, y, w1, w2):
    correct = 0
    for i in range(len(X)):
        h = np.tanh(X[i] @ w1)
        out = h @ w2
        pred = int(np.argmax(out))
        true = int(np.argmax(y[i]))
        if pred == true:
            correct += 1
    return correct / len(X)


# Run all schedules
schedules = ['constant', 'warmup', 'cosine', 'decay']
print("\nRunning experiments...")

results = {}
for schedule in schedules:
    final, avg_last5, all_accs = run_experiment(schedule)
    results[schedule] = {'final': final, 'avg_last5': avg_last5, 'all': all_accs}
    print(f"{schedule:10s}: Final={final:.1%}, Avg(last5)={avg_last5:.1%}")

# Find best
best_schedule = max(results.keys(), key=lambda x: results[x]['final'])
best_final = results[best_schedule]['final']
baseline_final = results['constant']['final']

improvement = best_final - baseline_final

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Baseline (constant): {baseline_final:.1%}")
print(f"Best ({best_schedule}): {best_final:.1%}")
print(f"Improvement: {improvement:+.1%}")

# Check target
target = 0.02  # 2% improvement
passed = improvement >= target
print(f"\nTarget (>= {target:.0%}): {'PASS' if passed else 'NOT PASS'}")

# Save
output = {
    'test': 'LR Schedule Comparison',
    'schedules': {k: {'final': float(v['final']), 'avg_last5': float(v['avg_last5'])} for k, v in results.items()},
    'best_schedule': best_schedule,
    'improvement': float(improvement),
    'target': target,
    'passed': passed
}

target = resolve_exploratory_results_path("lr_schedule_results.json")
with target.open('w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved: {target}")
