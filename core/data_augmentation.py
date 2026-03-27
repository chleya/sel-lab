# Data Augmentation Test (B.3)
import numpy as np
from sklearn.datasets import load_digits
import json

print("=" * 60)
print("Data Augmentation Test (B.3)")
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


def augment(x):
    """Apply random augmentation"""
    x = x.copy()
    # Random shift (±1 pixel)
    x = np.roll(x, np.random.randint(-1, 2))
    # Random scale (0.95-1.05)
    x = x * np.random.uniform(0.95, 1.05)
    return np.clip(x, 0, 1)


def run_experiment(use_augmentation):
    """Run with or without augmentation"""
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.5
    W2 = np.random.randn(hidden_size, output_size) * 0.5
    feedback = np.random.randn(output_size, hidden_size) * 0.1
    
    epochs = 50
    lr = 0.01
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        for i in perm:
            x = X_train[i]
            if use_augmentation:
                x = augment(x)
            
            h = np.tanh(x @ W1)
            out = h @ W2
            error = y_train[i] - out
            fb = error @ feedback
            
            W1 = W1 + lr * np.outer(x, fb)
            W2 = W2 + lr * np.outer(h, error)
            
            np.clip(W1, -2, 2, out=W1)
            np.clip(W2, -2, 2, out=W2)
        
        # Evaluate
        train_acc = accuracy(X_train, y_train, W1, W2)
        test_acc = accuracy(X_test, y_test, W1, W2)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    
    return train_accs[-1], test_accs[-1], np.mean(train_accs), np.mean(test_accs)


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


# Run experiments
print("\nRunning experiments...")

# Without augmentation
print("Training without augmentation...")
no_aug_train, no_aug_test, no_aug_avg_train, no_aug_avg_test = run_experiment(False)
print(f"  No augmentation: Train={no_aug_train:.1%}, Test={no_aug_test:.1%}")

# With augmentation
print("Training with augmentation...")
with_aug_train, with_aug_test, with_aug_avg_train, with_aug_avg_test = run_experiment(True)
print(f"  With augmentation: Train={with_aug_train:.1%}, Test={with_aug_test:.1%}")

# Results
improvement = with_aug_test - no_aug_test
train_gap_no = no_aug_train - no_aug_test
train_gap_with = with_aug_train - with_aug_test

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Without augmentation:")
print(f"  Train: {no_aug_train:.1%}, Test: {no_aug_test:.1%}")
print(f"  Overfitting gap: {train_gap_no:.1%}")
print(f"\nWith augmentation:")
print(f"  Train: {with_aug_train:.1%}, Test={with_aug_test:.1%}")
print(f"  Overfitting gap: {train_gap_with:.1%}")
print(f"\nTest improvement: {improvement:+.1%}")
print(f"Overfitting reduction: {train_gap_no - train_gap_with:+.1%}")

# Check targets
target_improvement = 0.01  # 1%
target_reduction = 0.02  # 2%

improvement_passed = improvement >= target_improvement
reduction_passed = (train_gap_no - train_gap_with) >= target_reduction

print(f"\nTargets:")
print(f"  Test improvement >= {target_improvement:.0%}: {'PASS' if improvement_passed else 'NOT PASS'}")
print(f"  Gap reduction >= {target_reduction:.0%}: {'PASS' if reduction_passed else 'NOT PASS'}")

# Save
result = {
    'test': 'Data Augmentation',
    'no_augmentation': {
        'train': float(no_aug_train),
        'test': float(no_aug_test),
        'avg_train': float(no_aug_avg_train),
        'avg_test': float(no_aug_avg_test),
        'overfit_gap': float(train_gap_no)
    },
    'with_augmentation': {
        'train': float(with_aug_train),
        'test': float(with_aug_test),
        'avg_train': float(with_aug_avg_train),
        'avg_test': float(with_aug_avg_test),
        'overfit_gap': float(train_gap_with)
    },
    'improvement': float(improvement),
    'gap_reduction': float(train_gap_no - train_gap_with),
    'improvement_passed': improvement_passed,
    'gap_reduction_passed': reduction_passed
}

with open('F:/skill/sel-lab/results/augmentation_results.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved: augmentation_results.json")
