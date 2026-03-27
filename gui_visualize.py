# -*- coding: utf-8 -*-
"""
SEL 可视化 - 强制窗口显示
"""
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 Tk 后端

import matplotlib.pyplot as plt
import numpy as np


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
    
    def forward(self, x):
        outputs = [m.forward(x) for m in self.modules]
        return np.mean(outputs, axis=0)
    
    def learn_all(self, x, target, lr=0.1):
        out = self.forward(x)
        err = target - out
        for m in self.modules:
            m.learn(x, err, lr)
    
    def predict(self, x):
        return int(np.argmax(self.forward(x)))
    
    def accuracy(self, X, y):
        return sum(1 for i in range(len(X)) 
                  if self.predict(X[i]) == int(np.argmax(y[i]))) / len(X)


def main():
    print("Opening visualization window...")
    
    # 数据
    np.random.seed(42)
    X = np.random.randn(200, 4) * 2
    y = np.zeros((200, 2))
    for i in range(200):
        if X[i, 0] + X[i, 1] > 0:
            y[i, 0] = 1
        else:
            y[i, 1] = 1
    
    X_train, y_train = X[:100], y[:100]
    X_test, y_test = X[100:], y[100:]
    
    # 网络
    net = SELNetwork(4, 2)
    
    # 训练并记录历史
    epochs = []
    train_accs = []
    test_accs = []
    
    print("Training (100 epochs)...")
    for epoch in range(100):
        for i in range(len(X_train)):
            net.learn_all(X_train[i], y_train[i], lr=0.1)
        
        if (epoch + 1) % 10 == 0:
            t_acc = net.accuracy(X_train, y_train)
            te_acc = net.accuracy(X_test, y_test)
            epochs.append(epoch)
            train_accs.append(t_acc)
            test_accs.append(te_acc)
            print(f"Epoch {epoch+1}: Train={t_acc:.1%}, Test={te_acc:.1%}")
    
    # 绘图
    print("Opening window...")
    
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('SEL Forward Learning Visualization', fontsize=14, fontweight='bold')
    
    # 1. 准确率曲线
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(epochs, train_accs, 'b-', linewidth=2, label='Train')
    ax1.plot(epochs, test_accs, 'r--', linewidth=2, label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # 2. 权重分布
    ax2 = fig.add_subplot(2, 2, 2)
    weights_all = np.concatenate([m.weights.flatten() for m in net.modules])
    ax2.hist(weights_all, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.set_xlabel('Weight Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Weight Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. 张力条形图
    ax3 = fig.add_subplot(2, 2, 3)
    tensions = [m.local_tension for m in net.modules]
    bars = ax3.bar(range(len(tensions)), tensions, color='lightcoral', edgecolor='black')
    ax3.set_xlabel('Module')
    ax3.set_ylabel('Local Tension')
    ax3.set_title('Module Tensions')
    ax3.set_xticks(range(len(tensions)))
    ax3.set_xticklabels([m.name for m in net.modules])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 网络状态
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    info_text = f"SEL Network Status\n"
    info_text += f"{'='*30}\n"
    info_text += f"Modules: {len(net.modules)}\n"
    info_text += f"Parameters: {sum(m.weights.size for m in net.modules)}\n"
    info_text += f"Final Accuracy: {test_accs[-1]:.1%}\n\n"
    info_text += f"Module Details:\n"
    for i, m in enumerate(net.modules):
        w_mean = np.mean(np.abs(m.weights))
        info_text += f"  {m.name}: |w|={w_mean:.3f}, T={m.local_tension:.2f}\n"
    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.show()  # 这应该打开一个窗口
    print("Window should be open now.")


if __name__ == "__main__":
    main()
