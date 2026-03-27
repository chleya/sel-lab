# SEL-Lab
Structural Evolution Learning Laboratory

## 项目目标
开发一个智能从结构演化中涌现的系统，基于局部前向动力学，而非全局反向优化。

**核心原则**：
- 无反向传播的学习
- 动力学导向的学习过程
- 结构作为核心实体
- 以适应性为评估标准

---

## 快速开始

### 1. 运行核心框架
```powershell
python F:\\skill\\sel-lab\\core\\sel_core.py
```

### 2. 运行实验
```powershell
python F:\\skill\\sel-lab\\orchestrator\\experiment_runner.py
```

### 3. 运行可视化
```powershell
python F:\\skill\\sel-lab\\gui_visualize.py
```

---

## 项目结构

```
F:\skill\sel-lab\
├── core/
│   ├── __init__.py              # 包初始化
│   ├── sel_core.py               # SEL 核心框架
│   └── environment.py            # 环境接口
├── orchestrator/
│   └── experiment_runner.py      # 实验运行器
├── protocols/
│   └── phase1_forward_feedback.yaml  # Phase 1 协议
├── analysis/
│   └── metrics.py               # 指标计算
├── results/                      # 实验结果
├── *_visualize.py               # 可视化工具
└── README.md                    # 本文档
```

---

## 核心组件

### SELConfig - 配置
```python
from core.sel_core import SELConfig, SELTrainer

config = SELConfig(
    input_size=4,
    output_size=2,
    initial_modules=3,
    learning_rate=0.1,
    epochs=100
)
```

### SELNetwork - 网络
```python
from core.sel_core import SELNetwork, SELConfig

config = SELConfig(input_size=4, output_size=2)
network = SELNetwork(config)
```

### SELTrainer - 训练器
```python
trainer = SELTrainer(config)
result = trainer.train(X_train, y_train, X_test, y_test)
```

---

## 研究阶段

| 阶段 | 状态 | 描述 |
|------|------|------|
| Phase 0 | ✅ 完成 | 概念分析 |
| Phase 1 | 🔄 进行中 | 前向学习验证 |
| Phase 2 | ⏳ 等待 | 结构演化优势 |
| Phase 3 | ⏳ 等待 | 长期累积 |

---

## 核心结果

### 前向学习
- 测试准确率：82%
- 无反向传播
- 模块化架构

### 结构演化
- 模块数量：3 → 5
- 张力驱动适应
- 局部更新

---

## 协议约束

### phase1_forward_feedback.yaml
```yaml
constraints:
  no_backpropagation: true
  no_global_loss_scalar: true
  no_backward_pass: true
  local_updates_only: true
```

---

## 可视化

### GUI 窗口
包含：
- 准确率曲线
- 权重分布
- 张力演化
- 网络状态

### 文本模式
包含：
- 训练进度条
- 准确率指标
- 模块数量

---

## 成功标准

1. ✅ 前向学习达到 >80% 准确率
2. ✅ 局部更新无需反向传播
3. 🔄 结构演化（进行中）
4. ⏳ 增量学习（未来工作）

---

## 下一步

1. 完成 Phase 1 协议实现
2. 测试更复杂任务
3. 实现零遗忘的增量学习
4. 添加完整文档

---

*SEL-Lab：结构演化，智能涌现。*
