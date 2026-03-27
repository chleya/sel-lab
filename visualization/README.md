# SEL-Lab Visualization Package

SEL 网络可视化工具包

## 功能

- **拓扑演化动画** - 展示网络结构动态变化
- **指标监控面板** - 训练指标实时可视化
- **多实验对比** - 多次实验结果对比分析

## 快速开始

```python
from core.sel_core import SELConfig, SELTrainer, create_task
from visualization.topology_animator import TopologyAnimator
from visualization.sel_visualizer import VisualizationConfig

# 训练网络
config = SELConfig(input_size=4, output_size=2, epochs=50)
trainer = SELTrainer(config)
X, y = create_task()
trainer.train(X[:100], y[:100], X[100:], y[100:])

# 创建动画
animator = TopologyAnimator()
for metrics in trainer.metrics:
    animator.add_history(metrics)

animator.create_animation()
animator.save_animation('topology')
```

## 组件

| 文件 | 功能 |
|------|------|
| `sel_visualizer.py` | 基类 |
| `topology_animator.py` | 拓扑动画 |
| `metrics_panel.py` | 指标面板 |
| `experiment_comparator.py` | 实验对比 |
| `main.py` | 主入口 |
| `examples/run_examples.py` | 示例 |

## 运行示例

```bash
cd F:\skill\sel-lab\visualization\examples
python run_examples.py
```

## 输出

- PNG 静态图像
- GIF 动画
- JSON 数据
