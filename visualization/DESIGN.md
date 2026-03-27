# SEL-Lab Visualization Enhancement Design
# SEL-Lab 可视化增强设计方案

## 1. 目标
开发一套完整的可视化工具，增强 SEL-Lab 网络拓扑演化过程的可理解性和交互性。

## 2. 推荐可视化类型

### 2.1 网络拓扑演化动画
- **类型**: 动态网络图（使用 NetworkX + Matplotlib animation）
- **展示**: 模块添加/移除、权重变化、连接强度
- **交互**: 播放/暂停、时间滑块、放大缩小

### 2.2 模块状态热力图
- **类型**: 时序热力图
- **展示**: 各模块在训练过程中的张力变化、权重范数
- **颜色映射**: 张力高→红，低→蓝

### 2.3 结构演化时间线
- **类型**: 时间轴 + 事件标记
- **展示**: 结构变化事件（添加/移除模块）
- **标记**: 准确率峰值、张力阈值跨越

### 2.4 多实验对比面板
- **类型**: 分面图表
- **展示**: 多次实验的准确率/模块数对比
- **统计**: 均值、置信区间

### 2.5 交互式权重探索器
- **类型**: 可折叠面板
- **展示**: 选中模块的权重矩阵详情
- **操作**: 拖拽、缩放

## 3. 技术栈选择

| 组件 | 选择 | 理由 |
|------|------|------|
| 可视化引擎 | Matplotlib | 广泛支持、动画成熟 |
| 网络图 | NetworkX | Python 原生、拓扑操作强 |
| 交互 | Matplotlib Widgets | 无需额外依赖 |
| 动画 | FuncAnimation | 内置支持 |
| 输出格式 | PNG/GIF/HTML | 通用兼容 |

## 4. 实现步骤

### Step 1: 基础架构
- [ ] 创建可视化输出目录
- [ ] 编写 SELVisualizer 基类
- [ ] 实现数据采集接口

### Step 2: 拓扑演化动画
- [ ] 实现网络图绘制函数
- [ ] 添加动画更新逻辑
- [ ] 实现交互控件（播放/暂停）

### Step 3: 状态监控面板
- [ ] 张力演化曲线
- [ ] 模块数量时间线
- [ ] 权重分布动态图

### Step 4: 高级可视化
- [ ] 结构演化桑基图
- [ ] 多实验对比箱线图
- [ ] 并行激活可视化

### Step 5: 输出功能
- [ ] 保存 GIF/PNG
- [ ] 保存 HTML 交互报告
- [ ] 命令行接口

## 5. 文件结构

```
F:\skill\sel-lab\visualization\
├── __init__.py
├── base_visualizer.py      # 基类
├── topology_animation.py    # 拓扑演化动画
├── metrics_panel.py         # 指标面板
├── multi_experiment.py      # 多实验对比
├── utils.py                 # 工具函数
├── main.py                  # 主入口
└── examples/                # 示例
    └── run_visualization.py
```

## 6. API 设计

### SELVisualizer 基类
```python
class SELVisualizer:
    def __init__(self, network, output_dir: str):
        self.network = network
        self.output_dir = output_dir
    
    def collect_metrics(self) -> Dict:
        """采集指标"""
        pass
    
    def render(self):
        """渲染可视化"""
        pass
    
    def save(self, filename: str, format: str = 'png'):
        """保存"""
        pass
```

## 7. 依赖
- numpy
- matplotlib
- networkx

## 8. 时间线
- Phase 1: 基础架构 + 拓扑动画 (2-3 小时)
- Phase 2: 指标面板 (1-2 小时)
- Phase 3: 高级功能 + 输出 (2-3 小时)

## 9. 风险与缓解
| 风险 | 缓解措施 |
|------|----------|
| 动画卡顿 | 减少帧数、提供预览模式 |
| 内存占用 | 分批处理、清理临时数据 |
| 兼容性 | 虚拟后端、无 GUI 模式 |

## 10. 验收标准
- [ ] 支持拓扑演化动画（GIF/MP4）
- [ ] 支持交互式播放控制
- [ ] 支持保存可视化结果
- [ ] 兼容现有 SELNetwork 数据结构
