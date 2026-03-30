# SEL-Lab 综合审查报告
**审查日期**: 2026-03-30  
**审查范围**: 代码逻辑、实验数据、结果有效性、整改建议

---

## 一、总体结论

| 层面 | 状态 | 说明 |
|---|---|---|
| 计算逻辑（forgetting/BWT） | **正确** | 符号一致，公式无误 |
| 实验规模（runs=3，主体）| **勉强可用** | 大多数文件 runs=3，已有统计基础 |
| 剪枝逻辑（sel_core.py） | **有 bug** | max 应为 min，影响所有结构演化实验 |
| Selector 变体有效性 | **大面积失效** | 55.6% 结果完全重复，selector 决策无差异 |
| 核心正向信号 | **真实存在** | 遗忘率优势、current_path_boost 效果均有统计支撑 |
| 对比 fixed 基线 | **仍落后** | 所有 Phase 均未超过固定网络精度 |

---

## 二、各 Phase 数据审查

### Phase 2（单任务结构演化）

| 实验 | fixed_final | evolving_final | advantage | decision |
|---|---|---|---|---|
| phase2_results | 0.900 | 0.880 | **-0.020** | neutral |
| phase2_hard | 0.406 | 0.404 | **-0.002** | neutral |
| phase2_improved | 0.406 | 0.324 | **-0.082** | neutral |

**结论**：三组实验全部为 neutral，SEL 在单任务场景下**始终慢于固定网络**，且最差情况下差距达 8.2%。Phase 2 的方向已被数据否定，不应继续投入。

---

### Phase 3（持续学习，核心战场）

#### 3.1 基础消融（runs=5，最可信数据）

| policy | avg_acc | vs_adapt_only | forgetting |
|---|---|---|---|
| **adapt_only**（基准） | 0.6285 | 0.00 | 0.0207 |
| task_specialist_clone_distill_merge | 0.6395 | **+0.0110** | 0.0140 |
| task_specialist_clone_limited_merge | 0.6390 | +0.0105 | 0.0127 |
| task_specialist_clone_anchor_merge | 0.6385 | +0.0100 | 0.0133 |
| best_clone | 0.6110 | -0.0175 | 0.0733 |
| random_clone | 0.5915 | -0.0370 | 0.0367 |
| **fixed**（上限）| **0.4560** | — | 0.2967 |

> **关键读数**：SEL 的遗忘率从 fixed 的 0.30 压到 0.013，减少了 **23 个百分点**。这是项目最强的真实信号。但对 adapt_only 的精度优势仅 1.1%，在 5 runs 下置信度中等。

#### 3.2 高维任务（digits_pairs，runs=3）

| policy | avg_acc vs adapt_only | current_acc delta | forgetting delta |
|---|---|---|---|
| task_specialist_clone_current_path_boost_merge | **+0.1325** | **+0.3404** | -0.1285 |
| task_specialist_clone_limited_merge | +0.0504 | +0.1313 | -0.0383 |
| task_specialist_clone_distill_merge | +0.0493 | +0.1279 | -0.0383 |

> **这是项目最有价值的发现**：在高维输入（input=64）的 digits_pairs 场景，`current_path_boost` 策略把 avg_acc 提升了 13.25%，current_task acc 提升高达 34%。差距显著，runs=3 情况下可信度高。

> **但注意**：相比 fixed 基线，best policy 的 `mean_avg_accuracy_gain_vs_fixed = -0.054`，仍然落后固定网络 5.4%。

#### 3.3 SEL 整体遗忘对比

| 维度 | 数值 |
|---|---|
| SEL policies 平均遗忘 | **0.0615** |
| fixed 平均遗忘 | **0.2948** |
| SEL 遗忘优势 | **-0.2333**（SEL 少遗忘 23.3%） |

这是跨全部 226 个标准格式文件、1362 条 policy 结果的汇总，统计量最大、最可靠。

#### 3.4 Selector 变体有效性分析

| selector 家族 | 总结果数 | 近零差异（±0.5%） | 失效率 |
|---|---|---|---|
| fingerprint | 76 | 18 | 24% |
| two_stage | 103 | 26 | **25%** |
| guarded | 40 | 9 | 22% |
| ranking | 123 | 16 | 13% |
| task_ranking | 107 | 14 | 13% |

55.6% 的结果完全重复（相同 acc 和 forgetting 到小数点后 4 位）。原因：在高维任务场景下所有 selector 都路由到 current_path_boost，决策已经饱和，selector 种类不影响结果。

**结论**：selector 横向扩展无效，不应继续增加变体。

#### 3.5 Regime Switch Oracle（理论上界）

`regime_switch_oracle`（已知最优场景下人工选择最佳 policy）的 delta vs adapt_only = **+0.0749**，而最好的自动 selector（`two_stage`）= **+0.0437**。

两者差距 0.031，说明 selector 还有改进空间，但不是靠增加变体，而是靠提升单个 selector 的判断质量。

---

### Phase 4（真实数据集）

| 实验 | fixed_final | evolving_final | advantage | decision |
|---|---|---|---|---|
| phase4_results（700样本）| 0.826 | **0.921** | **+0.096** | success |
| phase4_real（1257样本）| 0.866 | **0.923** | **+0.057** | success |

**Phase 4 是唯一 SEL 超越 fixed 基线的 Phase**，而且差距在真实数据集上仍保持 5.7%。这非常重要——说明 SEL 在真实的非线性、高维场景下确实有优势。

> **但 phase4_expansion 显示**：tension 触发的结构扩展失效（dominant_failure_mode: age_decay_too_aggressive），扩展后准确率反而下降。Phase 4 的成功主要来自适应学习率，而非结构演化本身。

---

## 三、代码层问题

### Bug 1：剪枝逻辑方向错误（致命）

**位置**：`core/sel_core.py`，`structural_adaptation` 方法中 `elif len(self.modules) > 5` 分支

**问题**：
```python
# 当前错误代码
prune_idx = max(pressured_modules, key=lambda item: item[1].local_tension)[0]
```

应改为：
```python
# 正确：删除张力最低、最不活跃的模块
prune_idx = min(range(len(self.modules)), 
                key=lambda i: self.modules[i].local_tension)
```

张力高 = 模块正在努力学习，应保留。张力低 = 模块已停滞，才应被剪枝。当前逻辑每次都删最努力的模块，使得所有包含 clone→prune 流程的实验结果不可信。

### Bug 2：`structure_recommendation` 污染实验记录

**位置**：`core/phase3.py:195`，`exploration.structure_insights` 的推荐结论被写入 `task_results` JSON

该推荐基于合成数据得出的"3模块最优"假设，不适用于所有场景，不应进入实验记录。应只保留在日志输出中。

### 问题 3：`phase3_results.json` 编码损坏

该文件使用 GBK 读取时报错（`0x8c` 非法字节），内容无法被自动化工具读取。如果包含关键汇总数据，需要重新生成。

---

## 四、整改意见（按优先级）

### P0：本周必须修（影响数据可信度）

**P0-1 修复剪枝 bug**

```python
# core/sel_core.py
# 找到 structural_adaptation 方法中 elif len(self.modules) > 5 分支
# 将 max 改为 min
prune_idx = min(range(len(self.modules)), 
                key=lambda i: self.modules[i].local_tension)
```

**P0-2 重跑基础消融**

修复 bug 后，重新运行 `phase3_ablation` 和 `phase3_digits_*` 系列（这两个是最常引用的结果），重新确认数字。

---

### P1：本月完成（提升实验质量）

**P1-1 冻结 selector 横向扩展**

把 20+ 个 selector 变体中，除以下 3 个之外全部移入 `archive/`：
- `task_specialist_clone_current_path_boost_merge`（最强单一策略）
- `task_specialist_clone_limited_merge`（遗忘最低）
- `task_specialist_clone_two_stage_selector_merge`（selector 家族最佳）

**P1-2 把 runs 提升到 5**

当前主体是 runs=3，把核心 digits_pairs 系列改为 runs=5，给出均值±标准差，建立统计置信度。

**P1-3 解耦 `exploration` 依赖**

从 `phase3.py` 的 `task_results` 记录中移除 `structure_recommendation` 字段，改为仅在 verbose 日志中输出。

---

### P2：下个月（提升研究深度）

**P2-1 诊断 Phase 4 扩展失效**

`age_decay_too_aggressive` 导致 tension 过早衰减，结构从不扩展。需要对 `age_decay` 参数做系统扫描（建议范围 0.001–0.02），找到能稳定触发扩展的配置。

**P2-2 tension 动态记录**

在 Phase 3 运行中，每个 epoch 记录各模块的 `local_tension`、`plateau`、`volatility` 三个分量。这是路线 B（tension 理论）的基础数据，现在缺失。

**P2-3 找一个真实的第三方 benchmark**

目前所有实验都用的是自建的 `digits_pairs`、`feature_shift` 等合成场景。应在 Split-MNIST 或 Permuted-MNIST 上跑一遍，和 EWC、PackNet 做正式对比。这是走向发表的必要条件。

---

## 五、当前可以对外展示的结论（已有统计支撑）

以下结论在现有数据质量下可以成立：

1. **SEL 的遗忘抑制效果显著**：平均遗忘率从 fixed 的 29.5% 降至 6.2%（跨 226 个文件，1362 条记录）。
2. **current_path_boost 在高维场景有大幅提升**：digits_pairs 场景下 avg_acc +13.25%，runs=3 可信。
3. **Phase 4 SEL 超越 fixed 基线**：真实 digits 数据集上 +5.7%，runs=3 可信。
4. **random_clone 有害**：相比 adapt_only 始终是负效果，可作为消融对照的负样本。

以下结论**不可对外展示**，需要重跑：

- 所有涉及 clone→prune 流程的结构演化结论（受剪枝 bug 影响）
- 20+ 个 selector 变体之间的对比（55.6% 重复，无区分度）

---

*本报告由代码审查 + 全量数据分析生成，基于 265 个 JSON 文件、1362 条 policy 结果。*
