# SEL-Lab Full-Auto Prompt

Use this prompt when running Codex in autonomous mode for this repository.

```text
你正在推进 SEL-Lab，一个持续实验系统。

主问题不要变：
forward-only structure reuse 在 continual learning 中何时有效，以及哪些机制带来收益？

当前主线：
围绕 Phase 3 推进，重点使用 digits_pairs 作为边界 benchmark。
目标不是证明某一个现有策略一定正确，而是找到在 richer sequential benchmark 上最有效的 reuse 机制，并解释它为什么有效或为什么失败。

当前已知事实：
- synthetic suites 上，specialist-merge 系列稳定优于 adapt_only
- digits_pairs 上，当前 leading candidate 仍优于 adapt_only，但仍落后 fixed
- merge-scale tuning 已经基本摸清，不能单独解决 fixed gap
- 轻量 retention heuristic 已被证明不够强

因此默认优先方向：
尝试更强的 retention / routing / gating / role-separation 机制，而不是继续重复参数扫描。

每一轮必须执行：
1. 读取 HANDOFF.md、AUTONOMY_PROTOCOL.md、UNIFIED_REPORT.md 和相关 results
2. 运行当前相关代码或测试
3. 记录结果变化
4. 找出当前最关键的问题
5. 做一个最小但高价值的修改
6. 再运行验证或实验
7. 更新结果文件、报告、交接文档和 memory

要求：
- 每轮必须有明确改进目标
- 不允许重复已经证明无效的轻微修改
- 如果连续 3 轮没有信息增量，必须改变策略
- 默认不要停下来问是否继续，直接推进
- 当前 leading mechanism 不是固定真理，可以被更强结果或更强负结果替换

但以下情况必须停止并汇报：
- 主线结论变化
- 出现强正结果，可以升级项目叙事
- 出现强负结果，说明当前方向不成立
- 当前 leading mechanism 被别的机制取代
- 连续 2 到 3 轮没有实质信息增量
- 需要更换 benchmark、重构研究问题或大改架构
- 测试失败，或报告与结果不一致

开始执行。
```
