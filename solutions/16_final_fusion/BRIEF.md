# 方案 16 · 终极融合：Stacker + 答案形态 + 算式重执行（收口）

**迭代动机（第四轮结论）**
- 当前最优：方案 11（OOF 分数级 stacking）→ validation `test_like` **87.32**，真实 test **83.30**。
- 方案 14 证伪了"题干数值可达性"（错误答案与正确答案一样可达），
  但发现**意外正增益**：「答案形态」特征（是否整数、量级是否合理）与嵌入**正交**，`emb+shape` 比 `emb` **+4.51**。
- 方案 15 发现「算式重执行」（LHS=RHS 重算）是**真实但微弱**的正交信号，
  在无投票配置下把 k=1 抬到 **69.23**（基准 66.67，McNemar p=0.12）。
- **关键缺口**：方案 11 的基础打分器集合里**没有** shape 与 NGSC（数值自洽）这两路证据。

**假设（可证伪）**：把「答案形态」与「算式重执行」作为**新的基础打分器**加入方案 11 的 OOF stacking 框架，
可以在不牺牲 k≥2 的前提下，同时提升 test_like 与真实 test，超过 87.32 / 83.30。

**方法骨架（必须复用方案 11 的架构，但要自包含实现）**
1. 基础打分器（≥6 路，全部在 train 的类 test 结构子集上、按 `1/k` 加权训练）：
   - 文本特征 LR
   - 题内去均值嵌入 LR
   - 成对排序（Bradley-Terry）
   - 投票占比
   - **新增：答案形态**（是否整数/分数/负数、量级相对题干数字的合理性、是否为题干数字本身等）
   - **新增：算式重执行**（解析候选文本中的等式，重算 LHS 与 RHS 是否一致、能否推出最终答案）
2. **K 折 OOF** 生成元特征（务必 OOF，方案 11 已量化 in-fold 会虚高 ~13 点）
3. 元特征：各基础分数的题内 z、平均秩、组内 top1、一致性统计量
4. 元学习器：**Bradley-Terry 成对损失**（方案 11 证明增益来自成对损失，而非"学习"本身）
5. 每题 argmax；推理纯 numpy，≤16 MiB，≤30 s

**必做消融（≥4 组）**
1. **方案 11 原配置（6 路，无 shape/NGSC）vs 本方案（8 路）** ← 核心对照
2. 只加 shape / 只加 NGSC，分别报告边际贡献
3. 逐个移除基础模型，给出完整边际贡献表
4. **按 k 分层报告**（k=1/2/3），确认 k≥2 没有被 k=1 的收益牺牲（方案 12 就栽在这里）

**主指标**：validation `test_like.accuracy`（当前最佳 **87.32**）；
真实 test 用官方 pipeline 仅跑一次（当前最佳 **83.30**）。
**同时报告 val k=1 子集准确率**（当前最佳 66.67；方案 15 的无投票配置达 69.23）。

**要交付**：`method.py`（纯 numpy）、`train.py`、`artifacts/`（≤16 MiB）、`results/validation.json`、
`results/test.json`、`REPORT.md`

**评测**
```bash
cd ../00_shared && python harness.py --method ../16_final_fusion/method.py \
  --checkpoint ../16_final_fusion/artifacts --split validation \
  --output ../16_final_fusion/results/validation.json
```

**铁律**：训练只用 train；val 只用于选型；test 参考不得用于训练/调参/选择；禁止位置/ID 先验。
若新证据族被强基线吸收（像方案 15 那样），**如实报告**——"该证据源已被 stacker 吸收"是有效结论。
