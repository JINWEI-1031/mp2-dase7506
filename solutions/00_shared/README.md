# DASE7506 MP2 — 方案实验共享基准

本目录是**所有方案共用的评测基准**。每个方案在 `solutions/<编号_名称>/` 下独立开发、互不干扰，
但**必须通过 `harness.py` 评测**，否则数字不可比。

## 1. 数据集事实（来自 `recon.py`，务必先读）

| 指标 | train | validation | **test** |
|---|---|---|---|
| 题量 | 4755 | 798 | 467 |
| 平均正确候选数 | 3.02 | 3.03 | **2.04** |
| 4 个全对的题 | 47% | 47.5% | **0%** |
| 选候选 0 的准确率 | 75.4% | 78.2% | **49.7%** |
| 众数投票准确率 | 85.3% | 86.2% | **69.2%** |
| 候选解析失败率 | 4.6% | 4.9% | 5.9% |

**两条铁律：**
1. **test 远难于 validation。** 一切以 `harness.py` 输出的 `test_like.accuracy`
   （类 test 子集）为准，`full.accuracy` 只是参考，会系统性偏乐观。
2. **禁止任何位置/ID 先验或查表。** 位置先验在 val 上虚高（78%），在 test 上只有 50%，
   在 audit 上必然崩溃。方法必须只用 `predict(bank, arrays, checkpoint)` 传入的内容。

## 2. 评测协议

```bash
# 从 solutions/00_shared/ 运行；--method 指向方案自己的实现文件
python harness.py --method ../03_logreg_text/method.py \
                  --checkpoint ../03_logreg_text/artifacts \
                  --split validation --output ../03_logreg_text/results/validation.json
```

输出字段：

| 字段 | 含义 | 约束 |
|---|---|---|
| `full.accuracy` | 全量 validation 选择准确率 | 参考值 |
| `test_like.accuracy` | **类 test 难子集准确率（主指标）** | 用于方案排序 |
| `prediction_seconds` | 进程内 predict 耗时 | 完整评测含启动须 ≤30s |
| `load_and_import_seconds` | 加载 + 导入耗时 | 计入 30s |
| `assets_bytes` | checkpoint 目录总字节 | **≤16 MiB (16777216)** |
| `choice_histogram` | 选择分布 | 用于检查位置崩塌 |

其他要求：
- 依赖必须固定版本；训练后预测与评分须**离线可跑**。
- 训练只用 `train` 划分；`validation` 仅用于开发与选型；**不得**把 test 参考用于训练/调参/查表。
- 只能在给定候选中选一个，不得调用外部模型或 API。

## 3. 每个方案必须交付的四件东西

放在方案自己的文件夹内：

| 文件/目录 | 内容 |
|---|---|
| `method.py` | 暴露 `predict(bank, arrays, checkpoint)`，返回长度 N 的 0..3 整数列表 |
| `train.py`（如有训练） | 复现训练的脚本，含固定随机种子 |
| `artifacts/` | 推理资产（权重/统计量/配置），**≤16 MiB**，附哈希 |
| `results/validation.json` | `harness.py` 输出 |
| `REPORT.md` | **设计—证据—结论**记录，见下 |

## 4. REPORT.md 模板（设计—证据—结论）

```markdown
# 方案 <编号> · <名称>

## 设计 (Design)
- 假设：一句话说明"我认为什么证据能让候选更可信"。
- 方法：输入 → 特征 → 打分 → 选择 的具体流程。
- 关键设计选择及理由（含被否决的替代方案）。
- 预期：事前预测会发生什么（可证伪）。

## 证据 (Evidence)
- 主表：full.accuracy / test_like.accuracy / n
- 消融：至少 2 组，说明每个组件贡献多少。
- 资源：prediction_seconds、assets_bytes（是否满足 16 MiB / 30 s）。
- 失败分析：典型错误样本 2-3 个（题干 + 所选 + 正确答案 + 出错原因）。

## 结论 (Conclusion)
- 假设是否成立？证据支持/反驳到何种程度？
- 与基线（众数投票 86.2% full / test-like __%）相比的净增益及其代价。
- 对 5 个研究问题（读推理是否有用 / 错答案为何可信 / 是否该跨候选比较 /
  能否识别不确定性 / 更多计算是否划算）的贡献。
- 下一步可迭代方向，以及**不建议**继续的方向。
```

## 5. 注意

- `00_shared/` 是只读共享区，方案不要往里写结果。
- 改动 `harness.py` 或 `recon.py` 会使所有方案数字失效；如需修改请先说明理由。
