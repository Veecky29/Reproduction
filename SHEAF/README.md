# SHEAF — Statistical Hypothesis–Edge Attribution Framework

> 2026-11-26 起的工作区。**结论是否定的**：精确可删除的加性读出层在这批基准上
> 没有带来任何预测增益。见 [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md)。
>
> Phase 0 SCP/LUR 诊断（独立于上面已收口的精度线，不重开 sheaf/mix/film）：[lur_pilot/README.md](lur_pilot/README.md)。

## 一句话

冻结一个 channel-independent 骨干，让**窗口统计量**通过一个**按（目标 × 预测块 × 源）
可加的读出层**进入预测，于是"第 j 个变量的统计，是否、在哪个预测步上，对第 i 个目标有用"
变成一个能被删除实验证伪的问题。

这个想法本身是干净的。**它失败在数据上**：能吃到跨通道信息的地方（Electricity/Weather），
自由通道混合比它好 9.4%（Electricity）；不能吃到的地方（Weather、ETT），所有方法都打平。

## 四个阶段

### 阶段一：检查 B —— 边先验 ρ 可信吗

`checkB/`。用 6 个标准基准检验"窗口内滞后相关 ρ"能否预测成对依赖（滞后互信息）。

主要修正（过程中发现旧结论有误）：

- **旧的 `pair_stats()` 算的不是相关系数**：它把跨窗口跨 lag 的全局最大 |cc| 除以
  `200×13`，值域 3e-4。改成真 Pearson 后，`ρ~MI` 从 0.291 变成 0.542（ETTh1）。
- **"逐窗口 ρ 优于全序列 ρ" 是反的**：0.77 才是全序列，0.29 是逐窗口。
- **ρ 必须在去趋势+去季节的残差上算**：原始域上 ECL 有 15% 的边 ρ≥0.90，
  是窗口内趋势造成的伪相关。resid 域上 ρ~MI = 0.69–0.84，不再饱和。

### 阶段二：三臂对照 —— 可删除读出层有用吗

`w12/`。同一编码器、同一协议下对照 `plain` / `concat` / `sheaf`，6 个数据集 × 3 种子
（Traffic 未跑通，原因见负结果 §2 的注）。

结论：`sheaf` ≈ `plain`（±2%）；`concat` 在 Electricity 上比它们好 **9.4%**（固定同一编码器后的公平对照；17.7% 是早期不公平对照的旧数，已作废）。

### 阶段三：可开关的跨通道混合（也失败）

`MIXING_SWITCH.md`。既然公平对照下混合只在 Electricity 上值 9.4%、在 ETT 上有害，
就试了 `h'_i = h_i + beta_i * Mix(h)_i`，让模型自己学开关。

**beta 在两个方向都不动**（0.1 起步停在 0.1，1.0 起步停在 0.97），
且 beta=1 也不等于稠密混合（低秩瓶颈补不回来）。这条也收口。

### 阶段四：窗口统计量本身有没有信息（试点）

`STATS_PILOT.md`。跨通道方向收口后，回头查 RevIN 抹掉的 `[mu, log sd]`。
三臂 `nostats` / `plain`(拼进头) / `film`(条件化)，同一编码器。

- 拼进头：ETTm1 −2.8%、ETTm2 −1.0%，小时级 ETT 略差
- 条件化：**Weather −11.9%**（全项目最大单点改进），但其余五个数据集全部变差

**做了判决性消融**（去掉 Weather 的 `rain` / `raining` 两列后重跑）：

| 通道集 | nostats | film | Δ |
|---|---|---|---|
| 21 通道（含 rain） | 0.1930 | 0.1701 | −11.9% |
| 19 通道（去 rain） | 0.1353 | 0.1237 | **−8.6%** |

两条方向相反的结论：

1. **"FiLM 只是在给退化通道装尺度门"被推翻**——去掉 rain 后增益只缩到 −8.6%，仍然很大。
2. **但 rain 两列本身是有害噪声**——同一个 `nostats` 模型去掉它俩改善 **30%**
   （0.1930 → 0.1353），验证集从 0.52 量级降到 0.19 量级。

FiLM 依然只在 Weather 上有效（ETT +3% ~ +12%，Electricity −1.2%），
一个六分之一命中且机制不清的效应不构成方法。**这条线到此收口。**

## 四轮尝试的总结

| # | 方向 | 结果 | 结论 |
|---|---|---|---|
| 1 | 加性可删除读出（`sheaf`） | ≈ `plain`（±2%），α 在 ECL 上被优化器压小 | 吃不到跨通道增益 |
| 2 | 可开关的跨通道混合（`mix`） | β 两端都不动（0.1→0.1，1.0→0.97） | 开关学不出来 |
| 3 | 统计量拼进预测头 | 仅 ETTm −1% ~ −3%，小时级变差 | 不足以成方法 |
| 4 | 统计量条件化（FiLM） | 仅 Weather −8.6%（去 rain），其余变差 | 数据集特异 |

**唯一确定可复用的产出**：Weather 的 `rain (mm)` / `raining (s)` 应从输入中移除
（改善 30%），而 TSLib / PAC-Former 的默认流水线都保留了它们。

## 目录

```
SHEAF/
  NEGATIVE_RESULT.md        # 尝试一（加性可删除读出）：设计、数字、三个协议坑
  MIXING_SWITCH.md          # 尝试二（可开关的跨通道混合）：同样失败
  STATS_PILOT.md            # 尝试三（窗口统计量的用法）：分数据集，Weather 异常待验证
  HANDOFF.md                # 项目交接（历史决策与修正记录）
  README.md
  checkB/                   # 阶段一：边先验 ρ 的可靠性
    checkB.py               # 主脚本（旧的 ρ 估计器，保留作对照）
    checkB_detail.py
    checkB_partial.py
    checkB_deseason.py      # 去趋势/去季节 + 同域重算 MI
    checkB_estimator.py     # 新旧估计器对照 + bootstrap CI
    checkB_scope.py         # 2×2：全局/逐窗 × 原始/残差
    results/*.csv, *.log
  w12/                      # 阶段二：三臂对照
    data.py                 # 切分、滑窗、resid 域 ρ 先验
    models.py               # plain / concat / sheaf
    train.py                # 单配置训练
    summarize.py            # 汇总
    driver*.py, run_all.sh  # 队列
    results/*.jsonl
  docs/report_review.md     # 对初版 report.docx 的评审（历史存档）
```

## 复现

数据不进 git（136MB 的 Traffic 等）。按 `data/SHA256SUMS` 校验原始 CSV，放在同一目录：

```bash
# 阶段一
cd SHEAF/checkB && python checkB_deseason.py

# 阶段二（需要 GPU）
cd SHEAF/w12 && python train.py --dataset Electricity --arm sheaf --horizon 96 \
    --lr 3e-4 --channels 9999 --stride 4 --batch 32
```

数据源（与 TSLib 官方一致）：

- ETT：`https://raw.githubusercontent.com/zhouhaoyi/ETDataset/.../ETT-small/ETT*.csv`
- Weather / Electricity / Traffic：
  `https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59.../<name>/<name>.csv`

切分必须用 TSLib 的 30 天月边界（`8640/11520/14400`），不是日历月。
详见 [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) §4.1。

## 相对既有工作的位置

同期的 DAED、PatchDecomp、TempShap、IB-Forecast 已经覆盖了"预测的可加分解"与
"忠实度保证"，本工作与它们的差别只在分解维度（源 × 目标 × 预测块）和无 softmax 竞争。
在精度上本工作没有优势，因此这条线不构成可发表的增量。
