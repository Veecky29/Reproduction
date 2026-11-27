# SHEAF 交接单（2026-11-26 21:45，W1.1 完成后）

> 重启 DSH 后从这一页开始读。细节在 `README.md`，数字在 `results/`。
> 上一条会话（session-23943c06）的原话已不可靠，见下面"已更正的错误"。

## 0. 项目定位

**SHEAF** = Statistical Hypothesis–Edge Attribution Framework。

冻结一个 channel-independent 骨干（PatchTST 主 / DLinear 副），让**窗口统计量**通过
**按（目标 × 预测块 × 源）可加的读出层**进入预测，于是"第 j 个变量的统计，是否、在哪个
预测步上，对第 i 个目标有用"成为可被删除实验证伪的问题。不追求 SOTA，追求"解释可验证"。
PAC-Former 已归档，只作黑箱上界参照。

## 1. 已更正的错误（重要，别再沿用）

| 旧说法 | 实际 |
|---|---|
| ρ~真值 = 0.77 / 0.52 / 0.76 / 0.46 / 0.70 / 0.73 | 那是 `checkB_partial.csv` 的**全序列** ρ；逐窗口是 0.291 / 0.883 / 0.274 / 0.628 / 0.797 |
| "局部 ρ 远好于全局 ρ（0.73 vs 0.27）" | 方向反了：0.77 才是全序列，0.29 是逐窗口 |
| 旧 `pair_stats()` 里的 ρ 是相关系数 | 不是：全局最大 \|cc\| ÷ (200×13)，量级 3e-4，`rho_uniq` 只有 0.05 |
| ETT 用"日历 12/4/4 个月"切分 | **错，会把结论抬高 20%**。官方 TSLib 用写死的 30 天月：`8640 / 11520 / 14400`。ETTh1 有 17420 行，最后 3020 行（2018-03..06）**不参与测试**。用日历月切到 17420，OLS 天花板从 0.3865 变成 0.4601，看起来"模型比基线差 20%"，其实是切分错 |

旧 ρ 估计器已修（真 Pearson：逐 lag 各自去均值、各自归一化；逐窗口取 max over lags，
再跨窗口平均）。`checkB_estimator.py` 里新旧并排 + 配对 bootstrap。

## 2. W1.1 的结论（已完成）

`scripts/checkB_deseason.py`，六个数据集，四个去混杂变体，ρ 与 MI 同域重算。

1. **混杂是窗口内的趋势，不是日周期。** 只去趋势 → ρ~MI 全面升到 0.73–0.96；
   只去季节 → 崩到 0.28–0.60（ETTh1/ETTm1/ECL）。机制是 96 步窗口里的伪相关。
2. **原始域确实饱和**：ECL 15%、Weather 11% 的边 ρ≥0.90；resid 域降到 0–7%。
3. **ρ 保留**：resid 域上 ρ~MI = 0.688–0.843，六个数据集全过 0.6 门槛。
   → **先验 ρ 必须在去趋势 + 去季节的残差上算**（已经写进 README 的设计决策）。

## 3. W1.1b 的结论（已完成，但留了一个决策）

`scripts/checkB_scope.py`，2×2（全局/逐窗 × 原始/残差）。在残差域上**全序列 ρ 在 5/6 个
数据集上优于逐窗口 ρ**（ETTm1、ECL 显著；只有 Weather 反向）。

→ 旧决策"ρ 必须逐窗口"**没有数据支撑**。注意 MI 是全局量，天生偏向全局 ρ，所以这不是
"逐窗口错了"，而是"这个指标测不了逐窗口要问的问题"。

**需要你拍板：先验是全局边分数，还是逐窗口分数？**
我的建议：先验用全局（稳定、CI 窄），窗口特异性交给学习到的门——这样归因粒度仍在门和
读出层，不在先验。

## 4. 下一步（W1 剩余）

1. **W1.2 平凡基线**：`[x̃_c ; μ_c ; logσ_c]` 直接拼进 iTransformer，看能否打平读出层。
   打平 → 读出层不必要，故事要改。（这一条最便宜且最能杀方案）
2. W1.3 逐通道读出对照（统计量不交互）。
3. W1.4 判定 ρ 先验能否在基线之外给出可验证增量。
4. 待重做："ρ 能延续到未来"（`ρ[t]~MI[t+96]`）——当时那两个数（0.51/0.68）**没有脚本产出**，
   口径不明，必须在残差域上重做。

判据先写后跑。W1 通过才进 W2（既有 checkpoint 审计）和模型实现。

## 5. 目录

```
SHEAF/
  scripts/checkB.py            # G / rho / MI_lag（旧估计器，保留作对照）
  scripts/checkB_detail.py     # top 边、三分位 MI、ASCII 散点
  scripts/checkB_partial.py    # 全序列 rho + 偏相关
  scripts/checkB_deseason.py   # W1.1 主脚本
  scripts/checkB_estimator.py  # 估计器对比 + bootstrap CI
  scripts/checkB_scope.py      # W1.1b 2×2 控制
  results/checkB_deseason.csv  # W1.1 主结果（另存 .log）
  results/checkB_estimator.log
  results/checkB_scope.log
  results/checkB_results.csv   # 旧结果：G 列可用，ρ 列以 W1.1 为准
  results/checkB_partial.csv
  README.md                    # 完整表格与解释
```

数据只读自：`C:\Users\Tim W\Desktop\PAC-Former\final\data\benchmark\*.csv`
（拷到 `SHEAF\data\` 后脚本自动优先用本地副本）

## 6. 参考复现仓库（用户的）

**https://github.com/Veecky29/Reproduction**

官方实现的可复现基线，3 个种子（2026/2027/2028），包含 DLinear / PatchTST / iTransformer：

```
Reproduction/
  DLinear/{official,data,configs,results,tools}
  PatchTST/{official,data,configs,results,tools}
  iTransformer/{official,data,configs,results,tools}
```

用途与结论：

1. **口径基准**：`iTransformer/results/summary.csv` 里 ETTh1/96 = 0.3875（论文 0.386），
   证明参考数字可复现——所以任何"复现不出来"都先怀疑自己的切分/标准化。
2. **切分来源**：`official/data_provider/data_loader.py` 的 `Dataset_ETT_hour`，
   边界是 `12*30*24 / 16*30*24 / 20*30*24`（见上面第 1 节的修正条目）。
3. 后面可能需要把 SHEAF 的代码和数据 push 上去（待用户给认证方式）。

## 7. 环境

```powershell
$py = "C:\Users\Tim W\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
cd C:\Users\Tim W\Desktop\SHEAF\scripts
& $py checkB_deseason.py     # 约 4 分钟，纯 CPU
```

环境里没有 scipy / statsmodels，分解是手写的经典加性分解（见 `classical_decompose`）。
