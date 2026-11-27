# W1.2 运行计划与状态（VM 上的自跑队列）

> 这个文件在 `/root/autodl-tmp/sheaf_w12/PLAN.md`，同时本地也留一份。
> 人不在、网断了、会话丢了，VM 都按这里写的往下跑。

## 现在在跑什么

`driver.py`（nohup 启动，pid 见 `logs/driver.log`）：

- 扫描 `results/*.jsonl` 里已完成的 `(dataset, horizon, arm, seed)`
- 只跑缺的，结果追加到 `results/matrix.jsonl`
- 每跑完一个就重写 `STATUS.md`（进度、当前任务、ETA、最近结果表）
- **幂等**：杀掉再启不会重复跑已完成的

矩阵：

| 数据集 | H | 臂 | 种子 | 次数 |
|---|---|---|---|---|
| Weather | 96 / 192 / 336 | plain, concat, sheaf | 2026–2028 | 27 |
| ETTh1 / ETTh2 / ETTm1 / ETTm2 | 96 | plain, concat, sheaf | 2026–2028 | 36 |

预计 **~2 小时、约 ¥4**（4090D @ ¥1.98/h）。

## 怎么停

```bash
kill $(pgrep -f driver.py)     # 当前这一轮跑完后退出
```

## 怎么读结果

```bash
tail -f /root/autodl-tmp/sheaf_w12/logs/driver.log
cat /root/autodl-tmp/sheaf_w12/STATUS.md
python summarize.py results/matrix.jsonl
```

## 已确定的结论（不要再推翻）

1. **ETT 必须用官方 TSLib 边界**：`8640 / 11520 / 14400`（12/4/4 个 30 天月）。
   用日历月切到系列末尾会让 OLS 天花板从 0.3865 变成 0.4601，凭空多出 20% 的"性能差距"。
2. **ETTh1 是饱和数据集**：三个臂全在 0.40–0.41，差异在种子噪声内；
   PAC-Former 消融里 `sca_only − backbone` 在 ETTh1 也是 0.0000。**不要用它做判定。**
3. **Weather 是判定战场**：`concat`（平凡基线）比 `plain` 好 18.4%，两个种子一致，
   远大于 PAC-Former SCA 的 4.7%。
4. **sheaf 已修两处结构缺陷**：聚合改均值（不随 C 放大）、消息分支零初始化
   （起点严格等于 plain，验证过 `msg_scale=0` 时扰动源通道对目标通道影响为 0）。
   修复前的 sheaf 比 plain 还差，那个结果**作废**。

## 判据（先写死，跑完照着判）

对每个 `(dataset, horizon)`，按种子配对比较 `test_mse`：

- `sheaf` 比 `concat` 好 **≥ 2%** 且三个种子方向一致 → WIN，读出层有价值，进 W2
- `sheaf` 与 `concat` 差在 **±2%** 内或方向不一致 → **TIE**，读出层在准确率上不必要，
  故事改成"同样准，但可精确删除、可归因"
- `sheaf` 比 `concat` 差 **≥ 2%** 且方向一致 → LOSS，回去改设计

注意：`concat` 是**故意做强的**对手（参数量对齐到 789k，统计量直接当 token 内容，
通道自由注意力），它赢是设计内的可能结果，不是意外。

## 已知待办

- `summarize.py` 的 `±` 在 Windows 管道里会显示成 `卤`（仅显示问题，不影响计算）
- ETT 的 H=192/336 没跑（饱和数据集，性价比低）
- ECL 没跑（40 通道，每次慢 3 倍）
