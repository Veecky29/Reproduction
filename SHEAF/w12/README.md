# W1.2 — 平凡基线 vs SHEAF 读出层

回答一个问题：**把窗口统计量直接拼进通道混合模型，能不能打平 SHEAF 的加性读出层？**
打平 → 读出层不必要，"跨通道交互"这个故事要改。

## 三个臂

| arm | 跨通道信息怎么进 | 作用 |
|---|---|---|
| `plain` | 不进（纯 CI 编码器 + 线性头） | 下界参照 |
| `concat` | `[z_c ; μ_c ; log σ_c]` 拼成 token，通道自由混合 | **平凡基线** |
| `sheaf` | CI 编码器 + 逐源加性读出 + ρ 先验 | 被检验的方案 |

`concat` 和 `sheaf` 用的是同一个 96→H 窗口、同一份数据、同一个训练预算（epoch / batch / lr），
所以"MSE 差多少"可以直接读。

## 跑之前的准备（AutoDL）

```bash
# 1. 数据（140MB，放数据盘）
mkdir -p /root/autodl-tmp/benchmark
# 从本地上传六个 csv 到这个目录

# 2. 环境（AutoDL 镜像一般已带 torch；确认一下）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 3. 代码
cd /root/autodl-tmp && git clone <repo>   # 或 scp -r sheaf_w12
```

## 先 smoke，再开跑

```bash
cd sheaf_w12
for arm in plain concat sheaf; do
  python train.py --smoke --dataset ETTh1 --arm $arm --horizon 96
done
```

smoke 只跑 2 个 batch，检查：窗口数量对不对、`params` 是否合理、有没有 shape 错、显存占多少。
**三个臂都过了再花 GPU 时间。**

## 正式矩阵

```bash
bash run_all.sh            # 全量：6 数据集 x 3 臂 x 3 预测步 x 2 种子
bash run_all.sh quick      # 只 ETTh1，H=96，2 种子
```

单个配置：

```bash
python train.py --dataset ETTh1 --arm sheaf --horizon 96 --seed 2026
```

输出追加到 `results/w12.jsonl`，每行一个配置，最后用 `summarize.py` 汇总（待写）。

## 判据（先写死）

按 `(dataset, horizon, seed)` 配对比较 `test_mse`：

- `sheaf` 相对 `concat` 的改进 **< 1%** 且两种子方向不一致 → **平凡基线打平，读出层不必要**
- `sheaf` 稳定好于 `plain` 但打平 `concat` → 跨通道信息有用，但**加性/可删除结构没带来额外价值**
- `sheaf` 稳定好于 `concat` ≥ 2% → 读出层有价值，进 W2

另外记录每个臂的 `params`：三者参数量要接近，否则比的不是结构而是容量。

## 已知限制

- 这里的三个臂都是**端到端从零训练**。最终 SHEAF 架构是"冻结骨干 + 只训读出层"，
  那是 W2 的事；W1.2 只是筛"统计量进来的方式重不重要"，不是最终架构的结论。
- `sheaf` 的 `delta`（ρ 先验权重）和 `eta`（水平差项）从 1.0 / 0.1 初始化后自由学习，
  没有网格搜索；如果 ρ 真的有用，这里应该能看出来。看不出来不代表 ρ 没用，
  只代表"在这个初始化 + 这个预算下没用"。
- ECL 默认抽 40 个通道（seed 2026），与 `SHEAF/scripts/checkB*.py` 的子集一致，
  保证先验和数据同源。
