# Phase 0：SCP、MSE_lb、LUR

这条线和已经收口的 SHEAF 精度实验（`sheaf` / `mix` / `film`，见上一级 `NEGATIVE_RESULT.md`）无关。这里不新增残差头、不改那些训练代码，也不把 ETT 当作默认结论。

Phase 0 只做测量：在 Electricity、Traffic、Weather 上，用**一个线性模型（DLinear）**和**一个已有深度模型（iTransformer）**产生预测，再调用 ICML 2026 官方实现计算

- **SCP**：谱相干可预测性，官方 CSV 里的 `P_lin_mean_overall`
- **MSE_lb**：线性 MSE 下界，官方 CSV 里的 `mse_lb_mean_overall`
- **LUR**：线性利用率。分频段的数是官方 `evaluate_band_LUR` 写出的 `XR_mean`；总体 LUR 是用官方能量权重 `weight_mean` 对这些 `XR_mean` 做加权后再对通道取平均

Welch 谱和相干本身不在本目录里重写。`setup.sh` 会把 [WanjinVon/TS_Predictability](https://github.com/WanjinVon/TS_Predictability) 固定在提交 `3bd624602c95506b6f9beedfc43c041fdb6afb36`。函数位置见 [UPSTREAM.md](UPSTREAM.md)。

默认预测长度 **H = 96**。官方诊断脚本把回看长度也设成同一个数（`seq_len = pred_len = H`）。本目录沿用这个约定。

## 预先登记的停止规则

下面这些数写在 `config.yaml`，也写在 `phase0.py` 的 `PREREGISTERED` 里。看完输出再改它们，这次运行就**不能**当成停止规则的答案（汇总里的结论会是 `not_preregistered`）。

在 **iTransformer** 上，一个频段是「实质低 LUR」当且仅当同时满足：

- 该频段的官方 `XR_mean` 在预登记通道上的算术平均 **< 0.5**
- 该频段的官方能量权重 `weight_mean` 在同样通道上的算术平均 **≥ 0.05**

预登记通道是 `even:16`：含第 0 路和最后一路在内、均匀取 16 个通道。频段是官方的 `auto:10`。每个通道最多用测试集前 `1000` 个窗口（官方参数 `max_samples`）。

**若 Electricity 和 Traffic 上都没有这样的频段，就停，不要再发明模块。** 只有 Weather 上出现低 LUR 频段，不够。Weather 上的增益不能单独支撑下一步。

本目录即使规则没有触发（结论 `not_stopped`），也**不会**训练残差头。Phase 1 不在这里。

## 安装

需要 Python ≥ 3.10（官方代码使用了 `int | None` 这种写法）。一块 GPU 即可（RTX 4070 Ti 或 4090 这一档）。官方脚本在进程里把 `--gpu` 写成 `CUDA_VISIBLE_DEVICES`；本目录的启动器会先按 `config.yaml` 的 `gpu`（或 `--gpu`）选物理卡，再向官方脚本传 `--gpu 0`。

```bash
cd SHEAF/lur_pilot
bash setup.sh
python -m pip install -r requirements.txt
```

`requirements.txt` 把 NumPy 限制在 2.0 之前，因为官方 `EarlyStopping` 还在用已经删除的 `np.Inf`。`torch` 在没有 CUDA 轮子时可能装成 CPU 版。训练机上请安装与本机 CUDA 匹配的 PyTorch。

克隆落在 `third_party/TS_Predictability/`，不进 git。重复执行 `setup.sh` 会回到同一个提交。

不训练的自检（给本机或 CI，不会开训练）：

```bash
cd SHEAF/lur_pilot
python -m unittest discover -s tests -t .
```

不要在 CI 里跑 `run_phase0.sh`。那条命令会训练。

## 数据目录

CSV 不在本仓库里。指向已经放好 TSLib 风格文件的目录，二选一：

```bash
export LUR_DATA_ROOT=/path/to/dataset
```

或把同一路径写进 `config.yaml` 的 `data_root`。命令行 `--data-root` 优先于环境变量，环境变量优先于 yaml。

会按这个顺序找文件（找到就停）：

| 数据集 | 接受的相对路径 |
|---|---|
| Electricity | `electricity/electricity.csv`，`electricity.csv`，`Electricity.csv` |
| Traffic | `traffic/traffic.csv`，`traffic.csv`，`Traffic.csv` |
| Weather | `weather/weather.csv`，`weather.csv`，`Weather.csv` |

Electricity 应为 321 个变量，Traffic 应为 862 个。对不上就退出。官方 `Dataset_Custom` 还要求有 `date` 列和目标列 `OT`（`features=M` 时 `OT` 只是被挪到最后一列，所有列都参与预测）。TSLib 文件里 `OT` 本来就在最后，通道编号与「去掉 date 之后的列顺序」一致。

**Traffic 找不到不会被悄悄跳过。** `prepare_data.py` 会列出找过的路径并退出。只想先跑另外两个数据集时，必须写明：

```bash
python prepare_data.py --datasets electricity,weather
```

这样做完之后，停止规则的结论是 `incomplete` 或 `not_preregistered`，因为 Traffic 没有测。

ETT 默认关闭。要加进去：

```bash
python prepare_data.py --include-ett
python train_baselines.py --include-ett
python diagnose.py --include-ett
```

ETT 的超参来自官方 `ETTh*.sh` / `ETTm*.sh` 在 H=96 的那一档，不作为本阶段的主要结果。

## Weather：去掉 rain / raining

这是**数据质量控制，不是方法**。`prepare_data.py` 在训练之前，从 Weather 里删掉列名（去掉括号中的单位、忽略大小写之后）恰好是 `rain` 或 `raining` 的列。TSLib 里通常叫 `rain (mm)` 和 `raining (s)`。21 列因此变成 19 列，`enc_in` 用删完之后的列数。

原始 CSV 不会被改。删过的文件写到 `work/data/weather.csv`。清单在 `work/data/manifest.json` 的 `columns_dropped`。若 21 列里认不出这两列，准备步骤直接失败，避免在未知列上训练。若文件里本来就没有这两列、并且只剩 19 个变量，会警告后继续，并在清单里写明。

不要把这次删除说成模型或损失的一部分。

## 训练和诊断

三条默认数据集、两个模型、H=96：

```bash
cd SHEAF/lur_pilot
export LUR_DATA_ROOT=/path/to/dataset

python prepare_data.py
python train_baselines.py
python diagnose.py
```

等价的一条命令（同样会训练，不要放进 CI）：

```bash
bash run_phase0.sh
```

只跑一个组合：

```bash
python train_baselines.py --dataset electricity --model DLinear
python train_baselines.py --dataset electricity --model iTransformer
python train_baselines.py --dataset traffic --model DLinear
python train_baselines.py --dataset traffic --model iTransformer
python train_baselines.py --dataset weather --model DLinear
python train_baselines.py --dataset weather --model iTransformer
python diagnose.py
```

`diagnose.py` 会读取对应 checkpoint，再调用官方 `evaluate_predictability()` 和 `evaluate_band_LUR()`。DLinear 走 `DLinear/run_longExp.py` 与 `DLinear/exp/exp_main.py`。iTransformer 走 `iTransformer/run.py` 与 `iTransformer/experiments/exp_long_term_forecasting.py`。损失是 MSE（`--use_ps_loss 0`），不跑官方脚本里的 PS-loss 扫描。

超参来源：

| 数据集 | DLinear | iTransformer |
|---|---|---|
| Electricity | 官方 `DLinear/scripts/ECL.sh`：lr `0.001`，batch 16，`enc_in` 321，`model_id` `Electricity_96_96` | 官方 `iTransformer/scripts/ECL.sh`：3 层，`d_model` 512，`d_ff` 512，lr `0.0005`，batch 16，`model_id` `ECL_96_96` |
| Weather | 官方 `DLinear/scripts/weather.sh`：lr `0.0001`，batch 16；`enc_in` 改为去掉 rain 之后的 19 | 官方 `iTransformer/scripts/Weather.sh`：3 层，`d_model` 512，lr `0.0001`，batch 用 `run.py` 的默认 32；`enc_in` 为 19 |
| Traffic | 官方仓库没有 Traffic 脚本。沿用上面 ECL 的 lr `0.001`、batch 16，`seq_len = pred_len = H`，`enc_in` 862 | 官方诊断仓库也没有。使用本仓库已有的 iTransformer Traffic 脚本：4 层，`d_model` 512，`d_ff` 512，batch 16，lr `0.001`，`enc_in` 862 |

DLinear 论文里 Traffic 的回看是 336、学习率 0.05。这里不沿用那一组，因为 Phase 0 对齐的是诊断仓库的 `seq_len = pred_len` 协议。

显存不够时只改 batch（它不在 checkpoint 目录名里）：

```bash
python train_baselines.py --dataset traffic --model iTransformer --batch-size 8
```

诊断必须使用同一套结构超参。batch 可以和训练时不同。

官方脚本内部的随机种子是写死的：iTransformer `run.py` 为 2023，DLinear `run_longExp.py` 为 2021。本目录不另设种子。

两处官方训练收尾不会挡住诊断，本目录认出来之后仍使用已经写好的 `checkpoint.pth`：

- DLinear 的 `exp/exp_main.py` 在早停之后用 `map_location='cuda:0'` 把权重再读回来。没有 CUDA 时这一步报错，权重本身已经写好。有 GPU 时这一步正常结束。
- iTransformer 的 `run.py` 在训练结束后调用 `test()`。那个函数用 `np.array` 把 batch 叠起来，最后一个 batch 更短时会报 `inhomogeneous shape`。SCP / LUR 不走这个 `test()`，而是直接加载 checkpoint。

先看将要执行的命令、不训练：

```bash
python prepare_data.py --dry-run
python train_baselines.py --dry-run
python diagnose.py --dry-run
```

### 改预测长度

H 同时改 `pred_len` 和 `seq_len`。三个步骤要用同一个 H，并且先重新准备清单（清单里记着 H）：

```bash
python prepare_data.py --horizon 192
python train_baselines.py --horizon 192
python diagnose.py --horizon 192
```

也可以改 `config.yaml` 的 `horizon`，再跑上面三条且不要另传 `--horizon`。H 不是 96 时，停止规则视为未按预登记协议执行。

## 写出的文件

训练日志和权重在 `work/`（不进 git）：

```
work/data/manifest.json
work/data/electricity.csv      # 指向原始文件的链接
work/data/traffic.csv
work/data/weather.csv          # 去掉 rain / raining 后的副本
work/checkpoints/<model>/<setting>/checkpoint.pth
work/logs/train_<dataset>_<model>_H96.log
work/official/<dataset>/<model>/H96/scp/     # evaluate_predictability 的原始 CSV
work/official/<dataset>/<model>/H96/lur/     # evaluate_band_LUR 的原始 CSV 和频段图
```

`diagnose.py` 再把官方列收成四份表，放在 `results/`（不进 git）：

| 文件 | 内容 |
|---|---|
| `results/phase0_metrics.csv` | 每个数据集 × 模型一行：`SCP`、`MSE_lb`、`NMSE_lb`、`MSE_model`、`LUR_overall` |
| `results/phase0_bands.csv` | 每个频段的 LUR。`channel=mean` 是通道平均；其它行是官方单通道 `XR_mean` |
| `results/phase0_summary.json` | 协议、停止规则、结论 |
| `results/phase0_summary.md` | 短摘要 |

`phase0_summary.json` 的 `verdict`：

| 值 | 含义 |
|---|---|
| `stop` | Electricity 与 Traffic 都没有实质低 LUR 频段。停止，不要加模块。 |
| `not_stopped` | 这两个数据集里至少一个有实质低 LUR 频段。本包仍然不训练新模块。 |
| `incomplete` | Electricity 或 Traffic 的 iTransformer 频段结果缺失。 |
| `not_preregistered` | 这次运行改了 H、数据集、通道、频段、样本上限、Welch 参数、rain 控制或阈值。 |

## 不在本目录里的东西

- 不重写 `w12/`、`checkB/`、`NEGATIVE_RESULT.md`、`MIXING_SWITCH.md`、`STATS_PILOT.md`
- 不加 Phase 1 残差头、APTF 损失、新骨干
- 不加 PAC-Former，也不加 LLM / 时序基础模型
- 不把官方仓库整棵复制进 `SHEAF/`
