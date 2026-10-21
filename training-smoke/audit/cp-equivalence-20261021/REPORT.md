# 2.7B CP1/CP2 数值对拍（2026-10-21）

真实权重、数据和 Megatron 分布式训练路径下，两组 mask 的对拍通过预设 BF16 容差；CP1 与 CP2 不是逐位一致。

| 检查 | 普通 mask | 不均匀 mask |
| --- | ---: | ---: |
| CP1/CP2 的输入 tokens、labels、position_ids、mask | 完全相同 | 完全相同 |
| 每个 CP2 rank 的有效 token 数/样本 | 2048 / 2048 | 618 / 1850 |
| 平均 CE loss 最大绝对差（两份 microbatch） | 0.00018692 | 0.00024891 |
| 全序列逐 token CE 最大绝对差 | 0.0358038 | 0.0358038 |
| CP 分界前后共 8 个 token 的 CE 最大绝对差 | 0.00730801 | 0.00730801 |
| 采样 logits 最大相对 L2 误差 | 0.4882% | 0.4882% |
| 全部参数梯度合并后的相对 L2 误差 | 0.4713% | 0.6075% |
| CP2/CP1 梯度范数比 | 1.000660 | 1.000747 |
| 单个参数张量最大相对梯度 L2 误差 | 3.6329% | 3.5985% |
| NaN / Inf | 无 | 无 |

没有观察到 CP 梯度 2 倍/半倍的系统性缩放问题。参数级误差与整体误差不同，最大参数误差已列出，不能把 0.61% 解释为每个张量都小于该值。

## 测试条件与覆盖范围

- 环境 `train_mobius_py312`，分支 `2026-10-10`，H800 80GB；CP1 使用 4 卡，CP2 使用 8 卡。
- 权重 `/mnt/data/Models/RWKV-7/rwkv7-qwen-vocab-2.7b-from-28b.pt`，全量 24 层、hidden 2304、248320 词表。
- 数据 `/mnt/data/Datasets/0056_qwen3.8/0056_qwen3.8_text_document`；tokenizer `/hikscale/models/Qwen/Qwen3.8-27B`。
- 固定 TP2、PP2、DP1、SP 关闭、BF16、seq4096、micro batch 1、global batch 2；每次两份相同训练样本，无 dropout，full recompute。
- 不均匀 mask 在前半段更稀疏，并屏蔽跨 CP 边界的一段 token。两个 CP shard 的有效 token 数不相等，检测归一化问题。
- 经过真实 `forward_step`、Megatron pipeline schedule、backward、`finalize_model_grads` 和分布式优化器 DDP reduce-scatter。
- 逐 token CE 使用全词表；logits 每个 TP vocab shard 固定采样 64 个列、覆盖所有 token。额外保存完整输出层输入，逐张量误差见 JSON。
- 比较 1356 个 TP/PP 参数梯度张量（含 TP 复制参数和未参与计算的零梯度）。还原每个 bucket 中已归约的优化器分片，不读取其它 stale buffer 区域。
- 梯度在 clipping/Adam 更新前采集。本次不执行参数更新，不保存训练 checkpoint。日志中的 `grad norm: 0` 是采集入口返回的占位值，真实对拍范数见 JSON。
- 接受标准：整体梯度相对 L2 < 3%、拟合缩放系数偏离 1 < 2%、采样 logits 相对 L2 < 2%、逐 token CE 最大绝对差 < 0.05；不以逐位相同为标准。

这是指定权重、两份样本、上述配置下的 BF16 容差验证；没有证明任意长度、精度或并行组合均等价，也没有测试 CP+SP 或 micro batch > 1。

## CP loss 归约修改

之前 `loss_func` 直接对带梯度的 tensor 执行原地 c10d all-reduce，PyTorch 会警告未注册 autograd kernel。
对拍未发现这条旧路径产生梯度倍数错误，但它依赖隐式 backward 行为。

现改为仅 all-reduce detached loss/token 统计，通过恒等梯度连接保留本 rank 的局部 numerator 导数，同时返回全局统计值。
若局部 numerator 为 L_c、全局有效 token 数为 N、microbatch 数为 M：scheduler 的局部 backward 缩放为 CP/(N×M)，DDP 的 CP 平均再乘 1/CP，最终得到 sum_c dL_c/(N×M)，与 CP1 相同。

不均匀 mask 的修改前/后参数误差指标完全相同；修改后两组均无 loss all-reduce autograd 警告。
新增 CPU 回归测试覆盖普通 mask、不均匀 mask、单 rank 全空、全部 token 被 mask，精确检查与全局 masked mean 的导数一致。相关测试共 34 passed。

## 复跑与文件

```bash
bash training-smoke/audit/cp-equivalence-20261021/run.sh 1 full
bash training-smoke/audit/cp-equivalence-20261021/run.sh 2 full
bash training-smoke/audit/cp-equivalence-20261021/run.sh 1 uneven
bash training-smoke/audit/cp-equivalence-20261021/run.sh 2 uneven

/mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/cp-equivalence-20261021/compare.py \
  --reference training-smoke/audit/cp-equivalence-20261021/cp1-full \
  --actual training-smoke/audit/cp-equivalence-20261021/cp2-full \
  --output training-smoke/audit/cp-equivalence-20261021/comparison-full.json
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/cp-equivalence-20261021/compare.py
```

- `summary.json`：汇总结果和源码/配置 SHA256。
- `comparison-full.json` / `comparison-uneven.json`：全部前向与参数级梯度误差。
- `before-loss-fix-comparison.json` / `before-loss-fix-cp2.log`：修改前证据。
- `cp1-*` / `cp2-*`：日志、trace 和原始梯度。大型 `.pt` 仅保存在本地，通过本目录 `.gitignore` 排除。

本次所有测试进程均正常退出；28B 首次 Adam 更新的 OOM 仍待单独解决。
