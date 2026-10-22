# 后续扩展：RWKV7 CP + SP（2026-10-21）

阶段编号更正：原计划第二阶段是“给 token shift 增加 CP 通信”，已包含在主仓库提交 0ca9d3a 中。本报告记录额外提前推进的 CP+SP 扩展；目录中的 stage2 为此前误编号，保留路径以便追溯。

CP 与 SP 已可同时开启。真实 2.7B 模型在 8 张 H800 上完成 TP2×PP2×CP2 的前向/反向对拍、3 步 Adam 更新和固定验证。代码位于主仓库 `2026-10-10` 分支的当前工作区；本阶段尚未提交或 push。

CP 基础实现已按要求先提交：主仓库 **0ca9d3a**，FLA 子模块 **261dbcf5**，两者均在 `2026-10-10` 分支。该扩展没有继续修改 FLA 或 Megatron 子模块。

## 实现

- `rwkv_megatron/token_shift.py`：第一次调用时在 Megatron 的 TP×CP 联合组收集 `(cp_rank, tp_rank)`，建立逻辑序列前驱/后继；后续复用缓存。组内只含同一 PP/DP 副本，不引入新 process group。
- `RWKV7TokenShiftSequenceParallel` 使用该逻辑邻居发送尾 token，反向沿相反边界发送首 token 梯度；CP1 时保持原 TP 路径。显式保留 send buffer 至异步通信完成。
- `pretrain_rwkv7.py` 允许 CP+SP，要求全局 `seq_length` 可被 `CP*TP` 整除；batch 继续使用连续 CP 切分。
- WKV 仍在列并行投影和 bottleneck all-to-all 恢复完整 CP-local 序列后运行，沿原 CP group 在相同 TP head 之间交换 recurrent state。多 microbatch 样本继续独立调用 FLA。
- SP 的 v0 继续以完整 CP-local 序列、本地 TP head 的不透明打包槽穿过 PP；不对这个槽做 token shift。PP shape 规则已支持依次除以 CP、TP，扩展回归覆盖 CP1/2/4，无需修改 Megatron。
- 对拍工具增加 SP head 输入的 TP gather，以及多 CP shard reference 合并。梯度仍在生产 DDP reduce-scatter 后、裁剪和 optimizer 更新前重建。

以 seq4096 / CP2 / TP2 为例，隐藏状态按以下逻辑序列排列：

| CP rank | TP rank | token 区间 |
| --- | --- | --- |
| 0 | 0 | [0,1024) |
| 0 | 1 | [1024,2048) |
| 1 | 0 | [2048,3072) |
| 1 | 1 | [3072,4096) |

因此 token shift 的跨 CP 边界是 `(CP0,TP1) → (CP1,TP0)`；WKV state 的跨 CP 边界则连接同一 TP head。这两种通信各自使用对应的组与顺序。坐标映射允许物理 rank 顺序变化，已用实际多卡测试验证。

## 回归

1. `tests/test_rwkv7_tp_cp_token_shift.py`：**5 passed**。TP2/CP1、TP1/CP2、TP2/CP2、CP-before-TP 的 rank 顺序、TP4/CP2。每组均检查 CP-only 和 SP 路径、局部长度 1/3、batch3、dim0/1 的前向值和输入梯度，与完整序列参考逐位相同。
2. CP loss、静态合同、v0 TP/PP/SP、非 SP v0 和参数标签：**46 passed**。pipeline carrier shape 扩展至 TP1/2/4 × CP1/2/4 × batch1/3。
3. 新增纯 Python 模块和分布式测试 Ruff 检查通过；Python 编译、shell 语法、Git diff whitespace 检查通过。

命令及日志：

```bash
OMP_NUM_THREADS=4 PYTHONPATH="$PWD" python -m pytest -q tests/test_rwkv7_tp_cp_token_shift.py
OMP_NUM_THREADS=8 python -m pytest -q \
 tests/test_rwkv7_cp_loss.py tests/test_rwkv7_static_contract_audit.py \
 tests/test_rwkv7_v0_tp_pp_sp.py tests/test_rwkv7_v0_tp_pp_no_sp.py \
 tests/test_rwkv_sequence_parallel_param_tags.py
```

见 `token-shift.log`、`contracts.log` 和 `lint.log`。

## 真实模型数值对拍

conda `train_mobius_py312`，torch 2.9.1+cu128 / Triton 3.5.1。模型、数据和 tokenizer 沿用此前 CP 基础验证：

- `/mnt/data/Models/RWKV-7/rwkv7-qwen-vocab-2.7b-from-28b.pt`
- `/mnt/data/Datasets/0056_qwen3.8/0056_qwen3.8_text_document`
- `/hikscale/models/Qwen/Qwen3.8-27B`

两侧均为 TP2×PP2×CP2、seq4096、BF16、full recompute、distributed Adam 设置；只切换 SP。每组两个 microbatch，MB1 对应 global batch2，MB2 对应 global batch4。MB1 reference 复用此前修复后的 CP2/SP-off 捕获；MB2 两侧重新捕获。

| 对拍条件 | 逐 token CE / 采样 logits / 完整 head 输入最大绝对差 | 整体梯度相对 L2 | 最差单参数相对 L2 |
| --- | ---: | ---: | ---: |
| MB1，完整 mask | **0 / 0 / 0** | **0.00586227%** | 0.209505% |
| MB1，不均匀 mask | **0 / 0 / 0** | **0.00596769%** | 0.269736% |
| MB2，完整 mask | **0 / 0 / 0** | **0.00585414%** | 0.190227% |

全部数据、标签、position 和 mask 检查相同；1356 个 TP/PP 参数梯度张量全部有限，SP-on / SP-off 整体梯度范数比约 1.0000001。指标包含 TP 复制参数，沿用之前定义，不能当成去重后的模型梯度范数。

采样 logits 为每个 TP 词表分片的 64 列，CE 覆盖全部词表。逐位相同仅针对实际捕获内容和这些条件；未声称所有词表 logits、所有 CP/TP size 均已证明逐位相同。小幅梯度残差符合 SP 改变参数梯度局部累加、BF16 舍入及 TP 归约顺序的影响，未逐元素归因所有残差。

结果在 `summary.json`、`comparison-mb1-full.json`、`comparison-mb1-uneven.json`、`comparison-mb2-full.json`，各自有 comparison 日志和原始运行日志。MB2 还验证了 CP state 不跨样本共享，以及 opaque v0 在 PP2 中保持样本/通道顺序。

复现示例：

```bash
bash training-smoke/audit/cp-sp-stage2-20261021/run.sh 1 1 full
bash training-smoke/audit/cp-sp-stage2-20261021/run.sh 1 1 uneven
bash training-smoke/audit/cp-sp-stage2-20261021/run.sh 2 0 full
bash training-smoke/audit/cp-sp-stage2-20261021/run.sh 2 1 full
python training-smoke/audit/cp-equivalence-20261021/compare.py \
 --reference training-smoke/audit/cp-state-fix-20261021/cp2-full --reference-cp-size 2 \
 --actual training-smoke/audit/cp-sp-stage2-20261021/cp2-mb1-sp1-full \
 --output training-smoke/audit/cp-sp-stage2-20261021/comparison-mb1-full.json
```

## 正常训练

新增可直接启动的配置和脚本：

- `configs/config-rwkv7-2p7b-qwen-cp2-sp-smoke.yaml`
- `training-smoke/launch-rwkv7-2p7b-qwen-cp2-sp-smoke.sh`

```bash
bash training-smoke/launch-rwkv7-2p7b-qwen-cp2-sp-smoke.sh qwen-rwkv2p7b-cp2-sp-stage2-20261021
```

| 步数 | LM loss | grad norm | 跳步 / NaN |
| --- | ---: | ---: | --- |
| 1 | 13.49903 | 74.311 | 0 / 0 |
| 2 | 13.37066 | 83.860 | 0 / 0 |
| 3 | 13.25385 | 97.700 | 0 / 0 |

固定验证 loss **13.15863**。日志为 `optimizer-smoke.log`，另存 `/mnt/data/Logs/outputs/qwen-rwkv2p7b-cp2-sp-stage2-20261021.log`。对拍比较的是相同权重更新前的结果；不同梯度舍入经过 Adam 更新后，不承诺后续步或参数逐位相同。未做受控性能测量，不据三步短跑宣称吞吐提升。

## 当前范围

已完成 CP+SP、TP2×PP2×CP2 的真实 2.7B BF16/Adam 验证；TP4×CP2 及不同物理 rank 排列完成的是 token shift 算子回归。28B 优化器显存、CP4 整模型和更长序列、CP+SP 保存恢复、interleaved PP、FP8/Muon 组合仍属后续工作。第一阶段已有的 28B Adam OOM 不因本阶段验证而自动视为解决。
