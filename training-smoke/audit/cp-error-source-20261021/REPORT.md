# CP1 / CP2 误差来源诊断（2026-10-21）

后续状态：该精度修复已合入本地正式 FLA kernel，并完成新一轮生产对拍及训练验证，见 `../cp-state-fix-20261021/RESULTS.md`。下文保留诊断时的实验记录和当时状态。

## 结论

主要来源已用真实输入和端到端干预确认：FLA DPLR 的 **CP 边界 state 汇总核提前把中间量 v2 从 FP32 舍入为 BF16**，而普通 WKV state 核保留 v2 的 FP32 精度。边界 state 因而不同，误差从后半段开始，经过各层传播。

诊断进程中只对齐这处精度后，真实 2.7B 模型两个 microbatch 的逐 token CE、采样 logits 和完整 head 输入与 CP1 全部逐位一致；1356 个 TP/PP 参数梯度张量的整体相对 L2 误差从 **0.47127% 降到 0.023258%**，差值范数下降约 **95.06%**。剩余误差主要来自分片后局部参数梯度的舍入及累加顺序。

本次没有修改正式 FLA 源码；干预仅存在于诊断进程，生成的对照 kernel 是 `kernel_fp32_v2.py`。正式修补与其他 CP 组合回归已记入仓库 `todolist.txt`。

## 条件与范围

- 分支 `2026-10-10`，conda `train_mobius_py312`，8 张 H800。
- 同一真实权重 `/mnt/data/Models/RWKV-7/rwkv7-qwen-vocab-2.7b-from-28b.pt`，同一 0056 数据前两个样本。
- TP2 × PP2 × DP1，CP1/CP2，SP 关闭，BF16，seq4096，MB1/GB2，完整重计算，无 dropout。
- 复用 `../cp-equivalence-20261021/config-cp{1,2}.yaml`，不更新参数。
- 逐层采样：第一个 microbatch，各 TP 的前 128 通道（两个完整 WKV head），覆盖全部 24 层。第一层投影和 learn bottleneck 额外捕获完整输入、输出和权重。
- 单算子：固定 CP1 的输入、固定同一输出梯度，重放第 0、6、23 层的两个 head。层号从 0 开始。
- 端到端干预：与先前 CP1-full 比较，两个 microbatch，完整 CE、完整 head 输入、每个 TP 分片 64 个词表 logits 采样、所有 1356 个 TP/PP 参数梯度张量；梯度在生产 DDP reduce-scatter 后、裁剪和优化器更新前重建。

## 第一个误差在哪里出现

`layers.json` / `layers.log`：

1. 第一层投影和 learn bottleneck 的完整输入、输出均逐位一致；第一层采样的六组 WKV 输入 r/w/k/v/a/b 也逐位一致。
2. 第一层 WKV 的前 2048 token 逐位一致；后 2048 token 开始不同，后半段相对 L2 为 **0.251284%**，最大绝对差 **0.00390625**。
3. 各 TP 采样的所有 24 层中，前半段的 norm、WKV 输入/输出、attention 输出、layer 输出始终逐位一致。后半段误差逐层传播，TP0 第 23 层 layer 输出相对 L2 达 **0.442501%**。

因此，本次首个误差源可定位到 WKV 的 CP 边界 state 计算。数据切分、第一层投影的 GEMM shape 变化、token shift 均没有在这些采样通道中先引入差异。

## 源码差异

普通 state 核：`fla/ops/generalized_delta_rule/dplr/chunk_h_fwd.py:108`：

```python
b_v2 = tl.dot(b_w, b_h.to(b_w.dtype)) + tl.load(p_u, ...)
b_hc = tl.dot(b_kg, b_v, b_hc)
b_hc = tl.dot(b_bg.to(b_hc.dtype), b_v2, b_hc)
```

`b_v2` 是 FP32 累加结果，`b_hc` 是 FP32，因此这里把 bg 转成 FP32，再与 FP32 v2 做 dot。

CP 汇总核：`fla/ops/cp/chunk_delta_h.py:198`：

```python
b_v = b_v.to(k.dtype.element_ty)  # BF16 输入时，这里提前舍入
# DPLR mode
b_h1 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
```

两者更新的是同一形式的 state：`h ← decay*h + kgᵀ*v + bgᵀ*v2`，但 v2 的精度不同。虽然 CP 汇总 state 和通信 buffer 都是 FP32，**汇总前已发生的 BF16 舍入不会因使用 FP32 通信而消失**。

诊断版只对 DPLR 的 `USE_BG` 分支保留 FP32 v2，并将对应 bg dot 输入转成 FP32。没有改 token shift、loss、DDP、模型权重或普通 WKV 核。

## 固定输入的因果验证

`replay-rank{0,1}.json`、`ablation-rank{0,1}.json`：

| 固定输入实验 | 第 0 层后半段输出相对 L2 | 第 6 层 | 第 23 层 |
|---|---:|---:|---:|
| 原 CP 汇总核 | 0.251284% | 0.271228% | 0.256339% |
| CP 汇总核保留 FP32 v2 | 0 | 0 | 0 |
| 用普通 WKV 前半段 FP32 final state 替换 CP initial state | 0 | 0 | 0 |

补充控制：

- 原 CP 汇总出的边界 state 与普通前半段 final state 的相对 L2 分别为 **0.131015%、0.227033%、0.249928%**。
- 对同样输入仅启用 varlen `[0,4096]`，输出误差为 0。
- 串行调用两个普通 WKV 半段，以可微 FP32 state 连接，输出和六组输入梯度均与完整序列逐位一致。
- 对齐 v2 精度后，这三个样本中两个 CP rank 的六组输入梯度也全部逐位一致；只替换前向 state 就已达到同样结果。另行用普通 backward 核替代反向边界汇总，没有进一步改变这些样本。
- `use_tf32x3_affine_chain=True` 没有改变原 CP 输出误差。CP2 中只有一个前驱段，初始零 state 经一次仿射合并；本次主因不是多段 affine chain 的 TF32 误差。这不能外推到 CP4。

## 整模型干预结果

`comparison-fp32-v2.json` / `summary.json`，与原 CP1-full 基线比较：

| 指标 | 原 CP2 | 仅对齐汇总核 v2 精度的诊断版 CP2 |
|---|---:|---:|
| 最大 mean CE 差 | 0.000186920166 | 0 |
| 全 token CE 最大绝对差 | 0.03580379486 | 0 |
| 采样 logits 最差相对 L2 | 0.488159% | 0 |
| 完整 head 输入差异 | 有差异 | 逐位一致 |
| 整体梯度相对 L2 | 0.471272% | 0.0232579% |
| 梯度范数比 CP2/CP1 | 1.00066046 | 1.00000414 |

整体梯度指标沿用原对拍定义，包含 TP 复制参数，不能当成去重后的模型梯度范数。诊断版这一轮使用完整 loss mask；未声称已完成诊断版的不均匀 mask、CP4 或优化器更新回归。

## 剩余梯度误差

前向相同不代表不同分片的参数梯度会逐位相同。普通 PyTorch BF16 参数的局部梯度会先以 BF16 生成，再进入 FP32 main_grad/DDP buffer。于是通常有：

`FP32(BF16(全序列梯度和)) != FP32(BF16(前半梯度和)) + FP32(BF16(后半梯度和))`。

Megatron TP linear 的 `gradient_accumulation_fusion` 则可以直接累加到 FP32 main_grad（`megatron/core/tensor_parallel/layers.py:810`），残差明显更小。

诊断版端到端剩余差值的平方范数构成：embedding **78.20%**，bottleneck **10.10%**，其他参数 **11.68%**，Megatron linear **0.0148%**。组内相对 L2 分别约为 0.3511%、0.1469%、0.0970%、0.0002847%。

独立控制实验进一步确认分片归约机制，均固定上游梯度，不引入任何 WKV 或 CP state 差异：

| 控制实验 | BF16 的全长 vs 两半局部梯度之和 | FP32 对照 |
|---|---:|---:|
| 真实第一层 learn bottleneck 输入，固定随机上游梯度 | wgrad 相对 L2 **0.234954%**；前向及 dgrad 差为 0 | wgrad 1.826e-7 |
| 真实首样本 token，固定随机 embedding 上游梯度 | embedding wgrad 相对 L2 **0.428676%** | 1.634e-9 |

embedding 的同条件完整 backward 重复运行差为 0。以上控制使用固定随机上游梯度，验证误差机制与量级，不是对真实训练残差的逐元素一一归因。

## 复现与产物

```bash
# 真实模型逐层捕获
bash training-smoke/audit/cp-error-source-20261021/run.sh 1
bash training-smoke/audit/cp-error-source-20261021/run.sh 2
python training-smoke/audit/cp-error-source-20261021/compare_layers.py

# conda train_mobius_py312；PYTHONPATH 指向仓库根目录
# 每次以下命令单独运行，均使用两张 GPU
OMP_NUM_THREADS=8 torchrun --standalone --nproc-per-node 2 training-smoke/audit/cp-error-source-20261021/replay_wkv.py
OMP_NUM_THREADS=8 torchrun --standalone --nproc-per-node 2 training-smoke/audit/cp-error-source-20261021/ablate_boundary.py

# 8 卡整模型干预，复用此前 capture.py 的生产梯度重建
bash training-smoke/audit/cp-error-source-20261021/run_fp32_v2.sh
python training-smoke/audit/cp-equivalence-20261021/compare.py \
  --reference training-smoke/audit/cp-equivalence-20261021/cp1-full \
  --actual training-smoke/audit/cp-error-source-20261021/cp2-fp32-v2 \
  --output training-smoke/audit/cp-error-source-20261021/comparison-fp32-v2.json

python training-smoke/audit/cp-error-source-20261021/replay_weight_grad.py
python training-smoke/audit/cp-error-source-20261021/replay_embedding_grad.py
```

原始大 tensor 仅保留在本机并由本目录 `.gitignore` 排除。诊断脚本、生成的精度对照 kernel、JSON 指标、日志和本文均可审阅。首轮探索性 FP16 重放因编译耗时被主动终止，正式保留的 BF16 重放全部完成。最初 native-backward 对照脚本的 h0 返回约定已修正并重跑成功。

本次证明的是上述 CP2 条件下与既有 CP1 实现的一致性；CP1 本身也有 BF16 chunk 运算误差，不能据此宣称与无限精度逐 token recurrence 完全一致。正式补丁仍应覆盖共享 CP kernel 的其他调用方、CP4、多样本/多长度和性能。
