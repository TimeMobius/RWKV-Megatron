# CP4 数值差异定位（2026-10-21）

> 后续更新：本文保留修复前的历史结果和当时的代码状态。生产修复已完成，CP4/8192与CP4+SP/16384全部层/head的WKV输出及六路输入梯度控制回放与CP1逐位一致，整模型参数梯度相对L2为0.0169253%。最新证据见 [CP4状态修复报告](../cp4-state-fix-20261021/RESULTS.md)。

已定位到 **FLA DPLR 的跨段状态汇总**。其中有两个可独立验证的数值来源：FP32 状态传播矩阵默认使用 TF32 累乘，以及固定 affine 汇总无法精确表示本地递推中的 BF16 舍入。前向和反向各自存在这一问题。当前生产代码未修改，严格门限未放宽。

用真实 2.7B/8192-token 输入复现原先最大误差后，只在诊断进程内替换边界状态传播：选定层的 WKV 输出和六路输入梯度全部恢复逐位一致；整模型捕获前向恢复逐位一致，参数梯度整体相对 L2 从 **0.466676% 降至 0.0169253%**。因此此次定位已经从“误差可能来自状态合并”推进到可复现、可干预的因果证据。

## 证据范围

- 主仓库 `0ca9d3a`、FLA `261dbcf5`、Megatron `d65baf60f`，均为 `2026-10-10` 分支；主仓库已有未提交 CP+SP 改动。两个子模块最终均干净，本轮未改生产算法、未 commit/push。
- `train_mobius_py312`，PyTorch2.9.1+cu128、Triton3.5.1、H800。详细环境见 `environment.json`。
- 真实权重 `/mnt/data/Models/RWKV-7/rwkv7-qwen-vocab-2.7b-from-28b.pt`，0056 数据，TP2/PP1/CP4、SP关闭、序列8192。
- 完整输入来自第一训练 microbatch，捕获零起始层 0/4/13/23 的全部 36 个 head。层4和层13分别包含原严格对拍最大的输出和输入梯度误差。
- 逐层控制回放使用同一 BF16 `sin(global_element_index * 0.013)` 上游导数；不是逐层实际 CE 上游导数捕获。共 4层 × 2TP × 4CP =32 个分片记录。
- 全模型对照仍使用正常 CE 和真实训练 backward，两条 microbatch，1356 个 TP 参数梯度张量。更新被跳过，比较相同初始化权重。

## 1. 边界替换消融：原误差可完整复现并消除

`ablate_states.py` 保持本地 WKV、六路输入、上游导数、CP切分和通信 group 一致，仅改变用于本地 WKV 的前向初始状态、反向终止状态梯度。

“真实递推状态”由原有普通 DPLR state kernel 逐 rank 计算，以 FP32 通过原 CP group 广播给后继；反向按相反方向传递。它是定位用的串行对照，没有使用全局 CP1 输出或梯度替换本地结果，也没有修改本地 WKV backward。

| 边界处理 | 最大 WKV 输出相对 L2 | 最大单路输入梯度相对 L2 |
| --- | ---: | ---: |
| 原始 CP4 affine 前向＋反向 | 0.674422% | 1.206028% |
| 仅前向改传真实状态 | **0，逐位一致** | 1.206028% |
| 仅反向改传真实状态梯度 | 0.674422% | 0.864645% |
| 前向、反向均改传真实状态 | **0，逐位一致** | **0，逐位一致** |

数据见 `ablation-tp{0,1}-rank{0,1,2,3}.json`。四个真实层、全部36个head均成立，且默认模式准确复现先前 0.674422%/1.206028% 的极值。

边界误差还有明确方向性：CP0/CP1 的前向初始状态及输出精确，误差从 CP2、CP3 出现；CP2/CP3 的反向终止状态梯度精确，误差出现在 CP0、CP1。最大前向边界状态误差为0.431451%，最大反向边界状态梯度误差为1.143068%。这与需要合并至少两个已处理分段时才暴露问题一致。

## 2. 第一项来源：矩阵链的默认 TF32 运算

状态汇总构造每段的传播矩阵 `M`，跨段计算 `H_next = M @ H + C`。当前 NVIDIA 路径默认将 `AFFINE_CHAIN_PRECISION` 设为 `None`，交给 Triton 的 `tl.dot` 默认精度；本环境 Triton默认值为 **tf32**。

具体位置：

- [chunk_delta_h.py:314](/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/3rdparty/flash-linear-attention-260115/fla/ops/cp/chunk_delta_h.py:314)：前向矩阵链累乘。
- [chunk_delta_h.py:758](/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/3rdparty/flash-linear-attention-260115/fla/ops/cp/chunk_delta_h.py:758)：反向矩阵链累乘。
- 同文件 CP merge 的 `tl.dot(M,H)` 也接收这一精度选项。

诊断中的 `torch.backends.cuda.matmul.allow_tf32=False` 只限制 PyTorch matmul，不会替 Triton 的 `tl.dot` 设置 `input_precision='ieee'`，所以之前关闭这个开关不能排除此项。

`affine_contract.py` 使用相同的层4/13、全部36个head及同一份已准备好的 `w/bg/gi`，独立以 FP64 计算理想的矩阵乘积。只改变 CP 矩阵链和 merge 的精度，结果如下，取所测分片最大值：

| CP 矩阵链精度 | 前向 M 相对 FP64 的 L2 差异 | 反向 M 相对 FP64 的 L2 差异 |
| --- | ---: | ---: |
| 默认 TF32 | 1.818969% | 1.851516% |
| FLA现有 `use_tf32x3_affine_chain=True` | 0.0158272% | 0.0157049% |
| 诊断强制 IEEE FP32 | 0.000491270% | 0.000496448% |

因此这里确实有矩阵链精度问题。表中是传播矩阵的差异，不是直接将 WKV 输出或训练 loss 的差异等同为1.85%。

## 3. 第二项来源：本地 BF16 递推不是严格 affine 映射

普通前向 state kernel 用 FP32 保存累计状态，但在下一次乘法中执行：

```python
v2 = W @ BF16(H) + U
H_next = D * H + K.T @ V + B.T @ v2
```

其中 `W/U/K/B/D` 是每个64-token块已经准备好的系数。见 [chunk_h_fwd.py:103](/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/3rdparty/flash-linear-attention-260115/fla/ops/generalized_delta_rule/dplr/chunk_h_fwd.py:103)。

CP 的矩阵汇总则按去掉状态量化后的形式构造 `M = product(D + B.T @ W)`，并用从零状态执行原递推得到常量 `C`。但是存在 `BF16(H)` 时，通常不满足：

```text
F(H) = F(0) + M @ H
```

这不是单纯交换加法顺序：状态舍入使数值函数不再严格线性，单个与输入状态无关的固定矩阵不能对所有状态精确表示它。

反向 state kernel 同时有 `BF16(dH)` 和 `BF16(dV + B @ BF16(dH))` 的舍入，再乘 `W.T`；见 [chunk_h_bwd.py:105](/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/3rdparty/flash-linear-attention-260115/fla/ops/generalized_delta_rule/dplr/chunk_h_bwd.py:105)。反向的固定矩阵汇总存在同一类不等价。

控制试验直接在已准备好的真实系数和真实边界状态上检验 `F(H)` 与 `F(0)+M64@H`，其中 `M64` 使用 FP64 矩阵参考，以排除上一节矩阵链精度的干扰：

| 本地 state 计算 | 前向 affine 残差最大相对 L2 | 反向 affine 残差最大相对 L2 |
| --- | ---: | ---: |
| 原生产递推 | 1.34051e-3 | 3.94729e-4 |
| 前向更新强制IEEE，但保留BF16状态操作数 | 1.27681e-3 | — |
| 去除递推状态链BF16转换，并用IEEE FP32的诊断控制 | 2.85222e-6 | 5.22079e-7 |

另外直接检查中点关系 `F(H) == 2F(H/2)-F(0)`：生产前向最大残差2.45637e-3，反向6.86712e-4；相应FP32控制降至4.74950e-6、5.82775e-7。

这些对照确认，即使把矩阵链精度提高到 IEEE，保留原 BF16 递推时仍有 affine 表示误差。单独打开 tf32x3 或 IEEE 不能保证恢复 CP1 的逐位结果。FP32 控制是为验证机制而在诊断进程中生成的 kernel，改变了本地参考计算，不能当作已经完成的生产修复。

同一试验还验证：原 CP 从零状态计算的常量 `C` 与普通 kernel 从零状态的结果逐位相同，前后向均成立；所有汇总和通信状态 dtype 均为FP32。全部发送端汇总 hash 与各接收端相应 rank 的汇总 hash 逐一一致，未发现通信丢精度或rank映射错位。

## 4. 为什么 CP2 精确而 CP4 不精确

CP2 的第二段只需要第一段**从零状态**算出的真实结果 `C0`；反向也只需要末段从零终止梯度计算出的结果。第一轮精度修复后，这条从零状态路径已经与普通递推对齐。

CP4 的第三段需要跨过两个前驱分段，必须用 `M1 @ C0 + C1` 重建边界。这时既开始受矩阵链精度影响，也开始依赖 `F1(C0) == M1 @ C0 + C1` 的近似；反向同理。因此 CP2 的逐位一致不能推导出 CP4 逐位一致。

## 5. 整模型因果确认

`capture_native_boundaries.py` 只在本进程替换前后向边界汇总，使用实际CP group传递真实FP32状态，其他模型、loss、梯度归约、TP、优化器梯度分片重建路径不变。

同一真实2.7B、TP2/PP1、seq8192、两个microbatch，对照原CP1捕获：

- 所有token的完整词表CE、完整head输入、每TP rank采样的64列logits **全部逐位一致**。没有捕获所有词表logits。
- 全部1356个参数梯度张量有限；整体相对L2 **0.466676% → 0.0169253%**。
- 所选4层/全部36head的独立WKV回放输出、六路输入梯度全部逐位一致。

数据见 `comparison-native.json`、`summary.json`。剩余0.0169%已降到此前BF16分片参数梯度累加的量级；本轮没有逐元素追踪它的全部来源，不能把这一残余也宣称为已经完全归因。

## 修复方向及当前状态

定位已完成，生产修复尚未实施。精度目标决定实现选择：

1. 保留当前CP1的BF16计算并要求逐位对齐：传递真实递推状态可作为可靠参考和精度优先路径，但引入跨CP段的串行依赖，需要实测代价。
2. 保留并行affine汇总：先明确矩阵链精度，再统一CP1和CP多段的状态计算/数值参考，评估更高精度状态乘法及性能。只改CP端矩阵精度不能消除原BF16递推的非线性舍入。

没有放宽原严格容差，也没有把诊断控制提升为默认配置。VPP仍按用户要求暂缓。以上机制证据限定于当前RWKV7/DPLR路径，不代表已经审计所有共享CP算子。

## 复现

从仓库根目录运行；所有脚本自动使用指定conda环境或明确Python路径：

```bash
# 输入已捕获在本目录的 layerN-rankR.pt；不修改用户权重或数据。
bash training-smoke/audit/cp4-error-source-20261021/run_ablation.sh --tp 0
CUDA_VISIBLE_DEVICES=4,5,6,7 bash training-smoke/audit/cp4-error-source-20261021/run_ablation.sh --tp 1
# 矩阵精度控制分别以 default / tf32x3 / ieee，TP0 / TP1 运行。
CP4_DIAG_ENTRY=training-smoke/audit/cp4-error-source-20261021/affine_contract.py \
  bash training-smoke/audit/cp4-error-source-20261021/run_ablation.sh --tp 0 --affine-precision ieee
# 核对已有全部证据，不占用GPU。
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/audit/cp4-error-source-20261021/summarize.py
```

`capture_inputs.py`、`capture_native_boundaries.py` 的启动配置分别存放在相邻的 `cp-validation-20261021/config-capture-cp4-{diag-inputs,native}.yaml`，均复制既有CP4/8192配置。原始大张量和自动生成的诊断kernel由本目录 `.gitignore` 排除，保留在本地；脚本可从已安装源码重新生成控制kernel。
