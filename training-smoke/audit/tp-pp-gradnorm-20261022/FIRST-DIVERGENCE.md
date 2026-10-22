# 最早差异定位：第0层Attention行并行输出投影

2026-10-22。沿用28B、seq4096、SP、MB1/GB2、BF16/full recompute、0056数据、`train_mobius_py312`和`2026-10-10`分支。固定TP2/PP4训练25步后的同一份BF16权重、同一第26步batch，比较TP2/PP2与TP4/PP2。本轮没有修改生产计算代码；所有通信精度变化仅用于独立诊断重放。

## 已定位的具体位置

第一次不一致出现在代码索引layer0（第1层）的`self_attention.linear_proj`：

- `rwkv_megatron/rwkv7_layer_specs.py:802`：`self.linear_proj(output * gate)`。
- `3rdparty/Megatron-LM-0.11.0/megatron/core/tensor_parallel/layers.py:1650`：局部`_forward_impl`，产生BF16投影部分结果。
- 同文件`:1667`：SP路径调用`reduce_scatter_to_sequence_parallel_region`，对BF16部分结果跨TP求和并切分序列。
- `.../tensor_parallel/mappings.py:190`：底层`dist_reduce_scatter_func`。

第0层输入LayerNorm、receptance/key/value投影、WKV六路输入和输出、GroupNorm、gate，以及最终送入`linear_proj`的输入，在两种TP之间均逐位相同。按TP分片重建的权重来自同一快照。

`linear_proj`输出相对L2差0.3571755%，20,971,520个元素中9,740,699个不同；token0输出相对L2差0.3888167%。后面的残差、FFN及后续层开始接收这些不同的数值。第0层FFN的`linear_down`也走同类行并行路径，但它的输入在此时已经不同，不能把它当作独立的首次差异。

## 局部矩阵乘法与通信分别重放

行并行投影的数学形式为：

```text
Y = Σ_r X_r W_rᵀ
```

当前生产路径先把各卡的局部矩阵乘法结果保存成BF16，再将这些BF16张量交给跨TP reduce-scatter。TP2和TP4的输入通道分组不同，因而被单独舍入的部分和也不同；4路BF16求和还会出现中间舍入。

保存了第0层该投影的原始输入、逐rank部分结果、真实collective输出及权重。`replay_row_precision.py`用相同输入/权重在独立GPU重放，所有2路/4路局部BF16矩阵乘法结果都与训练捕获逐位相同。

以TF32关闭的FP32矩阵乘法为独立局部参考：FP32分片乘法、FP32求和相对完整FP32矩阵乘法的差分别为1.2732e-6（TP2）和1.3706e-6（TP4）。说明当前权重拼接、分片数学关系可对齐到约1e-6；低精度误差主要在BF16投影输出路径及归约中引入。

两种TP的固定输入对照如下：

| 独立诊断路径 | TP2/TP4输出相对L2差 |
|---|---:|
| FP32部分结果＋FP32求和，最后转BF16 | **0.0035344%** |
| 训练捕获的BF16部分结果，改用FP32求和，最后转BF16 | **0.3048763%** |
| 生产BF16部分结果＋实际BF16 reduce-scatter | **0.3571755%** |

第一行与生产差异相比缩小约101倍。三行是控制实验，误差有方向和抵消关系，不能相减后作为彼此独立的误差占比。

局部BF16矩阵乘法结果与“独立FP32局部矩阵乘法后转BF16”仍有约0.0065%–0.0163%的相对L2差。不同GEMM精度/内核路径也可能改变舍入；因此不能把局部全部差异严格归为一次纯dtype转换，更准确的表述是“局部BF16 GEMM输出路径＋BF16归约”。

## 独立NCCL通信重放确认

`replay_collective.py`分别用2进程和4进程，只加载已捕获的部分结果，重新执行reduce-scatter：

- 两组所有rank的BF16通信输出均与真实训练捕获逐位一致。
- TP2的BF16结果等于“先精确求和这两份BF16值，再一次转BF16”的结果。
- TP4的生产输出与“FP32求和后一次转BF16”相比，有6,926,038个元素不同，相对L2差0.2899641%。这证明通信归约中的中间舍入确实有可测影响。
- 同样已舍入的TP4输入，FP32 reduce-scatter相对独立FP64求和的每rank相对L2差仅约2.86e-10–3.61e-10；BF16结果相对该求和的差约0.219%–0.293%。
- FP32通信仍不保证所有元素与独立精确和最后舍入逐位一致：TP4/rank1有1个元素不同，其余rank为0。不能宣称FP32通信绝对无舍入。

## 与第26步28.4%范数差的关系

细化捕获时总范数仍精确复现：TP2为27.448297211244277，TP4为21.427622352437325。记录过程没有改变被诊断轨迹。

本轮确定了进入网络的第一个前向差异源。上一轮已通过PP控制、WKV固定输入head切分、敏感head/token定位、首token loss消融及局部独立FP64梯度公式检查，建立后续数值敏感性放大的证据，见`REPORT.md`。

结论应表述为：**第0层行并行Attention输出投影的BF16局部结果和BF16跨TP归约引入首个差异，后续同类投影继续引入差异；这些差异在少数首token近抵消head上放大，最终形成观察到的真实梯度范数差异。**

本轮没有将28.4%全部量化归因于第0层的这一次投影，也没有证明只提高这一个投影的精度就能修复整模型误差。后向TP通信也可能贡献额外差异。独立FP32局部重放不能替代完整28B高精度参考或收敛验证。

## 改进方向与范围

若继续做精度修复，应针对行并行投影评估：局部累加/输出精度，以及跨TP归约精度。仅把已经舍入的BF16部分结果转为FP32通信，无法找回之前丢失的信息；仅保留FP32梯度或主权重也无法修复前向差异。提高通信精度会增加通信量和临时张量占用，正式方案需要重新验证显存/速度和完整梯度，而不是直接更改全部生产通信。

## 材料和复跑

- `fine_hooks.py`：前3层首microbatch重计算的模块边界及reduce-scatter前后捕获；只读张量，不改变输出。
- `capture.py`增加可选`TPPP_FINE_TRACE`入口，`launch.sh`沿用原启动环境。
- `fine-tp2pp2.log`、`fine-tp4pp2.log`及对应目录：真实28B一次前后向捕获。优化器更新跳过；四卡TP2/PP2使用CPU优化器存储避免无用GPU主权重占用，不代表四卡全GPU训练通过。
- `compare_fine.py` / `fine-boundary-comparison.json`：逐模块对照；`fine-layer0-wkv-all-inputs.json`覆盖全部六路WKV输入。
- `replay_row_precision.py` / `row-precision-replay.json`：相同投影输入/权重、局部矩阵乘法与不同精度的归约重放。
- `fixed-row-layer0-self_attention.linear_proj.pt`：独立行投影重放输入、权重及结果，约数百MiB。
- `replay_collective.py` / `collective-replay-tp{2,4}.json`：独立NCCL对照。
- `FIRST-DIVERGENCE.json`、`provenance-fine.json`：结论摘要、生产和诊断源码哈希。

```bash
TPPP_REPLAY_WEIGHTS="$PWD/training-smoke/audit/tp-pp-gradnorm-20261022/tp2pp4" \
TPPP_FINE_TRACE=1 TPPP_NPROC=4 \
bash training-smoke/audit/tp-pp-gradnorm-20261022/launch.sh fine-tp2pp2 \
 training-smoke/audit/tp-pp-gradnorm-20261022/config-fine-tp2pp2.yaml

TPPP_REPLAY_WEIGHTS="$PWD/training-smoke/audit/tp-pp-gradnorm-20261022/tp2pp4" \
TPPP_FINE_TRACE=1 \
bash training-smoke/audit/tp-pp-gradnorm-20261022/launch.sh fine-tp4pp2 \
 training-smoke/audit/tp-pp-gradnorm-20261022/config-replay-tp4pp2.yaml
```

`compare_fine.py`初次运行对layer的空位置参数列表处理失败，原因是其输入通过keyword传入；修正比较脚本、跳过该层级的input字段后完成，未重跑或更改训练捕获。其余有位置参数的模块输入、全部模块输出均已比较。
