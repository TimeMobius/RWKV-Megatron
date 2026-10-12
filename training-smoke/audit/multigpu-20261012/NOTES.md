# 多卡审计过程记录（已完成）

以下保留当时的排查进度；最终状态以 ../MULTI-GPU-20261012.md 和 completed-checks.json 为准。

- 环境及源码：复用单卡已验证的 Python 3.12 / Torch 2.9.1 / TE 2.19、当前 Megatron 和 FLA；不更新子模块。
- 旧 ISSUES.md 中的多卡结果来自 FLA 布局、LayerNorm、key 精度与 L2Wrap 修改之前，不能沿用为当前结论。
- 真实多进程小模型覆盖 TP2/4、PP2/4、SP、batch2、两微批次；全部 135 个导出参数梯度。
- 完整 3B 覆盖全部 1,059 个导出参数梯度；FP32 独立递推使用 seq32，BF16 FLA 使用 seq128；不混合两种输入长度比较。
- 初始审计脚本使用 NCCL gather_object 汇集 tensor 记录时停住；改为逐 rank 文件记录。后续切换过多并行组合时反复新建但未彻底销毁通信组，显存累积；每种组合改为新进程。上述是审计工具限制，不是据此判定生产通信错误。
- 非交错 PP 的发送形状已带上 v0；交错 PP 的 tensor_shape 仍是三维，尚未适配 v0，不建议开启 virtual_pipeline_model_parallel_size。
- 当前 selective/core_attn 没有接入 RWKV7TimeMix.core_attention 的重计算调用；配置不会触发 WKV 重算。整层 uniform/block 为本轮验证目标。selective/mlp、layernorm 需独立测试，不能由此推断都不可用。
- 原生 checkpoint 不含第零层未使用的 bottleneck_v 三个参数，from_rwkv_lm 会跳过；构造时 perform_initialization=false 的这些占位参数没有被显式初始化。通信哈希排除这些未使用参数，最终 checkpoint 对照仍包括它们，若出现差异将单独记录，不能伪装为完全相同。
- 本轮不把与单卡逐位一致设为 TP 精度目标；高精度用于验证公式/通信，BF16 用于记录实际误差。短跑不能证明长期收敛或 FP8 多卡质量。

## 完整 3B BF16 的新发现

- 单独增加 PP2，并非所有 BF16 前后向都完全相同。当前 seq128 的第一处记录差异在 layer16 WKV 输出，局部相对差约 5.8e-6，后续层放大后 logits 0.697%、整体梯度 4.890%。全部已记录 PP 发送/接收仍逐元素相同，v0 数值在各层保持不变。正在补齐该层所有 WKV 输入，不直接归因于网络通信。
- TP2 第一处差异在 layer0 输出矩阵乘法的部分结果跨卡相加，之前 norm1/raw k/v/r/WKV 全部相同。局部相对差 0.247%，最终 logits 0.977%、整体梯度 14.939%；CE 2.143411→2.132133。
- 完整 BF16 重计算不能沿用小模型“所有参数完全相同”的结论。TP2+SP+PP2 uniform1 相对不重算：CE 差 0.006829，logits 0.648%、整体梯度 6.055%。高精度重计算 1,059 个参数完全相同。正在核对 checkpoint 首次执行与其真实反向重放的算子输入/输出。
- TP4+SP+PP2+uniform1 相对单卡：整体梯度约 21.323%，不能直接写成已达到训练精度要求；与数学/通信错误的判断分开。

## 保存恢复配置修正

第一次 save 试验误把 train_iters 从 4 改成 2，虽然 LR decay 长度显式保持 4，Megatron 的 weight-decay scheduler 总长度仍由 train_iters×globalbatch 推导。resume 到 4 会按接口要求拒绝加载（wd_incr_steps 8 vs 16）。改为 save/resume/continuous 全部 train_iters=4，save 设置 exit_interval=2，保证原计划不变。错误尝试的 checkpoint 仅为本助手新建目录，已记录 manifest 并删除；失败日志另存，未修改用户资源。

## 可优化但本轮未改的 v0 传输

当前 SP 每个 TP rank 携带完整 `[S,B,H]` v0；隐藏状态本身是 `[S/TP,B,H]`。因此每个 PP 边界的传输量是 `(TP+1)×S/TP×B×H` 个元素，v0 占大头。各 rank 的 value 混合实际只读取本地 head 对应的 `H/TP`，理论上可只携带 `[S,B,H/TP]`，把 PP 包装由 `TP+1` 份缩到2份。TP2理论减少1/3、TP4减少3/5的该激活/梯度传输体积，但不是承诺整体训练提速；需重做序列/样本布局、v0梯度以及TP/PP/recompute/save-resume认证。本轮不擅自改这套协议。

## SP 与 PP 单项补充

- 真实 NCCL token-shift：TP2/TP4、local_seq=1/2/32、batch=1/3，各组合前向与反向均逐元素正确。
- cropped4层只用一个微批次：PP2 logits/CE 与单卡相同，整体梯度相对差2.10e-9（135个导出参数中的133个逐元素相同）；两微批次PP的微小差异与参数梯度累加顺序有关，不能单纯视为传输损坏。
- selective/core_attn 真实调用计数：每层仅一次with_grad，无no_grad和重放，证实此配置没有触发WKV重算。

## 确认的 LayerNorm selective 适配 bug 与修复

TP2+SP+PP2，selective/layernorm：第二个PP阶段两个rank的反向报 `AttributeError: CheckpointWithoutOutputFunctionBackward has no attribute inputs`。RWKV7TransformerLayer.forward 直接调用 `_run_input_layernorm`，却没有继承标准TransformerLayer在attention之后调用的 `discard_output_and_register_recompute`。已在RWKV层补上，hook放在完整attention carrier上，涵盖隐藏状态和第零层v0的两路梯度。仅影响配置为selective/layernorm的路径，不修改默认full重算、FLA或Megatron子模块。失败日志保留；正在重跑真实多卡验证。

## 重计算自身重放

完整3B单卡seq128，在真实checkpoint首次forward与其backward重放中，32层的全部WKV六项输入和输出逐元素相同，默认compiled key与Torch eager key两种路径都通过。重计算和普通forward的精度差并不能直接解释为“重放用了不一致的输入”。正在补测TP2+SP+PP2同样的自身重放。

输出部分结果FP32试验第一次尝试torch.mm(out_dtype=FP32)直接autograd，Torch2.9.1不提供该重载的derivative；该试验退出1，没有修改生产。追加独立审计后向，保持BF16矩阵后向，与生产优化器/融合main_grad路径不同；只能用于精度机会评估，不认证可直接投产。

## 修复验证与性能

- selective/layernorm回调修复：TP2+SP+PP2、小模型FP32 batch2/两微批次，全部135个导出参数梯度、logits、CE与不重算逐元素相同；BF16多卡已跑通。完整3B真实训练正在验证。
- selective/mlp实际发生重放，FP32同样逐元素梯度对照通过。
- 无SP的distribute_saved_activations=true、TP2+PP2、整层重算：135个参数梯度、logits、CE与不重算逐元素相同；SP+此开关上游明确禁止。
- seq2048/globalbatch8、BF16/Adam/full uniform1、共4步（warm2–4三步中位数），总TPS：single6065、DP8 34500、TP2+SP+PP2+DP2 13368、TP4+SP+PP2 7704。开启审计数据hash，结果仅短跑趋势，不是长期benchmark。3B放得下时DP更合适，过度模型切分增加通信。
- CPU回归128passed、5CUDA-dependent skipped、2历史已修复negative-control xfail、30subtests；CUDA专项待实际跑。


## 最终状态

所有GPU任务结束，八张卡空闲。LayerNorm selective修复后完整3B八卡两步/eval通过；FP32135梯度严格相同。最终根仓库CUDA回归133passed、2历史negative-control xfail、30subtests。多卡compiled-key自身重放在第二PP阶段存在微小a/b差；eager-key四个rank全部WKV输入/输出逐元素相同。最终PP2训练图复测与单卡CE/logits及1,059个导出参数梯度逐元素相同，早先PP差异保留为历史，根因不能强行断言完全定位。完整报告包含全部残留精度和功能限制。没有commit/push，没有修改两子模块。测试checkpoint和已认证FP32快照共清理约405.33GiB，BF16后续排查材料保留。
