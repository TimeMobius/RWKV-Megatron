# RWKV7 FP8 检查、修复与单卡测试

2026-10-10，独立工作区 `RWKV-Megatron-upstream-test`。Python 3.12.14、Torch 2.9.1+cu128、TE 2.19.0，实际硬件 H800。原工作区、用户配置/启动脚本、模型和数据未修改。

## 结论

原来的常规 FP8 配置没有真正开启 RWKV 的大矩阵乘法：六个投影分别进入了 `fp8_autocast(enabled=False)`。现在已修复，实际检查 GEMM 输入能看到 FP8 张量，而不只是配置显示 FP8。

但“能正确调用 FP8”不代表“精度和速度已经适合正式训练”。完整 3B 的结果仍有明显误差，128 token 的短序列也没有提速。**目前建议正式训练继续使用 BF16，FP8 保留为可配置的实验功能。**后续值得研究精度和长序列性能，不应把这些误差直接忽视。

固定 eval 已在前一提交 `11eaa50` 完成；BF16 和本次 FP8 的实际断点恢复都继续使用同一组验证数据。

## 正确的接法及本次改动

- 使用 Megatron 的 TE 并行 Linear，让 decoder 管理 FP8 格式、缩放、通信组、重计算和状态保存。删除 RWKV 自己每次新建的 HYBRID/DelayedScaling 配置；它既未传通信组，也不服从 YAML 的 recipe。
- 增加 YAML/CLI `rwkv_fp8_scope`：`ffn` 是开启 FP8 后的默认范围，只量化 FFN 的 up/down 两个大矩阵乘法；TimeMix 全部保持 BF16。`all` 开启六个大矩阵乘法，保留给精度实验，不推荐当前正式使用。
- 无论哪个范围，FLA WKV 输入/输出仍是 BF16，key 的敏感中间计算仍是 FP32，归一化和普通 Torch 的 LoRA 小矩阵乘法不被自动转换成 FP8。词表输出层、CE 和 L2Wrap 保留现有精度路径；v0 的载体仍是 BF16。
- 拒绝 `fp8: true` 等不明确格式、FP8 搭配 local Linear、TE 不可用时要求 FP8、以及 RWKV 不支持的 TE attention FP8 开关。FFN 范围要求权重保存在 BF16，避免 TimeMix 虽然关闭 FP8 计算、权重却已在初始化时被量化。
- 当前 TE 2.19 已移除 `override_linear_precision`。Megatron 仍传这个旧参数，但 TE 忽略它，所以 `fp8_wgrad: false` 不会让权重梯度用 BF16。这点由代码与实验同时确认，本次加入明确提示；没有声称这个开关能提高当前环境的后向精度。
- 关闭 FP8 时保留 TE/Apex 的归一化兼容，投影继续按原设置选择 local/TE。

代码位于 `rwkv_megatron/precision.py`、`rwkv_megatron/rwkv7_layer_specs.py`、`rwkv_megatron/arguments.py` 和 `pretrain_rwkv7.py`。没有修改 Megatron/FLA 子模块或训练环境。

## 配置

正式 BF16 训练继续设置：

```yaml
language_model:
  fp8: false
fixed_eval_samples: true
```

希望复现实验时，在现有完整配置中修改以下项：

```yaml
transformer_impl: transformer_engine
rwkv_fp8_scope: ffn
fixed_eval_samples: true
language_model:
  fp8: hybrid
  fp8_recipe: tensorwise
  fp8_wgrad: true
  fp8_dot_product_attention: false
  fp8_multi_head_attention: false
```

以上是配置片段，不是完整 YAML，不应覆盖其他模型参数。`--rwkv-fp8-scope ffn/all` 可以覆盖 YAML。未配置 FP8 时不会因增加 scope 而自动开启 FP8。

`tensorwise` 每次按当前输入确定缩放；`delayed` 使用先前的缩放状态。在这次相同数据的试验中前者误差较小。H800 没有测试或认证 Blackwell 专用的 MXFP8。

## 完整 3B 精度结果

同一份原始 3B 权重、同一批 MiniPile tokens，batch 1、seq 128、无优化器更新。进行 3 次前向/后向预热，再重复测量；FP8 不要求逐位等于 BF16。整体梯度差为所有 899 个实际产生梯度的参数的差值总量除以 BF16 梯度总量。

| 模式 | CE loss | 相对 BF16 的 logits 差 | 整体梯度差 | 第 0 层 value 权重梯度差 |
| --- | ---: | ---: | ---: | ---: |
| BF16 TE | 2.143411 | — | — | — |
| 恢复旧版显式关闭 FP8 的写法 | 2.143411 | 0 | 0 | 0 |
| all + delayed | 2.763053 | 16.56% | 43.71% | 89.27% |
| all + tensorwise | 2.224777 | 6.09% | 237.60% | 见原始 JSON |
| all + tensorwise，首尾层 BF16 | 2.220644 | 2.97% | 59.77% | 见原始 JSON |
| ffn + delayed | 2.277053 | 6.42% | 67.90% | 64.87% |
| ffn + tensorwise | 2.201098 | 3.63% | 17.77% | 30.42% |

每项前向和参数梯度均为有限值。首尾层保持 BF16 的方案并不能解决全模型的误差。只比较 loss 不够，梯度更能暴露 FP8 对训练的影响。

仅量化 FFN 单个矩阵乘法的额外实验也没有形成一致改善：up-only loss 2.181853、整体梯度差 41.42%；down-only loss 2.159932、整体梯度差 24.06%。这两项仅在审计脚本中实现，没有继续增加生产配置选项。

**这些差异直接来自降低矩阵乘法输入和后向输入的精度，以及误差经过多层传播后的放大；目前没有发现新增的公式或梯度连接错误。**重计算开/关时，FFN FP8 的 logits、loss 与全部参数梯度完全相同。第 0 层 value 权重梯度有实际传播，但 FP8 与 BF16 的数值差仍值得优化。以上单批结果不能证明长期训练收敛。

## 速度结果及范围

完整 3B，H800 单卡，batch 1、seq 128，已预热，无层级重计算，测量前向＋反向；不包含优化器、保存和 eval。每个方案测 5～7 次，以下用同一组 FFN 对照的中位数：

| 模式 | 时间 | TPS | 相对同组 BF16 |
| --- | ---: | ---: | ---: |
| BF16 TE | 234.6 ms | 545.7 | 基准 |
| ffn + delayed | 244.5 ms | 523.6 | TPS 下降 4.1% |
| ffn + tensorwise | 247.4 ms | 517.4 | TPS 下降 5.2% |

短序列下量化和调度开销抵消了矩阵乘法的收益。不能把这组数字外推到长序列、大 batch 或未来 TP/SP/PP。没有宣称 FP8 已提速，也没有把开重计算的 FP8 时间与不开重计算的 BF16 时间混着比较。

## 功能与续训测试

- 主仓库 116 项测试、30 个 subtest 通过，覆盖格式、YAML/CLI 优先级、关闭 FP8 的兼容、TE 缺失/local 冲突、权重保存精度和失效的 wgrad 选项，并回归固定 eval、key、L2Wrap、归一化、转换器和 sharding。
- 四层裁剪模型确认 FP8 范围、首尾层 BF16、全部参数有限，检查实际 GEMM 的 `Float8Tensor` 输入。完整 3B 单矩阵投影试验也检查了 GEMM 输入，不能仅凭配置判断开启成功。
- 完整 3B 的 FFN FP8，delayed/tensorwise 两种方案均确认重计算开/关的 logits、loss、全部梯度完全相同。
- 完整 3B 真实训练使用 Adam、分布式优化器、global batch 2、层级重计算、key FP32 合并、L2Wrap、每步固定 eval。训练 2 步保存，重新启动恢复至 4 步，与连续 4 步比较。
- delayed 方案的模型、优化器、学习率、随机状态与 64 份 TE 缩放状态完全一致，训练 loss/梯度大小与对应 eval loss 也一致。对 TE 的序列化状态先解码再比较真实数值，而不是要求含序列化标识的字节逐位一致。
- TE 2.19 默认拒绝加载 delayed 状态里的 pickle。仅在恢复**本次自己生成的测试检查点**的进程中设置 `NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1`；没有在生产代码或原启动脚本中全局放开。使用 delayed 恢复自己确认来源的训练检查点时也需要这个 TE 选项。tensorwise 不保存这类历史缩放状态。
- 单独重跑关闭 FP8 的 3B 训练，4 步 loss 和梯度大小与 `11eaa50` 的固定 eval 基准一致，验证数据也一致。

tensorwise 的真实恢复结果保存于 `fp8-tensorwise-training-validation.json`，delayed 位于 `fp8-training-validation.json`。本轮只认证单卡功能；TP/SP/PP 的 FP8 实际训练、长序列吞吐、FP8 参数存储和长期收敛仍需独立测试。

## 复现与数据清理

```bash
# 相同权重的裁剪模型、全部 FP8 范围，以及 FFN 范围
bash training-smoke/audit/launch-fp8.sh
bash training-smoke/audit/launch-fp8.sh --ffn-only

# 完整原始 3B；保留参数梯度于内存，只输出 JSON，不生成大梯度快照
bash training-smoke/audit/launch-fp8.sh --full-3b
bash training-smoke/audit/launch-fp8.sh --full-3b --ffn-only --repeats 7
bash training-smoke/audit/launch-fp8.sh --full-3b --ffn-projections --repeats 7

# tensorwise 保存/恢复/连续训练，再比较
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/run-fp8-tensorwise-training.py
OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/validate-fp8.py --tensorwise
```

delayed 使用复制的 `config-3b-{save,resume,continuous}-fp8-ffn.yaml`，同一审计 launcher，设置 `FIXED_EVAL_CAPTURE=fp8-ffn-{phase}-batches.json`，resume 进程加上述 TE 选项；执行 `validate-fp8.py` 比较。检查点清理后需要重新运行训练才能再次比较。

代码/配置/launcher 的修改前备份：`training-smoke/reference/before-fp8-integration.tar.gz`。当前所有配置都是新副本，用户原件 SHA256 再次核对一致。

清理清单及来源证据在 `TEST-CHECKPOINT-CLEANUP.json`。旧的已完成测试检查点释放约 774 GiB；本轮新生成的测试检查点在完成比较后也清理，保留配置、日志、JSON 与仍用于精度比较的审计材料。用户 `/mnt/data/Models` 下的检查点以及原工作区未触碰。
