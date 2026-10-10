# 第 0 层 v0 后向汇合定位

最新FP32汇总实测见`PRECISION-DIAGNOSIS.md`：离线0.545%→0.165%已在真实完整3B后向复现，前向/CE和固定入边不变。它对整个模型相对独立FP32的误差只带来小幅改善，未启用为生产默认行为。

2026-10-10，独立工作区，单卡 H800。当前 Megatron c2876aaa、FLA 40850297、TE 2.19.0、Python 3.12.14 / Torch 2.9.1。这里讨论 BF16；不是 FP16，也不是前一轮新旧 FLA decay kernel 的差异。

后续参考修正：本文“原参考”指我编写的历史共享原语诊断脚本，不是 BlinkDL 官方代码。当前默认已改为加载 BlinkDL 源码 forward/CUDA 算子；见 `BLINKDL-ALIGNMENT.md`。历史共享 view 仅保留为 `--shared-value-reference` 负向控制。下面0.1943%的解释不代表与实际官方BF16算子已对齐。

## 结论

原报告剩余的 v0 后向差异，已定位到第 0 层 value 的**加法分组**。本次固定完整 3B 样本没有发现漏传、重复传播、detach 或错误公式。无需为了逐位对齐改变生产模型。仅对齐参考脚本的分支结构，就能恢复全部参数梯度一致；默认独立参考仍保留原写法。

生产代码分别为 WKV 和输出修正建立 value 的 reshape/permute 分支；参考脚本原先让两者共享同一个 reshape 后的 value。还存在后续层 v0 carrier 的回传分支。令 F 为后续 31 层回传梯度，A/B 为本层 WKV 和输出修正两路梯度，本次实际执行为：

```text
Megatron: ((F + A) BF16 + B) BF16
原参考:   (F + (A + B) BF16) BF16
```

每次 BF16 加法都会舍入，因此结合律不成立。这次已经通过捕获梯度重放，证明具体的分组能逐元素重构两边结果，不只是笼统推测浮点误差。

## 因果证据

完整原始 3B，32 层，hidden2560/head64，MiniPile 固定前 128 token，batch1，普通 CE，TE 归一化，同一版 FLA。诊断参考共享相同底层原语，用于检查连接；独立 FP32 公式验证仍见 `full-fp32-naive-vs-recurrent-all-gradients-fla-40850297.json`。

使用现有 autograd node 的只读 backward hook 观察边，不添加恒等节点，不替换生产后向。新捕获结果与先前没有这些 hook 的记录完全相同，避免观察工具改变累加分组。

| 检查 | 结果 |
| --- | --- |
| 第 1–31 层 lerp 对 v0 的分支梯度 | 31/31 逐元素相同 |
| F，即后续层汇总梯度 | 两实现逐元素相同 |
| 本层两路梯度的 BF16 和 A+B | 与原参考本层梯度逐元素相同 |
| 原始 v0 总梯度差 | relative L2 0.194296%，max abs 0.00390625 |
| 用捕获边梯度重放两种加法分组 | 分别逐元素重构两边总梯度 |
| 只让参考分别建立 WKV / correction 的 value view | v0 总梯度完全相同 |
| 上述控制实验的前向 / 后向记录 | 546/546、290/290 完全相同 |
| 上述控制实验的全部导出参数梯度 | 1,059/1,059 完全相同 |

另用四层、hidden256、batch2、seq128 的裁剪模型复查：原分支结构 v0 梯度差0.185674%，同样被加法分组重放完整解释；参考分支对齐后全部135个参数梯度完全相同。这项额外检查覆盖了batch>1，但不代表任意规模或输入的误差上限。

生产分支位置：`rwkv_megatron/rwkv7_layer_specs.py` 的 `RWKV7TimeMix.forward`，第 0 层 `linear_value` / `gather_from_tensor_model_parallel_region`，`self.core_attention(...)`，以及独立 `rearrange(value, ...)` 的输出修正。WKV 内部还独立转换 value 布局。参考脚本增加 `--split-value-reference` 诊断开关；不改变默认参考或生产代码。

## 值得优化的是累加精度

两个实现的后续层 v0 梯度，都能用捕获的 31 个分支按反向层序逐次 BF16 相加精确重构。对**固定、已经计算并舍入的 33 个入边梯度**再用 FP64 求和，可以单独估计累加误差：

| 汇总方式 | 相对固定 33 路 FP64 和的 relative L2 |
| --- | ---: |
| 当前 Megatron BF16 分层汇总 | 0.545287% |
| 原参考 BF16 分层汇总 | 0.546307% |
| FP32 求和后仅最终转一次 BF16，离线估计 | 0.164751% |

离线 FP32 汇总本身对 FP64 和的 relative L2 为3.20e-8，最终BF16结果与FP64求和再转BF16逐元素相同。这说明 FP32 梯度汇总值得作为后续精度优化候选，但最后一行只是固定入边的离线估计，**没有实现或认证 FP32 carrier**。该 FP64 和也不是完整模型的独立数学 oracle：FLA/线性层已经产生的低精度算子误差不包含在这里，不能将这些数字说成整个模型的真实梯度误差或收敛提升。

实际改善需要处理 carrier dtype、hidden/value 打包、checkpoint 重计算、未来 TP/SP/PP 通信以及额外显存。直接把整个 carrier 转 FP32 会连带改变 hidden 的 dtype，不能作为局部修复。建议将原“v0 梯度链路待查”标记为本样本已解释；保留 FP32 汇总作为独立优化项。

## 复现与证据

从独立工作区运行，GPU任务串行：

```bash
# 原参考结构 + 不改变计算图的梯度边观察
bash training-smoke/audit/launch-single.sh --full --bf16 \
  --reference-core fla --aligned-reference --shared-pointops \
  --te-final-reference --carrier-chain-reference --te-reference-norms \
  --audit-v0-edges --shared-value-reference --result-tag=-v0-edge-audit

# 控制实验：仅对齐参考的 value 分支，检查全部参数梯度
bash training-smoke/audit/launch-single.sh --full --bf16 --all-gradients \
  --reference-core fla --aligned-reference --shared-pointops \
  --te-final-reference --carrier-chain-reference --te-reference-norms \
  --audit-v0-edges --split-value-reference --result-tag=-v0-confirm

OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/analyze-v0-edges.py
```

`v0-accumulation-summary.json` 保存因果断言、累加误差与版本来源；两个 `*-v0-edges.pth` 保留原始分支梯度，日志在 `training-smoke/logs/audit-v0-*.log`。没有改生产模型、FLA、Megatron、环境或原工作区。这项证据不替代多卡、FP8、长序列、长程收敛或 checkpoint 重计算认证。
