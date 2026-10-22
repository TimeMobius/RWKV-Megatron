# CP 补充验证（2026-10-21）

> 后续更新：本文保留修复前的历史结果和当时的代码状态。生产修复已完成，CP4/8192与CP4+SP/16384全部层/head的WKV输出及六路输入梯度控制回放与CP1逐位一致，整模型参数梯度相对L2为0.0169253%。最新证据见 [CP4状态修复报告](../cp4-state-fix-20261021/RESULTS.md)。

真实 2.7B 权重、0056 数据上的四卡用例、CP2 WKV 回放、CP4 长序列训练、Adam 保存恢复及 FP8/Muon 功能验证已完成。**CP4 的严格 WKV 数值门限未通过，不能标为全部数值验证通过。** 用户已要求暂缓交错 PP/VPP，本轮试验性的 VPP 生产代码已撤回，不作为当前支持能力。

## 环境和范围

- 工作区：`/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test`，主仓库基线 `0ca9d3a`，分支 `2026-10-10`。
- Megatron `d65baf60f`、FLA `261dbcf5`，子模块均在 `2026-10-10`；最终两个子模块均无未提交修改。主仓库已有 CP+SP 修改和本轮验证工具仍未提交，未 push。
- Conda `train_mobius_py312`，8 × H800 80GB。
- 权重：`/mnt/data/Models/RWKV-7/rwkv7-qwen-vocab-2.7b-from-28b.pt`，24 层、hidden2304、36 heads × 64。
- 数据：`/mnt/data/Datasets/0056_qwen3.8/0056_qwen3.8_text_document`；tokenizer：`/hikscale/models/Qwen/Qwen3.8-27B`。
- BF16、microbatch1、整层重算、关闭 dropout。CP4 的 8K 用 GB2、16K 用 GB2；FP8/Muon 组合用 GB4。保存恢复用 GB2。
- 本目录 `summary.json` 是最终机器可读汇总；`jobs.json` 保留最初批次的失败和成功记录，后续单独补跑以对应目录日志/JSON 为准。

## 训练功能验证

下列各项均完成真实优化器更新和固定验证，无跳步，loss/grad norm 均有限。每项每个 rank 的完整记录在 `<用例名>/rankN.json`。

| 用例目录 | TP/PP/CP | SP | 序列长度 | 更新步数 | 最后训练 loss | 验证 loss |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `tp2-pp1-cp2` | 2/1/2，四卡 | 关 | 4096 | 3 | 13.253924 | 13.158319 |
| `tp2-pp1-cp4-8k-measured` | 2/1/4，八卡 | 关 | 8192 | 3 | 13.252439 | 13.146093 |
| `tp2-pp1-cp4-sp-16k` | 2/1/4，八卡 | 开 | 16384 | 3 | 13.206945 | 13.107233 |
| `fp8`，tensorwise FFN | 2/2/2，八卡 | 开 | 4096 | 3 | 13.274155 | 13.122652 |
| `fp8-delayed`，delayed FFN | 2/2/2，八卡 | 开 | 4096 | 3 | 13.277982 | 13.127340 |
| `muon` | 2/2/2，八卡 | 开 | 4096 | 3 | 13.469355 | 13.396429 |
| `fp8-muon`，tensorwise FFN + Muon | 2/2/2，八卡 | 开 | 4096 | 3 | 13.464369 | 13.391548 |

16K 的 CP4+SP 最高 PyTorch 已分配显存为 15.774 GiB/卡。各用例拓扑、GB、诊断开销不同，不能据此作固定条件下的 CP 显存/吞吐比较。

## CP2/CP1 WKV7 输出和输入梯度

`tp2-pp1-cp2/rank{0,1,2,3}.json`：覆盖全部 **24 层、36 个 head**，共 96 个层/TP/CP 分片记录。

第一训练 microbatch 的真实六路 WKV 输入全部用于独立算子回放。先按 CP 拼接完整输入，再用同一个生产 `WKV7.forward` 执行 CP1 参考及 CP2 前向/反向，六路输入为 `r/k/v/w/a/b`。上游导数是确定性的 BF16 `sin(global_element_index * 0.013)`，两侧完全相同；这是控制变量的算子验证，不是所有层在端到端 CE 反向中的实际上游导数捕获。

**输出和六路输入梯度全部逐位相同，最大绝对差及相对 L2 均为 0。** 每个 TP rank 各负责 18 个 head，合起来覆盖全部 36 个 head。当前回放样本数为 1；MB2 的样本独立性证据仍见之前的 CP+SP 报告。

## CP4 数值结果：整模型容差通过，严格 WKV 门限失败

同一初始化权重、同一两条 8192-token 输入、TP2/PP1、SP关闭，对比 CP1 与 CP4。`captures/cp{1,4}-8k` 捕获真实训练前向与梯度，跳过更新。分布式优化器梯度从 reduce-scatter 的有效分片重建，未读取失效的未拥有区域。

`comparison-cp4-8k.json`：

- 全部 **1356 个 TP 参数梯度张量**有限，整体梯度相对 L2 **0.466676%**，norm ratio **0.998583**。
- 全部 token 的完整词表 CE 最大绝对差 **0.0473442**。
- 每 TP rank 采样 64 个词表列、覆盖全部 token 的 logits 最大相对 L2 **0.493570%**；没有导出全部词表 logits。
- 通过此前对整模型设定的容差：整体梯度 <3%、scale-fit 偏离 <2%、采样 logits <2%、逐 token CE 绝对差 <0.05。没有放宽这组既有门限。

`tp2-pp1-cp4-8k-measured/rankN.json` 对第一 microbatch 的全部 24 层、36 个 head 做相同输入/相同上游导数的 CP4/CP1 WKV 回放，共 192 个分片记录：

- 严格门限为输出相对 L2 <0.1%、每路输入梯度相对 L2 <1%。**99/192 个分片记录至少一项超限。**
- 最大输出相对 L2 **0.674422%**：零起始层 4，global rank6；最大绝对差 0.0029296875。
- 最大单路输入梯度相对 L2 **1.206028%**：零起始层 13，global rank0，`k` 梯度；最大绝对差 0.375。
- 所有回放输出、梯度仍有限。第一次启用严格断言的 `tp2-pp1-cp4-8k` 因超限退出；后续 `-measured` 用例继续完整采集并训练，保留 `passed:false`，没有修改原门限。

补充控制 `replay_affine.py` 使用已有真实层 0/6/12/23 的各两个 head、序列4096，四 rank 比较默认与 FLA `use_tf32x3_affine_chain=True`：默认最大输出/输入梯度相对 L2 为 0.822617%/0.462992%，tf32x3 为 0.906330%/0.457822%。它未消除误差，不能认定单独打开此选项即可解决，也不能宣称所有剩余差异均已证明仅来自累加顺序。后续定位已确认矩阵链默认TF32精度，以及本地BF16状态递推不能被固定affine映射精确表示，见 [CP4误差定位报告](/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/training-smoke/audit/cp4-error-source-20261021/REPORT.md)。诊断改传真实前后向状态后，所选四层WKV输出/输入梯度逐位一致，整模型梯度误差0.466676%→0.0169253%；生产FLA算法仍未修改。

## CP / CP+SP 保存恢复

两组均为真实 2.7B、TP2/普通PP2/CP2、seq4096、Adam、torch checkpoint，分别 SP关闭和开启。

连续训练至第 4 步，在第 2、4 步保存。新进程显式 `ckpt_step: 2` 恢复，继续第 3、4 步；恢复读取的是第 2 步而非最后的第 4 步。

`resume-sp{0,1}-comparison.json` 在全部 8 个 rank 上严格比较：

- 第 3 步开始前、以及第 4 步完成后的全部模型参数 SHA256；
- 所有实际归属本 rank 的 FP32 主权重、Adam tensor states（含 `exp_avg/exp_avg_sq`）、optimizer group 配置和 scheduler 状态；
- 第 3、4 步实际数据/输出 hash、完整精度 loss、grad norm、跳步计数及对应验证结果。

**两组均逐项相同。** 该结论限定于相同拓扑、Adam 和当前 torch 格式；没有验证改变 CP/TP/PP 拓扑恢复、Muon/FP8 的 CP 保存恢复、故障注入或长跑。

VPP 代码撤回后，又运行 `ordinary-regression-sp1` 四步普通 PP+CP+SP。`ordinary-regression-comparison.json` 验证它与先前 SP开启连续运行的第 3、4 步事件及前述完整状态 hash 全部一致。相关 CP loss、PP payload、token layout、静态合同回归 **45 passed**，见 `regression-tests.log`。

## FP8 / Muon 的验证边界

全部 8 个 rank 均确认 FFN 的 `TELinear` 启用真实 Float8 GEMM，前向与反向都执行；TimeMix/WKV 的输入始终是 BF16。tensorwise 与 delayed 分别实跑，未测试 FP8 `scope: all`。

Muon 使用兼容布局 `use_layer_wise_param_layout:false`、`muon_tp_mode:duplicated`，逐参数断言矩阵使用 Muon、其余使用 Adam；组合记录含 CP/TP 副本时共 1344 个 Muon 路由、1368 个 Adam 路由。没有宣称新布局 `true` 已验证。

这里只认证上述短步功能和执行路径。FP8 相对 BF16 的训练精度、长期收敛，以及此前 delayed+DP 的保存恢复统计问题，不能由这次 CP2/DP1 短跑推导为已解决。

## 复现

在仓库根目录，使用新实验的 checkpoint 目录运行，保留现有证据：

```bash
# 所有配置均为新副本，未改用户原配置/模型/数据。
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/audit/cp-validation-20261021/prepare.py
CP_VALIDATION_WKV=1 bash training-smoke/audit/cp-validation-20261021/launch.sh tp2-pp1-cp2 4
CP_VALIDATION_WKV=1 CP_VALIDATION_WKV_ENFORCE=0 bash training-smoke/audit/cp-validation-20261021/launch.sh tp2-pp1-cp4-8k-measured 8
bash training-smoke/audit/cp-validation-20261021/launch.sh tp2-pp1-cp4-sp-16k 8
# checkpoint 连续/恢复运行：对 SP0、SP1 分别执行。
CP_VALIDATION_STATE=1 bash training-smoke/audit/cp-validation-20261021/launch.sh resume-sp0-continuous 8
CP_VALIDATION_STATE=1 bash training-smoke/audit/cp-validation-20261021/launch.sh resume-sp0-restored 8
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/audit/cp-validation-20261021/compare_resume.py 0
# 完整已有证据复核，不重新占用 GPU。
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/audit/cp-validation-20261021/summarize.py
```

大型 checkpoint/梯度/trace 保留在本地并由 `.gitignore` 排除。VPP 历史试验文件只用于追溯，当前启动矩阵不包含交错 PP。
