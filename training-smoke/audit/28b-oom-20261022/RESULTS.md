# 28B OOM 定位与恢复训练（2026-10-22）

环境：`train_mobius_py312`，8×H800 80GB；主仓库、Megatron 和 FLA 分支均为 `2026-10-10`。
真实权重 `/hikscale/models/XiaoKe/rwkv7_28b_88_mapped.pt`；88层/hidden5120/80heads/FFN17408。
数据 `/mnt/data/Datasets/0056_qwen3.8/0056_qwen3.8_text_document`，tokenizer `/hikscale/models/Qwen/Qwen3.8-27B`。
全部训练 BF16，MB1/GB2，seq4096，full recompute，AdamW语义，lr1e-6，FP32梯度/主权重/一二阶矩。

## 原因一：八卡中用 CP 替换了部分 TP，常驻显存超过容量

原始可运行配置为 TP4/PP2/CP1/SP；首次失败配置为 TP2/PP2/CP2/SP关闭。
每卡物理参数数目分别为3,824,343,040和7,256,913,920（包含TP复制参数，不等于真实模型总参数除以卡数）。
CP复制模型参数；FP32梯度buffer仍保留完整的本TP/PP分片。
分布式优化器的DP+CP分片正常生效：实测组大小从1变为2，不是漏了CP优化器分片。

| 常驻tensor，GiB/卡 | TP4/PP2/CP1 | TP2/PP2/CP2，无offload |
|---|---:|---:|
| BF16参数buffer | 7.1234 | 13.5171 |
| FP32梯度buffer | 14.2468 | 27.0341 |
| FP32主权重 | 14.2468 | 13.5171 |
| FP32 Adam一阶矩 | 14.2468 | 13.5171 |
| FP32 Adam二阶矩 | 14.2468 | 13.5171 |
| 合计 | **64.1105** | **81.1023** |

表格由真实buffer/storage统计；CP2矩状态尚未全部分配即OOM，其完整需求由已确认的参数shard大小计算。
单卡容量为79.17GiB，尚未计入激活、临时tensor、CUDA/NCCL等开销就已超限。
因此缩短序列、仅打开SP、empty_cache或改碎片策略都不能使原CP2配置的完整常驻tensor装入。
当前代码重跑原TP4/PP2/CP1完成10步+固定验证；CP2原布局再次在第一次Adam状态分配时OOM，所有8rank均捕获。
证据：`baseline.log`、`baseline/rank*.jsonl`、`reproduce.log`、`reproduce/rank*.jsonl`。

## 原因二：offload初始化先分配完整主权重，再重建分片优化器

首次直接启用50% CPU offload仍在首次Adam更新OOM。
`_get_megatron_optimizer_based_on_param_groups`首先用未分片参数构造HybridDeviceOptimizer，
其中立即创建FP32 masters和带闭包的hook；DistributedOptimizer完成参数分片后再创建第二个HybridDeviceOptimizer。
第一个实例被hook引用环暂时保留，直到Python cyclic GC触发才释放；实际多占约13.48GiB GPU主权重（另有无用CPU副本）。
实测rank0 optimizer_built allocated：修复前65,137,490,432 bytes，修复后50,665,216,000 bytes，减少14,472,274,432 bytes。

修复：HybridDeviceOptimizer增加内部defer_init构造选项，分布式优化器工厂先只建参数组/defaults，
真正的主权重、子优化器和hooks在既有的分片后重建步骤中初始化。
此外补齐offload续训路径：dummy_step只分配状态，零学习率/零梯度且禁止compute-weight回写，避免旧主权重覆盖已经加载的模型；同步超参数时不再用外层checkpoint step覆盖子优化器正在推进的Adam计数。
非分布式构造默认仍立即初始化；defer_init不写入defaults或checkpoint，分片后的实例不继承deferred状态。

修改位于Megatron子模块：
- `megatron/core/optimizer/__init__.py`：分布式offload使用defer_init。
- `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`：延迟分配。
- `megatron/core/optimizer/distrib_optimizer.py`：DP-zero CPU scatter后重绑Hybrid子优化器状态，恢复GPU moments的设备归属。
- `megatron/core/safe_globals.py`：注册Hybrid checkpoint中保存的已知CPU/GPU优化器类，兼容PyTorch weights_only默认加载。
- `tests/unit_tests/optimizer/test_cpu_offload_initialization.py`：GC关闭时无额外CUDA分配、旧实例立即释放、只建owned shards的FP32 master，offload比例0/0.5/1的两步更新与纯GPU FP32参考一致，工厂实际传入defer_init。

修复前offload失败原始证据保留在 `cp2-offload-before-init-fix.log` 和同名目录。

## 已完成实跑

| 配置 | 训练/验证 | 结果 | 稳态步速中位数 | 训练审计CUDA allocated峰值 |
|---|---|---|---:|---:|
| TP4/PP2/CP1/SP，无offload | 10步+固定验证 | 无NaN/跳步/OOM，末步CE8.940693，验证CE8.548809 | 2.077s | 67.982GiB |
| TP2/PP2/CP2，无offload | 首次更新 | Adam状态分配OOM | — | 分配未完成 |
| TP2/PP2/CP2/SP，50%offload，初始化修复 | 30步+每10步固定验证 | 无NaN/跳步/OOM，首步CE12.42638，末步CE7.762780；验证CE8.548290→7.841588→7.502898 | 4.530s | 64.696GiB |

时间统计为step>2的日志中位数；每步global tokens=8192。
表内峰值是审计点捕获的PyTorch训练max_memory_allocated，不含NCCL等非PyTorch显存；`gpu.csv`保留每秒nvidia-smi采样，reserved与实际显卡占用会更高。CP2的nvidia-smi采样峰值为79595MiB/卡（约77.73GiB）；分阶段逐卡峰值见`gpu-summary.json`。
第一行和第三行改变了TP布局、CP、offload，不能用它们的耗时差直接量化CP开销，或视为严格数值等价对照。
30步短跑是功能/有限数值验证，不能替代长期收敛测试。

修复后的rank0常驻GPU tensor共60.4293GiB；CPU主权重和两组矩共20.6731GiB（不含CPU梯度副本、缓存、模型加载峰值）。
CPU/GPU各自抽样FP32 master在首次更新后最大绝对变化均约9.9e-7，确认两个部分均实际更新。

## 追加验证状态

同TP2/PP2/SP、相同50% offload、相同global batch2/seq4096、相同train_iters30数据索引与seed：
CP1使用DP2、每rank一个microbatch；CP2使用DP1、每rank两个microbatch。
前10步日志loss最大绝对差0.001407，grad norm最大相对差0.283540%；step10固定验证CE为8.549002与8.548290，差0.000712。
CP1对照step>2中位4.0576s，CUDA allocated训练审计峰值66.9799GiB。
该验证只比较日志loss和梯度范数，没有捕获28B全部参数梯度张量，也不代表严格逐位等价。

Step3完整checkpoint保存成功，含模型、Adam及RNG；保存耗时218.085s。
路径：`/hikscale/models/RWKV-28B/rwkv7-28b-cp2-offload-validation-20261022/iter_0000003`，4组TP/PP文件，逻辑数据约406GB。
首次恢复暴露Hybrid参数组携带AdamW/FusedAdam类但weights_only allowlist缺失的问题，保留失败日志`cp2-resume-before-safe-globals-fix.log`；已补已知类注册并通过序列化回归。
最终使用新正式启动脚本恢复成功，完整加载optimizer/RNG/data进度，在step4/5完成Adam更新，退出码0，无NaN/跳步/OOM。

| Step | 连续训练CE | 恢复训练CE | 连续/恢复grad norm |
|---|---:|---:|---:|
| 4 | 9.976159 | 9.976159 | 159.442 / 159.442 |
| 5 | 9.748903 | 9.748903 | 134.133 / 134.133 |

独立保存运行的step1/2/3同样与连续训练日志一致。
上述28B对照精度仅为日志精度（CE约7有效数字、grad norm小数3位），没有声称28B全部模型/Adam/RNG张量严格等价；全张量逐元素续训对照由前述两rank小模型回归提供。
原始v3恢复日志`cp2-resume-before-copyback-fix.log`记录了首步异常；这是状态初始化把构造时旧masters回写覆盖已加载模型造成，最终dummy状态初始化完全禁用回写后消除。
`cp2-resume-before-state-sync-fix.log`保留GPU Adam读取CPU矩状态引起的非法访问，最终DP-zero load增加Hybrid同步后消除。
两步小模型精确对照还暴露并修复了恢复后GPU Adam step反复被外层checkpoint step覆盖的问题；同步超参数时现跳过运行时计数。

优化器及序列化相关回归：单rank 22 passed / 3 skipped（新增CP2测试与既有layer-wise Muon需多rank）；新增测试另外以两rank运行5 passed/rank，包含完整保存恢复的两步精确对照。
以单rank torch.distributed.run运行，--noconftest并通过local_test_fixtures.py提供单rank临时checkpoint目录，避免与本次无关的全局测试数据下载fixture；新测试不依赖外部数据。
覆盖Hybrid checkpoint的weights_only序列化往返、已知类型加载、未知类型拒绝及safe numpy/pickle加载。两rank回归按真实加载顺序（先创建optimizer，再加载model）执行，检查compute weights及CUDA RNG加载前后保持一致，连续两步的BF16权重、FP32 masters和Adam两组矩全部逐元素相同；同时检查GPU状态归属设备正确。
完整日志 `optimizer-tests.log`。未修改其他审计工作的ISSUES.md。

## 配置与复跑

新配置 `configs/config-rwkv7-28b-qwen3.8-0056-cp2-offload.yaml`，新脚本 `training-smoke/launch-rwkv7-28b-qwen-cp2-offload.sh`。
默认开启W&B、TensorBoard和checkpoint，使用新的输出目录；脚本拒绝将新任务写入非空checkpoint目录。
正式模板沿用一天时限；本次只实际运行audit目录中的短测配置。

```bash
bash training-smoke/launch-rwkv7-28b-qwen-cp2-offload.sh
```

短测：
```bash
bash training-smoke/audit/28b-oom-20261022/launch.sh baseline
bash training-smoke/audit/28b-oom-20261022/launch.sh reproduce  # 预期OOM
bash training-smoke/audit/28b-oom-20261022/launch.sh cp2-offload
bash training-smoke/audit/28b-oom-20261022/launch.sh cp1-control
```

原始成功/失败日志均保留；`summarize.py`生成`summary.json`。
本次尚未commit/push；Megatron修复必须与新增配置一同使用。

## 用户指定布局追加复验：TP4/PP2/CP1，无offload

2026-10-22再次显式关闭offload，当前最终代码完成30步和固定验证；全部8rank主权重及Adam两组矩均在CUDA。详细结果见`TP4-PP2-CP1-NO-OFFLOAD.md`。

## CP2全GPU追加试验

2026-10-22通过seq2048、TE主权重余数存储及受限NCCL配置完成30步，完全关闭CPU offload；FP32梯度及Adam矩保留。seq4096多种设置仍OOM，不能将该试验视为原配置直接关闭offload成功。详见`CP2-NO-OFFLOAD.md`，此路径checkpoint/长期收敛尚未认证。
