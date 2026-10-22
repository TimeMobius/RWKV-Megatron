# 28B 显存定位与 CP2 恢复训练

结论、配置和限制见 [RESULTS.md](RESULTS.md)。

- `baseline`：原 TP4/PP2/CP1/SP，10步。
- `reproduce`：TP2/PP2/CP2，无offload；预期第一次Adam更新OOM。
- `cp2-offload-before-init-fix`：直接启用offload、修复前，仍OOM；原始证据保留。
- `cp2-offload`：初始化修复后，TP2/PP2/CP2/SP、50% offload，30步。
- `cp1-control`：同TP/PP/SP/offload，CP1/DP2对照10步，train_iters30保证数据索引相同。
- `cp2-checkpoint` / `cp2-resume`：step3保存、从新进程继续4/5步。
- `entry.py`：读取storage/memory并抽样记录主权重更新，不改模型/优化器计算。
- `summarize.py` / `summary.json`：日志统计和同TP对照。
- `provenance-before-init-fix.json` / `provenance.json`：修改前/后源码哈希与仓库信息。
- `megatron-deferred-offload.patch`：Megatron生产修复，可供审阅。
- `optimizer-tests.log`：新增初始化/更新回归及已有相关检查。

本目录短测不启用W&B，不自动启动半天/一天训练。正式新脚本默认启用在线W&B。

- `tp2pp4cp1-no-offload`：真实28B、TP2/普通PP4/CP1/SP、seq4096、MB1/GB2，全GPU FP32 Adam主权重/矩，30步及固定验证通过；报告见 [TP2-PP4-CP1-NO-OFFLOAD.md](TP2-PP4-CP1-NO-OFFLOAD.md)。
