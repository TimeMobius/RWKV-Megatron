# W&B 指标补传与后续配置（2026-10-16）

按用户偏好，项目的 Mobius workspace 已将 28 个训练/性能/验证图表统一放入
`Charts` 栏目，保留原有 System 和 Hidden Panels 设置；关闭按指标前缀自动生成栏目。
指标名称和历史数据不改。关键 loss/梯度/学习率/吞吐量/步耗时排在前面。
展示布局由 `tools/consolidate_distill_workspace.py` 管理，保存前后的 JSON 备份位于
`training-smoke/qwen-rwkv-qwenvocab/workspace/`；新指标需要加入展示时可以重跑该脚本。

原先只设置了 `log_throughput: true`，Megatron 因此在 stdout 打印吞吐量。
W&B 的 `throughput` 和 `iteration-time` 还受 `log_timers_to_tensorboard: true`
控制，这个开关未开启，造成训练曲线里缺少性能指标。

已给可复用的 single、long、async-single 配置开启 timer 和 memory logging。
历史运行专用配置保持原样，表示其真实执行配置。没有修改 Megatron Core 或 FLA。

对原长跑 `ter66hd1`（197 步）、最终同步短测 `m7cckbse`（16 步）和最终异步短测
`mcwrilzp`（16 步），从原始日志恢复以下曲线，追加到同一个 W&B run：

| 指标 | 含义 |
| --- | --- |
| `throughput` | 日志中的 Megatron 估算 TFLOP/s/GPU，原日志精度为 0.1 |
| `iteration-time` | 每次更新的秒数，含该次等待 teacher 的时间 |
| `throughput/tokens_per_sec` | global batch × 输入序列长度 ÷ iteration-time，训练区间吞吐，排除验证 |
| `throughput/samples_per_sec` | global batch ÷ iteration-time |
| `train/consumed_samples`、`train/consumed_input_tokens` | 实际消费样本及输入 token 数 |
| `train/nan_iterations`、`train/skipped_iterations` | 原训练日志里的对应计数 |
| `distill/kl_loss` | CE 权重为 0 时从训练目标恢复的 KL |
| `memory/*_mib` | 仅 stdout 中实际记录步上的 PyTorch allocator 快照 |

补传曲线使用 `train/iteration` 作为横轴。W&B 不能回填过去的内部 `_step`，所以追加
新 history 行，并为这些新指标设置真实训练步数轴；原来的 loss/grad 等 history 保留。
不要用补传产生的内部 `_step` 或 wall time 计算训练耗时。每个 run 的 config 有
`metrics_backfill` 来源/单位说明，并有 `metrics-backfill` artifact 保存恢复明细及日志 SHA256。
恢复脚本为 `tools/backfill_distill_metrics.py`，相同源日志重复执行时会跳过已补传的数据。

异步 run 的 `teacher/final_*` summary 补入最终 submitted/consumed/completed/failed/
active/peak_active/buffered 统计。这是最终快照，不是逐步 teacher 延迟或并发曲线。

W&B 的 System 页原本已有部分运行的 GPU 使用率、显存、功耗等系统采样；短测的
System 数据不保证完整。它与这里的 PyTorch allocator 显存指标口径不同。

后续在线训练通过 loss reporting 新增 `teacher/wait_seconds`、`teacher/*` 预取快照、
`memory/allocated_mib`、`memory/max_allocated_mib`、`memory/reserved_mib`、
明确命名的 `distill/kl_loss`；启用 CE 混合时也会记录 `distill/ce_loss`。
这些报告使用 Megatron sum/count 协议，随原来的 loss/梯度曲线上传，不额外启动训练。
`teacher/wait_seconds` 只表示 student forward 结束后等待 teacher 结果的主线程时间，
不等于后台 HTTP 请求的完整耗时。

没有采集的历史逐请求延迟、逐步预取统计、真实 CE/PPL 无法补回。当前 loss 是 KL，
Megatron 打印的 exp(KL) 不能作为语言模型困惑度。TFLOP/s 仍是 Megatron 的理论估算，
没有实现针对 RWKV 的 FLOPs 公式校准；比较蒸馏速度优先看 tokens/s 和含预热/验证的总耗时。

验证：24 项原异步/API/固定验证测试通过；另有 3 项日志恢复测试和 1 项 telemetry
loss/梯度不变测试通过。此次仅补传指标和修改后续日志配置，没有恢复模型训练。
