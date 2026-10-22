# TP4×PP2×CP1，不使用 Adam CPU offload（2026-10-22）

结论：当前代码可完成真实28B的30步训练和固定验证，不需要Adam CPU offload。

- 环境：`train_mobius_py312`，分支`2026-10-10`，8×H800 80GB。
- 28B原始权重：`/hikscale/models/XiaoKe/rwkv7_28b_88_mapped.pt`，88层/hidden5120。
- 数据：`/mnt/data/Datasets/0056_qwen3.8/0056_qwen3.8_text_document`。
- TP4/PP2/CP1，SP开启，seq4096，MB1/GB2，BF16，Adam，FP32梯度/主权重/矩，full recompute。
- 显式`optimizer_cpu_offload: false`、`optimizer_offload_fraction: 0.0`、`use_precision_aware_optimizer: false`；启动日志确认生效。
- 全部8rank的主权重及Adam一二阶矩均在CUDA；无CPU优化器状态。
- 30步、每10步固定验证及结束验证通过，退出码0，loss/grad norm有限，零NaN/跳步/OOM。
- 稳态耗时中位数（step>2）：2071.20ms/步。
- PyTorch训练审计max allocated：67.9824GiB/卡。
- nvidia-smi每秒采样最高：75829MiB/卡（74.05GiB，包含reserved/非PyTorch开销）。
- 末步训练CE：7.755666；step30固定验证CE：7.497223。

模型、梯度和优化器常驻tensor共64.1105GiB/卡，因此该布局可以使用全GPU Adam。
此结果适用于上述SP/序列/批量/重计算配置，是短跑验证；本次未另做该布局的checkpoint或长期验证。
没有修改生产代码；短测关闭W&B及checkpoint。

复跑：
```bash
bash training-smoke/audit/28b-oom-20261022/launch.sh tp4pp2cp1-no-offload
```

配置：`config-tp4pp2cp1-no-offload.yaml`；完整日志：`tp4pp2cp1-no-offload.log`；逐rank显存/状态：`tp4pp2cp1-no-offload/rank*.jsonl`；采样：`tp4pp2cp1-no-offload-gpu.csv`；统计：`tp4pp2cp1-no-offload-results.json`。
