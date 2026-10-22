# CP2 不使用 Adam CPU offload（2026-10-22）

结论：存在完全不使用CPU offload的28B CP2短跑配置，但本次通过的是seq2048，并且需要改变FP32主权重的存储表示以及NCCL设置。
不能将该结论解释为“原配置直接关闭offload即可跑”，也不能将本次seq4096的失败配置标为通过。

## 成功配置与结果

- `train_mobius_py312`，8×H800 80GB；主仓库/Megatron/FLA均为`2026-10-10`分支。
- 同一真实28B：`/hikscale/models/XiaoKe/rwkv7_28b_88_mapped.pt`，88层/hidden5120/80heads/FFN17408。
- 同一0056数据：`/mnt/data/Datasets/0056_qwen3.8/0056_qwen3.8_text_document`。
- TP2/PP2/CP2、SP开启，BF16、seq2048、MB1/GB2，full/uniform逐层重计算。
- `optimizer_cpu_offload: false`、`optimizer_offload_fraction: 0.0`。
- `use_distributed_optimizer: true`、`use_precision_aware_optimizer: true`、`store_param_remainders: true`。
- FP32梯度累加，Adam一阶/二阶矩均为FP32；主权重采用TE提供的BF16参数+int16余数表示，不另存一份完整FP32副本。
- `language_model.fp8: false`；`fp8_recipe: delayed`在此仅用于使当前Megatron选择TE管理主权重的precision-aware路径，启动日志`fp8=None`确认并未启用FP8。
- `empty_unused_memory_level: 2`，每步释放未使用CUDA缓存。
- 环境：`NCCL_MAX_NCHANNELS=1`、`NCCL_MIN_NCHANNELS=1`、`NCCL_BUFFSIZE=262144`、`NCCL_NVLS_ENABLE=0`。

真实运行完成30步，step10/20/30固定验证和结束验证全部通过，退出码0。
训练日志loss/grad norm均有限，零NaN、零跳步、零OOM；8个rank均记录step30更新完成，所有记录到的tensor storage均在CUDA，没有CPU optimizer状态。

| 指标 | 结果 |
|---|---:|
| 稳态step耗时中位数（step>2） | 5669.4ms |
| PyTorch训练审计max allocated，所有卡最大值 | 77.4923GiB |
| nvidia-smi每秒采样最高，所有卡最大值 | 80797MiB，约78.90GiB |
| 第30步训练CE | 7.878817 |
| step30固定验证CE | 7.754669 |
| step30固定验证PPL | 2332.437 |

每卡容量79.17GiB，当前余量很小；改变序列、batch或通信配置后必须重新测量。
限小NCCL缓冲区及频繁empty_cache会影响性能，此次全GPU方案未表现出速度优势；这里没有做受控性能归因。
短测不保存checkpoint且关闭W&B；没有验证该余数主权重模式的checkpoint续训，也没有进行长期收敛验证。
既有50% offload保存恢复结果属于另一条优化器存储路径，不能转用为该模式的认证。

## 常驻张量显存

rank0逐storage去重统计，其他rank存储类型相同：

| 张量 | GiB/卡 |
|---|---:|
| BF16参数buffer | 13.5171 |
| FP32梯度buffer | 27.0341 |
| FP32 Adam一阶矩 | 13.5171 |
| FP32 Adam二阶矩 | 13.5171 |
| int16主权重余数 | 6.7585 |
| 合计 | **74.3438** |

完整FP32主权重方案的合计为81.1023GiB，余数表示节省6.7585GiB/卡。上述合计不包含激活、临时buffer、CUDA/NCCL等开销。
普通FP32主权重方案单常驻需求就超过单卡容量，因此仅缩短序列而不改变其存储方式仍不能装下。

## 失败尝试保留

| 试验 | 结果 |
|---|---|
| 原TP2/PP2/CP2，完整FP32主权重、无offload | 先前已复现首次Adam状态分配OOM，见`reproduce.log` |
| seq4096，余数主权重、默认NCCL、逐层重计算 | rank0等部分卡第一次更新完成，rank4/5/6/7初始化Adam仍OOM，整步未完成；`cp2-no-offload-remainder.log` |
| seq4096，余数主权重、4通道/1MiB NCCL缓冲、逐层重计算 | 完成1步，第2步前向OOM；`cp2-no-offload-low-nccl.log` |
| seq4096，余数主权重、1通道/256KiB缓冲、两层一组重计算 | 完成1步，第2步交叉熵将logits转FP32时OOM，申请970MiB、剩余约865–905MiB；`cp2-no-offload-4k-recompute2.log` |

第一轮默认NCCL试验期间GPU0曾并行启动小张量对照，内存紧张导致小对照失败；其他rank的OOM仍被完整记录。
后续内核对照及真实模型运行分开执行，成功的seq2048与最后seq4096测试期间GPU仅由该训练任务使用。
日志和统计保留了失败，不将“第一次能更新”视为多步训练通过。

## 数值限制

TE文档说明BF16参数+16位余数用于重建FP32主权重，但它调用独立的Adam kernel，不能因此宣称与普通FP32 master路径逐位等价。
`check_remainder.py`以三个尺度共196608个参数、同一固定梯度序列、lr1e-6和相同Adam设置对照100步：

- Adam一阶/二阶矩逐位相同（这是固定相同梯度的对照）。
- BF16计算权重不完全相同，最早step2在一个65536元素tensor中出现3个不同元素。
- 所有步/张量的计算权重相对L2差最大6.12827e-5，约0.00613%；每张量最多3个元素不同。

这不是完整模型梯度对拍或长期收敛对照；真实训练使用不同kernel后梯度也可能随之出现差异。
原始对照见`remainder-parity.json`与`remainder-parity.log`。

## 复跑与证据

新增已通过的短测配置：`configs/config-rwkv7-28b-qwen3.8-0056-cp2-no-offload-smoke.yaml`。

```bash
bash training-smoke/launch-rwkv7-28b-qwen-cp2-no-offload-smoke.sh
```

该脚本创建带时间戳的audit目录，明确固定上述NCCL参数；默认30步、seq2048、无checkpoint和W&B。
本目录`launch-cp2-no-offload.sh cp2-no-offload-2k`可重放原试验配置。
`launch-cp2-no-offload.sh cp2-no-offload-4k-recompute2`为已知失败的4096对照，预期OOM。

逐rank显存/optimizer dtype：`cp2-no-offload-2k/rank*.jsonl`；日志`cp2-no-offload-2k.log`；采样`cp2-no-offload-2k-gpu.csv`。
`cp2-no-offload-results.json`由`summarize.py`及`summarize_no_offload.py`生成；源码hash及环境见`cp2-no-offload-provenance.json`。
本轮仅新增配置、启动/审计工具及记录，没有修改生产计算代码；未commit/push。
