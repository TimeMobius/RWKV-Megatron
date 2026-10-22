# TP2×普通PP4×CP1，全GPU Adam短跑（2026-10-22）

结论：真实28B在8×H800上完成30步训练、每10步固定验证和最终验证，退出码0。没有OOM、NaN或跳步；无需CPU offload或余数主权重存储。不过当前GB2配置比TP4/PP2显存更高、每步更慢。

## 配置与范围

- Conda `train_mobius_py312`，主仓库、Megatron、FLA均为`2026-10-10`分支；沿用当前未提交的工作区代码。
- 模型：`/hikscale/models/XiaoKe/rwkv7_28b_88_mapped.pt`，88层、hidden5120、80 heads。
- 数据：`/mnt/data/Datasets/0056_qwen3.8/0056_qwen3.8_text_document`，95/5划分，seed1234。
- TP2、普通PP4（每stage 22层）、CP1、DP1，SP开启，seq4096、MB1、GB2，BF16计算、full uniform recompute（每组1层）。
- FP32梯度累积、FP32主权重和Adam m/v；`optimizer_cpu_offload=false`、`optimizer_offload_fraction=0`、`use_precision_aware_optimizer=false`。
- 全部8rank的step30审计存储都在CUDA，没有int16主权重余数。日志的默认`store_param_remainders=True`不表示启用；本次走普通FP32主权重路径。
- 相对于之前TP4/PP2基准，YAML只改变TP=4→2、PP=2→4。没有添加限制NCCL、逐步empty_cache等补救设置。
- 本轮只新增配置、启动脚本、统计和文档，没有修改生产计算代码。短测未启用W&B、checkpoint，也没有验证长期稳定性或该布局的保存恢复。

## 实测结果

| 指标 | TP4/PP2基准 | 本次TP2/PP4 |
|---|---:|---:|
| 训练步数 | 30 | 30 |
| step>2耗时中位数 | 2071.20 ms | 2400.35 ms |
| PyTorch审计allocated峰值 | 67.9824 GiB | 71.5645 GiB |
| nvidia-smi每秒采样整卡峰值 | 75829 MiB / 74.05 GiB | 79969 MiB / 78.09 GiB |
| 最后一步训练CE | 7.755666 | 7.758949 |
| step30固定验证CE | 7.497223 | 7.499003 |

本次每步耗时增加约15.89%。两次均为相同GB2/seq4096，因此每步处理token数相同，但仍是先后进行的短跑对比，未做通信/流水线分项profiling。普通PP4配合仅2个microbatch会有流水线气泡，不能把耗时差全部归因于某一项。

逐卡显存采样峰值：rank0–7分别为78449、78541、67003、66963、67003、66963、79969、79849 MiB。

首尾stage各rank常驻存储约66.1552 GiB，中间stage约55.4983 GiB。尽管各stage均有22个RWKV层，首尾还分别承担embedding/output；本次首尾每个TP rank约39.463亿参数，中间约33.106亿。因此模型切分总份数相同不代表各卡显存峰值相同；当前首尾stage已接近整卡容量。

## 数值观察的边界

30步日志loss最大绝对差为0.003757，最终验证CE差为0.001780。并未做全参数梯度张量或严格数值等价验证。

日志grad norm差异多数小于几个百分点，但step26为21.373（TP4/PP2）与27.448（TP2/PP4），相对差28.42%；第一步分别480.600、481.104。该差异来源未定位，不应据本次loss接近就认定两种布局梯度等价或把差异全部归结为累加顺序。

后续定位（2026-10-22）：见 [TP/PP梯度范数定位报告](../tp-pp-gradnorm-20261022/REPORT.md)。独立范数/输入一致性、固定权重交叉重放、PP控制及WKV head切分控制已完成；指向TP BF16数值路径在首token少数近抵消head上的放大。该差异影响实际梯度裁剪，生产精度优化与完整梯度/收敛认证尚未完成。

## 复跑与证据

可复用配置：`configs/config-rwkv7-28b-qwen3.8-0056-tp2-pp4-no-offload-smoke.yaml`。

```bash
bash training-smoke/launch-rwkv7-28b-qwen-tp2-pp4-no-offload-smoke.sh
```

该入口生成带时间戳的逐rank审计目录，训练日志输出到终端。本次固定phase的实际命令为：

```bash
bash training-smoke/audit/28b-oom-20261022/launch.sh tp2pp4cp1-no-offload
```

本目录证据：

- `tp2pp4cp1-no-offload.log`：完整训练日志。
- `tp2pp4cp1-no-offload/rank*.jsonl`：全部rank存储和更新审计。
- `tp2pp4cp1-no-offload-gpu.csv`：每秒nvidia-smi采样，整卡峰值是采样值。
- `tp2pp4cp1-no-offload-exit.json`：退出码及整个启动耗时。
- `tp2pp4cp1-no-offload-results.json` / `summarize_tp2pp4.py`：统计及复算脚本。
- `tp2pp4cp1-no-offload-provenance.json`：源码哈希、仓库分支和HEAD。

运行结束后8张GPU显存均已释放。
