# 当前版本的 RWKV7 3B 本地参考配置

配置：`config-rwkv7-3b-reference.yaml`。对应代码基线是本工作区 `2026-10-10` 分支的 `fb5352b`，当前环境 `train_mobius_py312`，实际硬件 H800。

之前实际训练使用的是 `training-smoke/config-3b-continuous-fixed-eval.yaml`，保存/恢复分别使用同目录的 `config-3b-save-fixed-eval.yaml` 和 `config-3b-resume-fixed-eval.yaml`；FP8 验证另用名称包含 `fp8` 的副本。不是直接用原 14B YAML 启动 3B 模型。

这些配置能启动，是因为使用了新的 `--rwkv-config` 入口，模型结构已经改为 32 层、hidden2560、40 个 head，四个 LoRA 尺寸也与 3B 权重匹配。`rwkv_megatron/arguments.py` 将有效字段放到新版参数容器中，转换初始化/归一化/激活等关键字段；其他旧字段跳过，缺失项采用新版默认值。旧配置中保留无效字段不妨碍启动，但那些字段不会控制实际行为。

新模板从上述已经测试的 BF16 配置整理，仅保留当前解析器识别的字段及明确支持的转换字段。不复制 MoE、Mamba、旧 RWKV bottleneck、废弃通信开关和内部回调占位项。`deallocate_pipeline_outputs` 等由框架推导的值用注释说明，避免在 YAML 中写一个实际上被忽略的开关。

默认设置为单卡 TP1/PP1/SP关闭、BF16、seq128、micro batch1/global batch2、4步训练，每步验证4个固定样本；L2Wrap为1e-4，FP32 key合并开启，FP8关闭。LayerNorm仍优先TE、不可用时回退Apex。可以在YAML中分别关闭L2Wrap/key合并/固定eval，或按注释开启实验性的FFN FP8。

两个有意调整：激活配置明确写实际 RWKV7 使用的 `squaredrelu`；短跑改用固定学习率 `1.56e-5`，不再沿用旧试验的长 WSD 调度。因此新模板用于建立清晰的新基准，不要求第1步更新之后的loss与旧调度运行逐位一致。默认 `save: null`，不会生成大测试检查点。

启动：

```bash
cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
CUDA_VISIBLE_DEVICES=0 bash training-smoke/launch-3b-reference.sh
```

launcher从已用过的单卡脚本复制，自动激活当前环境、配置CUDA库路径、定位本工作区，使用生产 `pretrain_rwkv7.py --rwkv-config` 入口。可以通过第一个参数指定日志名、第二个参数指定另一份配置；`REFERENCE_MASTER_PORT` 可改端口。默认日志在 `training-smoke/logs/3b-reference.log`。

正式训练请先复制此配置，再设置实际训练步数、seq_length、batch、学习率及调度，开启自己的 `save` 输出目录。恢复训练时设置 `load` 为完整训练检查点目录，保留优化器和随机状态；`load_pretrained_rwkv` 是原始模型初始化权重，不是断点恢复。固定eval的可比性依赖相同的数据、tokenizer、seed、序列长度及eval批次配置。

最终配置已通过完整原始3B的4步训练和5次eval，零NaN、零跳步，未生成检查点。配置解析、实际生效参数与短跑验证结果保存在 `training-smoke/audit/3b-reference-config-validation.json` 和 `3b-reference-training-validation.json`；训练日志为 `training-smoke/logs/3b-reference-final-20261010.log`。这份模板是单卡短跑参考，修改为TP/SP/PP或长训练后仍需对应验证。

配置和launcher中的绝对路径对应当前本地环境；换机器或工作区时需修改环境、仓库、模型、数据及tokenizer路径。
