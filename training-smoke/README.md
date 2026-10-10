# RWKV 3B 上游更新与适配验证

> 2026-10-10再次将FLA更新到40850297（29个提交），仍为上游原样。独立公式、原算子容差与短训练检查通过；初始前向逐元素一致，BF16后向/训练轨迹有已定位差异。见[audit/FLA-UPDATE-20261010.md](audit/FLA-UPDATE-20261010.md)。历史表不能认证此更新的性能或续训。

> 后续已统一 RWKV7 的 ln0、ln1/ln2、ln_out：TE>=1.9.0 时使用 TE，否则统一回退 Apex。无需旧 checkpoint 兼容开关。变更与新验证见 [audit/TE-UNIFICATION.md](audit/TE-UNIFICATION.md)。下面的历史单卡/性能结果不能替代新后端验证。

> 2026-10-09 单卡公式审计发现并修复了 FLA 点算子的 token 布局/分派问题。下面的历史 30 步性能和 loss 表均为该修复之前的结果，不能视为当前代码的重新认证。单卡结论与复现见 [audit/SINGLE-GPU.md](audit/SINGLE-GPU.md)，待查问题见 [audit/ISSUES.md](audit/ISSUES.md)。本轮暂停多卡排查。

所有工作位于 `/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test`。原仓库 `RWKV-Megatron-newest`、原环境 `train_mobius` 的软件包，以及原始配置、启动脚本、数据、tokenizer 和模型权重均未修改。参考配置与启动脚本完整复制在 `reference/`，原仓库已有的用户改动也保留原样。

## 冻结的更新版本

- FLA：`3cf180339b8a1cbad823f553541cd531d18670ea` → `79d12e8ec0ae72bc7f2189b4244d6532ce8c268c`。
- Megatron：原分支 `2cb1b2e172b035cd2f4038b14c8d9f5ecfa1125c` → NVIDIA 上游 `d113016bb2850c4b7d804f24ba72c055d7d4f861`，附带本工作区的兼容补丁。旧版回放包含 vocab 宽度修复，快照版本为 `c11ea0b3cd8121a289b27985fdcd402322444996`。
- 新环境：`/mnt/data/anaconda3/envs/train_mobius_py312`，Python 3.12.14、Torch 2.9.1+cu128、Triton 3.5.1、TE 2.19.0、Apex CUDA/CPP 扩展。

这些是本次测试冻结的目标提交，不代表远程仓库当天的最新提交。2026-10-10按用户要求在主仓库和两个子模块建立同名本地分支并保存提交，尚未推送。分支、远程、提交身份见 `COMMIT-STATE.md`；补丁和恢复说明见 `patches/README.md`。

## 适配内容

1. 新版 Megatron 训练入口、参数容器、model provider、显式 TP/CP 批次广播与 loss 汇总。
2. 独立 RWKV TransformerLayer 承载层零 value residual，并保留跨层、跨 PP 的梯度；PP 通信尺寸按 RWKV carrier 扩展，普通 GPT 尺寸保持原契约。
3. TimeMix、ChannelMix 构造与调用签名，MLP spec，以及新版全局层编号和 TE API。
4. 恢复旧分支的多数据源 exhaustive blend shuffle 功能。
5. vocab 按最高 token ID 加一计算，避免 HuggingFace 重复注册特殊 token 导致 embedding 变大。
6. tokenizer 的 PAD 与 BOS 共用 65532，EOS/EOD 为 65531。新版包装器未暴露 special_tokens_dict 时需补查 BOS，使用 -1 padding 哨兵，保留真实 BOS 输入和标签。

词表文件确实有 **65,534 个条目**，最高 ID 为 65,534；ID 索引需要 65,535 行，按 128 对齐后为 **65,536**，与 checkpoint 一致。HF `len()` 的 65,539 不作为 embedding 尺寸依据。原 tokenizer 文件未改。

## 固定性能与数值测试

- 权重：`/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth`，约 2.85B 参数。
- 数据：`/mnt/data/Datasets/minipile/minipile_text_document`；tokenizer 为 `/hikscale/models/XiaoKe/Xiaoke-5-13B-2608`。
- 8 张 H800；TP1、PP2、DP4；BF16、完整层重计算、分布式优化器；FP8 关闭。
- seq4096、microbatch1、globalbatch8，32,768 token/步；seed1234。
- 训练 30 步，第 15/30 步及训练结束后各验证 5 次；不保存 checkpoint，不连接 W&B/TensorBoard。
- 吞吐排除前 5 步；总 TPS = 32,768 / 平均单步秒数。TFLOP/s 是 Megatron 的 Transformer 架构估算，不用于 RWKV 实际算力结论。

| 版本（各 2 次） | 平均单步 ms | 平均总 TPS | 相对旧版 | 两次 TPS 波动 | 验证 loss：15 / 30 / 最终 |
| --- | ---: | ---: | ---: | ---: | --- |
| 旧版（词表宽度修复） | 743.312 | 44,084 | +0.00% | 0.14% | 1.912290 / 1.966091 / 1.981133 |
| 仅更新 FLA | 718.994 | 45,575 | +3.38% | 0.01% | 1.902394 / 1.967026 / 1.997927 |
| FLA＋Megatron＋新环境 | 720.010 | 45,511 | +3.24% | 0.33% | 1.903288 / 1.962515 / 1.984971 |

所有六次运行均 exit 0、零 NaN、零跳步；每组两次的训练和验证 loss 在日志精度下完全一致。全量更新平均 TPS 相对 FLA 单独更新为 -0.14%，没有测出额外提速。

相对旧版，全量更新最大逐步训练 loss 绝对差为 0.029142；验证 loss 差（15 / 30 / 最终）为 -0.009002 / -0.003576 / +0.003838。仅更新 FLA 的最大逐步差为 0.011524，验证差为 -0.009896 / +0.000935 / +0.016794。这些是数值差异，不能从短跑推断长期精度相同。


## 已完成的稳定性检查

- 新版 TP1/PP2：两次完整 30 步训练、验证与 launcher 正常退出；逐步训练 loss 和三次验证 loss 在日志精度下完全一致，零 NaN、零跳步。
- 新版 TP2/PP2/DP2、sequence parallel：10 步训练、每 5 步验证 2 次，exit 0，零 NaN、零跳步。日志为 `logs/upstream-tp2-pp2.log`。不同并行配置的吞吐不与固定 TP1 工作负载混比。
- checkpoint：`torch` 格式、第 2 步保存并退出，加载模型/优化器/RNG/进度后继续第 3/4 步；与不中断的 4 步训练比较 loss、梯度范数、LR、样本进度，全部在日志精度下相同。3 次启动均 exit 0，见 `results-resume.json`。检查点保存在 `checkpoints/resume-test/`，约 77GB（第 2 与第 4 步）；重跑保存会更新本测试目录。
- 旧/新版 train 和 valid 的 document/sample/shuffle 共六组索引数组完全相同，见 `results-data-alignment.json`。
- 回归测试：33 passed。覆盖 carrier 梯度、最终 PP 才移除 carrier、TP/SP 尺寸、vocab/PAD-BOS 兼容，以及多源 blend shuffle 的样本守恒、缓存与随机种子。
- `pip check`、shell 语法与 Git diff 空白检查通过；原配置、脚本和原仓库用户文件的校验和通过。

常规回归命令排除 `tests/test_rwkv7_bottleneck_tp_batchmix_repro.py`：该文件是**修复前的故障复现器**，文件说明明确要求修复后失败；已有修复后回归测试验证 batch mixing 被消除。未删除它或放宽测试容差。

## 复现

从工作区根目录执行（GPU 测试串行运行）：

```bash
bash training-smoke/launch-3b.sh my-upstream-run
bash training-smoke/launch-3b.sh my-tp2-run training-smoke/config-3b-tp2.yaml

/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/summarize.py \
  training-smoke/logs/baseline-warm.log training-smoke/logs/my-upstream-run.log

PYTHONPATH=.:tests /mnt/data/anaconda3/envs/train_mobius_py312/bin/python -m pytest -q \
  tests training-smoke/test_compatibility.py \
  --ignore=tests/test_rwkv7_bottleneck_tp_batchmix_repro.py
```

`launch-3b.sh` 使用新的 `--rwkv-config` 入口；旧参考脚本保留在 `reference/`。首次启动可能花数分钟进行 Triton 编译和 autotune；最终隔离缓存首轮第一步约 363 秒，此时间不代表稳态训练速度。不要在 Bash 执行期间修改它正在读取的脚本。

```bash
# 如 replay 快照不存在，先重建；存在时脚本会退出以保护现场。
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/prepare-replays.py

# 在实际 exit code 记录下依次重跑所选阶段。
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/run-validation.py \
  baseline-warm baseline-warm-repeat fla-only-warm fla-only-warm-repeat

/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/run-validation.py \
  checkpoint-save checkpoint-resume checkpoint-continuous
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/compare-resume.py

# 六次固定基准全部正常退出后生成汇总。
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/build-report.py
```

## 2026-10-09 converter 提交同步

已同步原仓库提交 `68c9991bf6e0bc787b7b6f298a81c311fed3da96`：`rwkv_conversion.py`、42 项 converter 测试、中英文 README。包含显式 `--iteration` 与直接 `iter_*` 目录选择、自动 LayerNorm gamma offset、完整 key/shape/有限值和 TP 一致性检查、原子保存。

在独立工作区补充新版 Megatron 参数名 `decoder_first_pipeline_num_layers`、`decoder_last_pipeline_num_layers` 和 `pipeline_model_parallel_layout` 的拒绝检查，保持 converter 仅支持均匀、非虚拟 PP 的限制。对应测试通过。源提交补丁在 `reference/converter-68c9991.patch`，额外兼容修改在 `patches/converter-upstream-compatibility.patch`；同步前 converter 和 README 在 `reference/converter-before-sync/`。

同步后的完整回归：**75 passed**（42 converter + 原有 33）。新版实际训练保存的 TP1/PP2 3B、iteration 4 checkpoint 已成功导出；1,059 个输出张量均与来源 checkpoint 按正确层号、shape 和 norm offset 对照，权重与 dtype 完全相同，head shape 为 `[40, 64]`。见 `results-converter-sync.json` 和 `logs/converter-real-upstream-3b.log`。

```bash
OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python rwkv_conversion.py \
  --load-dir training-smoke/checkpoints/resume-test --iteration 4 \
  --save-dir training-smoke/checkpoints/converter-sync-68c9991
```

输出 `checkpoints/converter-sync-68c9991/iter_0000004/rwkv_state_dict.pth` 约 5.90GB，位于本工作区；原 checkpoint 和原仓库未修改。测试中的 TP2/TP4 转换使用合成分片；真实 3B 转换检验的拓扑是 TP1/PP2。

## 排除的诊断运行

`upstream-adapt-08.log` 发现 PAD/BOS 冲突导致真实 BOS 被替换并掩蔽部分标签，修复前结果不纳入最终对照。早期 `megatron-upstream-30-repeat.log` 的训练完成，但执行期间修改脚本导致 Bash 尾部读取错误、exit 2，不能视为启动成功；最终独立重跑替代它。

`baseline-today-cache-error.log` 是共享 cubin 缓存失败的首次尝试；`fla-only-today-source-path-error.log` 是入口符号链接导致导入旧 FLA 的错误对照，均保留用于诊断，不纳入最终统计。旧版及 FLA 的冷启动日志也单独保留；最终比较使用 warm 与 warm-repeat。

## 结论边界与环境记录

短期 loss 对照和断点续训证明已测路径能运行并恢复；它们不能证明长时间收敛或下游任务精度相同。旧版冷启动回放与预热回放也出现了不同的确定 loss 轨迹，说明编译/autotune 条件需一并记录；最终表使用重复可复现的预热结果。

Megatron 全量更新同时更换了 Python 和 TE，速度差应视作组合变更。未验证 14B、多机、CP、交错 VPP、FP8、CUDA Graph、torch_dist checkpoint 或改变 TP/PP 拓扑后的 checkpoint 重分片。保存与恢复已验证的格式是原配置使用的 `torch`。

环境清单、构建差异和 SHA256 见 `environment/README.md` 与 `environment/provenance.json`。TE 主源码来自 Megatron 锁定提交，但使用了本机 CUTLASS 4.4.2 源码，未复现上游锁定 CUTLASS 提交；这是本地验证环境。源码、构建产物保存在忽略的 `vendor/`；日志、缓存、测试 checkpoint 均在本工作区。
