# Qwen 词表 RWKV7 2.74B 单卡蒸馏测试

后续已完成 16 路 teacher 异步预取，24 项测试与完整模型冻结响应精度检查通过。
最终同步/异步各 16 步短测均正常退出、不保存 checkpoint；包含预热和验证的耗时
分别为 121.95 秒与 90.19 秒（1.35 倍）。实现范围、复跑方式和报告见 [ASYNC.md](ASYNC.md)。

## JSONL 文本输入

蒸馏也支持标准 JSONL 文本数据：每行默认读取 `{"text": "..."}`。例如
`/mnt/data/Datas/minipile/jsonl` 可以直接作为 `data_path`：

```yaml
data_path:
  - /mnt/data/Datas/minipile/jsonl
rwkv_jsonl_text_field: text
rwkv_jsonl_max_records: 128   # 测试时限制记录数；0 表示不限制
rwkv_jsonl_chunk_chars: 512
```

JSONL 数据先从原始文本切分，再直接用 Qwen tokenizer 编码，并把原始文本随 batch
保留下来。teacher 请求使用这段原始文本，student 使用同一次 tokenizer 得到的 token
ID，因此不经过 token ID decode，不会触发 packed BIN/IDX sample 从 UTF-8 或 byte-level
token 中间开始造成的 round-trip 问题。短文本右侧 padding，padding 位置的 loss mask 为 0。
测试配置是 `configs/config-rwkv7-qwen-distill-jsonl-minipile-test.yaml`。JSONL 分支不启用
旧的 round-trip filter；BIN/IDX 分支仍保持原来的整 sample round-trip 过滤策略。

从 `/hikscale/models/XiaoKe/rwkv7_28b_88_mapped.pt` 的前 24 层取子张量，生成
`/mnt/data/Models/RWKV-7/rwkv7-qwen-vocab-2.7b-from-28b.pt`。
checkpoint 为 BF16，共 **2,741,382,144** 参数。所有输出张量已逐项与源 checkpoint
的对应切片比较；源文件保持不变，检查记录见 `model-manifest.json`。

| 配置 | 数值 |
| --- | --- |
| 层数 / hidden | 24 / 2304 |
| heads / head size | 36 / 64 |
| FFN | 9216 |
| D_AAA / D_DECAY / D_GATE / D_MV | 128 / 128 / 288 / 64 |
| embedding / head | 248320 × 2304 |
| tokenizer | `/hikscale/models/Qwen/Qwen3.8-27B` |
| 数据前缀 | `/mnt/data/Datasets/0056_qwen3.8/0056_qwen3.8_text_document` |

保留全部词表行，只裁剪通道和层数。这是用于重新蒸馏的小模型初始化，不是已经训练好的
3B 模型，也不保证保留 28B 的函数或质量。词表补齐倍数为 256：248077 补齐至 248320。
旧 Xiaoke tokenizer 的 3B 模型不能直接用于同 token ID 的 Qwen logit KL，因此移除了
先前不兼容的 3B 试验启动配置。

## 复跑

使用现有 `train_mobius_py312`，没有新增包。生成 checkpoint（输出路径必须尚不存在）：

```bash
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python tools/build_rwkv7_small_from_checkpoint.py \
  --source /hikscale/models/XiaoKe/rwkv7_28b_88_mapped.pt \
  --output /mnt/data/Models/RWKV-7/rwkv7-qwen-vocab-2.7b-from-28b.pt
```

训练和精度检查在仓库根目录执行：

```bash
bash training-smoke/qwen-rwkv-qwenvocab/launch.sh
bash training-smoke/qwen-rwkv-qwenvocab/launch.sh precision
```

使用 `configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-single.yaml`：单卡 BF16、
seq128、micro/global batch 1、Adam、3 次更新、整层重算、不保存训练 checkpoint。
数据已经是 Qwen token ID，无需再处理 MiniPile。teacher 使用外部 HTTP 服务，
不会在本机运行 vLLM。每个完整 prompt 仅请求一次 `prompt_logprobs=100`，
HTTP temperature=1.0，本地 KL temperature=2.0。

GPTModel 无 labels 时返回 `[batch, sequence, vocab]`。teacher 第 `t+1` 个输入位置
的分布与 student 第 `t` 个输出对齐，使用 Megatron 的 `loss_mask[t]`，最后一个
student 位置不参与这次 prompt KL。请求前检查本地 decode/encode 往返一致性，响应检查
位置数量，loss 检查 teacher ID 范围；不兼容时明确失败，不静默丢弃越界候选。

Qwen 的有效词表是 `248077`，Megatron 为满足 256 对齐将 student head 补齐到
`248320`。蒸馏 KL 显式将 `rwkv_distill_valid_vocab_size` 设为 `248077`，只在有效
词表上计算 student softmax；最后 243 个 padding 行不会吸收概率或产生 KL 梯度。
未显式配置时也会从 teacher tokenizer 的最大 token ID 自动推导该值。

## 实测（2026-10-16）

| 步数 | KL | 梯度范数 | 毫秒/步 |
| --- | --- | --- | --- |
| 1 | 27.19118 | 66.304 | 8711.4 |
| 2 | 27.99094 | 55.700 | 6909.5 |
| 3 | 27.55949 | 60.039 | 6916.2 |

全部更新成功，无跳步、无 NaN。训练末尾验证 KL=34.94241，结束后再次请求 teacher
验证得到 34.93656。两次使用相同验证样本，在线 teacher 输出未缓存，不承诺数值完全
一致。峰值分配显存 49397.14 MiB（48.24 GiB）。这些是短跑链路结果，不构成收敛结论。
Megatron 的 `lm loss` 在这里指 KL；其自动打印的 `lm loss PPL` 不是真正的困惑度。

W&B project：<http://dian.sea-group.org:9098/mobius/rwkv-qwen-distill-2p7b>

本次 run：<http://dian.sea-group.org:9098/mobius/rwkv-qwen-distill-2p7b/runs/zaxuuzyi>

配置开启 TensorBoard writer，因为当前 Megatron 将 W&B 的 loss、grad-norm 日志也放在
该分支里；已从服务端确认 run finished，数据路径及训练/验证指标均已上传。

精度检查使用生成模型和 0056 第一条文档的 64 个 token：

- FP32 全模型前向与独立 Torch 公式参考 logits 相对 L2=5.15e-7，KL 差=0。
- FP32 初始 LayerNorm 和最终 LayerNorm weight 梯度相对 L2 分别为 1.51e-6、8.29e-8。
- 生产 BF16/FLA 前向输出上的 top-k KL 与原蒸馏代码差=1.91e-6，logits 梯度完全一致。
- loss 还通过独立 FP64 dense KL 的值/梯度测试，以及缺失位置和词表越界回归测试。

精度报告见 `precision-results.json`，训练报告见 `training-results.json`。
FP32 检查使用 Torch WKV 替代 FLA 来验证公式；BF16 检查是源 loss 对齐，不能据此
宣称整个 BF16 模型与官方 CUDA 实现逐位一致。尚未做长序列、保存恢复或多卡蒸馏测试。

## 观察长跑（2026-10-16）

`bash training-smoke/qwen-rwkv-qwenvocab/launch.sh long` 使用独立的
`config-rwkv7-qwen-distill-qwenvocab-2p7b-long.yaml`，仍为单卡 BF16、seq128、
batch1、LR=1e-6；最多 500 步或 60 分钟，每 50 步验证 2 个样本，
结束时写入独立的 Megatron checkpoint 目录（包含优化器和随机状态）。
从裁剪后的初始化 checkpoint 开始，因为之前的 3 步 smoke 没有保存训练权重。

本次后台任务 PID、启动时间、日志路径见 `long-20261016-1536-process.json`；
`run-long.sh` 在任务退出后记录 `long-20261016-1536.exitcode`，0 表示正常完成。
后台任务独立于聊天运行，日志为 `long-20261016-1536.log`。
当前 W&B run：<http://dian.sea-group.org:9098/mobius/rwkv-qwen-distill-2p7b/runs/gfarbmls>。

长跑启用 `rwkv_distill_filter_roundtrip`：在构建数据集时确定性地选择可解码再编码成
相同 token ID 的样本。检查候选时跳过了原始训练样本索引 84（首 token 是不完整 UTF-8
字符的后半部分），由后面的有效样本补足 500 项；验证保留 22 项，未拒绝样本。
完整映射保存在数据 cache 目录的 `roundtrip-train-selection.json` 和
`roundtrip-valid-selection.json`，不改源数据和训练目标。

本次长跑按用户指令在第 231 步后直接停止，GPU 已释放，没有自动续跑，也没有到达
500 步/60 分钟的保存点；因此未生成本次训练的 checkpoint。

## Teacher API 请求约定

当前 teacher 服务不依赖鉴权才能响应，但蒸馏请求统一显式携带：

```yaml
rwkv_teacher_api_key: sk-distill
rwkv_teacher_seed: 1234
```

请求头使用 `Authorization: Bearer sk-distill`，请求 JSON 使用 `seed: 1234`。
seed 用于让连续训练和从 checkpoint 续训时的 teacher 请求尽量保持一致；teacher
响应本身不写入 checkpoint，严格逐位恢复对比仍应使用冻结 teacher 响应。
训练配置优先使用 YAML/命令行中的显式值，不依赖本地环境变量中的 key。

W&B 性能指标需要同时开启：

```yaml
log_throughput: true
log_timers_to_tensorboard: true
```

这会把 `throughput` 和 `iteration-time` 写入 W&B；本地两小时 run 的 summary 已包含
这两个指标。关闭 `log_timers_to_tensorboard` 时，Megatron 原生路径可能只在 stdout
显示性能数据，因此后续蒸馏配置统一保留这两个开关。

## 第 500 步恢复对比（2026-10-20）

连续训练 run `3zpuqom8` 从初始化运行到 600 步，第 500 步保存 checkpoint。
恢复 run `k8vr551s` 使用 `ckpt_step: 500` 选择该 checkpoint，恢复模型、Adam、
scheduler 和 RNG；保留 `train_iters: 600`、数据构建及过滤配置，执行 501–600 步。
恢复测试设置 `save: null`，不覆盖原 checkpoint；teacher 请求均携带 `seed: 1234`
和 `Authorization: Bearer sk-distill`。

W&B 未采样、完整精度的 100 步记录对比：loss 平均绝对差 `0.00706444`，最大绝对差
`0.02763367`；平均相对差 `0.043604%`，最大相对差 `0.175316%`。grad norm 平均
绝对差 `0.0760094`，最大绝对差 `0.5670052`。学习率和消费样本计数逐步一致，两个
run 均无 NaN、无跳步，恢复进程退出码为 0。

相同 128-token 诊断输入、关闭缓存、相同 seed 连续请求三次，teacher 的 top-100
候选集合及 logprob 仍不完全一致。按相同 token ID 比较，重复请求的 logprob 平均
绝对差约 `0.036–0.037`。此诊断输入不同于第 501 步样本，因此只能确认服务仍有
波动，不能将所有训练差异都归因于 teacher，也不能据此证明 checkpoint 逐位恢复。
当前结果表明在线续训 loss 接近连续训练；严格恢复误差需要固定 teacher 响应另测。

完整逐步结果见 `seed1234-resume500-comparison-20261020.json`，恢复配置为
`configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-seed1234-resume500-600-20261020.yaml`。

## 带鉴权的一小时训练

2026-10-16 16:30（北京时间）重新启动，预计 17:30 左右停止训练并保存。
新任务从初始化 checkpoint 开始，60 分钟限时，步数上限 1000；每 50 步验证 2 个样本，
每 250 步及退出时保存模型、优化器和随机状态。启动时保留 1000 个有效训练样本，
拒绝索引 84、711；42 个验证样本均通过 tokenizer 往返检查。

本次 W&B：<http://dian.sea-group.org:9098/mobius/rwkv-qwen-distill-2p7b/runs/ter66hd1>。
已确认训练进程加载了本地 teacher API key，W&B 服务端状态为 running。
最新进程、配置、日志和 checkpoint 目录记录在 `latest-onehour.json`。
此任务的配置名、日志和 checkpoint 输出目录均独立于先前已停止的长跑。
