# 独立 Teacher Pool（2026-10-20）

训练进程可通过独立 CPU Pool 请求外部 vLLM。Pool 统一调度多个客户端的请求、执行
重试和限流退避，并暂存在内存中；不启动 vLLM，也不加载 teacher 模型权重。

```text
Megatron dataloader -> 本进程异步预取 -> 独立 Teacher Pool -> 外部 vLLM
                           <- CPU top-k ID/logprob 结果 <-
                    student forward -> 等待对应结果 -> KL -> backward
```

这是请求调度和结果缓冲的进程隔离。Pool 不自主遍历训练数据，也不是持久化的离线
logits 数据库。关闭 `rwkv_teacher_pool` 即恢复原来的直接 HTTP 路径，默认关闭。

## 启动

在仓库根目录、`train_mobius_py312` 环境中执行，未增加依赖。两个进程使用相同配置。
先在一个终端启动 Pool：

```bash
python tools/teacher_pool_server.py \
  --rwkv-config configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-pool-smoke.yaml
```

看到 `teacher_pool_ready` 后，在另一个终端运行训练：

```bash
RWKV_DISTILL_CONFIG=configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-pool-smoke.yaml \
RWKV_DISTILL_LOG=/tmp/rwkv-teacher-pool-smoke.log \
bash training-smoke/qwen-rwkv-qwenvocab/launch.sh
```

此示例为 2.74B、BF16、seq128、4 次更新，不保存 checkpoint，不创建 W&B run。
训练结束不会关闭独立 Pool，可以继续服务其他客户端；测试完成后在 Pool 终端停止它。
训练使用的其他配置可以加入以下字段，无需修改 Megatron Core 或 FLA：

```yaml
rwkv_teacher_pool: true
rwkv_teacher_pool_host: 127.0.0.1
rwkv_teacher_pool_port: 50055
rwkv_teacher_pool_authkey: teacher-pool
rwkv_teacher_pool_workers: 16
rwkv_teacher_pool_queue_size: 256
rwkv_teacher_pool_max_results: 512
rwkv_teacher_pool_result_ttl: 300.0
rwkv_teacher_pool_timeout: 2700.0

rwkv_teacher_async_prefetch: true
rwkv_teacher_prefetch_workers: 16
rwkv_teacher_prefetch_lookahead: 32
rwkv_teacher_prefetch_min_workers: 1
rwkv_teacher_prefetch_latency_target: 2.0

rwkv_teacher_api_key: sk-distill
rwkv_teacher_seed: 1234
rwkv_teacher_top_k: 100
```

Pool 的 worker 上限控制所有客户端的总并发；预取 worker 控制单个训练进程的在途
请求数量。Pool 根据实际 upstream 请求延迟和 429 反馈调整全局并发，复用已有
`OnlineTeacherClient` 的重试、指数退避和 LRU 缓存。HTTP temperature 固定为 1.0，
本地 KL temperature 独立处理。`sk-distill` 是发往 vLLM 的请求标记；Pool 的 RPC
authkey 是另一个参数。

## 缓冲、错误和恢复语义

- 使用标准库 BaseManager RPC，默认仅监听本机回环地址，供可信的同节点训练进程使用。
  不依赖 torch tensor IPC，不传递 CUDA tensor；ID/logprob 用 CPU int32/float32 bytes 传输。
- 启动时核对 teacher URL、模型名、top-k、seed、HTTP temperature、有效词表大小和
  vocabulary hash。不一致立即报错，避免 Pool 沿用另一个 teacher 的设置。
- 每次请求使用 UUID，多个客户端、重复验证和恢复后的训练不会按 iteration ID 串样。
  JSONL 原始文本和输入 mask 一起转发，沿用原来的 token 对齐检查。
- `queue_size` 限制等待执行的任务。`max_results` 为所有未消费请求预留结果容量
  （包含排队、执行、已完成任务）；满时拒绝新提交，客户端等待后再提交，不驱逐旧结果。
- 已完成结果取走即删除。未取走的结果在 `result_ttl` 秒后过期；TTL 应大于预取窗口
  加验证暂停的最长等待时间。过期会明确失败，不会替换或跳过样本。
- 客户端的 `pool_timeout` 覆盖排队和等待结果，超时会取消任务；已在执行的 HTTP 请求
  由原 HTTP timeout 约束，不能强制中断。Pool 进程退出时 RPC 断开会直接报错。
  该超时不承诺中断一个仍保持连接、却完全失去响应的 RPC 服务。
- upstream 最终失败会传回训练并终止，不回退直接 HTTP，不返回零损失，不静默跳过。
- checkpoint 仍只记录已消费样本的游标，不记录 Pool 队列。恢复后重新提交后续 batch；
  内存 LRU 只在 Pool 存活时可复用，停止 Pool 后缓存消失。
- per-step 仅增加少量 Pool 队列/错误指标，沿用已有 teacher 指标组；完整统计可通过
  `RemoteTeacherPoolClient.get_stats()` 查询，不修改远端 W&B workspace。

Pool 可以接收多个客户端，但这不意味着 TP/PP/CP 蒸馏已经适配；训练入口仍要求
TP=PP=CP=1，多 GPU 蒸馏仍需单独完成分布式 loss、teacher 分发和精度验证。

## 测试和精度

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests
```

最终结果：205 passed、2 xfailed，另有 76 项 subtests passed。

新增测试覆盖独立进程、多个客户端、多线程、乱序配对、队列/结果容量、背压、TTL、
排队取消、客户端超时、服务进程退出、错误传递、配置/词表校验以及原始文本/mask 转发。
FP32/BF16 下，同一 teacher 响应经过 RPC 后，KL 和完整 logits 梯度逐位相同，243 个
padding 类别梯度为 0；小模型 Adam 连续更新及消费游标恢复也与直接 teacher 完全一致。

完整模型测试从已有第 500 步 checkpoint 恢复，使用独立 Pool 和 4 路异步预取，完成
501–505 五次全参数 Adam 更新。teacher 使用此前保存的真实 Qwen top-100 响应，测试
Pool 禁止 HTTP，缺少对应样本立即失败。与此前直接同步 teacher 的基线相比，五步的
完整 logits、KL、裁剪前 grad norm 差值全部为 0，无跳步。没有写新 checkpoint；这里
未重新比较最终全部 Adam 状态。报告为
[teacher-pool-validation-20261020.json](teacher-pool-validation-20261020.json)。

回放服务入口为 `pool_replay_server.py`，接受正常的 `--rwkv-config` 以及
`--teacher-dir`（`frozen_teacher_resume.py` 保存的响应目录）；只用于诊断。该脚本和
训练侧冻结响应诊断沿用同一输入签名，验证时绝不拿其他样本的响应替代当前样本。
本次 Pool 配置、完整训练日志、trace/logits 位于 `/tmp/rwkv-teacher-pool-validation-20261020`。

实服务检查有明确限制：本次 `Qwen3.8-27B` completions 返回 503，报错
`All upstream providers ... are cooling down (Tried: [])`；`/v1/models` 只列出
DeepSeek-V4-Flash-0731、DeepSeek-V4.1-Flash 和 xiaoke-5。真实 HTTP 训练短测因此
在更新前失败，确认了重试和错误透传，但不能声称当前远端 Qwen 已可用于在线训练。
没有替换成其他模型或改动远端服务。

## 与原蒸馏项目的功能对照

以 `/mnt/data/Codes/RWKV/qwen3_5_to_rwkv7_distillation` 的实际调用链为准，不把 README
中的规划或未接线的 loss 函数算成已完成的在线功能。

| 功能 | 原项目实际情况 | 当前 Megatron 蒸馏分支 |
| --- | --- | --- |
| 独立 Teacher Pool | `src/teacher_pool.py`、`scripts/teacher_pool_server.py` | 本次加入，支持同步及异步客户端 |
| Pool 和训练的一键 launcher | `scripts/launch_distill_with_teacher_pool.py` 管理子进程 | 尚无同类 launcher，当前分别启动 |
| 按组件/名称冻结参数 | `FreezeConfig` 控制 embedding、head、norm、FFN、time-mix/LoRA，支持自定义匹配 | 尚无这些训练配置，目前全参数训练 |
| 更换冻结策略后恢复 | 加载时重新应用冻结设置，含 optimizer 重建代码 | 尚无此流程；原项目这部分也不应直接视为严格恢复保证 |
| 独立 CE/PPL 监控 | 每步即使只用 KL 优化，也计算 next-token CE，再按有效 token 汇总 PPL | 仅在 `rwkv_distill_ce_weight` 非零时计算/报告 CE，尚无独立 CE/PPL 开关；`exp(KL)` 不是 PPL |
| 大规模 JSONL 懒加载 | 递归 glob、按文件缓存、getitem 时 tokenize，支持 text/content/prompt 回退 | 当前 JSONL 预先 tokenize 并保存在内存，目录只找顶层文件；可显式选择字段，尚无相同懒加载路径 |
| 本地 teacher / 完整词表 KL | 可加载本地 HF teacher，走 dense KL | 当前 teacher 后端仅外部 API，使用 top-k KL；这是当前选择，不影响 top-100 目标 |
| token ID 直接作为 API prompt | client/Pool 有可选路径，失败可退回文本；BF16 训练直接 client 硬编码关闭 | 当前使用原始文本或经过 round-trip 校验的 decode 文本，不发送 ID prompt |
| 多 rank 在线蒸馏 | torchrun + DeepSpeed，有跨 rank teacher 有效性检查 | Pool 多客户端已测试，真实多 GPU 蒸馏未验收；TP/PP/CP 仍有显式限制 |
| teacher 失败后继续 | 原训练器同步有效性标记并执行零 loss 路径 | 保持重试耗尽后报错，尚无经过验证的跨 rank 失败继续策略 |
| checkpoint 保留策略 | 最近 N 份以及 best/final 命名 | Megatron 有 `save_retain_interval` 等机制，没有直接移植该 best/最近 N 策略 |
| Qwen 原始权重转换 | `src/weight_converter.py` 等提供 Qwen 到 RWKV 映射 | 当前读取已映射 RWKV 权重，并支持裁剪小模型；Qwen 转换仍由原项目完成 |

不属于“原项目已实现、Megatron 漏掉”的项：

- 在线 hidden-state/attention 蒸馏：原项目虽然有 MSE 函数，但在线 TopK 分支只计算
  KL，另一路调用也传 `student_hidden=None, teacher_hidden=None`；当前 vLLM API
  也不返回这些张量。两边都没有完整的在线训练接线。
- 持久化 teacher 数据库：原项目 Pool 也只存内存，两边都没有正式离线数据生成/读取
  系统；冻结响应文件属于诊断工具。
- 分布式优化器、CPU offload、梯度累积、LR scheduler：Megatron 已有对应机制，
  当前单卡配置没有开启全部选项，不能说框架不支持。DeepSpeed ZeRO 参数也不能直接
  当作 Megatron 的配置使用。
- 下游 ARC/BoolQ/HellaSwag 评测：原项目 AGENTS 明确标记待实现，未发现对应
  `scripts/evaluate.py`，不是现成可移植的能力。
