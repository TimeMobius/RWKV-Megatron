# Qwen teacher 异步预取验证（2026-10-16）

Megatron 训练进程现在支持并发 teacher 请求。接口继续使用已有的 vLLM 服务，
`prompt_logprobs=100`、请求 temperature=1.0、本地 KL temperature=2.0。
API key 通过现有 `RWKV_TEACHER_API_KEY` 环境变量加载，每个请求线程使用自己的
HTTP Session；配置、代码和 W&B 中不保存密钥。没有新增依赖。

```yaml
rwkv_teacher_async_prefetch: true
rwkv_teacher_prefetch_workers: 16
rwkv_teacher_prefetch_lookahead: 32
rwkv_teacher_prefetch_min_workers: 1
rwkv_teacher_prefetch_latency_target: 2.0
```

预取器在 Megatron 的 `RerunDataIterator` 内包装原 CPU DataLoader iterator。
只有训练主线程读取数据，后台线程只处理独立的 CPU token 副本。
每个 batch 保存自己的 Future，按原数据顺序等待结果；HTTP 返回顺序不影响目标对齐。
student forward 与请求重叠，随后等待该 batch 的 teacher 结果，执行原来的 KL 和 backward。
重试耗尽后异常才传播，不跳过 batch，也不会对训练样本重复计算 loss。首次请求加载 tokenizer 有锁，HTTP 调用计数也有锁。
teacher HTTP 请求现在对连接异常、408/425/429/5xx 做最多 3 次重试，使用指数退避、
`Retry-After` 和少量 jitter；401/403 等客户端错误不重试。相同 CPU token batch 使用
线程安全 LRU 缓存（默认 256 条）。预取器在最大 16 个线程内根据请求延迟和 429 自动降低或
恢复并发，统计记录在 `teacher/*` 中。
逐位置 logprobs 使用整行构造张量，减少并发解析中的逐标量 Tensor 操作。

队列中的 batch 不计入 Megatron consumed samples。恢复时由原 sampler 按实际完成样本数
重建，丢弃未消费的 Future。固定 batch 数时还根据 `train_iters`、`exit_interval`
限制提交数量，不请求已知退出点之后的数据。按时间或信号退出时取消尚未执行的请求、
等待在途请求完成，并释放原 DataLoader iterator，让其进程正常结束。

当前范围为 `dataloader_type: single`、`rerun_mode: disabled`、TP/PP/CP 都为 1。
实际训练只验证了单 GPU。Future 不进入 checkpoint；rerun 模式明确拒绝开启异步。
验证阶段保持同步，兼容固定验证样本的重置。

## 复跑

32 步独立 smoke 从 2.74B 初始化权重启动，不保存 checkpoint：

```bash
RWKV_DISTILL_CONFIG=configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-async-single.yaml \
  bash training-smoke/qwen-rwkv-qwenvocab/launch.sh
```

单元测试与完整模型精度检查：

```bash
bash training-smoke/qwen-rwkv-qwenvocab/launch.sh test-async
bash training-smoke/qwen-rwkv-qwenvocab/launch.sh async-precision
```

24 项测试通过，覆盖乱序完成、并发/缓存边界、异常传播且不重复请求、冻结 teacher
的 loss/梯度一致性、恢复后的样本顺序、取消排队请求、CPU token 隔离、配置覆盖、
已知退出点的请求上限、多进程 DataLoader 释放，以及已有的 KL/API/固定验证回归测试。

完整 2.74B BF16/FLA 模型使用 0056 的 128 个 token，冻结一次真实 top-100 teacher
响应，重复四次通过异步队列取回。与同步参考的 logits、KL loss、首尾 LayerNorm weight
梯度最大绝对差均为 **0**。只检查所列参数梯度，不代表检查全部参数梯度。
报告为 `async-precision-results.json`。

## 真实训练对照

两组最终短测都从原训练的第 197 步 checkpoint 恢复模型、Adam 和 RNG，执行
198–213 共 16 次更新，batch1、seq128、LR=1e-6；第 200 步验证两个固定样本。
两组都设置 `save: null`。保持 `train_iters: 1000`，以保持数据构建和过滤映射一致，
通过 `exit_interval: 213` 结束短测。

异步最终统计：submitted=consumed=completed=16，failed=0、peak_active=16，退出后
active=buffered=0。两组最终进程退出码及完整性能结果记录于 `async-validation-results.json`，
运行路径记录于 `latest-async-validation.json`。训练与报告上传同一个 W&B project：
<http://dian.sea-group.org:9098/mobius/rwkv-qwen-distill-2p7b>。

预取命中时的步耗时约 0.31 秒，首批预取的准备时间明显更长。比较速度必须包含首批
等待和验证时间；短测不能证明长跑持续吞吐或收敛。在线 teacher 独立请求的数值可能不同，
真实同步/异步训练的 loss 不要求逐位相等；数值对齐使用上面的冻结响应检查。

初版真实短测暴露了退出时强引用 DataLoader 导致 worker 未释放的问题，现已修复并加入
多进程 worker 退出回归测试；最终训练复测验证正常退出。初版产生的额外预取请求也已
通过按已知退出点限制 batch 数解决。更早的同步测试因运行时修改 launcher 导致 wrapper
退出码 2，其 16 步训练已经完成；最终对照重新执行，避免将该 wrapper 结果当作通过。
