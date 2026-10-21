# 蒸馏功能验证：2026-10-21

环境：Conda `train_mobius_py312`；分支 `codex/qwen-rwkv28b-distill`。无需新增依赖；Megatron 和 FLA 子仓库未修改。本记录只说明新增功能的正确性及短训练集成测试，不代表模型能力评估或多 GPU 验证。

## 单元及进程测试

```bash
conda run -n train_mobius_py312 python -m pytest -q \
  tests/test_distill_telemetry.py tests/test_jsonl_dataset.py \
  tests/test_online_teacher.py tests/test_teacher_pool.py \
  tests/test_teacher_prefetch.py tests/test_distill_launcher.py \
  tests/test_checkpoint_retention.py tests/test_distill_metrics_backfill.py
```

结果：81 passed。覆盖独立 CE、FP32/BF16 KL 梯度不变、token 数加权后求 PPL、续训后的验证日志步数、懒加载原分块算法、随机访问/重复/多 worker 续读、递归 split、token-ID 传输、pool 配置不一致、启动/失败/取消进程清理及 checkpoint 保留边界。

## GPU 数值对齐

使用 `[2, 8, 248320]` logits、有效词表 248077、11 个有效 next-token 标签，KL 温度 2，teacher top-100。输入为固定 seed 1234 的合成数据，用以隔离新增监控逻辑的数值影响。

| 输入精度 | student CE | FP64 NLL 参考 | CE 绝对误差 | KL / 梯度对齐 | padding 行梯度最大值 |
|---|---:|---:|---:|---|---:|
| FP32 | 13.0877962112 | 13.0877949819 | 1.2294e-6 | 均逐位一致 | 0 |
| BF16 | 13.0888614655 | 13.0888621409 | 6.7549e-7 | 均逐位一致 | 0 |

CE 监控使用 FP32 计算，参考使用 FP64 `logsumexp - target logit`。梯度比较基线是原 `kl_divergence_topk`，不是开启/关闭 telemetry 的两个相同实现。PPL 是优化步全体有效 token 的 CE 的指数；验证 PPL 在完整验证批次汇总后计算。训练日志区间展示的 PPL 为各优化步 PPL 的均值。

## JSONL 容量与对齐

合成 100000 条 JSONL，构造阶段 tokenizer 调用次数为 0，行索引为 3200000 字节（每行 32 字节），不保存原文或 token IDs。第一次访问才创建字符窗口索引并 tokenize。初始化仍需顺序扫描文件字节；这是带索引的 map-style dataset，不是完全不扫描文件的流式读取。随机访问靠后样本需要先建立前面窗口的索引；多 worker 各自构建索引。

独立参考算法逐字符分块、递归二分超长块并保留文档末尾，与 lazy loader 的 prompt、tokens、labels、mask 及重复顺序一致。窗口 S+1 个 token 时，student 使用前 S 个作为输入；teacher 文本请求可评分全部 S+1 个位置，客户端只返回前 S 个位置，不增加较早预测位置的上下文。

## 实际 2.7B 单卡训练

- Student：`/mnt/data/Models/RWKV-7/rwkv7-qwen-vocab-2.7b-from-28b.pt`。
- 数据：`/mnt/data/Datas/minipile/jsonl`，最多 12 条原始记录，字符块 128、序列长度 128。
- Teacher：外部 `http://192.168.10.121:8000`，`Qwen3.8-27B`，`Bearer sk-distill`、seed 1234、请求温度 1、top-100。
- 通过新 launcher 启动真实 pool、异步预取，4 步、micro batch 1 / global batch 2，单卡 BF16。TensorBoard 开启；此次回归测试关闭 W&B 上传。
- 训练、验证和测试正常退出；零 NaN、零跳步，teacher 请求零失败。
- 最后训练步 KL 为 34.34106，独立 CE 为 13.53465，PPL 约 755132.8；最终固定验证 CE 为 13.67083，PPL 约 865295.2。这些是短流程测试值，不是泛化能力结论。
- 已检查 TensorBoard 中的 CE/PPL、throughput 和 iteration-time；它们的单位分别为 nats/token、无量纲、TFLOP/s/GPU 和 s/step。

后续两步小模型测试直接读取 TensorBoard event 文件核对了独立的验证/测试指标：验证 CE 12.4459657669 / PPL 254222.65625，测试 CE 12.4860019684 / PPL 264607.25；`log(PPL)` 与对应 CE 的差小于 1e-6。使用 `distill/validation_student_*` 和 `distill/test_student_*` 避免 Megatron 通用日志将 test CE 写入相同的 validation 标签。恢复训练的优化步指标正确从 step 5 开始；框架有意不输出每个进程首次训练迭代的 iteration-time，随后迭代正常输出。pool 退出后两个测试端口均已关闭。

## Checkpoint 保存与恢复

使用 2 层、hidden size 128、Qwen 有效词表的小模型隔离文件管理测试，保持相同 JSONL/API/pool 链路。配置 `rwkv_checkpoint_keep_last: 2`。

1. 每步保存 step 1–4，完成后仅有 `iter_0000003`、`iter_0000004`，tracker 为 4。
2. 加载 step 4 的模型、optimizer、RNG 和数据进度，训练到 step 5；本次 `save_interval: 1000`，故 step 5 使用训练结束后的最终保存路径。
3. 完成后只保留 `iter_0000004`、`iter_0000005`，tracker 为 5，step 3 被最终清理删除。

最初延长训练总步数但同时更改 scheduler 时触发 Megatron 的 scheduler 一致性检查；恢复原衰减配置并启用 `use_checkpoint_opt_param_scheduler: true` 后成功恢复。清理不会改变 checkpoint 内容。该测试没有比较在线 teacher 再请求后的逐位 loss；固定 teacher 响应的断点续训对齐已有此前测试覆盖。本次实际保存测试使用单卡、同步 torch 格式；分布式格式仅验证识别/删除边界，未验证多 GPU。

## token-ID API 限制

2026-10-21 实际 `/v1/models` 和字符串 `/v1/completions` 正常，但数组 `prompt: [token_id, ...]` 返回：

```text
HTTP 422: Request body validation failed: ExpectedString at character 0
```

`rwkv_teacher_prompt_mode: token_ids` 的请求、右侧 padding、pool 握手和 FP32/BF16 KL/梯度一致性均有测试服务覆盖；当前网关不支持实际 token-ID 训练。默认继续使用 `text`，不静默回退。没有启动或修改 vLLM。

## 本机详细产物

本轮临时配置、原始日志和数值记录保存在 `/tmp/rwkv-distill-tests-20261021/`：`production.yaml`、`production.log`、`tiny.yaml`、`tiny.log`、`tiny-resume.yaml`、`tiny-resume-final.log`、`quality.yaml`、`quality.log`、`metrics-check.json`、`precision.py`、`precision.json`、`large-jsonl.json` 及 TensorBoard 文件。该目录是临时测试产物，不纳入 Git；复现时应重新选择空的保存路径和空闲 pool 端口。
