# 一小时蒸馏观察运行

状态：已启动，进程状态和最终退出码以 process.json / exitcode 为准。

- 开始：2026-10-21T10:15:19.058867+08:00
- 预计结束：2026-10-21T11:15:19.058867+08:00（60 分钟时限后保存最后一个 checkpoint，再清理进程）
- 分支：codex/qwen-rwkv28b-distill，代码提交：2547f8fd89917805ac99de49c6783b853fc1c94f
- 环境：train_mobius_py312；GPU 0；初始 2.7B student。
- 数据：/mnt/data/Datas/minipile/jsonl；不限制原始记录数；全部可用文档分块保留。
- seq length 128；micro batch 1；global batch 2；lr 1e-6；单卡 BF16。
- 开启 JSONL lazy/recursive、独立 CE/PPL、teacher pool/异步、重试退避、动态并发、缓存。
- teacher：Qwen3.8-27B；text prompt；top-100；seed 1234；Bearer sk-distill；请求温度 1；KL 温度 2。
- 并发上限 16、动态下限 4、lookahead 32。
- 每 50 步验证 4 个 batch；每 250 步保存；普通 checkpoint 保留最近 2 个。
- W&B 在线上传已由服务器 API 验证；CE/PPL 四个图表已追加到原 Charts 区。
- W&B：http://dian.sea-group.org:9098/mobius/rwkv-qwen-distill-2p7b/runs/f4c9de46
- 配置：/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/configs/config-rwkv7-qwen-distill-2p7b-onehour-features-20261021-101425.yaml
- 日志：/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/training-smoke/qwen-rwkv-qwenvocab/onehour-features-20261021-101425/train.log
- checkpoint：/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/training-smoke/checkpoints/qwen-distill-2p7b-minipile-onehour-features-20261021-101425

尚未实现的多 GPU 蒸馏/跨 rank 失败处理未启用；当前 API 拒绝 token-ID 数组，使用文本模式。
