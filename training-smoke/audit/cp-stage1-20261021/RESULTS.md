# CP 第一阶段真实模型短跑（2026-10-21）

环境：`train_mobius_py312`，分支 `2026-10-10`，8 张 H800 80GB。
两次运行均使用 TP2 × PP2 × CP2、SP 关闭、seq4096、micro batch 1、global batch 2、BF16、Adam、full recompute。
数据为 `/mnt/data/Datasets/0056_qwen3.8/0056_qwen3.8_text_document`；tokenizer 为 `/hikscale/models/Qwen/Qwen3.8-27B`。
本次关闭 checkpoint 保存与 W&B/TensorBoard；所有测试进程均已退出。

## 2.7B：3 步训练与验证通过

权重：`/mnt/data/Models/RWKV-7/rwkv7-qwen-vocab-2.7b-from-28b.pt`。
由权重确认 24 层、hidden 2304、36 heads × 64、FFN 9216；LoRA a/w/g/v 为 128/128/288/64。
词表 248320，日志统计总参数约 2.67B。

| Step | CE loss | Grad norm（裁剪前） | 耗时 |
| --- | --- | --- | --- |
| 1 | 13.49889 | 74.360 | 256.920 s（含首次编译） |
| 2 | 13.37054 | 83.848 | 0.620 s |
| 3 | 13.25378 | 97.787 | 0.579 s |

3 步均为 0 skipped / 0 NaN。每步使用不同训练样本，不能把短跑 loss 下降当作收敛证据。
固定验证 2 个样本，iteration 3 与训练结束后的验证均为 loss 13.15858，PPL 518440.6。
进程退出码 0。轮询 nvidia-smi 最高观测约 14.1 GiB/卡（非连续峰值采样）；日志 PyTorch 最大 allocated 为 10440.70 MiB。

复跑：`bash training-smoke/launch-rwkv7-2p7b-qwen-cp2-smoke.sh`。
配置：`configs/config-rwkv7-2p7b-qwen-cp2-smoke.yaml`；完整日志：`2p7b-smoke.log`。

## 28B：首次 Adam 更新 OOM

权重：`/hikscale/models/XiaoKe/rwkv7_28b_88_mapped.pt`。
真实模型加载、数据加载、forward、backward 已执行到 optimizer.step。
首次初始化 Adam `exp_avg_sq` 时 CUDA OOM，未完成任何训练 step；显存约 79 GiB/卡。
这是优化器状态分配失败，不能据此宣称 28B CP 训练通过。更短序列也不减少模型和 Adam 状态本身的显存。

配置：`configs/config-rwkv7-28b-qwen3.8-0056-cp2-smoke.yaml`；完整日志：`28b-smoke.log`。

## 尚待验证

- 已完成 2.7B CP1/CP2 logits 采样、完整逐 token CE 和全部参数梯度对拍；完整/不均匀 mask 下整体梯度相对 L2 误差为 0.47%/0.61%，通过 BF16 容差。见 `../cp-equivalence-20261021/REPORT.md`。
- 已消除 CP loss 的 all-reduce autograd 警告；使用 detached 全局统计和本地恒等梯度，并已复验梯度缩放。
- checkpoint 保存/恢复、长跑稳定性和 micro batch > 1。
- 28B 模型/优化器状态显存优化。
