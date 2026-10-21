# DPLR CP state 精度正式修复（2026-10-21）

已将此前诊断验证的精度对齐合入本地 FLA 子模块的正式 kernel。日常训练直接使用该补丁，无需诊断 monkeypatch。

## 改动

- `3rdparty/flash-linear-attention-260115/fla/ops/cp/chunk_delta_h.py`：DPLR `USE_BG=True` 时保留 v2 的 FP32 累加结果，bg 转成 FP32 后参与 state 更新；四个 K 分片均覆盖，支持 head size 至 256。
- GDN/KDA/GDN2/gated delta product 的 `USE_BG=False` 分支继续将 v_new 转回输入 dtype；接口、state 格式、affine-matrix 计算与反向 kernel 均未改动。
- 同子模块 `tests/context_parallel/test_cp_dplr.py` 新增 12 个 prefix-state 回归，使用普通 DPLR final state 作参考；BF16 K64/128/192/256，FP16 K64 V32/96，均覆盖 dense/varlen。阈值固定 5e-5，没有放宽原有测试阈值。
- 子模块和父仓库保持在用户指定的 `2026-10-10` 分支；保留工作区原有修改。完整补丁另存 `fla-cp-state-fp32.patch`，未创建提交。

## 真实 2.7B CP1 / CP2 对拍

8 张 H800，conda `train_mobius_py312`，torch 2.9.1+cu128 / Triton 3.5.1。使用同一真实 2.7B 权重、0056 数据、TP2×PP2×CP2 / seq4096 / MB1 / GB2，BF16，SP 关闭，完整重计算。CP1 基线复用前一轮捕获；CP1 路径未被本补丁改变。每组检查两个 microbatch 的全部逐 token CE、完整 head 输入、每个 TP 64 个词表 logits 采样，以及生产 reduce-scatter 后、裁剪/更新前的全部 1356 个 TP/PP 参数梯度张量。

本轮运行 `../cp-equivalence-20261021/capture.py`，直接调用修改后的生产 kernel，全程无 kernel 替换。

| 指标 | 完整 mask | 不均匀 mask |
| --- | ---: | ---: |
| 数据、标签、position、loss mask | 与 CP1 完全相同 | 与 CP1 完全相同 |
| 全 token CE 最大绝对差 | **0** | **0** |
| 完整 head 输入最大绝对差 | **0** | **0** |
| 采样 logits 最大绝对差 | **0** | **0** |
| mean CE 最大绝对差 | **0** | **0** |
| 整体梯度相对 L2 | **0.0232579%** | **0.0228621%** |
| 原实现整体梯度相对 L2 | 0.471272% | 0.607534% |
| CP2 / CP1 梯度范数比 | 1.00000414 | 1.00000053 |
| 非有限梯度 | 0 | 0 |

不均匀 mask 的两段有效 token 数为 618/1850，跨越分片中点的区间也被屏蔽。梯度指标沿用此前包含 TP 复制参数的对拍定义；剩余误差符合局部 BF16 参数梯度舍入后再做 CP 归约的差异，见前一轮 `../cp-error-source-20261021/REPORT.md`。这里的逐位一致只针对实际捕获内容和上述条件，不代表所有形状、CP size 或全词表 logits 均已证明逐位一致。

详细结果：`cp2-full/comparison.json`、`cp2-uneven/comparison.json`，配套 `run.log` 和 `comparison.log`。

## 训练验证

通过正常启动脚本 `training-smoke/launch-rwkv7-2p7b-qwen-cp2-smoke.sh` 跑完 3 步真实 Adam 更新及固定验证：

| 步数 | LM loss | grad norm | 跳步 / NaN |
| --- | ---: | ---: | --- |
| 1 | 13.49903 | 74.311 | 0 / 0 |
| 2 | 13.37066 | 83.810 | 0 / 0 |
| 3 | 13.25395 | 97.630 | 0 / 0 |

验证 loss **13.15849**。日志为 `optimizer-smoke.log`，另有 `/mnt/data/Logs/outputs/qwen-rwkv2p7b-cp2-state-fix-20261021.log`。本轮测试期间其他 GPU 测试并行运行，迭代耗时不用于训练吞吐性能结论。

## 回归与限制

- 新增 state 回归：**12 passed**，`regression-after.log`。修改前在同一阈值下，head64 BF16 dense 测试失败，相对误差 0.001067；证据见 `regression-before.log`。
- 父仓库 CP loss、静态合同、v0 TP/PP/SP 及参数标签测试：**34 passed**，`model-regressions.log`。
- DPLR/RWKV7 CP2/CP4 前向及六组输入梯度：**8 passed / 1 skipped**，见 `dplr-cp-regressions.log`。覆盖跨 rank 分段、分片边界对齐、CP4 多序列、CP4 单序列、多短序列、safe_gate 和 wrapper。TileLang route 因未安装可选后端而跳过。
- GDN 的现有端到端 CP 测试未能完成：当前 Triton 3.5.1 在 Hopper 上被其非 CP `chunk_bwd_dqkwg` 明确拒绝执行，错误要求 Triton >=3.7.1 或安装 TileLang。异常清理卡在 NCCL communicator destroy，相关进程已终止。独立非 CP 重现的完整错误在 `gdn-reference.log`；这条路径不调用本次修改的 CP kernel。没有升级训练环境或更改这些无关算子。
- 共享 `USE_BG=False` 分支另用旧、新 kernel 直接对拍，**6 组结果均逐位一致且全部有限**，覆盖 scalar g、per-channel gk、无 gate、dense/varlen 以及 grouped heads；见 `shared-prefix.json`。该检查只验证共享 prefix kernel，不能替代以上算子的完整端到端 backward 测试。
- Ruff 0.14.10 检查通过，autopep8 2.3.2 无格式差异，Python 编译和两层 Git diff whitespace 检查通过。工具安装在 `/tmp/rwkv-cp-lint-tools`，未改动训练 conda 包。

CP4 目前验证的是算子回归；未声称完成 CP4 整模型训练、CP+SP、28B Adam 显存问题或断点恢复验证。

## 同机性能成本

`benchmark_prefix.py` 在空闲的 GPU7 上比较固定旧提交 `ff481ea6862c99a818858081571373694de7b4e7` 与当前代码，均使用真实第一层的两个 head 输入，独立重复至 TP-local H18；B1/T2048/H18/K64/V64，BF16，BT64，varlen `[0,2048]`。测量完整 prefix 汇总 kernel（同时计算 h 和 affine matrix），不包含通信及其他 WKV 阶段；每次预热 100 ms、测量 300 ms，交替顺序共三轮，报告中位数。

| kernel | 延迟中位数 | 对普通 prefix final state 的相对 L2 |
| --- | ---: | ---: |
| 修复前 | 49.963 μs | 0.131015% |
| 修复后 | 53.695 μs | 0 |

该 kernel 增加约 **3.73 μs / 7.47%** 的计算开销，换取与普通 WKV state 更新的精度一致性。此结果不等于整个训练步骤变慢 7.47%。原始数据、重复测量、硬件和精度环境见 `benchmark.json`。

## 复现

在仓库根目录、指定 conda 环境下执行，`PYTHONPATH` 指向仓库根目录：

```bash
bash training-smoke/audit/cp-state-fix-20261021/run.sh full
bash training-smoke/audit/cp-state-fix-20261021/run.sh uneven
FLA_CI_ENV=0 OMP_NUM_THREADS=8 python -m pytest -q \
  3rdparty/flash-linear-attention-260115/tests/context_parallel/test_cp_dplr.py -k prefix_state
FLA_CI_ENV=0 OMP_NUM_THREADS=4 python -m pytest -q \
  3rdparty/flash-linear-attention-260115/tests/context_parallel/test_cp_dplr.py \
  3rdparty/flash-linear-attention-260115/tests/context_parallel/test_cp_rwkv7.py -k 'not prefix_state'
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 python training-smoke/audit/cp-state-fix-20261021/benchmark_prefix.py
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 python training-smoke/audit/cp-state-fix-20261021/check_shared_prefix.py
```

大 tensor 和导出的旧 kernel 源码仅保留在本机并由 `.gitignore` 排除。修复及覆盖合同见 `CONTRACT.md`。未来若向 FLA 上游发布 precision-changing PR，需按其 AGENTS.md 的 RFC 流程处理；本任务只完成已授权的本地修复。
