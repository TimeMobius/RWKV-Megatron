# RWKV7 测试与代码状态入口（2026-10-22）

最新提交状态：CP生产代码已分别提交（主仓库`f8bc564`、Megatron `f30bff508`、FLA `e932aadd`），测试报告另有归档提交；作者均为shiroko98。蒸馏恢复原名`codex/qwen-rwkv28b-distill`，主仓库路径隔离修正为`f979779`，两个子仓库仍是`d65baf60`/`ff481ea6`且干净。本轮未push。推送命令与未纳入的历史材料见[COMMIT_AND_PUSH.md](COMMIT_AND_PUSH.md)。下方“未提交/临时蒸馏分支名”叙述是提交前的历史检查记录，最新状态以本段和BRANCH_WORKSPACES.md为准。

远程复核完成：TimeMobius蒸馏分支为`87116dc`，其Megatron/FLA指针分别等于各自远程`2026-10-10`的`d65baf60`/`ff481ea6`。CP相对远程release仅有主仓库`0ca9d3a`、FLA `261dbcf5`两个既有额外提交；Megatron无额外提交，后续脏改动保留。见[远程核对报告](training-smoke/audit/branch-isolation-20261022/REMOTE-VERIFICATION.md)。

分支隔离更新：当前目录及两个子仓库已切换到新分支`codex/rwkv7-cp`，原`2026-10-10`指针未移动。新蒸馏工作区为`/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-distill`，三仓库使用`codex/rwkv7-distill`，子仓库固定Megatron `d65baf60`、FLA `ff481ea6`。本轮没有新增commit/push，CP改动仍未提交；下方旧分支表是隔离前检查记录。最新路径、分支和使用方式见[BRANCH_WORKSPACES.md](BRANCH_WORKSPACES.md)。

这是本轮CP、28B显存、TP/PP试验及数值定位的固定入口。先看本文件，再按下表进入报告；执行计划见[todolist.txt](todolist.txt)。报告中的日期是测试日期，不表示后续代码自动保持相同状态。

仓库绝对路径：`/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test`。
环境：`conda activate train_mobius_py312`，Python：`/mnt/data/anaconda3/envs/train_mobius_py312/bin/python`。
硬件：单机8×H800 80GB；主仓库、Megatron、FLA工作分支均为`2026-10-10`。

**用户最新决定：TP数值精度优化暂不实施，保留当前生产精度设置和诊断证据。** 这不撤回已经完成的CP及offload功能修复。

## 28B实际跑通的配置

模型：`/hikscale/models/XiaoKe/rwkv7_28b_88_mapped.pt`（88层，hidden5120，80heads，FFN17408，日志约27.46B）。
数据：`/mnt/data/Datasets/0056_qwen3.8/0056_qwen3.8_text_document`，95/5划分；tokenizer：`/hikscale/models/Qwen/Qwen3.8-27B`。

下表均为SP开启、BF16计算、FP32梯度及Adam m/v、full recompute、MB1/GB2、30步训练＋固定验证。显存是nvidia-smi采样的最高整卡值，耗时是step>2中位数，不能作为长期稳定性或单项优化的受控性能结论。

| 布局 | seq | 优化器存储 | 整卡峰值 | 步耗时 | 最终验证CE | 原始结论 |
|---|---:|---|---:|---:|---:|---|
| TP4/PP2/CP1 | 4096 | 全GPU，完整FP32主权重 | 75829MiB / 74.05GiB | 2.0712s | 7.497223 | [报告](training-smoke/audit/28b-oom-20261022/TP4-PP2-CP1-NO-OFFLOAD.md) |
| TP2/PP4/CP1 | 4096 | 全GPU，完整FP32主权重 | 79969MiB / 78.09GiB | 2.40035s | 7.499003 | [报告](training-smoke/audit/28b-oom-20261022/TP2-PP4-CP1-NO-OFFLOAD.md) |
| TP2/PP2/CP2 | 4096 | 50% Adam CPU offload，完整FP32主权重 | 79595MiB / 77.73GiB | 4.5298s | 7.502898 | [报告](training-smoke/audit/28b-oom-20261022/RESULTS.md) |
| TP2/PP2/CP2 | 2048 | 全GPU，TE int16主权重余数 | 80797MiB / 78.90GiB | 5.6694s | 7.754669 | [报告](training-smoke/audit/28b-oom-20261022/CP2-NO-OFFLOAD.md) |

注意：

- TP2/PP2/CP2的完整FP32主权重全GPU方案常驻约81.10GiB/卡，超过本机容量，首次Adam状态初始化OOM。
- CP2全GPU余数模式4K试验仍OOM（首次更新或第2步前向/CE）；只有上述2K在限制NCCL、逐步empty_cache下完成30步，余量很小。
- 余数是TE已有功能，余数本身在GPU；没有把梯度和Adam矩降为低精度。CP1及50% offload模式未启用余数存储。
- 50% offload已有保存step3、新进程续训step4/5的28B日志一致性验证；严格小模型状态逐位对照也通过。不能把它当作余数模式、TP2/PP4或完整28B状态的逐位认证。
- 这些28B短测没有上传W&B；此前2.7B长跑和质量评测有W&B记录。

## TP精度差异：事实、结论和暂停决定

原始两条30步轨迹第26步梯度范数21.372573与27.448297，相对差约28.42%，loss接近。输入哈希、独立FP64范数、确定性复跑、同权重交叉重放均已完成。

- 固定TP2只改PP4→PP2，2470个参数平方范数完全一致；固定WKV输入及入边梯度，三层真实4K的40heads与2×20heads输出/六路梯度逐位一致。
- 最早前向不一致在第0层Attention行并行`linear_proj`。之前的LayerNorm、r/k/v、WKV六路输入及输出、GroupNorm、gate、投影输入均逐位一致。
- 同输入同权重下，生产BF16投影输出的TP2/TP4相对L2差0.35718%；BF16部分结果改FP32求和后为0.30488%；部分结果与求和均用FP32、最后转BF16后为0.0035344%。局部BF16矩阵乘法和独立NCCL重放均逐位复现训练。
- 后续差异在MB0/token0的少数近抵消head上放大。第15层head62的r·k从-0.02264变成-0.00969，而逐项乘积绝对值合计约29.8。屏蔽该token直接loss后，同权重范数差由约28.10%降至8.66%。这是诊断消融，不是生产修改。
- 所测GroupNorm/receptance backward与局部独立FP64公式最大差0.385%/0.427%，未发现本次PP/范数统计/WKV切分逻辑错误。不能证明所有路径无错、两布局长期等价，或把全部28.4%量化归因于第0层一次投影；反向数值贡献及完整28B高精度参考仍未完成。
- 该差异影响真实梯度裁剪。**用户决定暂不做精度修复**，当前没有把生产TP通信/部分结果切到FP32。

证据入口：[初轮报告](training-smoke/audit/tp-pp-gradnorm-20261022/REPORT.md)、[最早差异报告](training-smoke/audit/tp-pp-gradnorm-20261022/FIRST-DIVERGENCE.md)、[机器可读摘要](training-smoke/audit/tp-pp-gradnorm-20261022/FIRST-DIVERGENCE.json)。后者是最新定位结论，前者保留较早的覆盖范围。

## CP功能、长跑及质量证据

| 工作 | 证据与边界 |
|---|---|
| CP token shift、MB>1、TP×CP+SP、普通PP payload | [验证矩阵](training-smoke/audit/cp-sp-stage2-20261021/VALIDATION-MATRIX.md)，VPP按用户要求暂缓 |
| CP4数值修复 | [recurrent状态报告](training-smoke/audit/cp4-state-fix-20261021/RESULTS.md)：2.7B全部24层/36heads控制回放WKV输出及六路输入梯度CP1/CP4逐位一致；整模型参数梯度相对L2约0.0169253%，不是逐位相同 |
| CP/CP+SP checkpoint、FP8、Muon短跑 | [验证报告](training-smoke/audit/cp-validation-20261021/RESULTS.md)，按报告列出的组合范围解读；不代表所有checkpoint/优化器组合认证 |
| CP4半天长跑 | [报告](training-smoke/audit/cp4-longrun-20261021/RESULTS.md)：**2.7B，非28B**，TP2/PP1/CP4+SP、16K，12小时49189步、约16.118亿token，无日志NaN/跳步；固定验证loss13.49113→4.067253；不是CP1收敛等价试验 |
| 长跑模型质量 | [报告](training-smoke/audit/cp4-quality-20261022/RESULTS.md)：heldout PPL显著下降，但生成仍重复，通用基准无一致提升，不能归因为CP |
| 数据来源 | [来源审计](training-smoke/audit/cp4-quality-20261022/data-source-audit.json)：0056为英文网页混合文本；确切上游集合未知，不能直接命名为FineWeb/CommonCrawl |

2.7B模型：`/mnt/data/Models/RWKV-7/rwkv7-qwen-vocab-2.7b-from-28b.pt`。
长跑W&B：[cp4-halfday-20261021](http://dian.sea-group.org:9098/mobius/Mobius-r7-28b-qwen-pretrain/runs/cp4-halfday-20261021)。

## 复跑入口和原始证据

所有命令在仓库根目录执行；会实际占用GPU。配置、模型/数据路径和训练步数先看对应报告。

```bash
# 28B TP4/PP2/CP1，全GPU，30步
bash training-smoke/audit/28b-oom-20261022/launch.sh tp4pp2cp1-no-offload
# 注意该固定phase会追加rank审计；复跑优先使用新phase并显式传同一config。

# 28B TP2/PP4/CP1，全GPU，30步，自动生成时间戳phase
bash training-smoke/launch-rwkv7-28b-qwen-tp2-pp4-no-offload-smoke.sh

# 28B CP2全GPU余数模式，仅2K，30步，自动生成时间戳phase
bash training-smoke/launch-rwkv7-28b-qwen-cp2-no-offload-smoke.sh
```

CP2 offload正式脚本`training-smoke/launch-rwkv7-28b-qwen-cp2-offload.sh`默认含在线W&B/保存和较长运行限制，不能当作与本轮30步完全相同的短测。短测用`training-smoke/audit/28b-oom-20261022/config-cp2-offload.yaml`和该目录`launch.sh`。

两处主要证据目录：

- `training-smoke/audit/28b-oom-20261022/`：完整日志、GPU CSV、逐rank JSONL、配置、退出状态和汇总；失败实验日志也保留。
- `training-smoke/audit/tp-pp-gradnorm-20261022/`：两条复跑轨迹、第26步实际权重快照、固定权重重放、前3层逐算子捕获、独立GEMM/NCCL及FP64局部公式。

`.pt/.pth`大快照、日志和部分缓存受gitignore排除，**不会因为提交源码自动上传Git**。文件路径、大小、mtime、轻量文件SHA256见[证据归档目录](training-smoke/audit/records-20261022/README.md)。保留这些大文件以便进一步定位，未删除。备份仅在本机，不能替代异机备份。

## 当前提交情况与提交建议

检查时尚未fetch远程；领先计数相对本地缓存的tracking ref。完整文件列表/HEAD/upstream/补丁见[版本现场记录](training-smoke/audit/records-20261022/README.md)。

| 仓库 | 分支 | 当前HEAD | 未提交的本次功能修改 |
|---|---|---|---|
| 主仓库 | 2026-10-10 | `0ca9d3a`，相对org同名分支领先1 | CP+SP token shift、recurrent CP选择及测试、配置/启动脚本/报告 |
| Megatron子仓库 | 2026-10-10 | `d65baf60`，相对origin同名分支无领先 | 4个已有文件＋1个新增测试：offload延迟初始化、加载状态设备同步/step及回写处理、weights_only兼容 |
| FLA子仓库 | 2026-10-10 | `261dbcf5`，相对origin同名分支领先1 | 3个已有文件＋2个新增文件：recurrent CP真实状态/梯度通信、context模式、TileLang路由及测试 |

**代码没有全部commit，两个子仓库都有修改。** 父仓库当前指向的子仓库HEAD与index一致，状态中的小写`m`主要表示子仓库内部脏改动，并不表示父仓库已经记录了这些改动。仅提交父仓库不足以保存子仓库工作区。

建议提交顺序：

1. FLA：独立提交recurrent CP功能与测试。
2. Megatron：独立提交offload初始化/保存恢复修复与测试；遵守子仓库要求的签名及格式检查。
3. 主仓库：提交CP+SP/recurrent接入、相关测试、可复跑配置和轻量报告，同时更新两个submodule指针。
4. 诊断材料可单独提交，只收脚本、配置、报告、轻量结果。大快照、训练checkpoint、缓存、W&B目录不进入普通Git。

这些功能已经有相应测试证据，适合整理成可审阅提交。用户本轮要求是记录与检查提交状态，本轮没有执行新的commit或push，没有改动index；**暂停的TP精度优化也没有新增生产修改需要提交**。`training-smoke/audit/ISSUES.md`、旧的10月12日/13日/14日诊断及不相关配置等其他工作区内容保留，不能不加区分地`git add -A`。
