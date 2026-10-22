# 按原计划逐项核对（2026-10-21）

五项功能均已实现，验证并非所有拓扑、所有张量覆盖均已穷尽。最新CP4修复与回归见 `../cp4-state-fix-20261021/RESULTS.md`；此前补充验证见 `../cp-validation-20261021/RESULTS.md`；交错PP/VPP按用户要求暂缓，试验性生产代码已撤回。

| 原计划功能 | 实现 | 证据范围 |
| --- | --- | --- |
| CP+TP，暂时关闭 SP | 已实现，0ca9d3a | 真实 2.7B TP2×PP2×CP2，SP-off，3 步 Adam + 验证 |
| token shift 增加 CP 通信 | 已实现，0ca9d3a | rank0 首 token 补零、前驱尾 token、反向边界回传；CP1 本地路径保留 |
| WKV 接入 FLACPContext | 已实现，0ca9d3a | 普通 CP group 的同 TP head state 交换，前向与反向 |
| micro batch > 1 | 已实现，0ca9d3a 的 per-sample WKV 调用 | 真实 MB2/GB4，SP-off/on 完整模型前向/反向对拍；独立样本 state，PP2 v0 carrier |
| CP+SP | 已实现，当前未提交 | 真实 TP2×PP2×CP2 的 MB1 完整/不均匀 mask、MB2 完整 mask，对拍与 3 步训练 |

| 原计划验证 | 完成情况 | 限定范围与证据 |
| --- | --- | --- |
| CP2/CP1 WKV7 输出和输入梯度 | 已扩展全部层/head控制回放 | 真实2.7B四卡TP2/PP1/CP2，seq4096，第一microbatch的24层/36head真实输入、同一确定性上游导数，六路输入梯度及输出全部逐位一致。不是所有训练batch/实际CE上游导数的捕获。`../cp-validation-20261021/summary.json`。 |
| CP2/CP1 参数梯度 | 已完成整模型对拍 | 真实 2.7B TP2/PP2，完整/不均匀 mask，1356 个 TP/PP 参数梯度张量；修复后整体相对 L2 为 0.0233%/0.0229%。`../cp-state-fix-20261021/cp2-{full,uneven}/comparison.json`。WKV 算子本身没有可训练参数。 |
| token shift CP 边界 forward/backward | 已完成 | 5 组实际多卡 token shift 测试，包含 CP-only、TP1/CP2、联合 TP×CP 和不同 rank 顺序；值和输入梯度与完整序列逐位相同。`token-shift.log`。 |
| TP1/PP1/CP2 小模型训练 | 已完成 | 两层 hidden128/head32，seq256，mock data，BF16；一步 Adam，loss12.44970，grad norm1.000，0 NaN/0 skipped。原始 `/tmp/rwkv7-cp2-tiny.{yaml,log}` 已归档 `../cp-stage1-20261021/tiny-tp1-pp1-cp2.{yaml,log}`。 |
| TP2/CP2 | 已补独立四卡用例 | 真实2.7B TP2/PP1/CP2，seq4096，3步Adam+验证，无NaN/跳步；另保留之前TP2/PP2/CP2组合证据。 |
| CP+PP pipeline payload | 已完成训练链路验证 | 早期 TP1/PP2/CP2 小模型一步 Adam，loss12.46001、grad norm1.003、0 NaN/0 skipped；原始配置和日志归档 `../cp-stage1-20261021/tiny-tp1-pp2-cp2.{yaml,log}`。另有真实 TP2/PP2/CP2 的 MB1/MB2、SP-off/on 对拍；不等于做过所有 PP 拓扑或逐消息字节审计。 |
| CP+SP | 已完成上述真实模型范围 | `RESULTS.md`、`summary.json`、`optimizer-smoke.log`。 |
| CP4 | 原生状态通信修复后严格WKV对拍通过 | TP2/PP1/CP4，SP关闭8192、SP开启16384各3步Adam+验证；全部24层/36head控制回放输出及六路输入梯度与CP1逐位一致。8192整模型捕获前向逐位一致，1356个参数梯度张量有限，整体相对L2由0.466676%降至0.0169253%。详见 `../cp4-state-fix-20261021/RESULTS.md`。 |
| CP/CP+SP保存恢复 | 两组严格一致 | TP2/普通PP2/CP2，Adam/seq4096，step2保存后新进程恢复step3/4；8rank完整参数、FP32主权重、Adam tensor states、scheduler及续训事件hash一致。 |
| FP8/Muon+CP | 功能短跑通过 | FP8 tensorwise/delayed、Muon兼容布局、FP8+Muon，与CP2+SP各3步训练+验证；核实实际调用，不代表FP8精度/长期收敛认证。 |
| 交错PP/VPP | 用户要求暂缓 | 试验性生产代码已撤回，当前支持范围仍为普通PP；不计入完成项。 |

明确剩余：CP4保存恢复、28B Adam显存问题及其后续验证、Muon/FP8与CP的保存恢复及长期收敛。交错PP/VPP按用户要求暂缓。此次短跑不能替代这些项目。
