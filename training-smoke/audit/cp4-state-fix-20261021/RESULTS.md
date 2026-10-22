# RWKV7 CP4 状态精度修复（2026-10-21）

RWKV 的 CP 路径已改用原生递推状态传递。真实2.7B、CP4/8192和CP4+SP/16384的全部24层、36个head，控制回放的WKV输出与六路输入梯度均与CP1逐位一致。原先0.674422%/1.206028%的严格对拍失败已消除，没有放宽门限，也没有修改CP1的BF16数学路径。

整模型CP4/CP1捕获前向逐位一致；全部1356个TP参数梯度张量有限，整体相对L2由0.466676%降至 **0.0169253%**。参数梯度不是逐位相同，本轮不宣称这一残余已逐元素完全归因。

## 代码行为

- 主仓库 [rwkv7_layer_specs.py:513](/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/rwkv_megatron/rwkv7_layer_specs.py:513) 对CP>1选择 `dplr_state_mode='recurrent'`。使用 `dataclasses.replace` 创建上下文副本，避免修改FLA缓存的上下文；CP1使用原有本地调用，每个microbatch样本仍独立调用WKV。
- FLA [context.py](/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/3rdparty/flash-linear-attention-260115/fla/ops/cp/context.py) 增加DPLR状态模式，并在 `copy_for_backward` 中保留它；FLA默认仍为原affine路径，其他算子不改变路线。
- FLA [chunk.py](/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/3rdparty/flash-linear-attention-260115/fla/ops/generalized_delta_rule/dplr/chunk.py) 在局部块内准备之后接收前驱的真实FP32状态，调用一次现有state kernel，再将真实末状态发给后继。反向接收后继的FP32状态梯度，用原有backward state kernel计算，再发给前驱。
- 新增 [context_parallel.py](/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/3rdparty/flash-linear-attention-260115/fla/ops/generalized_delta_rule/dplr/context_parallel.py) 使用显式CP group及其全局peer rank做相邻点对点通信。新文档起始/结束处停止状态传播，支持同一rank内含多条序列的contiguous varlen。
- 新模式显式拒绝zigzag、缺少process group或未知模式；TileLang verifier将该模式交给Triton，避免选择仍使用affine汇总的实现。

这条路径保留普通kernel中BF16操作数和FP32累计状态的原运算顺序，绕过不能精确表达状态舍入的固定矩阵汇总，也绕过该汇总的TF32矩阵链。局部准备、输出和其余梯度kernel仍并行；状态kernel沿CP rank形成串行依赖。相比定位时的广播消融，生产实现没有重复执行state kernel来求边界。

没有新增模型参数、checkpoint键或用户配置项。普通PP和原有TP/SP逻辑继续使用；VPP按用户要求暂缓。

## 真实模型验证

相同权重 `/mnt/data/Models/RWKV-7/rwkv7-qwen-vocab-2.7b-from-28b.pt`、0056数据、H800、`train_mobius_py312`。各项均完成3步Adam和固定验证，无NaN、无跳步。

| 用例目录 | TP/PP/CP | SP / microbatch | seq | WKV严格对拍 | 验证loss |
| --- | --- | --- | ---: | --- | ---: |
| `strict-cp2-4k` | 2/1/2，四卡 | 关 / 1 | 4096 | 输出＋六路输入梯度逐位一致 | 13.158319 |
| `strict-cp4-8k` | 2/1/4，八卡 | 关 / 1 | 8192 | 输出＋六路输入梯度逐位一致 | 13.145907 |
| `strict-cp4-sp-16k` | 2/1/4，八卡 | 开 / 1 | 16384 | 输出＋六路输入梯度逐位一致 | 13.107208 |
| `tp2-pp2-cp2-sp-mb2` | 2/2/2，八卡 | 开 / 2 | 4096 | 输出＋六路输入梯度逐位一致 | 13.128908 |

每项覆盖全部24层和36个head。WKV对拍使用第一训练microbatch的完整真实输入，在相同确定性上游导数下回放CP1/CPn，保持之前原严格门限，并额外检查逐位一致；它不是逐层实际CE上游导数的全量捕获。MB2用例同时验证样本状态独立和普通PP的v0链路。

整模型对照为TP2/PP1、seq8192、相同初始化权重及两个真实microbatch，保留生产loss、backward与分布式优化器reduce-scatter梯度重建，跳过更新：

- 所有token的完整词表CE、完整head输入、每TP rank采样64列logits，全部逐位一致；没有导出全部词表logits。
- 1356个参数梯度张量全部有限，整体梯度相对L2 **0.0169253%**；原值为0.466676%。
- 保持原模型对拍容差不变；新汇总脚本还要求整体梯度误差<0.05%。

原始记录为各用例 `rankN.json`、`comparison-model.json`，汇总为 `summary.json`。大张量在本地 `captures/` 下并由 `.gitignore` 排除。

## 算子和回归测试

[test_cp_dplr_recurrent.py](/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/3rdparty/flash-linear-attention-260115/tests/context_parallel/test_cp_dplr_recurrent.py) 的6项pytest通过，其中一项包含6组四卡数值用例：

- CP2的非连续物理rank group `[0,2]` / `[1,3]`，以及CP4；
- 密集序列、跨rank延续文档及rank边界恰好换文档；
- BF16、FP16，K/V=64/64、64/32、128/64；
- backward重计算及缓存两条路径；
- 输出、全部六路输入梯度逐位一致且有限；上下文copy/default、非法模式和TileLang路由拒绝也有检查。

首次运行这些新dtype/geometry的Triton编译耗时较长，栈采样确认发生在原有backward kernel的编译阶段，最终全部通过；缓存编译结果后的最终复跑为6 passed / 10.67秒（`recurrent-final-tests.log`）。未修改训练conda环境；临时调试工具仅安装于 `/tmp/rwkv-cp-debug`。

现有CP context、DPLR默认affine CP2/CP4、RWKV7包装器及TileLang相关回归：**54 passed、36 skipped、7 deselected**。跳过的是本环境不可用的TileLang执行等用例，筛除的是未在本轮选入的其他数值用例；不将它们计为通过。见 `fla-regressions.log`。依赖发现列表见 `dependent-tests.txt`；共享context被广泛引用，但本轮没有修改其他算子的kernel或默认路由，没有重跑整个FLA仓库。

主仓库CP loss、TP/PP/SP payload、token layout和静态合同回归：**45 passed**。FLA和新增审计脚本Ruff、Python编译、bash语法、git diff空白检查通过。

## 耗时对照

8张H800，真实2.7B，TP2/PP1/CP4，SP关闭，seq8192、MB1/GB2、BF16、整层重算、Adam，每个模式8步；关闭WKV诊断回放。两个配置副本逐字相同，全部rank各步实际batch hash一致。旧路径只在审计进程中显式选择affine，生产默认始终选择修复后的recurrent。

排除前2步，取第3～8步迭代耗时中位数：

| 状态路径 | 中位耗时 |
| --- | ---: |
| 旧affine | 778.95 ms/步 |
| 修复后recurrent | 772.75 ms/步 |

本次差异约-0.8%，属于单次短跑的小差异，不能宣称整体训练提速；本轮未见明显减速。本轮仅测量整体迭代时间，没有分解各项开销；跨CP的状态依赖确实存在，不能由本次测量推导多机、更高CP或其他模型的扩展效率。原始八步数据和比值在 `summary.json`。

## 文件与复现

主仓库和FLA子模块均在 `2026-10-10` 分支，本轮修改尚未commit/push。Megatron子模块无修改。FLA的完整补丁（含新增helper及测试）保存为本目录 `fla-recurrent-state.patch`，适用于FLA基线261dbcf5；实际生效源码位于上述子模块工作区。

```bash
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/audit/cp4-state-fix-20261021/prepare.py
CP_VALIDATION_WKV=1 bash training-smoke/audit/cp4-state-fix-20261021/launch.sh strict-cp4-8k 8
CP_VALIDATION_WKV=1 bash training-smoke/audit/cp4-state-fix-20261021/launch.sh strict-cp4-sp-16k 8
# 复核已有结果，不重新占用GPU。
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/audit/cp4-state-fix-20261021/compare.py
```

当前证据是上述短步训练和数值回归。CP4保存恢复、长期收敛及更高CP不由这些测试替代；之前的CP2/CP+SP checkpoint证据仍见前一轮验证报告，本轮没有重新认证全部checkpoint组合。
