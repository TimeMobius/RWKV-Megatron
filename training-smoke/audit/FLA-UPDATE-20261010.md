# FLA 上游再次更新（2026-10-10）

工作区仍为RWKV-Megatron-upstream-test；原工作区、原材料、环境软件包不改。主仓库和子模块分支均为2026-10-10，旧muon分支保留。

## 版本和影响

- 从79d12e8ec0ae72bc7f2189b4244d6532ce8c268c快进到40850297f9f977d7d33eb9d87fcf9a460554b553，增加29个提交。
- 目标是本次fetch得到的fla-org/flash-linear-attention main，提交时间2026-10-10T07:50:31+08:00。并非自动跟随未来上游更新。
- origin仍为TimeMobius/flash-linear-attention，主仓库.gitmodules URL与branch设置不变。FLA无本地自定义代码、无需merge冲突处理。
- 与当前调用相关：RWKV7 fused key update长路径网格重排、程序ID/int64地址修复、DPLR/cumsum同类修复、公共backend注册与normalization布局重构。原模块路径保留兼容导出。
- 还包含FLA自身RWKV7低精度正交初始化修复和L2Wrap dtype修复；本项目使用自己的初始化与CE，不因此新增L2Wrap训练目标。
- Python/Torch/Triton最低依赖未变，不重装环境。TE LayerNorm、FLA调用参数、v0载体和Megatron生产代码未修改。

## 冻结验证契约

生产点算子使用[1,T*B,D]，core使用[B,T,H,D]，BF16、head64、scale1、safe_gate=True、chunk_size64。独立recurrence误差门槛沿用原先output0.007、state/七种gradient0.008，没有放宽。

旧版本先重新捕获同一完整3B权重、真实MiniPile固定prefix128、batch1的546处前向、logits、选定参数梯度、v0总梯度及反向中间量；更新后重新捕获比较。参考wiring模型共享原语只用于确认连接，独立kernel FP32参考才用于算子数学误差检查。

额外运行上游test_rwkv7.py的fused_addcmul、fused_k_update、chunk_size wrapper原有用例，包含短/长序列、不同dtype和coeff形状；不改上游用例或容差。运行主仓库回归，以及完整3B真实入口4步BF16训练（seq128/full recompute/distributed optimizer），与更新前同配置日志比较。模型与数据原文件不改，复制的新训练config关闭checkpoint保存，历史续训checkpoint保留。

此轮不重新开启多卡或FP8认证，也不以编译首轮耗时作TPS结论。测试记录见fla-update-provenance.json、fla-update-comparison.json、fla-update-exits.json与kernel-results-fla-40850297.json。

## 结果和已定位的跨版本差异

- 主仓库88项回归、30个subtest通过；上游相关31个RWKV7用例通过，11个无关用例未选中。原有上游容差没有改变。
- 独立FP32递推核对完整3B全部1,059参数：CE差0，整体梯度相对L2差9.201e-6，最差单参数2.821e-5。公式参考不使用FLA core。
- FLA WKV vs独立FP32 recurrence：seq64/128/129、非零initial state、final state，七种梯度通过冻结门槛。该检查使用真实FLA core。
- 旧/新addcmul和key update：真实hidden2560、seq128/4096，输出和梯度逐元素一致；相同合法固定mix-backward配置的诊断也一致。各自同时通过FP32公式参考门槛。
- 完整3B旧/新版本的546处前向记录、logits与CE逐元素一致；CE均2.1354751586914062。BF16反向并非逐元素一致：v0总梯度相对差0.9384%，356个选定参数中最大相对差5.4152%。严格跨版本逐位检查确实失败，失败日志与exit1保留；没有将此检查重标为通过。
- 首个core梯度差异位于末层decay，其他r/k/v/a/b相同；decay相对差1.767e-6，最大绝对差2.980e-8，327680个BF16元素中363个不同。随后在多层BF16反向中放大。
- 因果诊断：只在测试进程中将新版chunk_dplr_bwd_dgk_kernel替换为旧源码，其余新版组件不动，546处前向、356个选定参数梯度、482处反向记录和v0总梯度全部恢复与旧版逐元素一致。该kernel在上游b4a0c734中将program_id(1)改为int64；数学表达式和precision契约没变，但重编译后的归约/BF16舍入不逐位相同。具体机器指令/全部autotune配置差异未逐项证明。生产未保留此override，FLA仍完全等于上游40850297。
- 各旧/新版本在fresh process重复后，logits和v0梯度分别逐元素复现。新版共享原语wiring参考546处前向逐元素一致，v0反向相对差仍约0.1943%、选定参数最大约0.3458%，与原先已记录的BF16多分支汇合问题分开记录。
- 实际4步训练完成、exit0、零NaN/跳步；LR和数据进度与旧版相同，loss/grad norm轨迹发生变化，见下表。此轮没有重跑保存/恢复，旧TE续训认证不冒充新版FLA续训认证。

| 步 | 更新前loss | 更新后loss | 更新前grad norm | 更新后grad norm |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.748121 | 2.748121 | 257.825 | 259.906 |
| 2 | 2.759416 | 2.759416 | 193.714 | 193.896 |
| 3 | 2.081873 | 2.083896 | 175.060 | 175.066 |
| 4 | 3.197702 | 3.199266 | 612.557 | 713.828 |

这里区分两个判断：独立公式、算子原容差、重复运行与短训练检查通过；跨版本BF16反向和训练轨迹不完全对齐。更新可用于继续验证，不能据此承诺长期收敛一致或精度提升，也不能把上述参数梯度变化隐藏为“全通过”。完整机器汇总见fla-update-summary.json。

## 复现

在独立工作区运行，GPU任务串行：

```bash
bash training-smoke/audit/launch-fla-update-tests.sh
OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/run-fla-update-validation.py
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/audit/validate-fla-update.py
```

旧版本捕获full-before-fla-40850297.pth和新版本捕获full-after-fla-40850297.pth留在忽略目录中。旧FLA分支codex/fla-before-20261010-update与reference/source-before-fla-40850297.tar.gz保留恢复路径。只更新本地分支/主仓库提交，不执行远程push。
