# RWKV7 统一 TE LayerNorm 与 Apex 回退

2026-10-09。仅修改 `RWKV-Megatron-upstream-test`，原工作区、环境与输入材料保留。按用户澄清，不添加旧 checkpoint 数值模式，也不检测或恢复旧 norm 后端。

## 实现

- `rwkv_megatron/normalization.py` 提供唯一默认选择规则：TE 可导入且版本>=1.9.0 时返回 Megatron TENorm，否则返回 Apex FusedLayerNorm 并告知回退。Apex 回退需要已安装的 CUDA 扩展。
- `rwkv7_model.py` 的 ln0 使用上述选择；构建 decoder 前显式传入 TransformerBlockSubmodules.layer_norm，覆盖上游仅按 TE 安装情况选择最终 norm 的默认值。继续沿用 submodules 构建入口，不修改上游全局 LayerNormImpl，也不在初始化后替换模块。
- `rwkv7_layer_specs.py` 的默认 local/TE 两个 spec 入口共享同一规则，ln1/ln2 与 ln0/ln_out 一致；Linear 后端仍由 transformer_impl 决定。外部自定义 spec 的层内模块仍由其作者负责。
- 移除 RWKV7 文件对 TE 的无条件导入，TE 缺失时 BF16 Linear 周围禁用FP8的context使用nullcontext，保证真正缺失TE时可以进入Apex/local计算路径。
- GroupNorm、FLA recurrence、v0传递、tokenizer、训练目标均未修改。没有添加完整FP8训练适配或宣称FP8认证；LayerNorm不会被强制改成FP8。

## 已验证

- 88项回归测试、30个subtest通过。新增11项覆盖TE可用、低版本、缺失，两种层spec入口，以及模型位置/关键字参数、PP rank/vp_stage和pre_process构建入口。
- 四层真实checkpoint裁剪模型：在正常TE、模拟低版本、fresh process阻止TE导入及可用性探测三种情况下，全部10个LayerNorm类型分别统一为TE或Apex；真实前后向有限，层零value权重有梯度，原生权重导出与135个张量shape验证通过。
- TE与Apex的norm参数保留sequence_parallel标记；zero-centered gamma分别通过前后向检查。本轮只测试标记，不重新启动多卡任务。
- 完整原始3B、MiniPile固定前128token，独立FP32普通torch recurrence和全部1,059参数梯度：CE差0，logits相对L2差8.215e-7，整体梯度相对L2差9.201e-6，最差单参数2.821e-5，v0总梯度7.359e-6。
- BF16连接对照共享相同TE norm、FLA点算子和core：546处前向记录、logits与CE完全一致，loss均2.1354751586914062。它是连接证据，不是独立数学oracle；BF16后向仍有v0汇合误差约0.1969%，所选参数最大约0.3323%，并未消除前轮记录的低精度累加差异。
- 完整3B真实训练入口：新目录single-resume-te/single-continuous-te，BF16、full recompute、distributed optimizer、seq128、micro/globalbatch1；2步保存退出再恢复到4步，与连续4步比较。四步loss/grad norm/LR/样本进度一致；最终903个模型张量、3个分布式优化器大张量、optimizer metadata、scheduler和RNG逐元素一致，零NaN/跳步。8个验证任务均exit0。最终记录在single-resume-te-results.json，汇总在te-summary.json。

| 步 | loss | grad norm |
| --- | ---: | ---: |
| 1 | 2.748121 | 257.825 |
| 2 | 2.759416 | 193.714 |
| 3 | 2.081873 | 175.060 |
| 4 | 3.197702 | 612.557 |

## 结果解释与边界

换归一化后端会改变训练轨迹。新的固定参考CE为2.135475，之前混合后端为2.133791；这两个值的变化不是续训丢失或模型公式改变。新后端应与同后端参考和连续训练比较，不能要求与旧后端逐位相同。

4步训练只能认证此短工作负载的前后向和恢复一致性，不能证明长期收敛或速度提升。SP/TP/PP执行和完整FP8训练尚未在此后端变更后重新验证，历史多卡数值/TPS不代表当前代码。

可用性回退针对TE缺失/版本不足，不吞掉已选TE后的CUDA运行错误。要求FP8或TE Linear时仍需要对应组件和硬件；Apex LayerNorm回退不能提供TE的FP8能力。

## 复现与恢复

```bash
OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/run-te-validation.py
bash training-smoke/audit/launch-single.sh --full --bf16 --reference-core fla \
  --aligned-reference --te-reference-norms --shared-pointops --te-final-reference \
  --carrier-chain-reference --result-tag=-unified
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/audit/validate-te-results.py
```

训练复现会写入本次专属te checkpoint目录。此次修改前源码备份在reference/source-before-te-layernorm.tar.gz，当前补丁/source archive及manifest见patches/README.md。本版已按用户要求保存为本地提交，尚未推送；见../COMMIT-STATE.md。
