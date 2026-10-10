# 单卡 RWKV7 公式、FLA 调用与续训审计

key合并计算现已正式接入默认训练，内部FP32、输出保持原类型，编译不可用回退纯Torch FP32。完整3B与原型的1,059个梯度/546处前向完全一致；实际四步训练的保存后恢复与连续训练最终模型、优化器、调度和随机状态相同。见`KEY-PRECISION-INTEGRATION.md`；下文“仅审计原型”等描述是历史状态。

后续key精度与合并试验见`KEY-PRECISION-TRIAL.md`：相同实际输入上的局部误差和额外保存减少；完整3B的128/两段512输入整体梯度差均改善，但不同精度指标不单调变化。2048长度前向/反向测速约提升2.9%，小模型重计算135个梯度一致；仅审计原型，默认训练未切换。

最新差异定位与FP32汇总实测见`PRECISION-DIAGNOSIS.md`：未发现本轮所查算子的公式/梯度逻辑错误；key中间精度候选将相对独立FP32的整体梯度差从12.6530%降到6.1086%。v0实际后向汇总误差0.545287%→0.164751%，前向/CE不变，但整体梯度误差几乎不变，属于次优先优化。注意这里的参照是独立FP32，和下文相对官方BF16的13.5362%不是同一指标。所有精度优化仍仅在审计进程中启用。

最新参考修正与精度测试：见 `BLINKDL-ALIGNMENT.md`。当前默认直接加载固定BlinkDL源码的forward及BF16 CUDA算子，v_first直接复用、value分支分开，不用carrier。完整3B的FP32公式/全部梯度通过；相同TE/FLA原语下直接v_first与carrier全部梯度一致。但实际官方CUDA vs 当前生产BF16的CE差0.00717878、v0梯度相对差7.6905%、整体参数梯度相对差13.5362%，尚未认证精度对齐。下文历史“原参考”是手写诊断脚本，不能当官方代码本身；之前共享view造成的0.1943%差异与实际官方算子差异分开记录。

2026-10-09。工作目录为 `RWKV-Megatron-upstream-test`；原仓库、原环境与输入材料未修改。按用户要求，本轮聚焦单卡，之前多卡差异另存 `ISSUES.md`，不以旧版结果作为正确性标准。

后续变更：本报告下方原始结果使用旧的 Apex 层内 norm + TE 最终 norm。当前默认构建已统一为 TE>=1.9.0，缺失或版本不足则统一回退 Apex；更新后的验证见 `TE-UNIFICATION.md`。历史结果不替代新后端验证。

2026-10-10 后向定位更新：第 0 层 value 的三路梯度与原参考两路梯度，因 WKV/correction 是否共享 reshape 分支而有不同的 BF16 加法分组。31 个后续层 v0 分支及本层两路之和均一致，捕获梯度重放能逐元素重构两边总梯度。仅对齐诊断参考的 view 分支，完整 3B 的全部 1,059 个参数梯度完全一致；生产代码无需为逐位对齐修改。详情与精度优化评估见 `V0-BACKWARD.md`。

## 参考源码

- BlinkDL/RWKV-LM，提交 `20be0f8cf0bdf506d8c0030c87a9a3416f95b143`：RWKV-v7/train_temp/rwkv7_train_simplified.py、train_temp/src/model.py、rwkv_v7_demo_rnn.py。
- TimeMobius/vllm，codex/rwkv7-sp-fast 分支，提交 `f3aa8bb7de0b87d6eecc3f64fd7ae30a110623c8`：vllm/model_executor/models/rwkv7.py、tmp_rwkv7_pure_torch_generate.py。

源码只作参考，保存在忽略的 `../reference-repos/`；没有安装或修改这两个仓库。vLLM 以推理为主，其 FP32 recurrent/key/correction 不能直接作为 BF16 训练逐位相同的目标。`reference-sources.json` 记录来源。

## 已确认的公式

| 项目 | Megatron 实现与参考关系 |
| --- | --- |
| token shift | 前一 token 减当前 token，序列开头填零，样本之间不混合 |
| six-way mix | `x + (shift-x) * x_r/w/k/v/a/g` |
| decay | 原版 `w=-softplus(-z)-0.5` 后的 `exp(-exp(w))`，等价于 `exp(-exp(-0.5)*sigmoid(z))`；FLA 接收 log-decay `-exp(-0.5)*sigmoid(z)` |
| LoRA | decay=tanh；a/v=线性两级后 sigmoid；gate=先 sigmoid 再 up projection |
| key | 先对原始 `k*k_k` 分 head L2 normalize，再 `k*(1+(a-1)*k_a)`；归一化不可使用更新后的 k |
| recurrent | FLA `r,w,k,v,-kk,kk*a`，`scale=1`、safe_gate=True、chunk_size64；状态为 K,V，与原版 RNN V,K 写法互为转置 |
| v0/v_first | 第 0 层 value 投影，后续层 lerp 到各层 value；不每层覆盖，不 detach |
| GroupNorm | 每 token、每 head，eps=head_size*1e-5；本 checkpoint head64，eps=64e-5 |
| output correction | `sum(r*k*r_k, head_dim)*v` 加在 GroupNorm 后，再乘 gate、output projection |
| block/FFN | ln0 仅最前，ln1→TimeMix→残差，ln2→shift→ReLU² FFN→残差，最后 ln_out/head |

## 找到并修复的接入 bug

原代码把 `[T,B,D]` 直接传给 FLA 的点算子，而 FLA 按 `[B,T,D]` 的 `shape[1]==1` 判断 decode。microbatch1 的长序列训练错误走备用 PyTorch 分支，microbatch>1 却走融合 Triton，影响 BF16 舍入与性能。

已在 `rwkv_megatron/rwkv7_layer_specs.py` 加入 `mix_inputs` 和 `update_key`，将独立 token 展成 `[1,T*B,D]` 后调用，再还原。点算子逐 token 独立，因此展平不引入跨样本或跨序列依赖。WKV 核心原本已正确转换 `[B,T,H,D]`，未改核心 recurrence 或 FLA 源码。

修复前统一前置计算后，首层更新 k 的相对 L2 差约 0.31%；修复后降到约 1e-8、首层 WKV 输出完全一致。`tests/test_rwkv7_fla_token_layout.py` 覆盖 batch1/3、数值/梯度及正确分派形状。完整回归 77 passed。

**历史 30 步 TPS/loss 表属于这项修复之前。** 不能将其作为当前版本的性能或数值结论；本轮没有继续多卡性能测试。

## 单卡实测证据

完整原始 3B、32 层、hidden2560/head40×64、真实 MiniPile 首个固定训练样本的前 128 token、batch1。权重路径与 tokenizer 未改；具体 token ID 在 `minipile-input.json`。此外使用裁剪四层、hidden256/head4×64、batch2 做便宜的全部参数对照。

1. 独立原生 torch FP32 递推 vs Megatron 将 WKV 核心同样换成 torch FP32 递推：完整 3B loss 差 `2.384e-7`，logits 相对 L2 `7.934e-7`。1,059 个输出参数梯度全部比较，整体相对 L2 `1.040e-5`，最差单参数相对 L2 `2.290e-4`（embedding）。第 0 层 value 权重梯度相对差 `5.434e-6`，first-value 总梯度 `6.535e-6`。这验证公式和梯度链路；不以同一 FLA 核心互比冒充独立参考。详见 `full-fp32-naive-vs-recurrent-all-gradients.json`。
2. FLA WKV vs 独立 FP32 recurrence/autograd：batch2、head64、seq64/128/129，前向、final state，以及 r/w/k/v/a/b/h0 七种梯度通过上游冻结误差阈值（output0.007，state/gradient0.008）。输出相对 L2 约 0.24–0.25%，w 梯度最差约0.70%。见 `kernel-results.json`。这些是低精度算子近似，不能要求 bit-exact。
3. 逐层定位 BF16 差异：原生 torch 与 Apex LayerNorm 在 ln0 即有差异；Apex CUDA 源码先将 normalized 值转输出 dtype，再做 gamma/beta。仅换 LayerNorm 不足以消除差异，还存在点算子融合/未融合、中间值精度和 FMA 差异。
4. 精确 wiring 对照：独立写出的原生层公式，诊断模式共享相同 FLA 点算子、Apex 层内 LayerNorm、TE 最终 LayerNorm、时间优先布局和计算顺序。完整 3B 的 **546 处中间记录、logits、CE loss 逐元素相同**，loss均 `2.133791208267212`。这只证明计算连接与相同原语一致；原生默认 FP32 公式对照才是独立公式证据。
5. BF16 后向不是逐位一致。直接复用 v_first 与 carrier 图的汇合/累加顺序不同；对照脚本对齐投影共享分支后，末层记录的梯度完全一致，剩余差异集中在 layer0/value 汇合。详细当前结果见 `full-bf16-fla-vs-fla-aligned-apex-shared-pointops-te-final-carrier-chain.json`；不隐藏这项差异，也未放宽 FLA 测试容差。

FLA 宣告 FP32 chunk 在部分平台不支持；本轮公式基准使用普通 torch recurrence，FP32 FLA 测试仅用于定位其内部 tensor-core 近似，不能当精确 FP32 oracle。

## 单卡保存、恢复一致性

分别测试修复前与修复后完整 3B 的真实 Megatron 训练入口：TP1/PP1/DP1，seq128、micro/globalbatch1、BF16、full recompute、distributed optimizer。第 2 步保存并退出，重启继续到 4 步，与从同一 pth 起步连续 4 步比较。

修复后结果：

| 步 | loss | grad norm |
| --- | ---: | ---: |
| 1 | 2.748408 | 248.902 |
| 2 | 2.765609 | 196.382 |
| 3 | 2.081463 | 173.755 |
| 4 | 3.203146 | 342.335 |

恢复/连续的 loss、grad norm、LR、消费样本进度在打印精度下相同；最终 **903 个模型张量、3 个分布式优化器大张量、scheduler、RNG 逐元素相同**。零 NaN/跳步，三次 launcher exit0。证据 `single-resume-fixed-results.json`、`single-fixed-exits.json`、`../logs/audit-single-*-fixed.log`。模型 key 数903与导出 RWKV key数1059不同，原因是 mix_coeff_input 拆成6个 key，以及去掉层零无用 v bottleneck。

只证明本工作负载4步和相同拓扑、torch checkpoint格式；不等同于跨版本或跨拓扑恢复认证。seq128 的训练索引与原seq4096样本索引不同，因此训练 loss 不与前面的独立128token片段直接混比。

## 原版训练目标差异

BlinkDL 当前训练的 L2Wrap 在 CE 后向注入最大 logit 正则：每 token 最大 logit 的梯度加 `grad_loss * 1e-4 * max_logit / token_count`，前向 CE 不变。当前 Megatron 仅CE，没有 L2Wrap。它不会解释同 logits 下的 CE 前向差异，但会导致对原版训练总梯度不一致。本轮所有公式对照统一普通CE，未擅自修改训练目标。

## 优化空间（尚未作速度承诺）

- LayerNorm 后端已有后续单算子评估，见 `LAYERNORM.md`：当前H800/BF16/hidden2560下Torch与TE前向比当前Apex更接近FP64参考，Torch单算子训练耗时最低。生产后端未切换，不能将该结果直接换算为完整训练TPS或多卡结论。
- 本次布局修复恢复融合算子分派，但实际端到端TPS需重新固定基准测量。
- 当前每次 TimeMix/ChannelMix 构建 TE FP8 recipe，即使 BF16/FP8关闭；可在不改变数值的前提下缓存或跳过无用 context。
- 最终 LayerNorm 当前对 hidden+v0 carrier 一起运行，然后丢弃 v0 norm结果；可以只规范 hidden，避免无用 norm/内存，需要复测checkpoint和后向。
- 完整词表 logits 的显存和 CE 可参考原版 chunk head/CE；梯度累加次序、L2Wrap开关、token mask必须单独验证。
- 六路输入混合、GroupNorm+correction+gate、FFN 可进一步融合；原版与vLLM有实现，但推理 kernel 不可直接当训练 kernel。先固定上述误差/梯度门槛，再测性能。
- v0若改为FP32 carrier/梯度汇合可减少低精度累加误差，但影响显存和未来PP通信，当前未改变。

## 复现

所有命令从独立工作区根目录运行，GPU任务串行。

```bash
# 完整3B独立FP32公式和全部梯度
bash training-smoke/audit/launch-single.sh --full --naive-core --all-gradients

# 独立写出的原生层公式，统一原语以检查精确前向连接
bash training-smoke/audit/launch-single.sh --full --bf16 --reference-core fla \
  --aligned-reference --apex-reference-norms --shared-pointops \
  --te-final-reference --carrier-chain-reference

# 单卡保存、恢复、连续训练（更新本工作区*-fixed测试目录）
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python training-smoke/audit/run-single-fixed.py
OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/compare-single-resume.py --fixed
```

`single.py` 默认不改变生产norm；`--torch-norms` 仅在本次实例中 monkeypatch 作归因。独立默认参考不使用Megatron/FLA，但精确连接诊断模式有意共享原语。测试没有导入vLLM运行时；从其源码核对公式和精度位置。
