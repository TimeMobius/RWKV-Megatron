# 新旧 FLA 的 RWKV7 算子与优化机会

2026-10-10，独立工作区，H800单卡，当前完整3B模型。此次只做源码对比和审计进程中的试验，默认训练实现、两个子模块和环境均未修改。

## 结论与优先级

最值得继续接入的是可配置的WKV内部重计算开关：保存中间结果能减少后向重复计算，本轮完整3B前向/梯度不变，模型前向＋反向速度提高约2.6%–3.8%；代价是保存更多中间数据，长序列/微批次及重计算配置需要单独评估。

输出修正融合算子可以用，但本轮完整模型速度持平，精度收益也不稳定，暂不建议默认切换。FLA GroupNorm可以用，局部精度更好，但当前完整模型慢约2%–3%，组合精度反而恶化，暂不推荐。新TileLang后端存在，但不适合直接替换当前H800、batch1、chunk64的训练调用。

当前WKV已经使用新版Triton核心，包括新版safe-gate计算；没有一个尚未启用、可以直接保证更快的“新版RWKV7训练算子”。未来长序列若输出logits成为显存瓶颈，FLA融合Linear＋CE＋L2Wrap是另一方向，但适配范围显著大于本轮的点算子试验。

## 对比的是哪些版本

- 原muon版本：`3cf180339b8a1cbad823f553541cd531d18670ea`，2026-01-23。
- 上一次FLA更新前：`79d12e8ec0ae72bc7f2189b4244d6532ce8c268c`，2026-09-28。
- 当前版本：`40850297f9f977d7d33eb9d87fcf9a460554b553`，2026-10-10。

从原muon到当前增加了TileLang DPLR/WKV后端、safe-gate相关实现改进、融合loss词表分片等。最后一次79d12e8→40850297的29个提交，RWKV7相关主要是程序ID/地址处理与共享后端重构，没有新增一套独立的RWKV7 point operator。不能把所有当前尚未使用的算子都称为刚新增。

| 功能 | 新旧关系 | 当前是否使用 | 评估 |
| --- | --- | --- | --- |
| WKV Triton chunk计算 | 新版更新了safe-gate求解、边界/地址、后端等 | 已用`safe_gate=True, chunk_size=64` | 保留，已有更新的核心收益 |
| TileLang WKV | 相对原muon新增，提交`5337b5ad` | 未用，未安装TileLang | 当前配置会被上游拒绝并回退Triton |
| WKV `disable_recompute` | 原muon底层DPLR已有；新版RWKV7入口暴露参数 | 默认不关闭内部重计算 | 本轮收益最明确，值得做可配置选项 |
| `gate_output_correction` | 原muon已有，后来修复mask/地址等 | 当前Torch多步计算 | 可接入，当前速度持平，不优先切换 |
| FLA GroupNorm | 原版已有，当前调整模块组织/后端 | 当前为Torch GroupNorm | 本轮速度较慢，暂不默认切换 |
| FLA整套ChannelMix | 原muon已有，后续修复越界 | 当前Megatron Linear＋Torch激活 | 不能直接替代TP/TE Linear路径，不优先 |
| 融合Linear＋CE＋L2Wrap | 原版已有，新版增加词表分片/归约等 | 当前保留Megatron输出层＋CE＋自有L2Wrap | 长序列显存方向，须单独适配和测试 |

## 完整3B速度：真正接入后比较

保持当前已完成的key FP32合并，batch1，使用真实MiniPile序列。三个point候选和WKV候选都仅在进程里替换forward，没有改持久参数。测量包含模型前向、普通CE、后向及清梯度；不含优化器更新、数据/检查点IO，L2Wrap关闭以固定精度对照条件。

预热后五种方案轮换，四轮、每轮每种方案四次；首次编译不计入。512长度不启用层级重计算，2048长度启用逐层完整重计算。

| 方案 | 512长度 tokens/s | 相对当前 | 2048长度 tokens/s | 相对当前 |
| --- | ---: | ---: | ---: | ---: |
| 当前生产实现 | 2328 | — | 6256 | — |
| 仅FLA输出修正融合 | 2324 | -0.2% | 6256 | 约0% |
| 仅FLA GroupNorm | 2270 | -2.5% | 6120 | -2.2% |
| GroupNorm＋输出修正融合 | 2265 | -2.7% | 6108 | -2.4% |
| WKV保存中间结果 | 2417 | +3.8% | 6421 | +2.6% |

部分轮次有明显波动，表中使用中位数；不能将一次结果承诺为任意模型/硬件/序列下的加速。2048长度，当前四轮约326.1–328.3ms，WKV保存方案约317.6–320.7ms，支持这一配置下有小幅收益。

WKV保存方案2048长度峰值分配显存约11.66GiB，当前约11.62GiB。这里启用了逐层重计算，只反映这一模型/配置的峰值；局部算子在2048长度、40head×64时，为后向额外保留约178.3MB。取消层级重计算或增加序列/微批次，保存成本会增长，不能只用约0.04GiB峰值差估计全部场景。

局部算子测试使用当前完整3B第15层真实128token输入，按时间维重复扩展至512/2048；这部分不是长上下文模型输出。关闭内部重计算的局部前向＋后向耗时约减少20%，但完整模型只提高约3%。融合输出修正/GroupNorm减少前向计算次数并不等于前向＋反向更快；Python调度、后向归约、布局和重算开销都需要计入，完整模型测试才是选择依据。

## 精度：只融合并不保证整体误差继续下降

从当前完整3B第0/15/31层取实际输入，断言重建的现有输出与保存记录相同。当前和FLA候选共享输入及后向输入，用独立FP64公式比较。

- 输出修正：当前输出相对误差约0.20%–0.28%，FLA约0.17%–0.19%；最差单项输入梯度约0.38%–0.47%，FLA约0.17%–0.24%。
- GroupNorm：当前输出约0.09%–0.29%，FLA约0.09%–0.17%；最差单项输入梯度约0.23%–0.50%，FLA约0.17%–0.21%。

这里是相同输入上的局部改善，不是完整模型的误差上限或训练收益。

完整3B、同一128token样本，比较全部1,059个参数梯度与此前保留的独立官方结构＋Torch FP32递推参考：

| 方案 | 全部参数梯度整体相对差 | logits相对差 | v0梯度相对差 |
| --- | ---: | ---: | ---: |
| 当前key合并后的生产实现 | 8.4914% | 0.7053% | 8.1523% |
| 仅输出修正融合 | 8.1380% | 0.7381% | 9.9350% |
| 仅FLA GroupNorm | 8.2407% | 0.7178% | 8.9610% |
| GroupNorm＋输出修正融合 | 19.8797% | 0.7061% | 7.8240% |
| WKV保存中间结果 | 8.4914% | 0.7053% | 8.1523% |

单独换两种point算子只小幅改善整体梯度差，其他指标可能变差；一起换时整体梯度差明显增加。组合方案不适合默认切换。增加FP32内部计算或减少舍入，并不保证其他BF16误差一起降低；本轮没有逐项追踪组合差异的完整传播链，因此不能把19.88%的原因说成已经完全定位。

为检查组合接入是否改变公式/漏传梯度，额外使用四层hidden256/batch2的独立FP32递推与官方源码体进行对照：CE差0，logits相对差2.10e-7，全部135个参数梯度整体相对差2.05e-7，最差参数3.84e-6。固定公式检查通过，没有在该FP32检查中发现公式/连接错误；BF16完整模型差异仍如实保留。

WKV保存方案与当前完整3B的CE、logits、v0总梯度、全部546处前向和全部1,059个参数梯度逐元素一致。局部WKV也通过此前冻结的独立FP32递推阈值，输出小于0.007、六种输入梯度小于0.008，没有放宽。这种“少做重复计算”的收益比改变point运算舍入更容易评估。

## 为什么TileLang目前不值得直接开启

当前环境没有TileLang。即使安装，当前FLA的`DPLRTileLangBackend`对H800（计算能力9.0）的chunk64也明确拒绝：上游实测该配置一般不比Triton更好，选择回退。当前`safe_gate=True`但未声明紧的`lower_bound`，也不能自动满足该后端的chunk64检查。

本轮在实际H800上直接执行了后端的形状资格检查，未安装/运行TileLang核：

- batch1、40head、head64，chunk16/32也因GPU并行块太少而被拒绝；batch1的40个块小于66的检查界线。
- 即使添加RWKV7已知的衰减下界`-0.6065306597`，chunk64仍因H800路径较慢而被拒绝。
- batch2、40head、chunk16/32可以通过资格检查；这只表示“允许尝试”，不证明在当前模型中更快。需要另装隔离依赖、比较不同chunk的精度/性能之后才适合决定。

因此这次没有安装额外依赖，也没有强行绕过上游判断。未来硬件、微批次或chunk策略变化时可以重新评估。

## 其他不能直接替换的算子

1. FLA `l2_norm`默认计算`x / sqrt(sum(x²)+1e-6)`；官方Torch `F.normalize`是`x / max(norm(x),1e-12)`。两者公式不是完全相同。真实输入的差可能很小，但极小输入测试中，Torch归一化后norm为1，FLA默认为约0.008。当前key合并已经保留官方公式，不建议为了一个单独归一化核换回默认FLA l2_norm，或重新增加中间BF16保存。
2. 泛用norm＋gate融合需要核对gate激活与运算顺序。RWKV7需要`(GroupNorm(WKV)+输出修正)*gate`，不能改成`GroupNorm(WKV+输出修正)`；gate本身也不能额外套sigmoid/swish。普通norm-linear融合不能直接跨过这个修正项。
3. FLA完整ChannelMix内部用普通矩阵乘法，绕过现有Megatron/TE TP Linear及梯度累积接口，还需要处理SP的shift边界。本轮未将其接入或测整套FFN，不把它作为现成训练替代；前序同输入FFN检查已没有公式错误。
4. `fused_mul_recurrent_rwkv7`用于推理递推，原muon也已有；不能作为chunk训练反向的直接替代。
5. 融合Linear＋CE＋L2Wrap新增了词表分片支持，可减少完整logits的保存。batch1、16384长度、65536词表的BF16 logits本身约2GiB，因此长序列时有评估价值；但当前Megatron输出Linear、TE/main_grad、TP、loss_mask/归一化及FP8要一起适配，本轮没有替换，仍建议后续独立做显存目标的试验。

## 接入限制与复现

FLA GroupNorm输出来自自定义反向函数的view，当前`output += ut`不能直接沿用。首次完整试验因此报错，审计原型将这项改成普通`output = output + ut`后通过；不是修改FLA公式。失败日志保留`audit-fla-candidate-model-benchmark-inplace-failure.log`和`fla-candidate-first-exits.json`，退出1没有被改成通过。

审计point算子把独立token展平为`[1,T*B,D]`，保持权重、v0载体和线性投影不变；全模型精度用普通CE，未将L2Wrap目标变化混入算子比较。默认生产代码仍保持提交`4c88b23`的行为。

```bash
# 真实输入局部精度和速度
bash training-smoke/audit/launch-fla-candidates.sh

# 源码历史、实际H800后端资格、l2_norm公式差
bash training-smoke/audit/launch-fla-features.sh

# 完整模型测速与四种候选的官方精度对照，串行
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/run-fla-candidates.py

# 组合候选的独立FP32公式/梯度检查
bash training-smoke/audit/launch-single.sh --naive-core --all-gradients \
  --reference-profile blinkdl --fla-candidate norm-correction \
  --result-tag=-fla-candidate-fp32-formula

# 需要之前保留的完整FP32参考及本轮原始快照
OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/analyze-fla-candidates.py
```

证据文件：`fla-features-results.json`、`fla-candidates-results.json`、`fla-model-benchmark-results.json`、`fla-candidate-summary.json`与`fla-candidate-exits.json`。精度对照和完整梯度原始快照保留在忽略目录。本轮没有新增优化器/断点恢复认证，也未重新开启多卡/FP8认证。
