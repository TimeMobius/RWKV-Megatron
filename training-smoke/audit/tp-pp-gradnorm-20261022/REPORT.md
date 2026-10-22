# 28B TP4/PP2 与 TP2/PP4 第26步梯度范数差异定位

2026-10-22，`train_mobius_py312`，分支`2026-10-10`。本轮只新增诊断材料、修改文档，没有修改生产计算代码；源码哈希与上一轮28B测试一致。

后续已进一步确定最早差异：第0层Attention的行并行`linear_proj`局部BF16输出及BF16 reduce-scatter。该投影之前的输入、WKV、GroupNorm和gate均逐位一致；真实通信可在独立进程逐位重放，详细证据见 [FIRST-DIVERGENCE.md](FIRST-DIVERGENCE.md)。下文保留初轮定位范围和当时未完成的事项，以便区分两轮证据。

## 结论

原日志21.373与27.448的差异可精确复现，是实际裁剪前梯度的差异。两次运行的第26步输入一致，独立FP64范数复算准确；保持TP2只改变PP划分时逐参数范数一致，固定WKV输入/入边梯度的head切分也逐位一致。

目前定位到TP改变后的BF16数值路径差异，在第一个microbatch的第一个token、少数接近抵消的head上放大，并通过后续层/递推反传扩大梯度。重点位置是全局layer15/head62、layer19/head12、layer23/head48（代码从0计数）。

不能把它解释为“只有日志精度”或“只是训练25步后权重不同”：固定第二条轨迹的同一份权重，改变TP仍有28.10%的范数差。当前证据支持数值敏感性，未发现本次PP划分、范数重复计数或WKV head分片公式错误；尚未将最早出现前向差异的每个算子逐一消融，也没有做完整28B的FP32/FP64参考或全参数梯度向量对照。不能据此认证两布局严格数值等价或长期收敛等价。

## 原运行的确定性复现与统计检查

- 配置、28B模型、0056数据、seq4096/SP/MB1/GB2/full recompute均沿用上一轮。
- `tp4pp2`与`tp2pp4`各重新训练30步，完成验证，正常退出。第1、24、25、26、27步保存每参数梯度平方范数、TP属性及输入哈希。
- 第26步独立FP64范数分别为21.372572778621585和27.44829721124429；优化器内部值分别为21.37257194519043和27.44829750061035。统计舍入远小于观察到的差异。
- 两布局首尾stage的tokens/labels/mask/positions哈希完全相同，两个microbatch顺序一致。
- 第26步前保存实际BF16计算权重，重新加载、跳过前50个训练样本后重放；同布局重放精确恢复原范数，逐参数平方范数相同。
- 所有范数都在优化器更新/裁剪之前读取；DP=CP=1，无读取未归约分片的问题。TP复制参数只计一次。

## 固定权重与固定数据交叉重放

两列使用相同第26步batch，但按对应布局重新分片同一份权重，不执行优化器更新。

| 第26步使用的权重来源 | TP4/PP2范数 | TP2/PP4范数 |
|---|---:|---:|
| TP4/PP2训练25步后的快照 | 21.3725727786 | 21.4450733605 |
| TP2/PP4训练25步后的快照 | 21.4276223524 | 27.4482972112 |

第一行差约0.339%，第二行差约28.098%。说明存在权重状态与布局数值路径的相互作用，不能只将28.4%归因于优化轨迹已经不同。

## PP控制与WKV切分控制

固定第二份权重、TP2、batch与SP，仅将普通PP4改为普通PP2：

- PP4范数27.44829721124429，PP2范数27.448297211244277。
- 2470个去除TP重复后的参数平方范数最大绝对差0。
- PP2控制只用4张卡。由于每卡模型切片更大，该控制使用Hybrid CPU优化器存储以避免无用的GPU主权重分配，但完全跳过optimizer.step；模型前后向仍BF16，main_grad仍CUDA FP32。该控制不是TP2/PP2全GPU训练成功的声明。

第15、19、23层真实4K/MB0 WKV输入及实际入边梯度，分别按40heads和两组20heads重放：输出及r/k/v/w/a/b六组梯度均逐位一致。见`wkv-head-split.json`。这排除了所测WKV输入上的head切分差异，不能替代全部层/全部数据的覆盖。

## 误差集中位置

固定第二份权重时，layer15的receptance权重梯度范数从0.48817变成9.74194，key权重梯度从0.49023变成8.80739。

MB0的layer15 receptance输出梯度，TP2整张量范数0.104514；单独head62就占0.104503，最大位置为token0。MB1的梯度范数约0.00186，两布局接近。layer19/head12、layer23/head48也集中于MB0/token0。

同权重、同输入时，layer0的key/receptance/value投影及WKV输出在两布局间逐位一致；layer3及之后已经有前向差异。layer15三路投影的整体相对L2约0.60%–0.67%。因此当前证据将前向差异的最早区间夹在layer0 WKV之后至layer3投影之前，尚未对区间中的GroupNorm、gate、行并行投影、MLP及BF16归约逐个归因。不能将全部差异量化地归给单一次NCCL求和。

## 为什么局部小变化可以产生明显的范数差

初始state为零时，单个head的首token WKV满足：

```text
S0 = k0 ⊗ v0
o0 = (r0 · k0) v0
```

第15层head62：

| 首token局部量 | TP4 | TP2 |
|---|---:|---:|
| r、k逐项乘积绝对值之和 | 29.81549 | 29.79460 |
| 使用捕获输入在FP64计算的r·k | -0.0226423 | -0.00968742 |
| 绝对项之和 / 点积绝对值 | 1316.8 | 3075.6 |
| 生产WKV输出范数 | 0.0573895 | 0.0198697 |
| GroupNorm逆标准差（含eps） | 38.0860 | 39.3469 |

乘积项很大而和接近零，BF16输入/中间值的小绝对变化可对应很大的局部相对变化。第23层head48的点积甚至从+0.0810163变成-0.0235901。

两布局第一个microbatch首token CE分别11.494982和11.369009。仅一个token的差异会被8192个训练token的平均loss稀释，但它的梯度可以影响前层权重及后续递推。

只屏蔽MB0/token0的直接loss，保持同一权重、完整序列前向、其他token及递推路径：

| 对照 | TP4/PP2 | TP2/PP2 | 相对范数差 |
|---|---:|---:|---:|
| 完整loss | 21.42762235 | 27.44829721 | 28.10% |
| 仅MB0/token0直接loss置零 | 21.42149325 | 23.27634451 | 8.66% |

额外平方范数差缩小约71.83%；mask同时使该microbatch有效token数少1，因此它不是严格的正交梯度能量分解。剩余差异尚未逐项分解，首token仍可经状态影响后续token的loss；不能把这个诊断mask当成生产修复。

## 独立局部FP64公式检查

对上述三个head、两种TP布局，使用捕获的GroupNorm输入、入边梯度和同一份gamma，在独立FP64算式中复算GroupNorm backward。

对receptance首token梯度，使用独立闭式公式，包含WKV与WKV外的RWKV输出修正：

```text
dr0 = k0 * dot(d_WKV_output0, v0)
    + k0 * rho * dot(d_GroupNorm_output0, v0)
```

六组检查中，GroupNorm backward相对FP64公式最大L2差0.38534%，receptance backward最大0.42728%。最异常的TP2/layer15/head62，实际receptance梯度范数0.104503，FP64局部公式0.104389；TP4对应0.00180489与0.00180444。

因此所测局部梯度放大与传入梯度及数学公式一致，没有发现28%量级的局部backward公式偏差。该检查固定的是已舍入的生产输入和入边梯度，不能当作端到端FP64训练参考。

## 影响与后续方向

原配置clip_grad=1。第26步两条原轨迹的裁剪系数约为1/21.3726与1/27.4483，后者低约22.1%。因此差异会影响优化器收到的真实梯度和后续状态，不能只忽略日志。

本轮没有切换生产通信或算子精度。若继续修复，优先对layer0 WKV之后的数值路径逐项做固定输入消融，评估TP归约、GroupNorm/输出修正融合等位置的精度；同时关注敏感首token/head。完整28B高精度参考、全梯度向量对照与长期收敛仍未完成，不能保证简单开启某一处FP32就解决全部差异。

## 复现材料

- `capture.py`、`launch.sh`：训练轨迹、固定权重/样本重放及首token loss消融。重放跳过optimizer.step；三种`config-*.yaml`保持train_iters30以保留数据索引，使用exit_interval提前结束。
- `tp4pp2.log`、`tp2pp4.log`：完整复跑轨迹。
- `replay-*.log`、对应目录：固定权重交叉重放。
- `ablate-*.log`、对应目录：PP控制、仅首token loss置零、逐token CE和GroupNorm捕获。
- `compare_stats.py`、`inspect_layers.py`：范数及head/token定位。
- `replay_wkv_split.py` / `wkv-head-split.json`：WKV head切分逐位对照。
- `check_token0_formula.py` / `token0-formula-check.json` / `token0-formula-fixtures.pt`：独立FP64局部公式及小型重放输入。
- `norm-step26.json`、`replay-common-tp2-comparison.json`、`same-weights-forward-differences.json`、`token0-sensitive-heads.json`、`SUMMARY.json`：机器可读结论。
- `provenance.json`：生产源码一致性及诊断文件哈希。

原始大快照保留，便于继续定位/修复。`before-config-fix.log`保留四卡CPU优化器控制缺少必需flag的启动失败，修正配置后通过；`wkv-head-split-before-rank-fix.log`保留诊断脚本全局rank路径错误，修正后3层全部通过，均未计入成功结果。
