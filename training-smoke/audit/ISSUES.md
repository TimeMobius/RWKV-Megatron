# RWKV7 审计待查问题（2026-10-09）

## 2026-10-10：FLA未使用算子的评估，默认训练未改

见`FLA-OPERATOR-OPPORTUNITIES.md`。WKV保存中间结果的完整3B前向/1,059个梯度与当前相同，模型测速约快2.6%–3.8%，可作为下一步配置候选，长序列显存成本待评估。输出修正速度持平；FLA GroupNorm慢约2%–3%；二者组合对独立FP32的整体梯度差约19.88%，高于当前8.49%，不适合默认替换。组合的小模型FP32独立公式/全部135梯度检查通过，未发现该检查范围内的逻辑错误，但完整BF16组合差异未逐点定位。TileLang对当前H800/batch1/head40/chunk64回退Triton，无需为当前配置安装。只修改审计材料，未改生产、子模块或环境。

## 2026-10-10：可配置L2Wrap接入并完成单卡训练验证

见`L2WRAP.md`。保留Megatron CE，新增有效token的最大logit梯度，随原mask/归一化/上游梯度缩放；先记录原logits最大值避免CE原地覆盖。YAML支持L2Wrap系数与key合并开关。官方CUDA loss同输入对照：FP32梯度相对差2.39e-7，BF16约0.0851%；真实两进程词表分片loss/并列最大值/mask/缩放通过。完整3B两微批次、重计算、Adam实际保存恢复与连续四步最终模型、优化器、调度、随机状态一致；另完成key兼容路径＋token汇总模式两步训练。L2Wrap默认关闭，旧普通CE数值不变；完整RWKV多卡/FP8仍待后续认证。

## 2026-10-10：key FP32合并计算正式接入默认训练

见`KEY-PRECISION-INTEGRATION.md`：正式实现与合并原型的全部1,059个参数梯度和546处前向相同；小模型重计算135个梯度一致；实际3B重计算/Adam/分布式优化器四步保存恢复对照，最终模型、优化器、学习率和随机状态全部相同。提供纯Torch FP32兼容路径及`RWKV7_KEY_FUSION=0`固定后端开关，完整3B兼容路径前向/反向通过。75个测试及30个subtest通过。默认训练已切换key计算；v0等其他精度候选仍为审计原型，SP/TP/PP/FP8实际训练认证仍留待后续。

## 2026-10-10：key FP32与合并计算试验完成

见`KEY-PRECISION-TRIAL.md`与`key-experiment-summary.json`。提高中间精度和减少中间保存是两件事：普通FP32减少BF16舍入，但局部保存增加，在完整3B的512/2048测速慢约7.0%/4.2%；合并FP32减少中间读写和反向保存，2048长度前向/反向约快2.9%。当前/普通FP32/合并FP32对官方BF16的整体梯度差：128输入13.54%/4.34%/8.55%，512偏移0为27.26%/5.92%/14.52%，512偏移512为5.73%/3.69%/3.02%。局部独立FP64公式/梯度检查通过，小模型重计算全部135个梯度一致。仍为审计原型，未切换生产默认或认证续训、优化器、多卡、FP8；不把与官方的距离当作训练质量证明。

## 2026-10-10：精度差异进一步拆解，v0 FP32汇总完成实际后向验证

见`PRECISION-DIAGNOSIS.md`。真实第0/15/31层同输入测试未发现key/value残差/输出修正/FFN公式或梯度逻辑错误；FP32公式对照通过。当前key中间计算多次BF16舍入值得优化；相同输入的FFN前向与官方逐元素相同，暂不优先修改。完整3B对独立FP32的整体梯度误差：生产12.6530%，key精度候选6.1086%，扩大FP32范围并加v0汇总9.3739%，官方BF16 3.4712%。不同指标不是单调改善，未认证训练收益或误差上限。

v0真实后向实验前向/CE及固定33路入边梯度不变，汇总误差0.545287%→0.164751%，复现离线估计；只影响层零value相连的7个参数，其他1,052个梯度不变。对整个独立FP32模型，v0误差8.3003%→8.2751%，总体影响很小。四层batch2重计算与非重计算全部135个参数梯度相同；完整3B重计算/optimizer/并行/FP8尚未认证。所有优化目前仅审计原型，生产代码/环境未修改。

## 2026-10-10：官方源码参考已校正，实际BF16精度仍有差异

详见`BLINKDL-ALIGNMENT.md`与`blinkdl-alignment-summary.json`。默认参考改为源码提取的BlinkDL forward及官方BF16 CUDA，直接复用v_first、不共享WKV/correction value view、不引入carrier。完整3B独立FP32公式对照通过全部1,059个梯度；相同TE/FLA原语下官方式直接v_first和生产carrier的全部梯度逐元素相同，支持本样本的传递正确。

实际官方BF16 CUDA对照：CE差0.00717878，logits相对差0.8403%，v0梯度相对差7.6905%，整体参数梯度相对差13.5362%；最大单参数43.1242%。这项精度问题保持开放，不能用旧诊断0.1943%的加法分组解释覆盖。统一Torch norm仍有CE差0.00553060/v0差9.6427%，后续应验证key-pre、v-residual、GroupNorm/correction/gate及FFN融合/拆分精度与后向，再判断优化。官方WKV与FLA在seq64/128上均通过原独立FP32输出/六种输入梯度阈值。原生产TE/FLA、环境和原工作区未改；未认证收敛、多卡或FP8。

## 2026-10-10：单卡 v0 梯度汇合差异已解释

见 `V0-BACKWARD.md` / `v0-accumulation-summary.json`。同一版 FLA 下，第 1–31 层回传 v0 的分支梯度与本层两路之和全部一致。生产为三路梯度累加，原参考共享 value reshape、先合并本层两路，BF16 加法分组不同；重放捕获梯度精确复现原先约0.1943%的总梯度差。只对齐参考分支后，完整3B全部1,059个参数梯度逐元素一致。该固定样本没有发现漏传、重复传播或detach；不为bit-exact改变生产结构。BF16后续层连续累加的精度仍有优化空间，FP32汇总尚未实现或认证。此结论与下述新旧FLA内核反向差异独立。

## 2026-10-10 FLA再次更新的数值记录

当前FLA已从79d12e8e更新至40850297，增加29个上游提交，没有自定义FLA代码。详情见FLA-UPDATE-20261010.md。完整3B前向仍逐元素一致，但BF16旧/新反向v0梯度相对差约0.9384%，选定参数最大约5.4152%；4步训练轨迹也不同。

最早core差异位于末层decay梯度，已通过只替换旧chunk_dplr_bwd_dgk_kernel的测试恢复全部记录梯度逐元素一致，定位到其int64程序ID变更触发的重编译数值变化。两版算子都通过原独立FP32参考容差，数学表达式未变，生产不保留旧kernel override。长期收敛影响、新版续训与多卡/FP8认证尚未验证；不与原先单卡v0多分支汇合差异混为一谈。

当前按用户要求暂停多卡排查，转向单卡原版公式、FLA 调用、梯度与恢复一致性核对。

## 1. 完整 3B：TP 改变后的 BF16 数值/梯度差异尚未解释

这是待排查的数值差异，尚不能确定是实现 bug 或低精度舍入放大。不能将“能训练”当作精度完全对齐。

固定原始 checkpoint、seq128、batch1、随机 token/随机标签：单卡 loss 12.33319664，TP2+SP+PP2 loss 12.32866096，差 -0.00453568。logits 相对 L2 差 1.7407%；选定参数梯度（包含第 0 层 value 全矩阵、各层归一化/mix/v-bias）相对 L2 差 10.7187%；第 0 层 value 权重梯度差 7.2170%；v0 梯度差 5.2719%。未比较完整 3B 的所有参数梯度。

已核实：v0 数值在记录的 18 处保持完全不变；PP 的发送/接收激活和反向梯度逐元素相等；没有看到 v0 的 detach 或传输丢失。原始捕获：full-3b-dense.pth、full-3b-tp2-sp-pp2.pth；日志 ../logs/audit-full-3b.log。

真实 MiniPile 首样本前 128 token：单卡 loss 2.13773966；TP1+PP2 相同；TP2+PP2（SP off/on）均 2.13636303。详细梯度对照尚待整理，输入在 minipile-input.json，日志 ../logs/audit-full-3b-minipile-partial.log。完整多卡 FP32 对照在暂停检查时已自行结束，日志保留；本轮报告聚焦单卡，暂不据此结案多卡问题。

较小的四层真实 RWKV 模型（原 3B 权重裁剪，head64、batch2、seq128）中：FP32 独立递推下 TP2/4+SP+PP2 与单卡的 logits 最大差约 1.2e-6，全部参数梯度相对 L2 差约 3.4e-7；单独加 PP 后输出和全部参数梯度完全一致。这个证据不能替代完整 3B 的精度结论。

## 2. 已修复的审计脚本问题：完整 FP32 参考未覆盖全部层

最初的完整 3B FP32 测试，脚本先筛选需要记录的层，再替换核心递推，导致只替换了 0/1/15/16/31 层，其余层仍用 FLA。发现后终止了该诊断运行；这些结果不作为独立 FP32 参考。

已调整为先替换全部层的核心递推，再筛选记录层。修正后的完整单卡 FP32 参考已完成，MiniPile loss 约 2.1372；该任务在收到暂停检查时已自行结束；结果保留，当前不继续多卡排查。此问题在审计脚本中，不是 RWKV 生产实现的修改。

## 3. 已确认的环境与链接

8 个 GPU rank 的实际模块路径均为独立工作区内新版 Megatron d113016、FLA 79d12e8；Python/Torch/TE/Apex 来自 train_mobius_py312。路径证据 runtime.json、distributed-runtime.json。基线对照有意导入旧快照和 Python 3.11，记录在 baseline-runtime.json，不能与新版启动入口混用。

## 接下来单卡排查

逐项对照 BlinkDL/RWKV-LM 与 TimeMobius/vllm codex/rwkv7-sp-fast：权重布局、token shift、decay、key update、v0/v_first、GroupNorm、输出修正、LayerNorm/ChannelMix、FLA 参数与 dtype；追踪首个前向和梯度差异；再验证单卡训练保存/恢复的 loss、参数、优化器、RNG 与数据进度。

## 4. 已确认并修复：FLA 点算子输入布局导致错误 decode 分派

`RWKV7TimeMix` 使用 `[T,B,D]`，FLA `fused_addcmul_rwkv7` 与 `fused_k_rwkv7` 约定 `[B,T,D]`，并以 `shape[1] == 1` 判断单 token decode。microbatch=1 的整段训练因此走备用 PyTorch 公式，microbatch>1 走 Triton。它不必然交换 token 数值（点算子无递推），但改变分派、BF16 中间舍入和性能；不能称为正确遵守算子契约。

根因证据：独立参考统一 ln0/ln1 与权重布局后，首层核心 r/v/log-decay/a/b 完全相同，原版传入 WKV 的 k 与参考相对 L2 差约 0.31%。layout 修复后 k 差降到约 7e-9、首层 WKV 输出逐元素一致。

修复仅在独立工作区：将这两个点算子的 token 轴展成 `[1,T*B,D]` 再调用，输出 reshape 回原形。这样无需交换样本，独立点运算和梯度保持原契约，SP 后本地 token 数和 gather 后全序列 token 数均适用。新增 `tests/test_rwkv7_fla_token_layout.py` 覆盖 batch1/3 的数值与梯度。FLA 核心 WKV 已显式转换 `[B,T,H,D]`，该调用布局正确。

旧版与先前更新版都有这个接入问题。修复会改变 microbatch=1 的 loss 轨迹，已有性能/loss 基准全部属于修复前，不能拿它们冒充本次修复后结果。当前只验证单卡；修复后多卡重新验证留待后续。

## 5. 归一化与 BF16 舍入差异已定位，但不等同于结构公式错误

完整 3B + MiniPile seq128：独立原生 FP32 递推 vs Megatron 同样替换为 FP32 递推，loss 差 4.77e-7，logits 相对 L2 7.74e-7，所选参数梯度最大相对 L2 1.73e-5，层零 value 权重梯度差 5.34e-6。数学公式通过这项对照。

BF16 独立原版慢公式与 Megatron 的第一处可见差异在 ln0（Apex 与 torch LayerNorm）；同样数学公式分步 BF16 中间结果和融合 FP32 内算一次舍入也会不同。已验证仅统一 LayerNorm 不能消除全部差异。继续逐层追踪具体舍入位置，不能以“能训练”或旧版一致作为正确性判据。

## 6. 原版训练目标有 L2Wrap，当前训练没有

BlinkDL 当前训练默认 CE 后向还给最大 logit 注入 `1e-4 * max_logit / token_count` 的梯度（缩放 grad_loss）；CE 前向数值不变。Megatron 当前只有 CE，没有这项 logit 正则。即使 logits 和 CE 相同，也不能要求它与原版含 L2Wrap 的总梯度完全相同。本轮数学比较统一为普通 CE，未擅自增加训练正则。

## 7. 单卡断点续训：修复前已逐元素验证，修复后正在重跑

原实现完整 3B、seq128、micro/globalbatch1，2 步保存→恢复到 4 步 vs 连续 4 步：loss/grad_norm/LR/样本进度相同，最终 903 个模型张量（2,948,065,280 元素）、分布式优化器 3 个大张量（8,844,195,840 元素）、scheduler、RNG 均相同。证据 single-resume-results.json。layout 修复后使用新的 `*-fixed` checkpoint 目录重新测试，避免混用。

Apex 差异源码证据：`training-smoke/vendor/apex-master/csrc/layer_norm_cuda_kernel.cu` 的 `cuApplyLayerNorm_` 写出 `gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i]`，在 gamma/beta 仿射前先转为输出 dtype。原生 PyTorch LayerNorm 舍入时机不同。此项尚未修改生产归一化，避免把数值行为切换与 FLA 布局 bug 修复混在一起。

## 本轮单卡结案记录

详情 `SINGLE-GPU.md`、机器可读汇总 `single-summary.json`。修复后的完整3B独立FP32对照覆盖全部1,059个导出参数梯度；CE差2.384e-7，整体梯度相对L2差1.040e-5，first-value总梯度差6.535e-6。没有发现v0覆盖、detach或断链。

统一BF16原语、相同投影分支和precision位置后：546处前向值、logits、loss逐元素一致。BF16反向末层和后续记录一致，剩余first-value梯度相对差0.1955%，所选参数最差0.3691%，集中在层零多路径梯度汇合；未宣称bit-exact。原生参考重算key投影会改变BF16参数梯度汇合顺序，诊断脚本已改成共享同一次投影，相关较大梯度差的早期日志不作为最后结论。

FLA布局bug已修复；修复后的单卡2步保存/恢复到4步与连续4步，模型、optimizer、scheduler、RNG全部逐元素相同。77项回归通过。多卡相关数字属于修复前，后续应在修复后的代码上重测；当前按用户要求没有重新启动多卡任务。

原生参考PyTorch/Apex/TE混用的数值位置已解释。生产归一化和L2Wrap训练目标没有在本轮切换；优化项见单卡报告，未报告修复后的TPS提升。
