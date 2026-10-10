# BlinkDL 官方结构与算子精度对照

2026-10-10，独立工作区 `RWKV-Megatron-upstream-test`，单卡 H800。生产代码、子模块、环境、原工作区与输入材料未修改；修改前审计脚本备份为 `training-smoke/reference/audit-before-blinkdl-alignment.tar.gz`。

## 参考已修正

此前“原参考”是我手写的审计实现，共享了官方没有共享的 value view；后来加入 carrier 对齐也只是 Megatron 连接诊断，不能作为官方 BF16 精度基准。之前约0.1943%的差异，是该历史诊断图的加法分组差异，不代表官方精度已经对齐。

当前 `single.py` 默认使用 `BlinkDLReference`，从 BlinkDL/RWKV-LM 固定提交 `20be0f8cf0bdf506d8c0030c87a9a3416f95b143` 提取源码中的 TimeMix、FFN、Block.forward 函数体，并用 AST 断言函数体未修改。用 checkpoint 装配模块、跳过随机初始化与训练脚本副作用；没有再手写这些 forward 公式。

- FP32：官方 `rwkv7_train_simplified.py` 的 TimeMix/FFN 函数体，官方 `src/model.py` 的 Block 函数体；仅将 demo CUDA core 换成独立 Torch FP32 递推。head64、32层与原始3B权重适配由模块装配完成，符合官方注释的语言模型设置。
- BF16：官方 `src/model.py` 的 TimeMix/FFN/Block 函数体，以及原始 WKV clampw、mix6、key-pre、a-gate、v-residual、GroupNorm+correction+gate、FFN CUDA 算子。选择官方默认 baseline clampw，未使用可选 H100 v3。
- v_first 按官方直接复用同一对象，各层之后断言对象身份不变；不添加 gather/carrier。WKV 与输出修正分别建立 value 分支。
- LayerNorm 使用官方 Torch 实现；最终只处理 hidden。所有对照统一普通 CE，不包括官方 L2Wrap 的额外后向正则。
- 审计执行函数体/包装器的 eager 版本，剥离 MyFunction/JIT 装饰器；不是完整官方 JIT/Lightning/DeepSpeed 训练入口认证。CUDA 源码与数值编译参数原样使用，只补充 Conda 分散的 include 路径、绝对源码路径和隔离 build 目录。

原 `NativeReference` 保留为明确的 `diagnostic` 模式；默认改为分开的 value view。共享 view 仅通过 `--shared-value-reference` 作为历史负向控制启用。显式的旧诊断开关仍自动选择 diagnostic；官方测试建议明确传 `--reference-profile blinkdl`。

## 完整 3B 精度结果

真实 MiniPile 固定首个片段，seq128、batch1，32层、hidden2560、head40×64。比较全部1,059个导出参数梯度。所有“相对差”均为 relative L2，整体梯度按全部参数元素合并后的 L2 计算，不是参数误差的算术平均。

| 对照 | CE绝对差 | logits相对差 | v0总梯度相对差 | 全部参数梯度整体相对差 |
| --- | ---: | ---: | ---: | ---: |
| Megatron FP32递推 vs 官方结构+独立FP32递推 | 4.768e-7 | 7.677e-7 | 7.137e-6 | 1.236e-5 |
| 当前生产 BF16 vs 官方 BF16 CUDA | 0.00717878 | 0.840332% | 7.690537% | 13.536157% |
| 仅诊断将 Megatron norm换Torch vs 官方BF16 CUDA | 0.00553060 | 0.821928% | 9.642735% | 5.029114% |

FP32 CE分别为2.13715219、2.13715267，最大单参数梯度相对差2.317e-5，最差为blocks.7.att.w0，满足原有公式验证阈值。该证据支持公式和梯度链路正确。

生产BF16 CE为2.13547516，官方BF16为2.12829638；最大单参数梯度差43.1242%，位于blocks.31.att.r_k。这些是真正不同算子实现之间的差异，**不能宣称当前 BF16 已与官方精度对齐**，也不能以“低精度通常有舍入差异”将其直接判为可忽略。

统一Torch LayerNorm后的控制实验中，ln0、层零ln1及r/k/value原始投影完全一致，但第一处后续可见差异在传入core的更新key与归一化key向量：k相对差0.1049%、a相对差0.3129%、b相对差0.3738%，随后WKV输出0.2622%。官方key-pre融合内算与Megatron/FLA拆分操作的精度路径不同。控制实验没有消除完整模型差异，v0误差反而增大；不能据此决定将生产TE切回Torch。

## v_first传递控制与核心算子验证

为隔离连接问题，用独立写出的诊断层共享生产TE/FLA点算子/WKV及精度位置，但**不启用carrier-chain-reference**，直接复用v_first、独立建立两条value view。完整3B的546个前向记录、logits、CE、v0总梯度以及全部1,059个参数梯度均逐元素一致。四层hidden256、batch2、seq128裁剪模型的全部135个参数梯度同样一致。

这说明在相同底层原语的这两个样本中，carrier与直接v_first传递没有造成漏传/重复传播差异；它不是官方CUDA精度认证，不能替代上表第二行。

另将官方WKV CUDA和生产FLA的decay转换/WKV调用，输入同一批BF16原始r/wraw/k/v/a/b，与独立FP32普通递推比较。batch2、head2×64、seq64/128、零初始状态；输出与六种原始输入梯度均通过原有阈值：output0.007、gradient0.008，没有放宽容差。原始w梯度包括sigmoid/decay转换的链式求导。

| seq | 官方WKV输出相对FP32差 | FLA输出相对FP32差 | 官方最差输入梯度差 | FLA最差输入梯度差 |
| --- | ---: | ---: | ---: | ---: |
| 64 | 0.1623% | 0.2585% | 0.2635% | 0.5186% |
| 128 | 0.1645% | 0.2628% | 0.2648% | 0.5288% |

官方WKV在这些固定随机输入上更接近FP32；不能推广成完整模型/所有算子的精度排序。该测试不检查官方没有公开返回的final-state梯度，也不能说明其他官方融合算子或生产拆分算子都已经通过独立梯度验证。

## 结论与剩余事项

参考结构已校正，并且现在直接执行固定官方源码的forward/CUDA算子。FP32公式测试通过；v0 carrier的连接在相同原语下通过；实际官方与当前生产BF16存在值得继续定位的完整模型精度差异。后续应优先独立验证key normalize/update、v-residual gate、GroupNorm+correction+gate和FFN融合/拆分后向的精度位置与梯度，再衡量32层累积影响；不以逐位一致作为生产修复目标，也不擅自更改FLA源码或TE选择。

本轮没有训练更新、断点恢复、多卡、FP8、长序列或长期收敛认证，也没有TPS结论。每种完整模型只测一个固定片段，数值不是误差上限。

## 复现与证据

从独立工作区运行，GPU任务串行：

```bash
# 官方结构FP32公式与全部梯度
bash training-smoke/audit/launch-single.sh --full --all-gradients \
  --naive-core --reference-profile blinkdl --result-tag=-official-fp32

# 官方实际BF16 CUDA与全部梯度
bash training-smoke/audit/launch-single.sh --full --all-gradients \
  --bf16 --reference-profile blinkdl --result-tag=-official-bf16

# 一键重跑完整矩阵和WKV算子验证
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/run-blinkdl-validation.py
OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/validate-blinkdl-alignment.py
```

`blinkdl-alignment-summary.json` 保存汇总、版本、CUDA源文件hash和认证限制；`blinkdl-kernel-results.json` 保存核心算子全部梯度误差；`blinkdl-validation-exits.json` 保存成功测试的命令/退出状态。前两次官方扩展编译缺cuda_bf16.h/cusparse.h的失败记录保留在`blinkdl-preflight-exits.json`与对应日志，补include后成功，未通过修改数值实现规避。

首次测试结果文件名原为vs-recurrent，为避免误读，已按真实backend改成vs-blinkdl-main / vs-blinkdl-simplified；数据内容未改，旧新文件名与SHA256映射保存在`blinkdl-result-filenames.json`。原始测试日志保留首次生成时的名称。历史诊断证据未覆盖。
