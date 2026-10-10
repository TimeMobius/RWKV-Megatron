# LayerNorm 后端评估

2026-10-09。在独立工作区与 train_mobius_py312 环境完成；生产实现、原仓库和输入材料未修改。单卡实际设备是 NVIDIA H800，并非根据目录名推断为 H200。

后续按用户的SP/TP/PP/FP8计划，生产默认后端已统一TE，TE缺失或低版本回退Apex。本文比较数据保留为切换前的单算子选型依据；变更与验证见 `TE-UNIFICATION.md`。

## 构建入口不是第三种实现

`self.submodules.layer_norm(config=..., hidden_size=..., eps=...)` 是选择和构建后端的入口。当前传入单层 ModuleSpec 时，上游 `_get_block_submodules` 默认填入 `LayerNormImpl`：安装 TE 时选择 TENorm，其次 Apex，最后 Torch。因此当前最终 ln_out 实际是 TE LayerNorm，与 TENorm 是同一个后端。

当前完整3B的实际类型再次捕获确认：ln0、层内 ln1/ln2 是 Megatron FusedLayerNorm（Apex）；最终 ln_out 是 transformer_engine.pytorch.module.layernorm.LayerNorm。

## 方法

- 当前代码完整3B、真实 MiniPile 固定前128 token，前向捕获 ln0、0.ln1、0.ln2、15.ln1、31.ln2、ln_out 的输入与 affine 权重。采集走当前生产路径，未切换任何模块。输入保存在 layernorm-inputs-current.pth。
- 后端比较使用相同 BF16 输入、BF16 gamma/beta、eps1e-5、hidden2560；关闭 persistent、zero_centered_gamma、sequence parallel，与已有单卡配置一致。
- 精度参考为相同量化输入/权重的 FP64 Torch LayerNorm 与 autograd。后向使用相同固定随机 BF16 输出梯度，目的是隔离算子；不是完整模型 CE 反向。
- eager 计时使用 CUDA events，5轮旋转测试顺序，每轮10次预热、40次测量，表中为中位数。包括 eager 调用/autograd 调度开销，不能当纯 kernel 时间。
- 再用 CUDA Graph 重放补充比较，每项5轮100次、10次预热，以减少主机调度影响。仅捕获单算子，未开启整模型 CUDA Graph。
- 长形状计时重复相同 ln0 输入值，不冒充实际长序列模型精度测试。

## 精度

前向输出相对FP64参考的相对L2误差：

| 捕获位置 | Torch | Apex | TE |
| --- | ---: | ---: | ---: |
| ln0 | 0.16980% | 0.57346% | 0.16980% |
| 0.ln1 | 0.17758% | 0.48662% | 0.17758% |
| 0.ln2 | 0.16661% | 0.28840% | 0.16661% |
| 15.ln1 | 0.16403% | 0.29515% | 0.16403% |
| 31.ln2 | 0.16559% | 0.30260% | 0.16559% |
| ln_out | 0.16727% | 0.27635% | 0.16727% |

当前Apex路径前向误差约为Torch/TE的1.7至3.4倍，符合之前定位的affine前转BF16行为。Torch与TE前向极为接近，但不逐位相同；两者相对L2差约1.36e-7至1.84e-5。

本次固定输出梯度下，三者的输入/gamma/beta梯度误差非常接近；相对FP64约0.159%至0.180%。不能由前向排序直接断言完整模型反向误差或长期收敛排序；不同前向会改变完整网络的上游梯度。

## 速度

单个LayerNorm前向加反向耗时，单位微秒，中位数：

| 输入形状 T,B,D | Torch eager | Apex eager | TE eager | Torch graph | Apex graph | TE graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128,1,2560 | 83.08 | 128.76 | 217.22 | 12.03 | 19.12 | 18.05 |
| 512,1,2560 | 84.41 | 128.77 | 215.49 | 14.25 | 22.25 | 30.33 |
| 4096,1,2560 | 84.09 | 127.96 | 218.56 | 77.53 | 99.46 | 110.26 |
| 4096,4,2560 | 261.33 | 317.93 | 359.84 | 258.23 | 311.44 | 354.97 |

当前形状下训练总耗时Torch最低。TE在短形状的graph纯前向上更快（seq128：TE4.34us、Torch5.84us、Apex8.40us），但这不能推导训练时TE最快。完整原始试验数据、纯前向计时和参考舍入到BF16后的额外精度指标见 layernorm-evaluation.json。

## 建议与限制

此前仅依据上游集成倾向TE，应根据实测修正：当前单卡BF16 hidden2560优先考虑统一Torch；TE作为多卡集成候选，Apex保留已有训练的兼容模式。当前没有依据直接把TE认定为性能最佳。

应统一ln0、层内ln1/ln2、ln_out的显式后端选择，避免由安装环境隐式决定最终归一化。继续使用submodules构建入口是合理的，但应传入明确选择的后端；不建议修改上游全局LayerNormImpl来影响其他模型。TimeMix中的GroupNorm必须保留，不属于本次后端统一。

多卡注意：上游WrappedTorchNorm明确拒绝sequence_parallel。因此生产采用Torch需要RWKV本地兼容封装，保留norm参数的sequence_parallel标记及已有梯度归约协议；不能直接用普通nn.LayerNorm替换然后认为SP自然正确。TE和Apex已有这方面的Megatron集成，本次没有重新进行多卡测试。

本结果只覆盖这台H800、当前环境、hidden2560、非persistent、BF16 affine参数。其他模型尺寸、FP32 affine权重、persistent kernels、融合LayerNorm+Linear、完整图捕获不在结论范围内。切换后端会改变数值轨迹，需重新做完整模型前后向与保存/恢复、端到端TPS验证；尚未证明长期收敛改善。

## 复现

从独立工作区根目录运行，单GPU任务串行：

```bash
bash training-smoke/audit/evaluate-layernorm.sh
bash training-smoke/audit/evaluate-layernorm-graph.sh
```

生产代码未切换，两个launcher均exit0，脚本语法与git diff --check通过。
