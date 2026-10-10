# 可配置 L2Wrap 与 key 合并开关

2026-10-10，独立工作区`RWKV-Megatron-upstream-test`。原工作区、原配置/启动脚本、模型和数据未修改；FLA、Megatron子模块与环境未修改。

## 配置方法

在`--rwkv-config`读取的YAML顶层添加：

```yaml
rwkv_l2wrap_coefficient: 0.0001
rwkv_key_fusion: true
```

- `rwkv_l2wrap_coefficient: 0.0001`启用官方系数；`0`关闭。不设置时默认关闭，方便保留普通CE与旧实验。
- `rwkv_key_fusion: true`使用FP32 key合并；`false`使用纯Torch FP32兼容路径。
- YAML明确配置的key开关优先于`RWKV7_KEY_FUSION`环境变量；未配置时仍沿用环境变量，默认开启。
- 命令行`--rwkv-l2wrap-coefficient`、`--rwkv-key-fusion`或`--no-rwkv-key-fusion`覆盖YAML；同时支持YAML的`language_model`节中这两个选项，但建议放顶层。
- 配置中的系数必须有限且不小于0；key开关须是YAML布尔值，不能写字符串`"false"`。

根目录`example_config.yaml`已包含两项示例，现有原配置不被自动改写。本轮新建的训练配置明确启用L2Wrap；兼容路径试验配置明确设置key合并为false。未来续训应保持相同系数、mask策略与key后端。

## 实际做了什么

`RWKV7Model.compute_language_model_loss()`保留原输出层和CE后端，在训练时给逐token CE附加一个后向项。每个有效token只对全词表最大logit的位置增加：

`1e-4 × 最大logit × 该token传回来的梯度`

因此显示的CE不变；CE之外增加梯度，更新参数后训练轨迹会变化。没有额外token数除法，原训练入口的mask、token数、微批次归一化和loss scaling自然作用于它。被mask的token不增加正则；这是明确的有效token策略，与官方普通wrapper对全部输入token加正则的策略需要区分。无mask时公式与官方一致。负的最大logit也保留其符号；没有擅自改成只惩罚正值。

验证/推理与关闭系数时不增加此项，模型loss接口返回原CE。没有替换FLA融合输出层，也没有改优化器。

重要实现细节：某些Megatron CE后端会原地覆盖FP32 logits。先记录最大值和位置，再调用CE，避免正则误用被改写后的数值。只保存逐token最大值及索引，不为正则保留一整份logits；后向仍需生成一份同shape梯度张量，这次没有宣称融合输出层那样的显存节省。

## 词表分片与后续并行

按实际logits宽度区分完整词表与TP词表分片。完整词表直接找最大值；词表分片在TP组做MAX归约确定全词表最大值，再做MIN索引归约选择第一个全局词表位置，仅所属分片增加梯度。分片间并列最大值也只计算一次，没有收集整个词表。

PP仍在原最后输出阶段计算loss，SP保持现有输出层/层内传播路径；本次不改变它们的通信。已做真实两进程Gloo loss测试，覆盖Megatron词表分片CE、跨分片并列最大值、mask、零/非零缩放及已收集完整词表的情况。这是loss组件的TP测试，没有认证完整RWKV多卡模型的SP/TP/PP或FP8训练。

## 公式与官方实际CUDA算子的精度

单元测试覆盖不同梯度缩放、部分mask、全部mask、负最大值、并列最大值、FLA标量wrapper等价、两微批次累积、两种token归一化路径、CE原地覆盖防护、关闭/验证模式及YAML优先级。共91个测试、30个subtest通过，原容差没有放宽。

另加载固定BlinkDL提交`20be0f8cf0bdf506d8c0030c87a9a3416f95b143`的`L2WrapCrossEntropyCUDA`类及原CUDA源码。使用同一份完整3B保存的128token logits、相同MiniPile labels，比较保留Megatron CE的新接口与官方融合CE＋L2Wrap。两者均使用系数1e-4，分别测试外部梯度缩放1和8。

| dtype | CE绝对差 | 输出logits梯度相对差 |
| --- | ---: | ---: |
| FP32 | 2.38e-7 | 2.39e-7，约0.000024% |
| BF16 | 2.38e-7 | 0.000851，约0.0851% |

固定FP32阈值1e-5、BF16阈值0.004均通过。BF16的CE梯度与正则梯度分开写回/相加，官方融合版一起算完再写回，舍入差异符合这一路径；无需追求逐位一致。本测试对齐的是loss公式与logits梯度，不是宣称整个模型与官方所有算子完全一致。

## 完整训练和恢复

完整原3B、batch1、128token、Adam、分布式优化器、逐层重计算，每步两微批次（global batch2），显式YAML开启L2Wrap和key合并。执行训练2步保存，再重新启动恢复到4步，并与连续4步比较；四步loss、梯度范数与最终模型、优化器、调度和随机状态的原始张量均一致，无NaN或跳步。

另外用完整3B显式设置`rwkv_key_fusion: false`和`calculate_per_token_loss: true`，同样每步两微批次，实际训练2步成功，确认兼容路径和另一种token汇总方式能通过训练入口运行。两个路径的训练数字不应相互当作逐位一致的目标。

关闭L2Wrap的完整3B与上一版key合并实现比较，loss、logits、546处前向和1,059个参数梯度完全一致；对现有普通CE运行不引入额外计算或梯度变化。实际完整3B启用L2Wrap的梯度链通过优化器/恢复检查；另在小模型中用独立可微`0.5e-4 × max(logit)^2`检查全部参数梯度，覆盖FP32、BF16与重计算。

这些结果说明本次接口接入及缩放正确，未证明长期训练收益或FP8、多卡模型正确性，也没有因为官方loss正则存在就将此前CE算子差异归因于它。

## 复现与原始证据

```bash
bash training-smoke/audit/launch-l2wrap-tests.sh

# 只测试loss的真实两进程词表分片
AUDIT_MASTER_PORT=51287 bash training-smoke/audit/launch-l2wrap-tp.sh

# 相同完整3B logits，与官方CUDA loss对照
bash training-smoke/audit/launch-l2wrap-official.sh

# 小模型全部梯度与独立可微正则对照
bash training-smoke/audit/launch-l2wrap-model.sh

# 新建检查点路径；不会覆盖本次已有检查点
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/run-l2wrap-training.py

OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/compare-single-resume.py --suffix=-l2wrap

OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/validate-l2wrap.py
```

`l2wrap-official-results.json`记录官方CUDA精度；`l2wrap-tp-results.json`记录真实词表分片对照；`l2wrap-model-results.json`记录全部小模型参数梯度；`single-resume-l2wrap-results.json`记录完整3B恢复对照；`l2wrap-validation.json`汇总验证断言。原始数GB快照和检查点保留在忽略目录中。

生产修改前备份为`training-smoke/reference/before-l2wrap.tar.gz`。配置与启动脚本均从工作区已验证版本复制，新输出目录为`single-{resume,continuous}-l2wrap`，不修改此前测试材料。
