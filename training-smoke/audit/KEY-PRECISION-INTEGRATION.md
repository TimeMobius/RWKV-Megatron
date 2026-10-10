# key FP32合并计算正式接入

2026-10-10，独立工作区`RWKV-Megatron-upstream-test`，当前Python3.12/Torch2.9.1环境，H800单卡，原始3B模型与MiniPile。原工作区、原始配置/启动脚本、模型和数据未修改；两个子模块也未修改。

## 已修改的行为

`rwkv_megatron/rwkv7_layer_specs.py`调用新的`rwkv_megatron/key_preparation.py`，将key归一化、key更新与a/b生成一起计算。中间用FP32，最后三个结果返回key投影输出的原类型。矩阵乘法保持原精度；没有把v0传递或其他中间计算改成FP32。

CUDA默认使用`torch.compile(fullgraph=True)`合并运算和自动生成的反向。每个head单独归一化，不跨样本、序列或TP分片归约；没有改参数、检查点键名或并行参数标记。CPU、没有编译器或无法编译时使用普通Torch的相同FP32公式。可在启动前指定`RWKV7_KEY_FUSION=0`固定使用兼容路径；编译失败会提示，并在本进程后续调用中保持兼容路径。只处理编译不可用/编译失败，形状错误、CUDA运行错误及后向错误不会被静默吞掉。

兼容路径保持公式和输出类型，但浮点执行方式不同，数值不保证逐元素相同。因此续训时应保持相同后端。默认路径第一次运行需要编译，实测首个训练迭代约10–11秒，后续约0.4秒；这些是128长度batch1的本次运行记录，不是速度保证。

## 验证结果

正式默认实现与上一轮测试过的合并原型，在同一完整3B、128token输入下：

- loss和logits完全一致。
- 第0层value总梯度完全一致。
- 全部546处前向记录和全部1,059个参数梯度逐元素相同。

因此此前合并原型的局部公式、精度、512样本及2048测速证据仍可用于解释这个默认实现，详见`KEY-PRECISION-TRIAL.md`。本轮没有重复测速，不承诺所有精度指标都变好或长期训练收益。

默认实现的小模型重计算对照：四层hidden256、batch2，启用与关闭重计算的loss、logits和全部135个参数梯度相同。

实际3B训练使用Adam、分布式优化器、逐层重计算。分别执行“训练2步→保存→重新启动→训练到4步”与“连续训练4步”，四步均无NaN或跳步。完整恢复对照结果如下：

| 步数 | 连续训练/保存后恢复的loss | 梯度范数 |
| --- | ---: | ---: |
| 1 | 2.744306 | 260.530 |
| 2 | 2.759152 | 192.713 |
| 3 | 2.084512 | 175.071 |
| 4 | 3.200983 | 483.346 |

loss和梯度范数在日志精度下相同；检查最终第4步的原始文件，模型、优化器、学习率调度、随机状态及`distrib_optim.pt`全部逐元素相同。这是同版本、同合并后端的续训验证，不意味着旧算法/不同后端的训练轨迹也相同。原始文件保留在新建的`training-smoke/checkpoints/single-{resume,continuous}-key-fusion`目录。

普通Torch兼容路径也完成了完整3B全部参数的前向/反向检查，所有梯度有限。此处兼容路径是纯Torch FP32，与上一轮依赖FLA key-update的普通FP32原型不是同一实现。对独立FP32参考，整体梯度差约12.20%，默认合并版约8.49%，原生产BF16约12.65%；兼容路径用于正常运行，不保证获得合并版的精度/速度收益。对官方BF16的差分别约10.68%/8.55%，这是另一个比较指标。

相关回归测试：75通过，另有30个subtest通过，包含独立公式/数值梯度、CUDA合并精度、编译器缺失与编译失败回退、运行错误传播、LayerNorm、FLA token布局、converter、并行参数标记与检查点分片元数据。后几项是本地结构测试，没有做多卡训练认证。

目前认证范围是单卡BF16及上述恢复路径；SP/TP/PP与FP8实际训练继续按后续阶段单独验证。检查点结构保持不变，所有持久参数仍由原模块维护。

## 复现与材料

生产源文件与旧审计代码已备份到`training-smoke/reference/before-production-key-fusion.tar.gz`。本次训练配置复制为`config-3b-single-{save,resume,continuous}-key-fusion.yaml`，启动脚本复制为`launch-training-key-fusion.sh`，指向新建的检查点目录。

```bash
# 正式默认实现，对固定BlinkDL官方源码/算子做完整单卡对照
bash training-smoke/audit/launch-single.sh --full --bf16 --all-gradients \
  --reference-profile blinkdl --capture-output full-production-key-fusion \
  --result-tag=-production-key-fusion

# 兼容路径：固定纯Torch FP32后端
RWKV7_KEY_FUSION=0 bash training-smoke/audit/launch-single.sh \
  --full --bf16 --all-gradients --reference-profile blinkdl \
  --capture-output full-production-key-eager --result-tag=-production-key-eager

# 已存在本次结果时不会覆盖旧检查点，重跑前请复制配置并选择新输出目录
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/run-production-key-training.py

# 比较实际续训结果
OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/compare-single-resume.py --suffix=-key-fusion

# 原型/生产完整梯度、重计算和兼容路径参考对照
OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/validate-production-key.py

# 相关回归
bash training-smoke/audit/launch-key-tests.sh \
  tests/test_rwkv7_timemix_sharded_state_dict.py \
  tests/test_rwkv_sequence_parallel_param_tags.py \
  tests/test_rwkv7_v0_tp_pp_sp.py tests/test_rwkv7_v0_tp_pp_no_sp.py
```

`production-key-validation.json`保存正式实现与原型相同的断言及兼容路径逐参数误差；`single-resume-key-fusion-results.json`保存完整续训对照；`production-key-training-exits.json`记录实际启动命令和返回值。

为保留旧对照能力，审计脚本仍能选择旧BF16和上一轮普通FP32公式；正常训练不需要审计脚本。新增测试首次构造模拟编译异常时漏传了当前Torch要求的参数，已修正测试，原失败日志保留在`audit-production-key-tests-mock-signature-failure.log`。生产代码并未因此改变公式或放宽容差。
