# 固定每次验证使用的样本

2026-10-10，独立工作区`RWKV-Megatron-upstream-test`，当前H800单卡环境。原工作区、原配置/启动脚本、模型和数据未修改。

## 使用

在`--rwkv-config`读取的YAML顶层添加：

```yaml
fixed_eval_samples: true
```

默认`false`，不设置时保留现有行为。命令行`--fixed-eval-samples`和`--no-fixed-eval-samples`可覆盖YAML。要求真正的YAML布尔值，不接受字符串`"true"`。

每次验证使用验证集开头的同一组样本，组大小仍由`eval_iters × eval_global_batch_size`决定；不同数据并行rank读取相应分片。断点恢复后也从开头读取，不因历史`consumed_valid_samples`跳过。训练数据和test迭代器不重置，验证样本数量、mask、loss计算和报告格式保持原方式。

保持数据、tokenizer、seed、seq_length、验证迭代数和验证批次配置相同，才是同一组的比较。训练中权重会变化，所以固定样本并不意味着每次eval loss相同；它用于排除“验证样本换了一组”造成的波动。

## 原来的行为及改动

原来内置`single`/`cyclic`验证迭代器一直往后读，后一次eval接着上次位置；恢复后按已消费验证样本数构建sampler。

新逻辑仅在开启参数时：

1. 构建验证DataLoader时从样本0开始。
2. 使用可重启的`FixedEvalDataIterator`，继承原`RerunDataIterator`。
3. 每次非test的`evaluate()`开始时，重置验证sampler位置、DataLoader自有随机generator及底层iterator，丢弃前次预取的数据。
4. 验证次数和已处理验证样本统计继续递增；这些统计不再作为固定验证的采样偏移。

没有缓存整套验证样本，每次通过原DataLoader读取。顺序和循环sampler、多worker预取、虚拟PP使用的嵌套iterator列表及无数据rank的None均处理。多验证集保留每个iterator各自的起点。外部自定义迭代器没有统一重启协议，开启参数时明确报错，不假装固定成功。

这里固定的是样本与内置GPT数据的内容；任意自定义随机数据增强还需自身保证确定性。当前MiniPile/GPT数据没有这类随机增强。

修改Megatron子模块的验证入口与新增iterator辅助文件，主仓库参数解析接入YAML。子模块仍指向TimeMobius/Megatron-LM-RWKV7的`2026-10-10`分支，提交与主仓库gitlink一同保存。FLA、模型算子、环境未修改。

## 测试

- 主仓库98项测试、30个subtest通过，其中新增固定验证测试覆盖single/cyclic、worker0/2、两个数据分片、部分验证中止后的重启、模拟新进程恢复、嵌套列表/None、全局随机状态不受影响、YAML和命令行优先级。
- Megatron子模块3项专用单元测试通过，使用torch.distributed.run执行。以`--confcutdir`运行数据无关的测试，避免顶层fixture下载无关数据集。
- 完整原始3B、MiniPile、batch1、global batch2、seq128，启用L2Wrap、key合并、层级重计算与Adam。每步做一次验证，每次2个eval迭代、总4个样本，记录实际返回的tokens/labels/mask/positions的哈希，而不只比较sampler索引。
- 固定模式训练2步保存，再重新启动恢复至4步，与连续4步比较：各次eval的四个样本哈希均相同，恢复后仍读同一组。
- 每个固定运行的第一次eval立即在相同权重下重复，样本哈希和原始eval loss完全一致。这个额外重复只在审计脚本中执行，之后恢复验证统计计数；正常训练不多做eval。
- 关闭参数的实际3B对照继续读取新组，连续各次eval组哈希不同；第一次与固定组相同，确认未改变验证集的开头。
- 固定模式保存恢复的eval loss与连续运行对应时刻完全一致；最终模型、优化器、学习率调度和随机状态原始张量均相同。没有改变训练进度或为了重启验证而扰动全局随机状态。

本轮仍聚焦单卡实际训练；两个数据分片与虚拟PP列表属于iterator级别检查，没有扩大为多卡RWKV/FP8训练认证。

## 文件与复现

```bash
# 主仓库回归与固定验证sampler测试
bash training-smoke/audit/launch-fixed-eval-tests.sh

# Megatron专用测试，使用一个torchrun进程
bash training-smoke/audit/launch-fixed-eval-megatron-tests.sh

# 实际训练固定/恢复/连续/原行为四组；不会覆盖已有新检查点
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/run-fixed-eval-training.py

OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/compare-single-resume.py --suffix=-fixed-eval

OMP_NUM_THREADS=8 /mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
  training-smoke/audit/validate-fixed-eval.py
```

配置从此前已验证的L2Wrap配置复制为`config-3b-{save,resume,continuous,rolling}-fixed-eval.yaml`；启动脚本也复制，原文件不改。样本哈希与loss记录保存为`fixed-eval-{phase}-batches.json`，汇总为`fixed-eval-validation.json`；检查点对照为`single-resume-fixed-eval-results.json`。

备份位于`training-smoke/reference/before-fixed-eval.tar.gz`。采用Megatron仓库的`mcore-testing`和`mcore-linting-and-formatting`技能；isort通过`uv run --no-project --with isort==5.13.2`执行，新文件通过Black检查，格式工具隔离运行，训练环境依赖不变。
