# CP4 12小时长跑状态

状态：completed；阶段：finished。
更新（UTC）：2026-10-22T00:41:10.712682+00:00。

配置：真实2.7B/0056，8×H800，TP2/PP1/CP4，SP开启，seq16384，MB1/GB2；BF16、Adam、lr=1e-6、整层重算。训练进程720分钟后保存退出。

最后训练记录：`{"iteration": 49180, "ms": 868.3, "loss": 3.86583, "grad_norm": 13.487, "skipped": 0.0, "nan": 0.0}`。
日志区间累计跳步=0.0，NaN步=0.0；记录的loss/grad norm有限=True。

初始固定验证：`[{"iteration": 0, "loss": 13.49113}]`。
最新周期验证：`[{"iteration": 48000, "loss": 4.080681}, {"iteration": 48500, "loss": 4.075543}, {"iteration": 49000, "loss": 4.069746}]`。
退出checkpoint重载验证：`[{"iteration": 49189, "loss": 4.067253}]`。

完整曲线见tensorboard/和训练日志；GPU显存/利用率/功耗采样见gpu.csv。status.json包含每100个日志记录的耗时中位数与全局tokens/s。计时沿用生产日志，会包括日志区间内的其他开销，不能替代CP1/CP4受控性能对照。

学习趋势根据固定验证loss判断；单次长跑不能证明与CP1长期等价。baseline和周期验证使用相同数据配置、固定前32条16K样本；最终验证重载checkpoint，但不等同于Adam/RNG连续训练恢复对拍。

本任务不改生产源码，不自动删除checkpoint，不自动重试失败的训练。

## 完成长跑结论

2026-10-22 05:57（北京时间）结束，完成49189步，处理1,611,825,152个token。训练在720分钟上限正常保存退出，baseline/train/final-eval进程退出码均为0。

固定32条16K验证样本loss从13.49113下降至4.067253，在该配置和样本范围内表现出明确的学习趋势。全部日志区间无NaN/跳步，10个checkpoint保存成功，最终checkpoint新进程重载验证通过。记录的生产源码起止hash一致。

第100步起日志耗时中位数865.9 ms/步，约3.78万tokens/s（8卡合计）；GPU采样峰值22315 MiB/卡。训练后段未见持续耗时增长或显存增长。这些是本次运行的观测，不代表CP1对照或更大规模扩展结论。

W&B：[长跑指标](http://dian.sea-group.org:9098/mobius/Mobius-r7-28b-qwen-pretrain/runs/cp4-halfday-20261021)。2026-10-22训练结束后补传，共4921条history记录，含100次固定验证（初始、98次周期和最终重载）。W&B导入耗时不代表训练耗时；训练原始起止时间已写入run config。`train/loss_instant`来自TensorBoard的每10步采样值，`train/loss_interval_mean`来自stdout的日志区间均值，两者定义不同。
