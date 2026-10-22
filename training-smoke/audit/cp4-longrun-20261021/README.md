# CP4 半天长跑

2026-10-21启动，环境为 `train_mobius_py312`，主仓库和FLA均在 `2026-10-10` 分支。使用已修复的recurrent CP路径，不改生产源码。

配置继承上一轮 `cp4-state-fix-20261021/config-strict-cp4-sp-16k.yaml`：真实2.7B权重、0056数据、8×H800、TP2/PP1/CP4、SP开启、seq16384、MB1/GB2、BF16、Adam、lr=1e-6、整层重算。优化器和学习率沿用短跑，主要观察该配置的长时稳定性和验证loss趋势，不代表调参后的最佳收敛速度。

执行顺序：

1. 初始权重的固定验证，16个eval iteration，共32条16K样本。
2. 训练进程运行720分钟，或先达到100000步上限。每500步验证同一批32条样本，每5000步保存完整模型、Adam与RNG；时间到后额外保存最终checkpoint。
3. 新进程重载最终checkpoint，验证同一批样本。此项检验checkpoint读取及模型验证，不是Adam/RNG连续恢复对拍。

`entry.py` 仅统一eval-only和训练阶段的数据集目标长度计算，避免 `skip_train` 改变validation shuffle index。模型、loss、前后向与优化器仍由原 `pretrain_rwkv7.py` 执行。各阶段使用相同train_iters、eval_interval、eval_iters、seed和cache目录，保证固定验证可比较。

后台任务使用tmux会话 `rwkv-cp4-halfday-20261021`，无需保持聊天或终端连接。`supervise.py` 每30秒记录GPU显存、利用率和功耗并刷新 `status.json`、`RESULTS.md`。baseline/train/final-eval任一阶段失败则保留日志并标记失败，不自动重跑。完成后自动更新汇总；不自动删除checkpoint。

查看当前状态：

```bash
cat training-smoke/audit/cp4-longrun-20261021/RESULTS.md
tail -f training-smoke/audit/cp4-longrun-20261021/train.log
```

详细训练/验证曲线写入 `tensorboard/`；GPU采样见 `gpu.csv`。`provenance.json` 和源码快照/补丁记录实验起点，`state.json` 记录起止源码hash。不要在运行期间修改生产源码。

耗时汇总排除前100步，按100条训练日志（通常1000步）分窗给出中位step耗时和全局tokens/s。日志区间可能包含验证/保存相关开销；本轮只有一条CP4长跑，没有同期CP1或旧affine长跑对照，不能据此宣称受控性能或长期收敛等价。半天长跑完成前，任务状态保持“运行中”。
