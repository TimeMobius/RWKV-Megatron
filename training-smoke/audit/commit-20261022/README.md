# 提交前复核（2026-10-22）

署名shiroko98，CP生产代码与蒸馏路径修改分别提交，测试结果见test-results.json。
主仓库基础43项、TP×CP多卡边界5项、FLA recurrent CP 6项、Megatron双rank每rank5项全部通过。
FLA修改文件Ruff和Megatron isort检查通过。蒸馏导入路径、shell语法和启动器帮助命令通过。
没有重新启动28B完整训练，也没有实施暂停的TP精度优化。

原始复核日志保留本目录main.log、token-shift.log、fla.log、megatron.log，受gitignore排除；轻量证据归档提交包含本说明及机器可读摘要。
Megatron的密码学签名要求经用户明确授权，本次只用DCO -s。

提交和推送命令见仓库根目录COMMIT_AND_PUSH.md；本轮未push。
