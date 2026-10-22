# 蒸馏工作区

本目录为`/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-distill`，主仓库及两个子仓库均使用原分支名`codex/qwen-rwkv28b-distill`。

主仓库从TimeMobius远程蒸馏提交`87116dc`创建，本地新增提交仅修正工作区隔离相关的启动路径与分支声明，没有引入CP训练功能。

两个子仓库仍保持原始版本且无源码修改，无需新增commit：

- Megatron：`d65baf60f3d715f313f644eeb68a4047f4cc42aa`。
- FLA：`ff481ea6862c99a818858081571373694de7b4e7`。

上述提交同时等于远程各自`2026-10-10`基线。`.gitmodules`声明蒸馏分支名，但实际依赖仍由父仓库的这两个gitlink固定；声明不会自动更新版本。

两个子仓库为独立本地克隆，与CP工作区的文件/index/分支配置隔离。若需要在TimeMobius发布蒸馏子仓库的同名分支，只需push现有分支指针，不需要创建空提交。CP改动及测试证据保留在兄弟目录`RWKV-Megatron-upstream-test`的`codex/rwkv7-cp`分支。

两个蒸馏启动脚本`training-smoke/qwen-rwkv-qwenvocab/launch.sh`和`run-long.sh`按自身位置寻找本工作区；不要使用仍包含旧目录绝对路径的历史审计脚本来代表当前蒸馏入口。

以下命令只检查依赖导入及启动器帮助，不启动训练：

```bash
bash with-local-env.sh
bash with-local-env.sh python tools/launch_distillation.py --help
```

2026-10-22分支隔离没有移动原`2026-10-10`指针。提交时使用`shiroko98 <512815652@qq.com>`，本轮未push。
