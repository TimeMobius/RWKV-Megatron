# 蒸馏工作区

本目录为 `/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-distill`，主仓库及两个子仓库均使用 `codex/qwen-rwkv28b-distill`。

主仓库从 TimeMobius 蒸馏提交 `87116dc` 创建，`f979779` 修正工作区隔离相关的启动路径与分支声明。2026-10-23 新增 TP/普通 PP/SP 蒸馏，分阶段提交和验收见 [实施记录](docs/distillation-parallel-20261023.md)。CP 不在本次蒸馏支持范围内。

当前依赖由父仓库 gitlink 固定：

- Megatron：`1cb0ccc336a0793153c54d687d53a9b0bca6e509`，在原 `d65baf60f3d715f313f644eeb68a4047f4cc42aa` 上加入此前验证过的 hybrid optimizer 初始化/恢复修复；父仓库 `45f72f2` 固定此依赖。
- FLA：`ff481ea6862c99a818858081571373694de7b4e7`，保持原版本。

`.gitmodules` 声明蒸馏分支名，实际依赖仍由这两个 gitlink 固定，声明不会自动更新版本。两个子仓库为独立本地克隆，与 CP 工作区的文件/index/分支配置隔离；兄弟目录 `RWKV-Megatron-upstream-test` 的 `codex/rwkv7-cp` 工作区保持独立。

蒸馏启动脚本按自身位置寻找当前工作区；不要使用仍包含旧目录绝对路径的历史脚本代表当前入口。多卡验证入口为 `training-smoke/parallel-distill-20261023/run.sh`。

以下命令只检查依赖导入及启动器帮助，不启动训练：

```bash
bash with-local-env.sh
bash with-local-env.sh python tools/launch_distillation.py --help
```

提交使用 `shiroko98 <512815652@qq.com>` 和 sign-off，按用户此前授权不做 GPG 签名；本轮未 push。
