# CP与蒸馏工作区（2026-10-22）

| 用途 | 主仓库路径 | 主仓库及子仓库当前分支 |
|---|---|---|
| CP | `/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test` | `codex/rwkv7-cp` |
| 蒸馏 | `/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-distill` | `codex/qwen-rwkv28b-distill` |

蒸馏继续使用用户原分支名，不再使用临时名`codex/rwkv7-distill`。两个主仓库为独立worktree；新蒸馏目录的两个子仓库为独立克隆，没有复制CP的脏工作文件，也不使用objects alternates。

## 当前提交

提交作者与提交者均为`shiroko98 <512815652@qq.com>`。本轮Megatron签名按用户明确授权只使用DCO `-s`，未做`-S`密码学签名。

| 仓库 | CP分支 | 蒸馏分支 |
|---|---|---|
| 主仓库生产代码 | `f8bc564`：CP+SP/recurrent接入、测试及28B布局 | `f979779`：启动路径、依赖分支声明及隔离说明 |
| Megatron | `f30bff508`：offload初始化及checkpoint恢复 | `d65baf60`：原基线，无源码修改/新提交 |
| FLA | `e932aadd`：原生recurrent CP状态通信 | `ff481ea6`：原基线，无源码修改/新提交 |

CP主仓库生产提交之后另提交轻量测试证据/索引；最新HEAD以`git log`和`COMMIT_AND_PUSH.md`为准。既有主仓库CP提交`0ca9d3a`、FLA `261dbcf5`均包含在CP分支历史中。暂停的TP精度优化没有实施。

三处原`2026-10-10`分支指针未移动：主仓库`0ca9d3a`、Megatron `d65baf60`、FLA `261dbcf5`。它们与远程同名分支的既有领先差异见[远程核对报告](training-smoke/audit/branch-isolation-20261022/REMOTE-VERIFICATION.md)。原始检查快照是历史现场，不会被当前提交覆盖。

## 依赖隔离与启动

两个工作区的`.gitmodules`及本地分支覆盖均指向各自分支名，但实际版本仍由gitlink固定。未来依赖升级需在对应子仓库提交，再回到对应主仓库提交gitlink；分支声明不会自动更新模型代码。主仓库的本地覆盖使用`git config --worktree`，避免共享config相互覆盖。

根目录`fla`、`megatron`软链接均为相对路径。蒸馏启动脚本按自身位置寻找根目录，PYTHONPATH优先使用本目录依赖；`with-local-env.sh`提供同样的环境入口。导入位置及bash语法已验证。

```bash
cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-distill
bash with-local-env.sh
bash with-local-env.sh python tools/launch_distillation.py --help
```

历史审计脚本可能仍有旧目录绝对路径。模型、数据、teacher服务、checkpoint、日志没有搬迁，也没有启动新训练。

## 发布与保留内容

本轮仅本地提交，未push。发布命令见[COMMIT_AND_PUSH.md](COMMIT_AND_PUSH.md)：先CP两个子仓库，再CP主仓库；蒸馏子仓库只发布现有分支指针，无需创建空提交，随后发布蒸馏主仓库。

CP目录中其他任务的`training-smoke/audit/ISSUES.md`、旧的10月12日/13日/14日诊断、未跟踪蒸馏实验和大张量等原地保留，没有不加区分地加入本次提交。训练脚本、测试、配置与轻量报告已按范围提交；原始大快照、日志、checkpoint和本地备份继续留在本机。不要使用`git clean`清除它们。

统一测试入口为[TESTING_STATUS.md](TESTING_STATUS.md)。分离前的源码哈希/基线核对见`training-smoke/audit/branch-isolation-20261022/`；原始现场备份见`training-smoke/audit/records-20261022/README.md`。
