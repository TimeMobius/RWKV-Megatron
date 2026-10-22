# 本地提交与推送（2026-10-22）

提交作者和提交者均为`shiroko98 <512815652@qq.com>`。本轮仅创建本地commit，没有push；没有修改三处旧`2026-10-10`指针。

| 用途/仓库 | 分支 | 提交 |
|---|---|---|
| CP主仓库生产代码 | `codex/rwkv7-cp` | `f8bc564` |
| CP Megatron | `codex/rwkv7-cp` | `f30bff508` |
| CP FLA | `codex/rwkv7-cp` | `e932aadd` |
| 蒸馏主仓库路径修正 | `codex/qwen-rwkv28b-distill` | `f979779` |
| 蒸馏 Megatron | `codex/qwen-rwkv28b-distill` | `d65baf60`，无新增commit |
| 蒸馏 FLA | `codex/qwen-rwkv28b-distill` | `ff481ea6`，无新增commit |

CP主仓库另有随后的证据归档提交；用`git log -2 --oneline`查看完整主仓库最新HEAD。本文件位于该归档提交中，因此不自引用其SHA。旧CP主仓库`0ca9d3a`和FLA `261dbcf5`是新CP分支祖先，push新分支时会一起发布；不会推到旧release分支。

蒸馏本来还有两个未提交启动路径修改，并不是只有CP目录有改动；本次已将这两项、环境入口、分支声明/说明单独提交为`f979779`。蒸馏没有引入CP训练算法，两个子仓库的源码与原基线一致。为了发布同名子仓库分支，只需推送现有提交指针，不需要空提交。

Megatron按用户本轮明确授权只使用`-s` DCO签名，未使用`-S`密码学签名。

## 提交前检查

- FLA recurrent CP新回归：6 passed；修改文件Ruff检查通过。
- CP主仓库loss、PP/SP payload、token布局和静态合同：43 passed。
- TP×CP token shift真实GPU边界forward/backward：5 passed，覆盖TP2/CP1、TP1/CP2、TP2/CP2、反序rank布局、TP4/CP2。
- Megatron新offload分片初始化/更新/序列化/严格checkpoint续训：两rank各5 passed；isort检查和diff空白检查通过。
- 蒸馏入口：三个shell脚本语法通过，导入解析到本蒸馏工作区，teacher launcher帮助命令通过；没有启动teacher或新训练。

本轮没有重跑28B完整训练，提交保留之前各报告中的实测范围和限制；没有实施暂停的TP精度优化。

## 推送命令

以下命令会写TimeMobius远程，提供给用户执行，本轮尚未运行。CP先推两个子仓库，最后推主仓库，避免父仓库引用尚未发布的提交。

也可以直接执行仓库根目录`push-recorded-branches.sh`，它按以下顺序推送，任一步失败即停止。本轮只做了脚本语法检查，没有执行推送。

```bash
cp_repo=/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
distill_repo=/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-distill

git -C "$cp_repo/3rdparty/Megatron-LM-0.11.0" push -u origin codex/rwkv7-cp
git -C "$cp_repo/3rdparty/flash-linear-attention-260115" push -u origin codex/rwkv7-cp
git -C "$cp_repo" push -u org codex/rwkv7-cp

# 蒸馏子仓库无新代码，只发布指向原基线的同名分支。
git -C "$distill_repo/3rdparty/Megatron-LM-0.11.0" push -u origin codex/qwen-rwkv28b-distill
git -C "$distill_repo/3rdparty/flash-linear-attention-260115" push -u origin codex/qwen-rwkv28b-distill
git -C "$distill_repo" push -u org codex/qwen-rwkv28b-distill
```

两个主仓库的`org`与子仓库的`origin`均指向TimeMobius，主仓库另一个`origin`指向其他fork，因此这里明确使用`org`。Git commit署名与HTTPS推送凭据是两回事，实际push使用机器当前GitHub登录凭据，不会因为作者名设置为shiroko98自动切换认证账户。

## 保留的其他工作区文件

CP目录仍可能显示`training-smoke/audit/ISSUES.md`修改，以及其他历史审计、临时结果、未跟踪蒸馏文件和本地W&B目录；它们原地保留，没有纳入本次功能提交。已选择的诊断源码、配置、轻量报告/顶层汇总纳入证据归档，大型tensor、日志、逐rank捕获、checkpoint和压缩现场备份留在本机。备份与路径清单见`training-smoke/audit/records-20261022/README.md`。
