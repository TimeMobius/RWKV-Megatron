# 本地证据与版本现场备份

[返回仓库根目录总入口](../../../TESTING_STATUS.md)。主仓库路径：`/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test`。

本目录`archive_evidence.py`在运行时创建一个带北京时间戳的目录，并保存：

- `repository-state.json`：主仓库及两个子仓库的分支、完整HEAD、upstream、相对本地tracking ref的领先/落后计数、未提交文件和gitlink状态。没有执行fetch。
- `main-worktree.patch`、`megatron-worktree.patch`、`fla-worktree.patch`：各仓库当前HEAD至工作区的完整已跟踪文件补丁。新增未跟踪源码另存入archive，不能只用这些patch恢复全部修改。
- `*-index.patch`：HEAD至index的原始差异，保留当时是否已stage的信息。脚本不会执行git add/commit/reset。
- `evidence-inventory.json`：所列测试目录中的文件路径、字节数、mtime。纳入轻量备份的文件另有SHA256；大型张量只记路径/大小/mtime，未读取全量内容计算hash。
- `reviewable-evidence.tar.gz`及`backup-summary.json`：当前未提交源码、配置、启动脚本、轻量报告/结果、日志和以上版本记录的压缩备份，以及archive自身SHA256和字节数。

`LATEST.txt`包含最近一次快照目录的仓库相对路径。需要查看时，先读LATEST，再读该目录的repository-state、inventory和backup-summary。文件是否后来改变，可用原始路径当前SHA256与inventory比较。

重要范围：

- `.pt/.pth`权重快照和算子输入、大型checkpoint、训练cache、TensorBoard/W&B文件不复制进轻量archive；原文件保留在测试目录或报告指定路径，inventory列出选定测试目录中的张量文件。
- `.log`、部分结果目录和archive被gitignore排除；仅commit报告不能把原始现场带到其他机器。
- 该备份位于同一台机器，只提高本地可找回性，不是异机容灾。没有删除或移动原始证据。
- `ISSUES.md`和其他历史工作区内容只作现场备份，不能视为本轮提交授权或本轮所有权。
- tar恢复时应先查看成员，再解压到临时目录核对。成员使用仓库相对路径，不自动覆盖工作区。

复用命令（在仓库根目录，生成新的快照）：

```bash
/mnt/data/anaconda3/envs/train_mobius_py312/bin/python \
 training-smoke/audit/records-20261022/archive_evidence.py
```
