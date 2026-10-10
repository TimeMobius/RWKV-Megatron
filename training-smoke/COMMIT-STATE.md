# 本地提交版本

2026-10-10。本工作区是同一主仓库的独立 Git worktree：

- 工作区：`/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test`。
- 主仓库与两个子模块均使用新本地分支：`2026-10-10`；原有 `muon` 分支保留。
- 主仓库基点：`34e447cdc186a2763ae3e56f3b776772be8f719c`，对应本次核实的 `org/muon` 远程头。
- `org`：`https://github.com/TimeMobius/RWKV-Megatron.git`；`origin`：`https://github.com/shiroko98/RWKV-Megatron-muon.git`。
- 原工作区仍在 `muon`、提交 `68c9991`，其中两个用户未提交文件原样保留。本分支以文件补丁同步了该 converter 提交。

这次只创建本地提交，尚未推送分支。Megatron 本地适配也单独提交后由主仓库 gitlink 引用；FLA 直接引用冻结上游 `79d12e8e`，没有额外源码提交。上游基点与 Megatron 本地补丁提交见 `environment/provenance.json`。

子模块 origin 已改为 `https://github.com/TimeMobius/flash-linear-attention.git` 和 `https://github.com/TimeMobius/Megatron-LM-RWKV7.git`，upstream 继续保留 NVIDIA/fla-org。主仓库使用 `org` 指向 TimeMobius，`origin` 保留原 fork；新分支的 pushRemote 指定为 org。`.gitmodules` 同步使用上述明确 URL，并将两个子模块的更新分支设为2026-10-10。

今后远程发布时应先发布两个子模块分支，再发布主仓库分支，否则远程克隆无法获取仅存在本机的Megatron提交。不能向 NVIDIA 上游直接推送。此轮未执行push，因此不能把本地新分支称为已存在于GitHub的远程分支。

提交包含生产适配、converter 同步、TE/Apex 选择、回归测试、配置和启动脚本副本、轻量审计结果及环境版本记录。模型、数据、checkpoint、native构建树、运行日志与压缩源码备份由ignore规则留在本机。

验证依据是此前已完成的88项回归测试（30个subtest）、完整3B全部参数FP32对照、统一TE的546处前向连接对照、TE低版本/缺失回退，以及新后端4步连续与恢复逐元素一致。提交前重新核对结果、源码散列、原输入散列和diff检查。新版多卡/FP8完整训练尚未认证。

本次新提交的Author、Committer和Signed-off-by均为用户指定的 `shiroko98 <512815652@qq.com>`。先前新建的Megatron本地提交已按用户要求amend修正身份；未改写原有muon历史提交。Megatron要求 `-s -S`，使用本机SSH密钥签名；不等同于GitHub账号认证签名，不将密钥加入仓库，不修改全局Git配置。通过提交时的配置和本地allowed-signers文件验证。
