# TimeMobius远程提交核对（2026-10-22）

使用`git ls-remote`直接读取三个TimeMobius远程，核对实际返回的分支SHA；再依据相同SHA检查本地Git对象树、祖先关系和子模块gitlink。没有执行fetch、更新remote-tracking ref、切分支、commit、push或重置工作区。原脏改动中2544个未有意更新的轻量文件SHA256仍与分支隔离前快照一致；另外两项是此前有意更新的todolist/测试索引文档。

## 远程基线与本地新分支

| 仓库 | 远程`2026-10-10` | 本地蒸馏`codex/rwkv7-distill` | 本地CP `codex/rwkv7-cp` |
|---|---|---|---|
| 主仓库 | `69aa22a55f2858f302bf60b42213c8b23da4c642` | `87116dcb9b81e72c253e7116a4f4463ed14a454b` | `0ca9d3aeea1cfbc64589f1dca3a08dbf7e6846fa` |
| Megatron | `d65baf60f3d715f313f644eeb68a4047f4cc42aa` | 同远程基线 | 同远程基线，另有未提交offload修复 |
| FLA | `ff481ea6862c99a818858081571373694de7b4e7` | 同远程基线 | `261dbcf5091dce3486c5bdee561328184b6920e0` |

远程主仓库蒸馏分支`refs/heads/codex/qwen-rwkv28b-distill`实际返回`87116dcb9b81e72c253e7116a4f4463ed14a454b`，与本地新蒸馏分支HEAD完全一致。

## 蒸馏核对

远程主仓库的`2026-10-10`（69aa22a）和蒸馏分支（87116dc）记录的两个gitlink相同：

```text
Megatron d65baf60f3d715f313f644eeb68a4047f4cc42aa
FLA      ff481ea6862c99a818858081571373694de7b4e7
```

它们又分别等于子仓库远程`2026-10-10`的HEAD。新蒸馏工作区两个子仓库的HEAD与此精确一致且工作区干净；未包含后来的CP提交或未提交CP/offload代码。

新蒸馏主仓库当前有两个未提交的启动脚本路径调整，以及本地环境/工作区说明文件；它们是上一轮隔离时创建的，不是CP算法改动。因而“提交HEAD一致”不表示工作目录完全零改动。

## CP核对

三个仓库的远程`2026-10-10`均为当前CP HEAD的祖先，没有缺失基线提交，也没有夹入蒸馏分支的后续提交。相对远程基线，CP已提交历史仅多出：

- 主仓库：`0ca9d3a`，`Add RWKV7 context parallel training and numerical validation`。
- Megatron：0个新增提交，HEAD仍为`d65baf60`。
- FLA：`261dbcf5`，`Fix DPLR context-parallel prefix state precision`。

CP+SP、recurrent CP、offload初始化/恢复修复及相关测试配置等后续修改仍未提交，原地保留；工作区还含历史诊断及其他未跟踪文件，不能把整个脏目录笼统视为只包含一组CP修改。后续commit需要按范围挑选文件。

## 远程与本地旧分支的区别

按用户要求，本地旧`2026-10-10`指针没有移动：主仓库仍为`0ca9d3a`，Megatron为`d65baf60`，FLA为`261dbcf5`。其中主仓库和FLA比远程同名分支各多出上面那一个CP提交；这个差别来自分离分支之前，不是本次核对新增的改动。

新`codex/rwkv7-cp`和`codex/rwkv7-distill`分支尚未发布到TimeMobius远程，本次ls-remote未返回它们。远程既有蒸馏分支和三个远程`2026-10-10`均未更改。

机器可读完整SHA、状态和验证结果见`remote-verification.json`。分支隔离说明见仓库根目录`BRANCH_WORKSPACES.md`。
