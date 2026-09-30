# Monolithic 开发历史与产物保留

整理日期：2026-09-30。清理前源码：`82a9295961a93ad991f16298cf63beb5652241e6`。这是既有证据摘要；用户要求不再跑测试，本批没有新数值验收。

## 关键决策和结果

| 阶段 | 决策及证据边界 |
| --- | --- |
| P1 | 局部固定软球/轻载接触为 PASS_LIMITED_SCOPE，不能推广到自由抓取。 |
| P2 | Close 的完整物理步未解决；阶段为 HALTED_REQUIRES_REARCHITECTURE。组件回归不等于物理收敛。保留质量、PCG、production/diagnostic 和 graph 经验，不继承整体 PASS。 |
| P3 | 建立完整 q+x formulation、tet/joint/contact 组件、三种摩擦 law、真实 trial、完整历史及事务发布。Stopped 与 Converged 分开；失败回滚不得污染原状态。 |
| P4 T0 | 固定原基线 `39f80c48`。用户将正式矩阵从54条收窄为Franka九条＋Sharpa一条，另44条延期。 |
| T1–T3 | 提取共享组件；公开 SolverMonolithic 直接继承 SolverBase。普通 model.state() 仅可作为初始化源；续算必须携带完整历史。checkpoint 经 scratch 校验再替换，最终接触力返回独立快照。统一 smooth SDF 构建，保留官方 mesh.sdf。 |
| T3退役 | `56d054ae` 收尾旧源码/诊断入口退役；78份快照及其哈希保留在压缩证据中，已退役实现不承诺可运行。有效数学、安全、collision/AABB、资产与独立 oracle 覆盖保留。 |
| T4 | `9ff616c5` 对应历史等价性/dev/full PASS：55917步，55830收敛、87 Stopped、0硬失败；409项组件中402通过、7跳过，另27组小矩阵。 |
| T5-A/B | `4866e1c2` 精度参数化，默认FP64；`96a9466f` 安全与兼容。439项组件432通过、7跳过，十条FP64全程通过。precision进入状态和配置身份；跨精度续算拒绝。 |
| T5-C原协议 | `304dfded`：十条FP32在第1–4步失败；独立审计1024次候选、259次后验，51次接受判据分歧。原资格结论FAIL。Close后由用户移出比较范围。 |
| Sharpa资产恢复 | 原文件被删除后重建并重新认证；新资产不是旧资产的逐位恢复。`72ca0842` 替代FP64参考8667步：8637收敛、30 Stopped、0硬失败；抓取成功仍未认证。Franka九条原FP64参考复用。 |
| FP32阈值实验 | `964da0d0` 将FP32四元数检查改为1e-6，FP64仍1e-10；`cd9a84c4` 提供显式FP32绝对目标1e-6，默认不改。`02207352` 十条实验55917步：8624收敛、47293 Stopped、0硬失败；NOT_AUDITED，不授予数值资格。 |
| T5-D | 用户收窄为默认Franka OR＋C-IE、production/graph，两精度各三次。受测源码 `e7e18416`，报告 `23a2afd5`。各5250步、无硬失败；FP64每次全收敛，FP32分别0/1/0步收敛。 |
| 最终文件收口 | `82a92959` 整理验收资料，未再运行测试；R1正式runner资产绑定仍OPEN。历史阶段PASS不能自动拼成当前提交的无条件PASS。 |

T5-D求解耗时中位数：FP64为251.975秒，FP32为583.445秒，约2.32倍。没有净速度优势；GPU频率变化、持续graph重捕获及求解质量不同限制性能结论。FP32三次终态最大分量差约0.342 mm，原因未定位。默认仍FP64。

## 清理与证据定位

唯一保留的参考集合见 [资产目录](../../asset/monolithic/README.md)：九条Franka加一条重建资产后的Sharpa，全部FP64。其余轨迹、逐迭代大日志、大JSON、重复源码zip与编译缓存按用户要求删除。状态数据没有抽帧或降精度；保留轨迹的日志仅丢弃嵌套明细，并单独记录压缩前后哈希。

workspace 下 `agents/integration/artifacts/history/` 是轻量证据包：

- `evidence.tar.gz`：原路径组织的小型JSON回执、Markdown报告、脚本/补丁及完整退役快照。入包后逐文件核对哈希，才删除原文件。
- `large-json-summaries.jsonl.gz`：大JSON的顶层标量摘要与原哈希；不能恢复被删除的数组和完整诊断。
- `pruned-files.csv.gz`：移出活动产物树的原文件路径、大小、SHA-256及归档/删除分类。
- `retained-files.csv.gz`：worktree资产与十条轨迹当前文件身份。
- `cleanup-receipt.json`、`retired-worktrees.json`：清理数量、保留规则和被移除工作树的分支/提交。

历史文档中未落在保留资产/参考上的原始证据链接，现应在压缩包内按原相对路径查找。例如：

```bash
tar -xOf agents/integration/artifacts/history/evidence.tar.gz p4/t4-v1/receipt.json
```

大轨迹与详细诊断已按保留策略删除，只有摘要/哈希的文件不能声称仍可逐帧重审。Git中的阶段源码和分支保留；不把哈希当作数据备份。

## 工作树与未关闭事项

保留当前 `Monolithic-refactor`、提供共享环境的 `monolithic-integration`、主checkout `newton`，以及仍被独立oracle引用的 `superdex-reference-54ae749a`。移除已干净的 articulation、collision、contact、linear、tet、verification 和 p28-baseline 工作树；保留原分支，detached p28提交另以 `archive/monolithic-p28-baseline` 固定。

本批不更改solver或runner。R1仍需正式入口明确接入重建Sharpa身份和保留参考；旧T0参考的重复录制已经清理，不能继续依赖旧完整证据树。Q1为FP32新阈值未独立认证，Q2为Sharpa抓取未认证，Q3为矩阵与性能范围限制。后续恢复正式runner时应利用这十条保留参考，不能绕过身份验证或将不同格式的Sharpa参考直接冒充P3录制。

## 后续R1修复（2026-09-30）

正式runner已绑定worktree内的重建Sharpa清单、资格目录与十条保留FP64参考；场景不再依赖agents资产路径链接。精度runner默认使用重建清单，历史observer未指定清单时仍选择原代次。Sharpa原生录制只读适配，Franka保留原冻结包络，重建Sharpa逐字段严格比较。409项输入与十参考的文件身份检查通过；新增回归检查未执行，GPU/长轨迹未运行。因此R1为代码修复完成、运行验收待执行。此前章节中的OPEN描述是当时状态。
