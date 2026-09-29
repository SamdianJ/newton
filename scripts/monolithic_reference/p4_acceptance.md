# P4 总验收与交付记录

> 2026-09-30 存储整理：保留九条Franka＋一条Sharpa完整FP64参考；其余大录制和日志按用户要求清理。下文是历史记录，部分原证据路径已归档，定位见[开发历史](development_history.md)。本次不关闭R1，也不产生新的测试通过结论。

日期：2026-09-30。文档收口基于 `23a2afd5`；性能受测源码为 `e7e18416`。开发分支为 `SamiulJ/monolithic-refactor`，固定原基线为 `39f80c48da2022187823f8218131a249807597a0`。

## 当前结论

**本次验收文件交付完成，复用既有证据；整体无保留验收仍有一项开放问题 R1。** 用户明确要求不再运行测试。本批仅整理文档、证据路径与哈希，不修改生产代码、runner、资产、预算或门槛，也不启动组件测试、discovery或长轨迹。文档静态格式检查不作为新的数值验收。

| 结论层次 | 状态与适用范围 |
| --- | --- |
| FP64结构重构与旧源码退役 | 已有证据通过；T3退役与T4受测版本绑定清楚 |
| FP64重构等价性、dev、临时full | T4及T5-B对应版本已有PASS；full为九条Franka＋一条Sharpa，另44条未执行 |
| 重建Sharpa资产 | 新资格和8667步FP64参考已有验证；不是原资产逐位恢复，旧清单/基线不覆盖 |
| T5精度及成本实验 | 已交付；最终T5-D范围仅默认Franka，两精度各三次 |
| FP32数值资格 | 原T5-C为FAIL；新四元数/绝对阈值实验为NOT_AUDITED，不自动推广 |
| 当前正式dev/full入口 | OPEN：仍绑定已被重建资产替代的旧清单/资格目录，见R1 |
| 本批测试执行 | NOT_RUN，按用户指示；没有把历史通过记录改成新提交的通过记录 |

## 范围和阶段证据

实现范围为公开完整q+x `SolverMonolithic`、直接继承SolverBase、完整状态/恢复/最终力快照、共享关节/tet/contact组件、统一smooth SDF构建和旧实现退役。具体API见[使用与迁移说明](../../docs/guide/monolithic.rst)，精度边界见[精度契约](p5_precision_contract.md)，退役规则见[退役说明](legacy_retirement.md)。阶段历史见[阶段日志](p4_stage_results.md)。既有changelog包括P4入口、退役、精度及显式FP32容差片段；本批未引入新的产品行为。

| 阶段 | 已有交付和验收事实 | 主要回执 |
| --- | --- | --- |
| T0 | 契约/输入/测试ID、基线驱动、十项准入及冻结比较包络 | [T0c](../../../../agents/integration/artifacts/p4/t0-v1/t0c-receipt.json) |
| T1–T2 | 中性定义与共享组件、完整事务、公有API及生命周期 | [T2-2](../../../../agents/integration/artifacts/p4/t2-2-v1/receipt.json) |
| T3 | 统一SDF、场景/runner、测试迁移及源码归档退役 | [退役收口](../../../../agents/integration/artifacts/p4/t3-closeout-v1/receipt.json) |
| T4 | 重构等价性/dev/full通过；55917步、87 Stopped、0硬失败；409项组件及27组小矩阵 | [T4](../../../../agents/integration/artifacts/p4/t4-v1/receipt.json) |
| T5-A/B | 精度参数化与安全；439项组件、27组小矩阵及十条FP64重新验收 | [T5-B](../../../../agents/integration/artifacts/p4/precision-comparison/t5-b-v1/receipt.json) |
| T5-C原协议 | 十条FP32首失败与独立审计；1024次候选、259次方向后验，51次接受判据分歧 | [T5-C](../../../../agents/integration/artifacts/p4/precision-comparison/t5-c-v1/receipt.json) |
| 后续FP32实验 | 用户授权四元数1e-6及显式绝对目标1e-6；十条55917步完整推进，8624收敛、47293 Stopped、0硬失败 | [完整实验](../../../../agents/integration/artifacts/p4/precision-comparison/fp32-relaxed-v1/receipt.json) |
| T5-D | 默认Franka，两精度各三次5250步，31500步无硬失败；成本、显存、重复性及限制记录 | [T5-D报告](../../../../agents/integration/artifacts/p4/precision-comparison/t5-d-franka-v1/RESULTS.md) |

组件检查数是各受测版本的实际计数，不能相加当成最终一次测试覆盖。discovery表示发现/导入检查，不表示执行了全部Newton测试。最近完整组件套件在原T5-C为443项、436通过/7既有跳过；后续修改有各自局部验证。性能批次的8项runner检查和pre-commit也只沿用原记录。本次不补跑。

## 最后长轨迹结果的使用边界

T5-D三次求解耗时（秒）：FP64为430.241/251.724/251.975，中位251.975；FP32为642.944/573.110/583.445，中位583.445。FP64每次5250步均收敛；FP32收敛0/1/0步，其他为Stopped。当前实验没有净速度或总显存优势；默认仍FP64。GPU频率变化、持续graph重捕获和求解质量差异已在报告中保留。

这些运行证明对应源码、资产与配置的完整推进，不代替十配置的新full，也不代替FP32独立数值资格。重复FP32终态坐标最大分量差0.342mm，原因未定位。原录制的坐标/力误差只是另一个明确绑定的实验结果。

## 证据继承与资产身份

1. 原T0/T4的同资产重构比较保持原结论；T3退役的运行文件/AST绑定证明单独记录，未改写受测源码标签。
2. T5-B提供参数化后十条FP64完整场景及组件验证。后续FP32专属变更、资产恢复与性能测量分别使用自己的回执，不能合成为一次未实际执行的最终全量测试。
3. Sharpa原手部文件误删后，已完成新SDF资格及一条替代FP64参考；见[重新认证回执](../../../../agents/integration/artifacts/p4/sharpa-asset-recovery-v1/requalification-receipt.json)。新运行使用`sharpa-asset-recovery-v1/frozen-inputs.json`，其409文件身份与原资产代次不同。Franka九条旧FP64参考的复用索引单独保留。
4. 旧FP32失败、原资格审计和新放宽阈值实验各自保留；旧审计的FAIL不证明新实验同样失败，新实验完成也不能覆盖旧失败或授予新资格。

完整回执索引及SHA-256见[证据索引](../../../../agents/integration/artifacts/p4/final-acceptance-v1/evidence-index.json)。只索引现有阶段回执，不声称本批重新审计了每份大体积轨迹或重新计算了数值结果。

## 开放问题与明确未认证范围

- **R1：正式回归入口资产绑定，OPEN，未获得豁免。** `run_p4.validate_inputs()`仍读取T0旧清单，2026-09-30前一轮只读核查已报告`Frozen input changed: …/hand/left_MP.npz`。`monolithic_sharpa_recorded.py`仍使用旧`p3/pr3/smooth-sdf-v1/geometry`；新身份已接入精度runner，未接入正式runner。即使选Franka，原入口仍先验证整份旧清单。本批仅记录，不修改代码、不再次执行该检查。后续应显式接入新清单、证书及替代参考，保留原重构基线；目前不能声称当前checkout的正式dev/full可原命令重跑通过。
- **Q1：实验FP32数值资格未授予。** 新阈值完整轨迹没有完整独立残差审计；重复性原因未定位，Stopped不计收敛。计划允许实验失败作为交付，不要求为结束结构重构而修算法。
- **Q2：Sharpa抓取成功未认证。** 完整推进、状态安全与替代基线检查不代表抓取物理成功。
- **Q3：范围限制。** 原54条矩阵中的44条仍延期；Close已由用户移出精度实验；T5-D仅默认Franka。未扩展到这些范围。

R1是当前入口可复现性问题，与历史FP64数值证据是否通过分开。用户要求补文件及不再测试，不解释为将R1视作已修复或批准无条件通过。本文交付完成不等于所有开放项关闭。

## 文档批次交付

交付包括本文、逐阶段证据索引、总回执、文档修改前快照和提交绑定；同步主计划、回归计划、Source Map与HANDOFF。生产文件与实验数据不改动，不复制大体积录制。每阶段已有提交保留，本批文档独立提交，不推送。
