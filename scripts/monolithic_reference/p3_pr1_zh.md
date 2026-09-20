# P3-PR1：私有 tet-only 变分基线

PR1 为无接触、单连通 tet 模型提供真实时间推进，支持 Kim/Smith 材料、lumped/consistent mass，以及显式选择的 Armijo、ResNorm、ArmijoOrResNorm。没有新增公开构造模式或生产 globalization 默认；PR2 articulation 与 PR3 contact/grasp 不在本交付内。

## 状态与数值契约

- `_create_tet_only` 要求零 articulation/body/shape，所有粒子由 tet 拥有，且至少一个动态节点。
- `_create_tet_state` 提供实际求解及提交使用的 FP64 particle_q/particle_qd；拒绝普通 FP32 输入/输出状态。model 资产及外力仍来自现有 FP32 数据，r/B/PCG 保留 FP32。
- 采用 FP64 状态是因为原悬臂释放的必要位置修正小于 FP32 坐标间隔，独立高精度解写回 FP32 后不能维持残差达标。不能将该私有状态视为已受 viewer、collision 或其他 solver 支持。
- 完整增量目标包含 BE 惯性、材料、冻结重力与外力；consistent gravity 使用完整质量矩阵行和。current 计算 E/r/B，trial 仅计算 E/r。
- exact H 与生产 B 分开：Kim 的 B 省略 cofactor 几何项；Smith 将完整节点 Hessian 投影后消元。
- 私有 global/x 停止目标为 absolute RMS=1e-7、relative=1e-6，small-step=1e-10。原停止准则在 FP64 修复后仍有自由落体累计误差及横向加载误差，用户批准独立停止精度设计后采用该配置。
- 保持 h=1ms、Newton10、回溯8、PCG200、rtol=1e-4/atol=1e-6、actor_block、原正则化与物理验收门槛。旧公开耦合入口默认配置不变。
- strict 默认只发布 SUCCESS；小步和预算停止不算收敛。legacy 发布仅用于对照。失败、被拒 trial 与发布异常遵循完整事务契约。

## 已有证据与已知限制

[v6 验收记录](fixtures/p3_pr1_acceptance_v1.json) 绑定审查前源码快照：144/144 基线、12/12 CPU 时间细化、25 项专项与104项回归通过。记录中的 `not_done` 是当时的历史状态，不代表包含本文件的后续提交状态。

独立 Astra-High 审查没有已证实 P0/P1，发现一个 P2：精度诊断将每1ms输出误当成每个细化物理步。当前已修复；按记录时间推进完整物理子步，缺少中间生产状态时逐步 residual 及积分结果明确不可用。新增5项回归通过，原1ms自由落体诊断的既有字段保持一致。

[诊断修复记录](fixtures/p3_pr1_diagnostic_fix_v1.json) 绑定修复后的脚本和新增测试。它补充而不改写 v6 源码哈希；solver 和正式轨迹/细化工具未因该修复改变。

完整原始材料位于开发工作区 `agents/integration/artifacts/p3/pr1/`，不随本仓库分发；本仓库保留可执行测试、研究工具及便携结果记录。未运行 nightly，不作性能或抓取验收声明。

此前 FP32/FP64 测试混合同一进程超过9分钟未完成，原因未定位；分别运行的专项和旧路径回归通过不代表消除此限制。

## 复现

在本仓库运行以下独立进程：

```bash
uv run --no-sync -m unittest newton.tests.test_monolithic_p3_tet newton.tests.test_monolithic_p3_accuracy_diagnostic
uv run --no-sync -m unittest newton.tests.test_solver_monolithic_tet newton.tests.test_solver_monolithic_material_mass newton.tests.test_solver_monolithic_step newton.tests.test_solver_monolithic_joint_terms newton.tests.test_monolithic_segment_graphs newton.tests.test_solver_monolithic_friction
```

`verify_p3_tet.py` 负责校准与轨迹，`summarize_p3_tet.py` 和 `summarize_p3_tet_refinement.py` 负责完整矩阵/共同时间网格汇总。使用新的输出目录，保留所有失败和原始记录；不得覆盖旧证据或将 sampled records 当作全部物理步。
