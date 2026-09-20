# P2 中止 / P3 规划：便携阶段记录 v4

> 当前导航（2026-09-20）：下文与v1–v4 manifest记录2026-09-19的历史规划状态；PR1现已完成，见 [PR1基线](p3_pr1_zh.md)。开发已回到 `SamiulJ/monolithic-integration`，旧Close debug实验从活动源码撤回；参考分支 `debug/cube-close-residual-20260915` 及工作区归档保留历史。PR2计划和积分状态ADR位于工作区 `dev/monothlic/`，尚未实现。

2026-09-19，P2 保持 **HALTED_REQUIRES_REARCHITECTURE**；原因：**当前架构遇到了不可解决的阻碍，需要评估重构。** P3 为 **PLANNED_NOT_IMPLEMENTED**。

工作区主计划为 `dev/monothlic/newton_monolithic_solver_p3_plan_zh.md`，配套公式为 `dev/monothlic/newton_monolithic_solver_p3_formulations_zh.md`。[v4 manifest](fixtures/p3_phase_transition_v4.json) 绑定文档、参考代码与基线哈希；[v1](fixtures/p3_phase_transition_v1.json)、[v2](fixtures/p3_phase_transition_v2.json)、[v3](fixtures/p3_phase_transition_v3.json) 及原迁移指南保留，不回写历史。

|单元|范围|状态|
|---|---|---|
|P3-PR1|tet-only；Kim/Smith objective、residual、exact H/生产 B；完整惯性/外力目标与三模式|NOT_STARTED|
|P3-PR2|参考 superDex link-space 能量与 reduced 映射；时间离散/架构 ADR；小系统及无接触 Sharpa|NOT_STARTED|
|P3-PR3|接触罚势及 superDex 式耗散；显隐式方案 ADR；组件→耦合 Close→新轨迹抓取|NOT_STARTED|

PR1 不要求未迁移的 articulation/contact。PR2 不单列 Coriolis 势，不在新惯性梯度上叠旧 bias force；superDex 的 Newton–Euler inertia 和空间 torque 近似 merit 不能冒充严格变分。其时间离散、predictor 和状态迁移为必做待办。

PR3 仅比较 superDex 支持的耗散 n/fₙ 组合 C-EE、C-IE、C-II，冻结方案待定。C-IE 是参考默认的隐式 n/显式载荷估计，使用当前几何估算 stage-start gap，不是严格缓存的 fₙ。显式 n/隐式 fₙ 被参考禁止，已排除研究范围。耗散 n 显式不代表法向罚力的 SDF 梯度冻结；非可积路径保留 merit/residual、真实 Dr 与对称 B 的区分及模式兼容 ADR。

PR1 必做 **Armijo / ResNorm / ArmijoOrResNorm**：能量充分下降、residual 非增、两者 OR。能量非下降方向不得预先阻断 R；NaN/非法 trial 不被 OR 掩盖。变分路径 s=rᵀp；非可积路径不能把它宣称为 surrogate 的实际斜率。外层只按原 active global/q/x 目标，缺失块记 N/A；STOPPED 不等于 CONVERGED，publish 独立。最终默认模式不自动改变。

P2 8H fixture、8I production、8J/K/L、G6/剩余 G7 保持未完成；历史 Schur 触发保留。8E/F 收尾和 8G NOT_APPLICABLE 不变，默认 actor_block；AMG/MAS/patch/性能优化后置。

旧侧推 cube 仅做 Close 回归；PR3 抓取等待用户新上方接近轨迹，未到位不能声称 PR3 抓取/P3 全部通过。nightly 由用户手动执行；本次只更新规划、核验公式，没有更改 solver、仿真、提交或推送。

历史基线 HEAD `fd4f095a6ed5782f0395e34ce9c30054ffd958b8`，分支 `debug/cube-close-residual-20260915`。12 个未提交研究文件不包含在该 HEAD；清理时完整归档于工作区 `agents/integration/artifacts/p3/debug-cleanup-20260920/untracked-backup/`，其中保留 `scripts/monolithic_reference/close_globalization_interpretation_zh.md` 检查汇总及 `close_globalization_results_zh.md` 20分支研究报告。原P2中止快照也保留。v4证据包位于工作区 `agents/integration/artifacts/p3/planning-20260919-v4/`；公式差分通过不等于实现或轨迹通过。历史manifest不回写。
