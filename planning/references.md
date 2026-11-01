# P9a 设计来源与核对边界

2026-11-02。只作架构依据，不是测试结果。源码在本地读取，未构建或运行 SuperDex。

## 1. 用户输入与本项目基准

- 首份 review：`Monolithic-refactor @ 8fa80ba7`，H1/H2、M1–M5、L1–L5、replay 异常状态与共享 scratch；重复段落已去重。问题与限定见 [PRD](PRD.md)。
- 用户 2026-11-02 的三层架构及接口/数学/生命周期/验收补充是本 PRD 的需求来源；图的关系保存在 [PR-01](PR-01.md) Mermaid 中。
- 用户引用的 [Producer 构建代码](https://github.com/SamdianJ/newton/blob/9aca3a08c88c3e2a7d12c97d9269dbbedce2cafc/newton/_src/solvers/monolithic/producers.py#L792-L807) 已用本地 `git show 9aca3a08:.../producers.py` 核对：一个 articulation producer、一个 rigid producer，cloth/cable 逐 block 建立。
- 当前起点是 `8fa80ba7ce3fd7ce2483f2ec098e66d410bc434e`，不能把引用版本当作实际 worktree 起点。上一轮核对中 admittance `db6c29bb` 的 monolithic 核心与此基准相同；其任务工具与实验结果不因此自动成为 P9a 的测试结果。

## 2. SuperDex 固定参考

本地目录 `worktrees/superdex-reference-54ae749a`，commit `54ae749a042709e897cad12da66822c71bcd1b97`，检查时工作区干净；origin 为 `facebookresearch/project_superdex`。

以下链接固定 commit，定位到已阅读的原始文件。借鉴职责与语义，不移植 C++ 框架、动态分配策略或默认停止规则。

| 来源 | 观察 | 在本项目的取舍 |
| --- | --- | --- |
| [snle_problem.h](https://github.com/facebookresearch/project_superdex/blob/54ae749a042709e897cad12da66822c71bcd1b97/superdex_physics/libraries/mochi/mochi_core/include/mochi_core/solvers/snle_problem.h) | solution size 可不同于 DoF size；increment 可非欧氏；actorResiduals/actorMatrices 与 interactionResiduals/interactionMatrices 分开；预条件器应包含 interaction 对 actor block 的贡献 | problem 与策略分层；保留类型 retraction/difference 和完整耦合；不把 actor 与 block/group/domain 等同 |
| [snle_problem.cpp](https://github.com/facebookresearch/project_superdex/blob/54ae749a042709e897cad12da66822c71bcd1b97/superdex_physics/libraries/mochi/mochi_core/src/solvers/snle_problem.cpp) | `ComputeFullResidual` 先放 actor residual 再加 interaction；更新 solution/increment 后使缓存失效 | 唯一 ownership + 多来源贡献；数值有效性与存储绑定分离；不照搬按需 resize 到 CUDA 热路径 |
| [newton_solver.cpp](https://github.com/facebookresearch/project_superdex/blob/54ae749a042709e897cad12da66822c71bcd1b97/superdex_physics/libraries/mochi/mochi_core/src/solvers/newton_solver.cpp) | PerActorWeighted 从完整 residual 切片取值、使用物理层权重；权重缺失时可回退 Global | 借鉴物理输入/求解策略分离；保留 Newton 当前逐行 force/torque 判据；缺必需元数据直接拒绝，不采用 weighted L2 或静默 Global fallback |
| [mochi_snle.h](https://github.com/facebookresearch/project_superdex/blob/54ae749a042709e897cad12da66822c71bcd1b97/superdex_physics/libraries/mochi/mochi_physics/src/mochi_snle.h) | ActorSnle 管 full/reduced residual、矩阵和预条件器；按 AssemblyParams 选择清零/计算目标 | 参考选择性 E/r/B 生命周期；本阶段不增加 reduced-coordinate 框架 |

没有证据表明 SuperDex 的这些 C++ 接口天然满足 Warp CUDA graph/no-D2H 合同；设备输出、typed batching 和分段归约是针对本项目的设计，需要独立验证。

## 3. 既有测试/流程来源

- workspace `dev/monolithic/project.md`：阶段与测试层级、源码身份、CPU/CUDA/精度、quick、云端 suite/stage 和完整场景流程。
- workspace `ci/newton-monolithic-ci/scripts/stage.py`：当前完整作业集；不为规划更改它。
- 本 worktree `scripts/monolithic_reference/cloth_recipes.json`（v2）、`cable_recipes.json`（v12）、`fixtures/physical_gates_v2.json`、`fixtures/quick_cases_v1.json` 与实际资产清单：验证前按使用的版本冻结 hash，不放宽。
- workspace `dev/monolithic/P9/performance-plan.md`：2026-10-27 已登记的 P9-P1/P2；P9a 是首个具体 worktree，不覆盖这些历史未完成项。

初始化身份与输入摘要见 [source-manifest.json](source-manifest.json)。文档中的拟新增测试和未来命令均为计划，不是当前已实现能力或 PASS。
