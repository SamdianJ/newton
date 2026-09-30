# Monolithic 项目与测试规范

本文件说明当前 `SamiulJ/monolithic-refactor` 分支的测试组织方式，与 [AGENTS.md](AGENTS.md) 配套。范围限于 Monolithic；不替换 Newton 其他模块的 CI 或 nightly。公开 solver 用法见 [Monolithic 指南](docs/guide/monolithic.rst)，开发历史见 [development_history.md](scripts/monolithic_reference/development_history.md)。

## 阶段开发流程

P4 开发阶段已结束；后续以明确命名的阶段推进，`P4j` 这样的子阶段也视为独立阶段，拥有自己的目标、PRD、worktree 和验收。当前项目集成主干为 worktree `worktrees/Monolithic-refactor` 对应的分支 `SamiulJ/monolithic-refactor`，不是 Newton 上游 `main`。

1. **制定阶段目标并创建 worktree。** 写清要解决的问题、交付物、不做什么和完成标准。从当前集成主干的明确提交创建该阶段的独立分支与 worktree；一个阶段使用一个 worktree，阶段内各 PR 在其中按依赖顺序推进。记录阶段 ID、分支、worktree 路径和起始提交；不在集成主干上直接开展下一阶段开发。
2. **根据目标制定阶段 PRD。** 放在 `dev/monolithic/<阶段>/PRD.md`，明确支持范围、行为/API 边界、资产/环境、验收指标与开放项。每份阶段 PRD **至少包含一个端到端（E2E）阶段测试**：从真实输入到最终输出，写明场景/配置、执行命令、完整运行长度、通过/失败条件和结果位置。组件测试通过或脚本能启动不能替代 E2E。E2E 可复用已有完整场景，但须说明它如何验证本阶段目标。
3. **按 PRD 拆分 PR。** 为每个 PR 明确目标、修改范围、依赖、组件测试、小型集成测试及对应 PRD 条目。阶段 E2E 和完整场景集合在开发前约定；不在看到结果后临时删减失败项。PRD 中的 E2E 若已属于完整场景集合，同一受验源码与输入的执行结果可复用，不重复跑同一条长轨迹。
4. **开发中只做组件测试，PR 收尾做小型集成。** 迭代期间运行受影响的组件测试和必要静态检查；每个 PR 完成时运行其约定的小型集成，验证组件连接、短步进和关键事务。不得把完整 Franka/Sharpa 轨迹作为日常调试或每个 PR 的默认检查。
5. **每个 PR 更新文档并提交。** 在 `dev/monolithic/<阶段>/` 更新实现说明、PRD/PR 对应关系、验证结果、偏差与开放项，然后将该 PR 的代码和可版本化记录提交到阶段 worktree 所属分支。失败或未执行的检查不能写成通过，未满足该 PR 退出条件不得标记完成。这里的“提交 worktree”指提交其 Git 分支，不是删除工作树或提前合并主干。
6. **阶段目标验收后跑完整场景，再合并。** 全部 PR 完成后，先核对阶段目标、文档、组件与小型集成结果；通过这一轮阶段目标验收后，执行 PRD 约定的 E2E 和完整场景回归。长场景安排手动夜间运行，结束后检查结果。所有必需门槛通过、开放项如实记录后，才将阶段分支合并到 `SamiulJ/monolithic-refactor`；完整场景检查仍属于最终合并门槛，不能先宣称最终 PASS 再补跑。若合并冲突修改了受验代码，须复验受影响范围。合并后更新阶段 HANDOFF 与主干状态；确认成果已保留后再清理阶段 worktree。

本流程规定后续开发的执行顺序；修改规则本身不创建新阶段、不启动测试或合并。显式的阶段范围调整、暂停或“不运行测试”指令优先，受影响门槛保留为待执行，不自动豁免。

## 文档、测试与结果目录

下面路径均相对于共享 workspace `/home/lightwheel/Desktop/newton/SamiulJ`，不是任意 worktree 的当前目录：

```text
dev/monolithic/
  PRD.md                         # 当前项目范围
  architecture.md                # 当前架构
  open_items.md                  # 跨阶段开放项
  <阶段>/                        # P0–P4 历史，P4j 等后续阶段
    PRD.md                       # 阶段目标、至少一个 E2E、验收门槛
    plan.md                      # PR 拆分、顺序与测试映射
    PR-01.md                     # 每个 PR 的实现/验证/偏差记录
    HANDOFF.md                   # 当前状态、运行方式、开放项
    acceptance.md                # 阶段目标验收、完整场景及合并结论
  test/
    results/
      <阶段>/
        <PR或stage>/<运行ID>/     # 命令、回执、摘要、必要失败数据
worktrees/<阶段worktree>/
  newton/tests/                  # 按功能组织的组件、小型集成和 E2E 测试
  scripts/monolithic_reference/   # 复用现有完整场景 runner
```

**开发文档统一维护在 `dev/monolithic/` 并按阶段组织；测试源码继续放在各阶段 worktree 的 `newton/tests/`，测试结果统一写入 `dev/monolithic/test/results/` 并按阶段分组。** 测试文件按功能命名，阶段归属通过 PRD 映射和结果目录表达，避免重复复制组件测试。阶段测试只调用受验 worktree 的代码，运行时记录并核对实际导入来源，不依赖隐含当前目录。新的测试结果不再散落到 `agents/` 或各 worktree 的临时 output 中；历史产物保持原身份，通过索引定位，不为改目录而重跑。

**测试源码不迁移。** 取消将测试搬到 workspace 的计划，继续使用仓库现有的 unittest 注册、Newton discovery 和 CI。完整场景 runner 保留在受验 worktree 的 `scripts/monolithic_reference/`；`dev/monolithic/test/` 仅存放结果与索引，不维护测试实现副本。测试修改随阶段分支提交、验收和合并。

`dev/` 当前位于 Newton Git 仓库之外。每个 PR 仍必须实际更新该阶段文档，并在提交说明或版本化回执中记录文档路径/版本或摘要哈希；提交 worktree 不等于已提交、推送这些外部文件。合并记录须能定位阶段 PRD 与验收结果，不能仅依赖聊天记录或旧 PASS 标签。

## 测试分层

| 层级 | 内容 | 运行时机 |
| --- | --- | --- |
| 组件 | 当前模块的数学、装配、输入校验、API、独立参考和工具行为 | 开发时运行受影响组件；普通 CI 通过现有 Newton runner 发现这些测试 |
| 小型集成 | 小规模 q+x、多步历史、真实 trial、发布/回滚、graph 生命周期、状态恢复 | 每个 PR 完成时运行约定的小型集成，按 CPU/CUDA 条件报告 |
| 夜间全量 | Franka 九条＋Sharpa 一条完整 FP64 轨迹 | 阶段目标验收后、合并前手动触发；结束后读取报告和回执 |

不要求每次提交运行十条轨迹，也不把 `run_regression --profile dev` 当轻量测试：它仍运行 Franka 和 Sharpa 两条完整轨迹。当前没有新增 GitHub 自动调度、远端 runner 或资产上传；夜间任务使用已具备 GPU 和本地资产的环境显式启动。

## 按功能选择测试

测试位于各 worktree 的 `newton/tests/`，继续使用现有 unittest 注册、设备选择和发现机制。以下前缀均可用 Newton runner 的 `--pattern` 选择：

| 文件模式 | 主要职责 |
| --- | --- |
| `test_monolithic_joint_*.py` | 关节参数、变分项、FK/惯性参考、精度 |
| `test_monolithic_tet_*.py` | 材料、质量、E/r/B、装配、精度 |
| `test_monolithic_contact_*.py` | 三种摩擦 law、几何、网格接触、独立积分参考 |
| `test_monolithic_collision_*.py` | 独立 collision pipeline 与 AABB |
| `test_monolithic_linear_*.py` | BSR、PCG、预条件、误差、容差、图捕获 |
| `test_monolithic_sdf_*.py` | 平滑场、导数、连续性、构建与资格身份 |
| `test_monolithic_solver_*.py` | 接受/收敛条件、完整 q+x、状态/checkpoint、发布/回滚、精度安全、graph 与小型集成 |
| `test_monolithic_assets_*.py` | 网格、材料资产、手部挂载及 cube attachment；缺少外部资产时明确报告跳过 |
| `test_monolithic_controls_*.py` | Franka 和记录轨迹的采样、时钟与控制插值，不运行完整抓取 |
| `test_monolithic_tooling_*.py` | 参考/录制校验、runner 选项、统计与报告；单元测试使用小型数据或 mock，不启动长轨迹 |

通用环境中的运行示例：

```bash
# 按修改选择一个组件；GPU 环境下先串行运行，避免测试争抢设备。
uv run --extra dev -m newton.tests --pattern 'test_monolithic_joint_*.py' -j 1

# PR 收尾按范围选择小型集成；下例为当前全部组件＋小型集成集合。
uv run --extra dev -m newton.tests --pattern 'test_monolithic_*.py' -j 1
```

当前本机共享环境可用 `env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=../monolithic-integration/.venv uv run --no-sync python -B -m newton.tests ...`，从本 worktree 根目录运行。CPU/CUDA、外部资产与可选依赖的检查/跳过应分别报告；不要把跳过算作覆盖通过。首次内核编译的耗时不代表测试规模扩大为完整场景。

## 历史测试退役规则与本次调整

先检查直接/间接导入、注册、脚本及构建/配置引用，再按测试实际守护的行为判断。仅守护旧开发记录的测试删除；仍覆盖当前功能的数学与安全断言先迁移。不因文件带有 `legacy` 或阶段名称而直接删掉有效覆盖，也不新增只用于证明退役模块不存在的历史名称清单。

本次实现提交为 `56be2f04`：

- 退役 `test_monolithic_p2_evidence.py` 和 `test_monolithic_p3_close_diagnostics.py`，共五个历史测试方法。
- 退役旧 baseline 中两个构造函数 AST 对比及一个固定旧源码/输入保护方法；当前参考身份、运行来源与输入变更拒绝覆盖保留。
- 54个既有模块改为功能名称；拆出求解器 oracle，将接受与原生收敛目标集中到 globalization 测试。当前共55个测试模块；后续维护将 joint_variational 改名为 joint_objective。
- 录制完整性与比较包络归入 recording；矩阵选择和来源检查归入 runner。控制测试直接调用当前场景实现，不再经过历史转发入口。
- 保留当前 API 对非法参数/外来历史的拒绝以及仍明确支持的 checkpoint/录制 schema 兼容检查；这些保护当前输入边界，不维护退役 solver 的可运行性。

旧文件快照、哈希、模块迁移表和静态核对记录在 workspace 的 `agents/integration/artifacts/history/test-reorganization-v1/`。静态计数为294个测试函数定义调整到288个：退役8个、新增2个；这是源码定义数，不是按设备展开后的运行用例数。静态导入/语法核对和 pre-commit 已通过；当时按“不重跑测试”的要求，运行测试、discovery 和完整轨迹均为 **NOT_RUN**。后续用户指定的默认 Franka 单条复验已通过，见 [验证记录](scripts/monolithic_reference/franka_default_validation.md)；不等于全部组件或完整矩阵通过。

## 手动夜间全量回归

当前完整回归集合为：Franka 的 Armijo/ResNorm/ArmijoOrResNorm × C-EE/C-IE/C-II，共九条，各5250步；Sharpa 的 ArmijoOrResNorm＋C-IE，一条8667步。全部使用 FP64、production、graph，单 GPU 串行执行。后续阶段在 PRD 中明确引用此集合及本阶段 E2E；经明确范围调整前保持这十项，不能把仅一条默认场景的 PASS 称为 full PASS。

保留资产及十条参考位于 [asset/monolithic](asset/monolithic/README.md)，大数据不随 Git 克隆。缺少资产应明确失败，不能跳过后声称 full PASS，也不能自动重建替代资产或改写物理身份。

```bash
# 只列出十项，不构建场景或启动 GPU 求解。
uv run --no-sync python -B -m scripts.monolithic_reference.run_regression --profile nightly --list

# 替换阶段/运行ID占位符；阶段目标验收后手动启动，output 必须是新目录。
uv run --no-sync python -B -m scripts.monolithic_reference.run_regression \
  --profile nightly --output "/home/lightwheel/Desktop/newton/SamiulJ/dev/monolithic/test/results/<阶段>/stage/<运行ID>"
```

当前本机执行时同样使用上述 `UV_PROJECT_ENVIRONMENT` 共享环境设置。`full` 作为原十项入口继续支持；`--case` 过滤只能得到 PARTIAL，不能替代夜间全量结论。不添加定时触发，也不因为组件测试失败而自动启动长轨迹。

运行结束后检查：

1. `summary.json`：是否完整收集十项、是否过滤、每项结果及提交/源码/输入身份。同一批次身份不一致必须为 FAIL。
2. `RESULTS.md`：逐项结果、Converged 与 Stopped 数量、录制保留状态及回执链接。
3. 各 case 的 `receipt.json`：GPU/软件环境、资产和参考身份、比较误差、场景物理指标及异常。失败录制用于查看失败位置、回滚状态或轨迹分歧。

单个 case 失败后继续收集其他 case。Franka保持原抓取指标和冻结比较包络；重建Sharpa使用当前严格比较策略，抓取成功仍未认证。不得通过放宽门槛、减少步数、调整预算或改成 xfail 取得通过。

## 结果与空间管理

默认 `--retain-recordings failures`：成功 case 保存回执、场景摘要和比较结果后删除本次完整录制，失败 case 保留录制供定位；已有十条参考绝不改写或删除。需要研究完整新轨迹时显式指定 `--retain-recordings all`。失败问题处理完再清理诊断数据，不永久累积重复录制。

分别记录组件检查与夜间回归结果；未执行写 NOT_RUN，不能沿用其他提交的 PASS。Stopped 单独统计，不当作收敛。FP32完整场景数值实验和性能实验独立于默认FP64验收；精度安全、跨精度拒绝等当前组件覆盖仍保留。

每个完成的可审查批次提交一次，按请求推送；显式“不运行测试”的指令优先。静态检查通过只证明静态检查范围，不能据此宣称运行验收通过。

## 当前维护入口

线性实现分为 `linear.py` 工作区、`linear_types.py`、`bsr.py`、`preconditioner.py`、`pcg.py` 及 production plan/kernel 两个模块。公开 solver API 不变；原 `articulation_variational`、`contact_variational`、`variational_iteration` 分别为 `articulation`、`contact`、`iteration`。活动参考工具的 P2/P3/P4/P5 模块名已换成功能名；历史报告仍记录当时名称。

本机设计文档已由 workspace 的 `dev/monothlic` 整理为 `dev/monolithic`，顶层为 `PRD.md`、`architecture.md`、`open_items.md`，一页交接位于 `P4/HANDOFF.md`。这些文档位于 Newton Git 仓库之外，不能声称随本分支提交或推送；仓库内开发历史与阶段记录保留便携摘要。
