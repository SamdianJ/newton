# P9a PRD：统一非线性问题接口与主机开销治理

2026-11-03。状态：**IN_PROGRESS / PR-01、PR-02、PR-03、PR-04、PR-05、PR-06 COMPLETE**。统一接口、PCG 复用、摘要融合、材料设备批处理、最终 Stats 快照与 L1/L2 合同已通过验收（L3 仅完成测量处置，合成限制保留）；`P9a-PERF-01` 与未复现观察 `P9a-OBS-01` 仍 OPEN。OBS02 已加入捕获边界防护并通过回归，依赖内部归因待查。证据见 [PR-01](PR-01.md)、[PR-02](PR-02.md)、[PR-03](PR-03.md)、[PR-04](PR-04.md)、[PR-05](PR-05.md) 与 [PR-06](PR-06.md)。阶段尚未完成；[PR-07](PR-07.md) 已进入验收，独立审查发现的 P9a-F1（非连续多维能量提前复制）已修复，最终 SHA 的全组件与完整 E2E 待验。

## 1. 身份与范围

| 项目 | 绑定 |
| --- | --- |
| 阶段 | P9 的第一个子阶段 P9a |
| worktree | `worktrees/Monolithic-p9a` |
| 分支 | `SamiulJ/monolithic-p9a` |
| 起点 | `SamiulJ/monolithic-refactor @ 8fa80ba7ce3fd7ce2483f2ec098e66d410bc434e` |
| SuperDex 参考 | `54ae749a042709e897cad12da66822c71bcd1b97`，见 [参考映射](references.md) |
| 文档 | 按用户本次要求，正文在 worktree 的 `planning/` 中版本化；workspace `dev/monolithic/P9/` 只维护入口及历史专项 |
| 测试/结果 | 测试留在本 worktree `newton/tests/`；结果在 workspace `dev/monolithic/test/results/P9/P9a/<PR-or-stage>/<run-id>/` |

用户确定 PR-01 从 H1 开始，并将 H1 定义为：**以 actor 为自由度所有者的统一非线性求解接口重构，让通用物理组合和设备执行组合成为同一套组合，最终完全去除旧的分开的实现。** 后续顺序见 [计划](plan.md)。本阶段不先实施 H2/M1 来替代这一顺序。

成功标准分为正确性、架构统一、数据传输和性能四项；不能仅凭结果相近或局部 kernel 更快关闭问题。

## 2. review 问题全集与处置

以下位置均相对 `newton/_src/solvers/monolithic/`，行号绑定起点版本。以下表格保留起点盘点与计划；本轮关闭状态见表后。H/M 的级别来自原 review，不表示已有量化性能排名。

| ID | 已核实问题及边界 | 主要位置 | 计划归属 / 完成证据 |
| --- | --- | --- | --- |
| H1 | 单树、有 tet、无刚体及 q+x 排序限制；多树/刚体/材料回到完整向量主机运算；两套物理组合容易漏贡献 | `solver_monolithic.py:494,914,1085,1240`；`iteration_device.py` | PR-01；所有当前受支持类型同一设备契约，旧双路径删除，完整覆盖与回读审计 |
| H2 | 新 assembly generation 下载 offsets 和整个 columns 容量后哈希；拓扑内容变化重建 PCG plan/graph | `pcg_production.py:417,455,836` | PR-02；固定绑定变更合法数值/稀疏内容不重建，存储或调度改变正确失效 |
| M1 | 后验多个向量、收敛评估分别归约并回读；正则化决定实际次数 | `iteration.py:124,228`；`iteration_device.py:306,388` | PR-03；按真实依赖边界融合 packet，前后同步次数与字节数 |
| M2 | cable 公共角零模检查下载因子并逐 factor `np.isin`；只在无固定 theta 且 Jacobian 判为潜在零模时进入 | `solver_monolithic.py:1104` | PR-01 完成设备迁移；PR-04 检查批处理成本，空段/抵消/非零净 torque 测试 |
| M3 | material raw reaction 经 host 相减再上传；循环内 residual 回读只在 reference 分支 | `producers.py:742` | PR-01 完成设备迁移；PR-04 按类型批处理反力及 xyz/theta 映射 |
| M4 | target position/theta/torque 每步下载校验再上传，position 下载两次 | `solver_monolithic.py:529,691` | PR-01 保留校验并迁至设备；PR-04 验证固定成本输入 packet 与多材料扩展 |
| M5 | 主机路径每轮 residual 转 list；发布时材料反力转 list，字段覆盖而非每轮累计 | `iteration.py:330`；`solver_monolithic.py:815,1393,1413` | PR-01 移除迭代中完整诊断回读；PR-05 处理最终快照与延迟序列化 |
| L1 | 反复遍历 owner 属性、构建 buffer signature | `solver_monolithic.py:921`；`segment_graph.py:22` | PR-06；显式绑定计划，仍能发现更换存储，测主机耗时 |
| L2 | dt 是多个 graph key 的一部分 | `solver_monolithic.py:1015` | PR-06；更新绑定的数值缓冲，A/B/A dt 无不必要重捕获 |
| L3 | contact order span/多种绑定超过 6 个驻留项后可能抖动；仅跨一个边界不必然发生 | `contact_pipeline.py:351`；`segment_graph.py:49` | PR-06；复现再选有界策略，包含内存、空算及总耗时 |
| L4 | rigid domain 每次下载 rotations 和步初 rn | `rigid.py:599` | PR-01；设备 domain 标志，旋转边界与失败语义不变 |
| L5 | cable gauge 投影下载/上传完整 delta | `solver_monolithic.py:1373` | PR-01；设备分段投影，保留公共角与当前 FP64 均值约定 |
| R1 | replay launch 抛异常后 `_in_flight` 保持 true，下一次使用被阻塞 | `pcg_production.py:745` | PR-02；失败注入、重入拒绝、清理与后续合法使用测试 |
| C1 | 多个 graph 共享 scratch，仅串行使用安全 | `iteration_device.py`；`pcg_production.py:188` | 全阶段保持单 stream、非重入合同；这是约束，不宣称当前已有并发错误 |

PR-06 后的状态：

| 条目 | 当前处置 |
| --- | --- |
| H1、L4、L5 | FIXED；全类型统一 problem/phase、旧双路径删除、设备 domain/gauge；CPU/CUDA E2E、两精度 quick 与回读审计通过 |
| M2、M3、M4 | FIXED；类型批处理/固定设备提交、同形状重绑与两精度物理 oracle 通过；48 组阶段对照、三组成对完整 step、CPU/CUDA/E2E 与两精度 quick 通过；仅元数据扫描仍归 L1，见 [PR-04](PR-04.md) |
| M5 | FIXED；最终 residual/reactions 为独立只读原生精度快照，旧 Stats 跨成功/失败步不变；JSON 格式与完整失败诊断保留；CPU 21/CUDA 14、小型 E2E、quick 60 项及三组成对无导出/完整导出测量见 [PR-05](PR-05.md) |
| H2、R1 | FIXED；固定存储内容变化复用 plan/graph，保留绑定失效与 standalone 校验；失败排空/失效/恢复机制、CPU/CUDA E2E 和两精度 quick 通过，8 组稳态布局回读为 0；见 [PR-02](PR-02.md) |
| M1 | FIXED；assessment 5→1、posterior 4/6→1 次读取，数学与依赖边界通过独立公式/CPU/CUDA/E2E/quick 验证；三组成对成本、FP32 字节与存储代价见 [PR-03](PR-03.md) |
| L1、L2 | FIXED；构造或首次绑定时注册、生产绑定检查、稳定 dt/alpha 数值缓冲；CPU 25/CUDA 23、64 步 E2E、quick 60 项及三组成对整步/机器人测量见 [PR-06](PR-06.md)。属性扫描消除不等于所有元数据耗时下降 |
| L3 | MEASURED / RETAINED；真实所选窗口无 span 重捕获，保留 power2 + LRU 6。合成七档循环仍会抖动，替代策略有空算/复合键代价；不标记通用修复，见 [PR-06](PR-06.md) |
| C1 | 持续约束；保留单 stream、非重入合同 |
| P9a-PERF-01 | OPEN；PR-06 已消除 Sharpa alpha 图抖动，但总耗时未全面改善：相对 PR-05，q+x FP64 无导出 +5.13%、trees FP64 无导出 +1.49%（均三对同向），trees FP32 JSON +4.81%；原始基线 q+x/rigid 与 Franka/Sharpa FP32 的最新三对结果及历史观察均见 [PR-06](PR-06.md)。不以捕获数归零宣称性能 PASS |
| P9a-OBS-01 | OPEN / 未复现观察；最初 PR-02 CUDA 批次一次 rigid FP32 checkpoint 逐位差异，随后小例与完整严格复跑通过。测试清理已修复，因果未证实，PR-07 回看；原 FAIL 不覆盖、不改门槛，见 [诊断](PR-02.md#5-一次未复现的-checkpoint-观察) |
| P9a-OBS-02 | MITIGATED；首次 CUDA graph 执行的 magnitude/reset 错误已在独立 Warp 程序复现；capture miss 同步防护、组件 red/green、200 次材料重放及最终 CPU/CUDA/quick 通过。依赖内部根因待查，PR-07 回看；见 [PR-04](PR-04.md) |

审查结论的限定：原“约 20 次/轮、5 次/trial”“host 比 kernel 更贵”未经 profiling；历史 11 solves/2 graph_rebuilt 不记录重建原因且可能含首次捕获；“无 O(n²)”不适用于每树 dense block 或全量配对元数据；有界缓存不等于已证明无泄漏。`_steps` 首次分配后复用，`_variational` 每步重建。上述表述纠正不另造求解器 bug。

原建议代码不可直接复制：现有 PCG row group 从 1 编号；`1+g` 会错位。`reduceat` 对末尾空段/全空段越界。5 个 graph 绑定再加一个仍未超过 6。仅放宽 H1 gate 会遗漏 rigid energy、实际位移和其他 q+x 假设。

## 3. 架构合同

### 3.1 ownership 与映射

复用 `ActorTable` 和 `LinearLayout`，不建立第二份 actor 真相源。初始化派生 actor identity/kind、state indices、tangent rows、local-to-global maps、physical unit、两套 group、typed batch offsets；验证自由行唯一 owner、无空洞、无越界，固定自由度保留约束与反力映射。

| 概念 | 职责 | 不可混同 |
| --- | --- | --- |
| Actor | 一棵树、独立刚体、tet/cloth/cable 对象的状态和 DoF ownership | interaction 不成为另一份状态 owner |
| 物理贡献 | 惯性、弹性、驱动、接触等向 owned rows 添加贡献 | 多项可作用于同一行 |
| 预条件器块 | 每树 dense、rigid 6×6、tet 粒子 3×3、材料专用块 | 一个 actor 可以有许多块 |
| 线性分组 | `LinearLayout.groups`，用于 PCG/posterior/步长统计 | 不等于物理验收分组 |
| 物理验收分组 | `convergence.RowGroups`，进一步区分 force/torque 等 | 编号与数量不要求等于线性组 |
| World/island | 确实无跨域耦合的独立求解、接受与回滚域 | 不按 actor 拆开耦合问题，不凭局部收敛冻结 actor |

起点未包含独立 P8c 分支的完整多 world 工作。本阶段保留已有 world/layout 身份与边界，不自动合入 P8c，不把“actor 分组”宣传为独立 world 求解。

### 3.2 类型执行器与 interaction

沿用 Producer 组织方式，热路径只提交计算并写入预分配缓冲。保留一个 articulation batch 处理多树、一个 rigid batch 处理多刚体；cloth/cable 可先在构建/捕获阶段展开 block 调用，稳态运行不能逐 actor 回读或作 host 决策。同类 SoA/offsets 与 kernel 合并的额外吞吐优化分开测量。

| phase | 输出或责任 |
| --- | --- |
| predict / begin_step | 冻结控制、步初数据与历史，构造预测点 |
| candidate | 按类型 retraction 构造 trial |
| difference | 实际有效位移，而不是无条件 `alpha * delta` |
| coordinates | rounding 操作数尺度 |
| domain | 状态/整条 trial 路径的合法性标志 |
| evaluate | E、r、absolute-energy/absolute-contribution、状态标志 |
| assemble | 组装项、matrix-free 算子及绝对算子贡献 |
| publish | 恢复速度、反力、历史与发布 generation |

每个启用项在初始化登记 energy/absolute-energy buffers、residual 与 rounding 贡献、tangent/matrix-free 贡献和 domain/nonfinite/capacity flags。明确可选输出的中性值；缺少必需 phase/贡献立即拒绝，不能静默回退。统一评价只使用这份计划，接触计入一次，rigid energy 不允许由散落的条件分支补齐。

### 3.3 私有 `_MonolithicProblem`

新增最薄的私有适配层（建议 `problem.py`），绑定 ownership/layout、类型执行器、interaction、original/accepted/trial 视图、E/r/B/scaling/direction、分段归约和状态摘要、storage binding 与 numerical generation。复用现有 buffers/算子，不复制一套状态或线性工作区，不引入通用插件框架。

职责为 `prepare_step`、`linearize`、`make_trial`、`evaluate_trial`、`qualify_direction`、`publish`；具体签名可按现有调用风格调整。计算方法返回设备视图或写入 outputs，host 摘要收集是显式边界。`iteration.py` 只控制 Newton、PCG 精度、正则化重试、line search 和发布策略，不判断 q/x 排序、tet 有无、actor 类型或树数。

CPU 和 CUDA、graph 开与关使用相同 problem/executor 计算流程；graph-off 仅是相同设备工作不捕获，不保留 NumPy 物理实现作为生产 fallback。离线独立数学 oracle 保留在测试/参考工具中。

## 4. 不可改变的数学与事务语义

1. 保留 C-IE、ArmijoOrResNorm、best-progress 约束、现有正则化及 PCG 后验资格；Stopped 不改标 Converged。
2. 保留 `tau_i = max(g_u, rtol_u * max_{j in (b,u)} |r0_j|, c_u * eps_T * a_i)` 与 `max_i |r_i|/tau_i <= 1`，r0 来自预测点首轮。非有限 residual/rounding 不能通过。借鉴 SuperDex 的物理层提供数据、策略层聚合，不替换为其 weighted L2。
3. 保留完整耦合 residual/operator、接触交叉块及对预条件器块的贡献。已有非保守物理项继续使用原 acceptance-energy 关系，不强制其成为同一势能的精确导数。
4. Rigid 保留 COM、左乘 SO(3) retraction、相对旋转 logarithm 与 domain；articulation/粒子使用保存的 step increments，避免大绝对坐标相减损失小位移。
5. Tet 保留 det(F) guard、动态/固定映射、质量和材料模型。Cloth 保留整条路径坍塌检查和固定顶点反力。Cable 保留 xyz/theta、frame history、gauge、contact torque、路径检查及 BSR + matrix-free operator/absolute product。
6. `build_tangent=False` 不覆盖 accepted tangent 或材料 element matrices；拒绝 trial 不污染 accepted 状态、历史、算子或有效性标记；接受时交换状态角色与配套 increments。
7. 保留现有发布/回滚与 generation 合同，包括状态快照和单调失效代次的区别。失败后旧因子不得重新成为有效因子；不得为了回滚检查强行倒退单调发布计数。
8. 稳定归约顺序、空组中性值、NaN/Inf 传播需显式定义。FP32/FP64 均按各自原物理门槛验收，不以跨精度逐位一致为要求，不放宽 frozen gates。

## 5. 执行、存储与回读验收

- 分段计划使用连续 `(start,count)` 或离散 row indices + offsets；大组分 chunks。对固定数量 K 个 metrics，scratch 上界为 `O(K*(N+C+G))`，不采用 dense `G×N` terms。全局归约不得重复计入 actor 或单位子组。
- actor 拓扑、maps、容量、数组绑定、dtype/device/stream 改变时重绑/重建；dt/control/步初状态是数值更新；evaluation/assembly/factor/trial 有效性分别维护。PR-01 明确生命周期，PR-02/06 才关闭对应缓存成本。
- PR-01 非线性运算不得回读完整 residual、delta、scale、effective displacement、材料因子、reaction 或逐 actor domain 数组。允许诊断最终快照暂在发布边界读取（M5）；H2 布局哈希暂单独登记。不能将整向量改名成 packet 规避审计。
- 接触刷新、PCG 完成、收敛判断、trial 决策、发布等边界分别计量。固定 Newton/PCG/trial 次数下，host 同步调用次数不随 actor 数增加；摘要字节允许随必要 group 数增长，但必须与完整行向量区分。
- H2 完成后，workspace-owned 固定绑定的稳态 solve 不再读取布局大数组；独立 plan 的外部输入验证合同保留。不能只比较 graph_rebuilt 次数而忽略首次建图、读取量或错误重用。
- 每次记录 D2H/H2D 次数与字节、kernel/graph 提交数、capture/replay/eviction、驻留图和 scratch 字节、Newton/PCG/trial 工作量、step median/P95。冷编译与初始化单列，稳态重捕获计入耗时。
- 性能对照同 GPU 串行、同配置/输入、预热后至少三组成对交替测量；FP32/FP64 分开。没有现成基线时先记录而不编造加速百分比。结构目标可验收；总耗时若有稳定退化须解释并解决或明确留项，不自动宣称性能 PASS。

## 6. 验收矩阵与阶段 E2E

组件新增项先建立能在旧实现暴露问题的机制用例；复用现有数学 oracle，不以 graph/eager 自比较代替物理正确性。命令与文件分工见 [PR-01](PR-01.md)。

| 用例 | 必须证明 |
| --- | --- |
| 单树 + tet | E/r/B、方向、接触及原逐行验收正确 |
| 两棵不同规模、混合 force/torque 的树 | 两套 group、packed rows 正确，小负载不被稀释 |
| 多树 + rigid + tet + 跨 actor 接触 | 完整能量、交叉块、力投影及全系统耦合 |
| Rigid-only、joint-only、tet-only | 不依赖 q+x 占位；零长可选 batch 安全 |
| actor 顺序置换 | 按身份映射回物理输出后等价，不要求同一行编号 |
| 接近旋转域边界、大坐标小增量 | retraction/difference/domain 与精度约定 |
| rejected trial、assembly/发布失败 | accepted tangent/history 不污染；回滚与失效代次正确 |
| cloth/cable 固定行、gauge、接触 torque | 约束反力、零模兼容性、matrix-free 及路径检查完整 |
| 非有限/空组/容量边界/更换 buffer | 正确拒绝、摘要中性值和图绑定失效 |
| FP32/FP64 × CPU/CUDA；CUDA graph 开/关 | 同一计算组合，各精度原门槛；能力限制显式 SKIP，不能冒充覆盖 |

**阶段专项短 E2E（PR-01 实现，拟新增功能测试 `test_monolithic_solver_problem.py`）：**

- 从 ModelBuilder/当前混合场景 fixture 构造、创建 state/control、连续 64 步，到 state_out/反力/Stats；dt=1 ms，CPU 与 CUDA 两精度，CUDA graph 开关分别运行。
- 组合 A：两棵不同规模的固定基座树 + rigid + tet，采用现有受支持接触类型；组合 B：articulation + rigid + tet + cloth + 两条 cable，复用 material-state fixture 的受支持布局。另有 rigid-only 变体。测试实现时固定具体参数并纳入提交，不采集或修改 frozen quick 快照。
- 覆盖加载/持有/卸载三个阶段（16/32/16 步），第 32 步保存状态并作后续 continuation 对照；用独立故障分支注入 rejected trial 和发布失败。
- 必须 64/64 完成，无硬失败、非有限值或容量丢弃，逐行结果与状态标签一致，固定行和反力平衡通过已有组件物理阈值；checkpoint continuation 按现有精度合同；注入失败分支完整回滚。无法完成构造或支持组合时先报告，不能用静默 fallback 或缩小矩阵代替验收。
- 执行入口：`uv run --no-sync python -B -m newton.tests --pattern 'test_monolithic_solver_problem.py' -j 1`。测试已经实现，实际设备/精度结果见 PR-01；它属于小型集成，不替代下面的长 E2E。

**阶段完整 E2E（PR-07）：**复用固定 CI stage 作业集，两精度，绑定最终 SHA；包括 Franka 5250 步、Sharpa 8667 步、64 刚体 2000 步，材料完整任务、cable cantilever 全矩阵（每例 600 步）、cloth patch/prefold（各 360 步）、tet 既有全矩阵；FP64 remote-tree 完整 2000 步。其配置及物理门槛以当前版本 recipe/fixture 和 CI stage manifest 为准，执行前冻结 hashes。完整 helix 360 保持独立 nightly 项，不用本阶段顺带启动。

例如完整材料 E2E 的单项入口（在获准的阶段作业环境中运行，两精度分别执行）：

```bash
uv run --no-sync python -B -m scripts.monolithic_reference.run_cloth_benchmark prefold \
  --device cuda:0 --pcg-mode production --use-cuda-graph --precision fp64 \
  --output /home/lightwheel/Desktop/newton/SamiulJ/dev/monolithic/test/results/P9/P9a/stage/<run-id>/cloth-prefold-fp64
```

该例完成加载、卸载、恢复及 checkpoint 检查，以 `cloth_recipes.json` 原残差、面积比、恢复位移/能量和折角门槛判断；完整 360 步才可 PASS。两精度原有差异/已知失败照实分类。

每个实现 PR：受影响组件 + 小型 q+x/新增混合集成 + 本机 quick 两精度 + pre-commit。纯规划文档提交不启动仿真。阶段结尾云端 suite 和获准的 stage；已知失败/重试需沿用项目处置规则，不自动新增豁免。

## 7. 非目标与仍保留的 P9 backlog

不新增公共 `MonolithicProblem` API、USD schema 或配置开关；复用 Model/State/Control。无新物理材料、接触律、容差、默认预算、强预条件器或公开 Stats 语义变更。公开序列化/状态兼容若受到影响，先按既有合同处理。

已有 P9-P1（池容量与全配对元数据）、P9-P2（完整接触执行规模/图缓存）、浮动基座、多 world 外层设备循环、world 压缩、P5-I2/P6b-P1 性能资格继续保留。P9a 的 L3 只覆盖实际命中的局部 graph 成本，不自动关闭整个 P9-P2。它们见 workspace `dev/monolithic/P9/performance-plan.md` 与 `dev/monolithic/open_items.md`。

P9a 收尾要求问题表逐项给出 FIXED/验证证据或明确未完成项；未完成 PRD 必需项不得标记阶段完成。PR-01 的最终 runtime、quick、结构审计和成对性能结果，PR-02 的组件/E2E、quick、成本对照与观察项，以及 PR-03 的 packet 机制、PR-04 的材料批处理与首捕获防护、PR-05 的快照/导出成本、各自的三组成对测量及两精度 quick，以及 PR-06 的绑定/参数、L3 对照与原始基线复核已记录；阶段 suite/stage 为 **NOT_RUN**，整体性能资格为 **OPEN**（含 `P9a-PERF-01`），`P9a-OBS-01` 留待阶段回看。
