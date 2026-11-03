# P9a 实施计划

2026-11-03。状态：IN_PROGRESS（PR-01/02/03/04/05 COMPLETE；下一步 PR-06；性能留项 OPEN）。需求与范围以 [PRD](PRD.md) 为准。实现顺序固定为 **PR-01 H1 → PR-02 H2/R1 → PR-03 M1 → PR-04 M2/M3/M4 批处理收尾 → PR-05 M5 → PR-06 L1/L2/L3 → PR-07 阶段验收**。

PR-01 可以按内部里程碑分多个可审查提交，但它仍是一个完整交付 PR。中间保留旧路径仅用于迁移对照；最终必须删掉。不能以“材料后续再接”关闭 H1。

## 1. PR 总表

| PR | 目标 | 依赖 | 完成证据 |
| --- | --- | --- | --- |
| PR-01 | actor-owned 统一非线性接口、分段归约、全类型设备 phase、删除旧双路径 | 起点 | COMPLETE；[最终验收](PR-01.md#7-最终验收与剩余性能项)：E2E/quick/回读通过；性能退化另列 OPEN |
| PR-02 | 固定绑定 PCG 复用；异常 replay 生命周期 | PR-01 的 binding/generation 合同 | COMPLETE；[验收](PR-02.md)：CPU/CUDA 机制与 E2E、两精度 quick 通过，8 组稳态布局回读为 0；保留未复现观察项 |
| PR-03 | 多向量后验与收敛结果 packet 融合 | PR-01 通用组归约，PR-02 稳定 plan | COMPLETE；[验收](PR-03.md)：两类摘要各一次读取，独立公式/CPU/CUDA/E2E/quick 通过，三组成对成本与额外存储已记录 |
| PR-04 | cable gauge、material reaction、control 校验的类型批处理收尾 | PR-01 已完成设备迁移 | COMPLETE；[验收](PR-04.md)：材料设备提交固定，48 组阶段对照、三组成对测量、CPU/CUDA/E2E/quick 通过；首捕获 reset 防护及性能留项单列 |
| PR-05 | Stats 完整向量只作最终快照，序列化按需 | PR-01 的发布边界 | COMPLETE；[验收](PR-05.md)：独立只读快照、失败诊断/JSON 兼容、CPU 21/CUDA 14 与 quick 60 项通过；无导出/完整导出三组成对成本已记录 |
| PR-06 | binding 元数据复用、dt 数值化、有界 graph 档位策略 | PR-01/02；M 项测量 | L1/L2/L3 实际触发证据、失效验证、总耗时/内存对照 |
| PR-07 | 所有 requirement 关闭审查、全组件与完整 E2E、性能报告 | 前述 PR | 最终 SHA suite/stage 及问题处置；不复用旧版本 PASS |

M2/M3/M4、L4/L5 中阻止统一设备契约的部分已经属于 PR-01，后续不重新实现另一套逻辑。PR-04 开始前审查 PR-01 的完成记录：若已达相应结构与成本目标，直接以证据关闭该项，只处理剩余批处理缺口，禁止为凑 PR 新造抽象。

## 2. H2 与 R1：PR-02

已在 `91b939e7` 完成受影响组件与两精度 quick 验收，详见 [PR-02](PR-02.md)。实际生产改动仅需 `pcg_production.py`；现有 assembly 验证和 buffer signature 满足合同，无须修改 BSR 构建或新增 epoch。以下保留原实施要求。

涉及 `pcg_production.py`、`linear.py`、`bsr_reduction.py` 及受影响 linear graph/fixed-storage 测试。

1. 用固定 buffers、合法新 assembly generation、相同与变化的 offsets/columns 内容分别复现哈希调用与 plan 重建，区分初建、binding 变更、schedule 变更、内容变更。记录读取的实际容量字节。
2. 区分 workspace-owned 与 standalone plan。前者在 binding/schedule/容量与布局所有权合同满足时，复用读取设备稀疏内容的 graph；后者保留外部可变布局校验。pointer/signature 足够时不额外引入无用途的 structure_epoch；若确需 epoch，明确谁在实际重分配时推进。
3. 保留 stale caller、scaling/factor、regularization、stream、material operator/preconditioner 绑定校验。不能将 `self._workspace is workspace` 作为绕过所有检查的捷径。
4. 为 replay 失败注入先写 red test，按“未提交/已提交/结果收集失败”的真实状态安排清理或使 plan 失效；不把所有异常视为 GPU 未执行。检查 enqueue/capture/collect 相邻失败边界和已有非重入拒绝。
5. 修改“拓扑变化必须重建”的旧测试预期，改测同存储合法变化读取新算子、解满足独立 residual oracle；存储替换或不合法布局仍拒绝/重建。覆盖 A/B/A、数值零导致的 nnz 变化、容量边界与 standalone plan。

验证：`test_monolithic_linear_graph.py`、`test_monolithic_linear_fixed_storage.py`、`test_monolithic_linear_internal_storage.py`、`test_monolithic_linear_production.py` 的受影响用例，CPU/CUDA 与精度变体；固定物理输入前后 D2H/重捕获/耗时对照，再执行本 PR 通用收尾。

## 3. M1：PR-03

已在 `afe86b7e433a2b32f24b6ba8e89e13b286268164` 完成验收，详见 [PR-03](PR-03.md)。assessment 回读 5→1、posterior 4/6→1；三组成对小场景中 q+x/rigid 两精度 median 均降低，材料波动、部分 FP32 字节增加照实保留。整体 `P9a-PERF-01` 仍 OPEN。以下保留实施要求。

同时复测 `P9a-PERF-01` 的 q+x/rigid-only：使用 PR-01 的相同输入与工作量对照，检查新增摘要读取是否实际解释耗时，不能仅以读取数量下降关闭性能退化。

涉及 `iteration.py`、PR-01 的分段归约/输出计划及 PCG 结果边界。不会恢复硬编码三个组，也不创建 `(5*G,N)` 中间矩阵。

1. 在一次后验计算可用后，对 error/rhs/magnitude，以及非零 regularization 时的 original/regularization term，共用一次固定顺序分段遍历，输出 `K×G` 小结果；遗漏向量用明确 schema，而不是依赖 Python array 身份数目。
2. 合并同一依赖边界内的 worst、rounding-limited、assessment sums/minimum 和 square sum。PCG 内部 norm 若已由结束 packet 收集，不再二次读取；不得提前读取尚未计算的数据来凑“一次同步”。
3. NaN/Inf、零/空组、大动态范围、force/torque 混合、regularization=0/非零都与独立公式对照；FP32 保留既有 rounding-floor 和 cap。

验证：solver reductions/convergence/direction/precision oracle 及 linear tolerances；审计每种路径的同步次数，单树与多树/材料各至少一例。M1 关闭依据为 packet 合并后的真实计数及数学结果，不是少了几个 Python 函数。

## 4. M2/M3/M4：PR-04

已在 `c3dc0714` 完成验收，详见 [PR-04](PR-04.md)。N=8 的 gauge/reaction/product+size/absolute/band+factor 提交分别为 20→5、8→1、32→3、24→2、9→2；control 保持 1 次校验、1 次状态回读、3 次 D2D。元数据枚举仍属 L1。8 cable 整步 median 降低 13.1%/15.6%（FP32/FP64）；FP64 trees 三组小幅上升、median +1.6%，保留于 P9a-PERF-01。OBS02 已蒸馏为组件 red/green：capture miss 同步防护通过最终组件和 quick，命中重放不加同步，依赖内部根因待查。以下保留原实施要求。

PR-01 已经必须完成以下设备语义，此 PR 只合并剩余按 block 展开的存储/launch，并补充分项测量。

| 项目 | 实施重点 | 机制验证 |
| --- | --- | --- |
| M2 gauge | 用预绑定 cable theta rows、factor offsets/count/weights 做设备分段求和/零模判断；按类型收集状态；避免 CPU `np.isin` 与 dense G×N | 无接触、count=0、中间/末尾空段、全空段、正/零权重、严格抵消、固定 theta、净 torque 不兼容；保持已有判零与求和精度，不能加经验阈值 |
| M3 reaction | 设备 scatter 使用已绑定 particle ids/theta offsets，更新 raw residual；保留 fixed rows 的反力与 reference oracle 分支 | 多 cloth/cable、不连续 ids、固定点、theta 接触 torque；独立力平衡和反力符号；contact residual 与力的映射不重复相减 |
| M4 controls | D2D 冻结 position/theta/torque，device finite/fixed-theta 检查，统一输入状态 packet | shape/dtype/device 非法仍在入口拒绝；NaN/Inf/固定 theta 非零 torque；失败不污染可发布状态；0/1/多材料及同一 control 连续更新 |

先测 1/2/4/8 个同类型对象。在固定计算阶段下，稳态 host 调度和同步次数不能线性跟随对象数；设备工作量正常随对象自由度增长。进一步 SoA packing 只有在数据表明有必要时实施，记录捕获阶段展开已有 block 与真正 kernel batch 的各自收益。

验证：cable operator/solver/linear、cloth solver/linear、material state、solver input boundary/publication/contact 相关用例；不是只跑 NumPy 向量化单测。原 review 的 `reduceat` 示例不用作实现模板。

## 5. M5：PR-05

已在 `d0fc86db` 完成验收，详见 [PR-05](PR-05.md)。最终 residual/reactions 保存独立、只读 NumPy 数组，现有 serial/save 按需导出列表，两个失败 runner 补完整 JSON。65,536 行快照保留存储减少约 87.5%/75.0%（FP32/FP64），但完整 JSON 仍约 39 ms；实际整步 q+x FP64 + JSON median +7.6%（三组两升一降）保留于性能项，不宣称全场景提速。以下保留实施要求。

涉及 `solver_monolithic.py`、`solver_types.py` 及现有 recording/serial 工具的必要消费者。

1. PR-01 中已取消每 Newton 轮 residual list 转换。盘点 Stats 消费者，明确最终发布、停止发布、硬失败分别保存哪个 residual/reaction 和对应 generation。
2. 优先用独立只读 NumPy 快照消除 Python float/list 分配，现有序列化入口按需转换；若已有消费者要求 list，则在兼容边界处理。不得保存会被下步覆盖的 solver scratch/device view。
3. 仅需摘要的运行不额外制造完整向量列表；不能删除用户诊断字段或悄悄更改公开格式来获得速度。若延迟序列化需要新的公共开关，先缩小到内部方案。

验证：保存前一步 Stats 后运行下一步/失败步，旧数据不变；JSON/recording roundtrip；最终 residual/reaction 数值与 snapshot 一致。分别测无序列化和实际序列化的耗时、host 内存及字节量，不把成本移到导出后隐去。

## 6. L1/L2/L3：PR-06

PR-05 新观察：q+x FP64 + JSON 三组 median 比值 0.958/1.115/1.084，汇总 +7.6%；对应无导出 -3.8%，输入/物理工作量/诊断向量和传输一致。PR-06 复核这项未归因的耗时上升及组间差异，当前不归因于快照或 graph；见 [PR-05](PR-05.md)。

同时处理 `P9a-PERF-01`：PR-04 新记录 FP64 trees 三组成对 median +1.6%（0.9%/0.9%/3.3%），独立插桩样本无新增显式 stream 同步，仍须归因；统一 phase 增加了 graph/binding 工作；须归因并处理稳定退化，复用 PR-01 三组成对测量合同，不把材料场景的收益抵消为其他场景已通过。

- L1：复用 PR-01 显式 buffer registry，消除热路径 `vars(owner)` 扫描。绑定变更仍走生产有效性检查，不能只留下会被关闭的 assertion。
- L2：将参与捕获的 dt 相关标量更新纳入稳定 buffers，检查各物理项与派生量实际使用新 dt；检查 A/B/A dt 下结果和 graph identity。不是只删除 key 中的 dt。
- 机器人补充已复现：`problem_candidate` 将 alpha 放入参数键，Sharpa close/lift 每轮测量捕获 2055/233 次（baseline 0），该段驻留上限 6 且大量淘汰。L2 的数值参数处理须覆盖 alpha，L3 同时覆盖此场景；先用小 q+x 多档 alpha＋角色切换提炼机制测试，再复测现有四窗口，详见 [补充证据](PR-01-robot-performance.md)。
- L3：先构造超过驻留集合的 span 交替模式，辨认 contact order、pool finish、BSR 等各 segment 原因。比较合并档位、固定预算 mask 与有界缓存工作集；不无限增大 LRU。报告空算/峰值 scratch/graph 驻留和整体 median/P95。局部无可复现回退时记录“未证实瓶颈”，不凭猜测改缓存。

L4/L5 已由 PR-01 的 domain/gauge 设备合同关闭，在本 PR 汇总其结果。C1 单 stream 非重入合同继续保持。

## 7. 验证、记录与提交

每个实现 PR 的实际测试清单由其修改范围选择，通用收尾不可省略：

1. 受影响 CPU 组件、所需 CUDA 精度变体；先修复前失败、后实现通过。CPU/CUDA 缺失或能力限制标为 SKIP/NOT_RUN。
2. 小 q+x 与 PR-01 新混合集成，失败回滚和 graph-off/graph-on 同流程。
3. workspace 入口（从 workspace 根目录；`<run-id>` 替换成新的唯一目录）：

   ```bash
   uv run --no-project python -B dev/monolithic/test/ci/ci.py quick \
     --worktree /home/lightwheel/Desktop/newton/SamiulJ/worktrees/Monolithic-p9a \
     --output /home/lightwheel/Desktop/newton/SamiulJ/dev/monolithic/test/results/P9/P9a/<PR>/quick-<run-id>
   ```

   两精度 frozen quick 不改清单/阈值/快照。新失败说明能否提炼为组件机制或短窗口；增加冻结窗口另按项目流程处理。
4. `uvx pre-commit run -a`、独立正确性/架构审查、更新本 PR 文档并本地提交。没有推送请求不推送；不为本次规划修改 CI 调度。

PR-07 最终 HEAD：云端 suite、经项目流程批准启动的完整 stage；提交命令与运行身份保存在报告。性能成对测量与物理验收分别报告，已有已知失败需具体处置，不能自行登记为豁免。PR-01 最终结果见其第 7 节；阶段 E2E 为 NOT_RUN，整体性能资格与 P9a-PERF-01 保持 OPEN。

PR-07 同时回看 `P9a-OBS-01`：PR-02 最初 CUDA 批次有一次 rigid FP32 graph-on checkpoint 的微小逐位差异；最小场景与后续严格复跑未复现，失败注入测试清理已补齐，但因果关系未证实。原 FAIL 和诊断证据保留在 [PR-02 第 5 节](PR-02.md#5-一次未复现的-checkpoint-观察)，不改门槛或登记为豁免。

PR-07 同时回看 `P9a-OBS-02` 的依赖版本处置：首次捕获 reset 问题在 Warp-only 程序复现；Newton 的 capture-miss 同步防护和组件回归已通过。依赖内部根因未归属，升级后须重验再考虑移除防护；不自动合并 OBS01 的原因。
