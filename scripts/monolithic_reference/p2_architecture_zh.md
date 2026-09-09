# Newton Monolithic Solver P2 Architecture

> 版本：v1，2026-09-09；状态：**DESIGN_FOR_REVIEW / IMPLEMENTATION_NOT_STARTED**。
> 范围：P2性能优化，以及继承的自由球G6、剩余G7与条件patch Schur。本文明确实现方案、研究准入和PR依赖，不将设计记录算作开发或验收通过。
> 阅读路径：§2 PR拆分 → §3–§9开发设计 → §10验证矩阵 → §11出口和证据。PR-8系列是本设计的新逻辑编号，不代表已创建远程PR；现有P2-0～P2-5任务编号及G0～G7编号保留。

## 1. Authority、当前状态与不变量

### 1.1 输入与优先关系

以工作区`dev/monothlic/newton_monolithic_solver_p2_plan_zh.md`及2026-09-09用户确认的优化顺序为P2实施输入；物理与验收继承PRD、P1架构及既有冻结报告。PRD/P1旧“AMG/graph归后续”的泛化描述，在本阶段仅被用户批准的**aggregation AMG研究及有收益的条件集成、稳定计算段与PCG分段capture**覆盖；不引入完整step graph、通用AMG后端或全设备控制。旧冻结文件不改写，输入哈希见附录A。

代码核查基线：`87d736b0389566a58631293775693cbef4d25c68`，worktree为`SamiulJ/monolithic-integration`。当前已有BSR有效前缀、actor-block/private PCG、Smith/consistent、joint、contact/history、AABB与真实Sharpa SDF。自有质量矩阵开关、PCG诊断模式、CUDA Graph、多节点块、AMG及patch Schur均未据此宣称实现。

P2-0报告状态为`DIAGNOSTIC_FINDINGS_WITH_VALIDATION_GAPS`；性能目标仍待补测与评审。Hold约一次Newton更新不等于可以直接把N改为1；r5加载683是每物理步累计PCG迭代p50，不是单次PCG调用。历史缺失的运行脚本哈希/检查日志保持缺失，后续新运行不能补造旧证据。

### 1.2 两条任务线

| 任务线 | 主场景 | 出口 |
|---|---|---|
| 性能 | Sharpa半径25mm r3 anchored-close，r2对照；独立Smith/consistent gravity r5，r3/r4/r6对照 | 补测、预算冻结、优化/研究决策及完整成本证据 |
| 抓取 | 完整22 DoF Sharpa + carriage + 25mm自由软球，三档网格 | 新半径/目标h校准、至少两指承载、G6与剩余G7、条件Schur关闭 |

固定球冠的4.5s研究与全动态自由球的抓取fixture独立命名、独立solver/history、独立manifest。前者通过不能替代后者。20mm为历史对照，30mm为失败压力样本。原高分辨率悬臂梁near-static问题保持开放，性能改善不能关闭该问题。

### 1.3 必须保持的契约

- 唯一production算子仍为`K_global_scalar_bsr`。令`D`为现有正对角尺度、`S=D^(-1/2)`，线性求解仍为`Ahat*y=-S*R`，`Ahat=S*K*S+lambda*I`，`delta_z=S*y`；K为当前production投影切线，不是完整raw Jacobian。
- 当前/试探按candidate重算FK、质量矩阵、真实残差与接触；目标步内冻结。BE、joint displacement恢复速度、绝对位置写入float32 State的路径保持。
- Smith材料、consistent mass、摩擦/history公式、固定key与静态pair容量、PR-7C有效前缀装配不变。默认Kim/lumped兼容路径与已有G1/G2/G3/G4均保留。
- PCG实际调用p95预算64、硬上限200、默认linear tolerance 1e-4和既有global/q/x门槛不变。regularization改变必须使数值factor失效；warm start只按现有generation规则使用。
- 正常收敛至少99%，soft stop单列；hard rollback/异常停止连续运行且时间不前进。内部detF guard 0.2与验收min(detF)>=0.1区分；初始穿透门槛2mm不放宽。
- State、joint/contact force generation、history epoch的发布与回滚仍作为同一事务；rejected trial和线性求解器scratch不是已提交物理状态。
- CPU保留数学/解析几何/控制/事务回归，真实手mesh SDF为NOT_REQUIRED。无新soft-only/通用rigid-only入口；G1H继续走原私有q-only工厂。

## 2. PR拆分、开发项与验证项

### 2.1 PR总表

“研究必做”要求交付可复现决策，不要求研究对象一定进入production。条件PR只有满足列明条件才实施；已有物理预算触发的Schur不能按普通性能候选随意豁免。

| PR | 性质/依赖 | 开发项与所有权 | 验证项/完成证明 |
|---|---|---|---|
| PR-8A 测量修复与预算基线 | 必做；起点 | 修复runner计时覆盖/FPS，真正固定candidate重放，补有效加载重复；冻结分负载性能/时间误差manifest | 先复现测量错误再修复；至少5次安静重复；P2-0缺口逐条关闭或明确保留，时间步/预算正式评审 |
| PR-8B 自有articulation质量矩阵 | 开发与评估必做；8A有效基线 | `articulation.py`私有并行kernel与构造开关；公共Newton函数不改 | 相同M、R/K/global parity，CPU/CUDA与共享生命周期；kernel/完整路径/整步成本；默认False |
| PR-8C PCG设备摘要与production | 必做；8A | `linear.py`设备摘要和紧凑必需状态包；solver传入固定mode | diagnostic参考一致、production无可选统计工作、强制检查一致；同步次数及完整solve/step成本 |
| PR-8D 分段CUDA Graph | 开发/捕获验证必做；8C | PCG四类图段、参数/指针失效、图外控制；其他稳定段按收益选择 | CUDA实际capture/replay、失败/重放/参数变化，CPU非graph回归；建图摊销与总收益，维护16项清单 |
| PR-8E 多节点块研究 | 研究必做；8A矩阵库；最终成本对照8C/8D | 相邻4/8节点块，离线oracle及最小实验apply，不接公开默认 | 主子块dense parity、SPD、真实残差；建块/factor/apply/连续轨迹总成本，独立GO/NO_GO |
| PR-8F Aggregation AMG-PCG研究 | 研究必做；8A；可复用8E矩阵库 | 弹性smoothed aggregation、软体A_xx、层次复用/更新实验 | 线性SPD apply、同残差解、建层/更新/转换/周期/显存总账；独立GO/NO_GO |
| PR-8G 通过准入的预条件器集成 | 条件；8E/8F各自GO且8C/8D对照完成 | 仅集成获准路线到workspace；每条路线独立子提交或子PR；冻结适用设备/规模与显式接口 | allocation-free apply、generation/lambda/模式/graph、连续轨迹与全套受影响回归；无候选GO则NOT_APPLICABLE |
| PR-8H 自由抓取校准与预算筛查 | 必做；8A；正式配置等待已选优化 | 复用`GraspCase`/资产loader/阶段控制；25mm与所选h重新校准，冻结两指承载fixture、G6物理门槛和G7预算 | 三档接触校准、控制映射、所有球节点动态、支撑力分账；失败样本；actor-block逐次PCG筛查及Schur触发记录 |
| PR-8I 条件normal patch Schur | 条件必做；8H触发 | 先关闭patch近似SPD设计项，再开发有界normal patch setup/factor/apply；不改变global切线 | tiny精确Woodbury、多个共享q/节点patch的SPD/容量、lambda refactor、相同物理质量与完整成本；触发后不能NO_GO代替完成 |
| PR-8J 剩余Smith/装配/接触优化 | **最后一批性能优化**；8B–8I开发或决策关闭 | 新profile后才选择材料求值、PSD投影、写入/重复计算优化；保持物理与投影规则 | 固定candidate parity、事务、至少5次匹配基线测量；无收益可NO_GO/DEFERRED |
| PR-8K 最终G6/G7与性能复验 | 必做；8H物理冻结；8B–8J最终决策/代码 | 固定最终配置运行正式抓取、friction-off和三档规模测试，补曲线/截图 | 同一最终代码/fixture上的G6、剩余G7、原组件/兼容、PCG预算与内存/耗时门槛通过 |
| PR-8L P2出口与记录 | 必做；8K | 报告、哈希索引、设计/实际状态清单、用法与integration日志 | 无未解释必做失败；P2性能线和G6/G7分别列状态；满足全部要求才宣称V0.2 Grasp MVP |

### 2.2 执行依赖

推荐顺序：8A → 8B → 8C → 8D → 8E → 8F → 8G（若GO）→ 8H → 8I（若触发）→ 8J → 8K → 8L。

8B与8C计算所有权独立；8E/8F离线矩阵研究及8H场景探索可在8A后独立开展，但候选最终准入必须相对已优化PCG/graph路径复测。此处独立性不表示要求subagents。8J始终排在其他性能开发与决策之后，8K/8L属于复验/收尾；若8K暴露新Schur触发，返回8I，重新做必要profile与8K，不带失败发布。

8H先探索再校准/冻结，8K只执行冻结门槛。8A的探索性结果不能直接冻结自由抓取预算；8H将负载特定预算增补到独立manifest。改变N/h的推荐与是否改变example默认分开决策，架构不把N=2/S=4作为既定默认。

每PR按“设计/反例 → 实现与数值验证 → 性能/验收记录”形成可审查提交。8E/8F先提交研究工具/结果，只有总成本证据GO后才开展8G正式运行时集成。

## 3. 模块与执行策略所有权

### 3.1 现有接口与增量

| 位置（相对仓库根） | 现有入口 | P2责任 |
|---|---|---|
| `newton/_src/solvers/monolithic/solver_monolithic.py` | `_initialize`、`_evaluate_current/trial`、`_iterate`、`_finish_step` | 构造策略快照与传递；保持非线性/事务，必要最小结果汇总 |
| `newton/_src/solvers/monolithic/articulation.py` | `_eval_passive_dynamics`、`MonolithicArticulationWorkspace` | 选择reference/owned mass kernel，共享M/J/inertia缓冲 |
| `newton/_src/solvers/monolithic/linear.py` | `finalize_assembly`、`factor_actor_preconditioner`、`solve_pcg` | 必需控制包、可选摘要、分段图、获准预条件setup/apply与失效 |
| `newton/_src/solvers/monolithic/tet.py`、`contact.py` | 现有candidate求值/scatter | 仅8J有证据时做等价计算优化；8I消费已有factor，不重算接触公式 |
| `newton/examples/softbody/monolithic_sharpa_grasp.py` | `GraspCase`、`stage_target`、`joint_mapping` | 独立fixture、控制/观测/负对照，不复制积分器 |
| `newton/examples/softbody/example_monolithic_sharpa_soft_ball.py` | 现有viewer/headless入口 | 显式实验配置、冻结文件、曲线；继续公开experimental模块导入 |
| `scripts/monolithic_reference/run_p2_*`、`analyze_p2_0.py` | 诊断runner和证据索引 | 测量修复、矩阵/成本研究、报告；不把离线工具放进candidate热路径 |

不为研究预先建立多后端框架。8G/8I若使`linear.py`难以审查，可在实际获准后增加一个专用私有模块；不得搬动无关代码。所有新执行策略是solver配置，不是材料/资产USD属性。

### 3.2 构造配置设计

以下是待实现接口设计，当前构造器没有这些参数：

| 选项 | 默认与取值 | 生命周期/错误 |
|---|---|---|
| `use_optimized_articulation_mass_matrix` | 严格bool，False | True只走私有实现；禁止用公共结果再重复计算；切换新建solver |
| `pcg_mode` | 嵌套封闭枚举，可由`diagnostic`/`production`字符串转换；默认diagnostic | 非法值构造期拒绝；不允许运行中切换 |
| `use_cuda_graph` | 本设计建议严格bool，False；8D冻结最终签名 | True在不支持设备/依赖上显式拒绝；CPU照常验证False；不静默声称已捕获 |
| soft预条件选择 | 8G准入后才确定接口；现有actor-block保持默认 | 显式选择已验证设备/规模；无自动调参/静默回退；未获准AMG不是可选值 |

执行配置写入manifest并绑定workspace寿命；物理topology/config/history identity沿用原契约，不用执行策略重新解释contact key。新增缓存绑定实际assembly/scaling generation、lambda、设备/stream、布局/指针和执行模式；数值更新、图更新与物理history提交是三种不同生命周期。

私有q-only工厂透传支持的执行选项，`n_x=0`时无x预条件buffer/kernel或AMG层，global/q判据保留，x为NOT_APPLICABLE。普通solver范围不放宽。

## 4. PR-8A：可信测量与预算冻结

### 4.1 必须修复/补测的已知问题

| 缺口 | 开发/补测 | 防回归验证 |
|---|---|---|
| 每步覆盖`_evaluate_current`，破坏窗口profiler包装 | runner内统一安装/撤销计时与曲线wrapper，不在循环覆盖已安装wrapper | 多步窗口current计数与真实assembly计数/调用序列核对，含异常清理；不假定恒定N+1 |
| `solver_only_fps_at_frame_dt`实际是实时因子 | 新schema分别输出FPS、RTF、wall/sim；保留旧原始字段/修正sidecar | 合成已知step耗时/帧步数测量：FPS=输出帧数/累计solver秒，RTF=仿真秒/solver秒；不混用分位数 |
| tet五次timing未加载重力 | 当前runner已有重力更新，验证实际执行并重新运行有效加载5次 | 与`ResponseCase.step`同一输入时间序列/重力斜坡/状态对照；不能仅assert源码出现重力函数 |
| 材料probe为8步演化 | 从真实加载轨迹导出固定candidate，在预分配scratch重复求值 | 重放前后输入指纹相同；明确Kim/Smith计算量对照不要求物理响应相同 |
| Sharpa仅4次安静重复、嵌套计时误解释 | 补至至少5次有效独立重复；同步插桩与正常运行分开 | 排他/inclusive时间、调用次数、未归因时间、污染运行标记 |
| 交付源码不等于旧运行源码 | 新运行保存实际源码、dirty diff、依赖/环境哈希 | 不覆盖旧报告；索引包含全部trace/NPZ，排除索引自身及digest |

矩阵导出只用于离线诊断，允许受控host拷贝；production candidate无新增完整数组回读。采集`K/R/D/S/lambda/generation`、owner矩阵、factor kind/key/权重、动态节点映射与rest/current坐标，绑定fixture及输入哈希。矩阵RHS对照不冒充物理history恢复或通用checkpoint。

### 4.2 工作负载与时间协议

- Sharpa：25mm r3主、r2对照，4.5s prepare/close/hold；现状N=10、h=1ms、S=10、frame_dt=10ms。继续P2计划既定N和S扫描，交叉只采用质量通过组合；h/S变化仍采样步末目标并按共同物理时间比较。
- Tet：Smith/consistent r5主、r3/r4/r6对照；h=20ms、6substeps/帧、36s、1800步。加载[0,2]、动态保持(2,34)、末段[34,36]分别计时。r5=1875tet/540动态节点；r6=3240tet/882动态节点，不能把规模差异归为材料差异。
- 真正GPU完成边界定义solver计时；JIT/资产/SDF、setup、正常step、测量/绘图/IO分开；控制更新成本声明范围。每档预热后从同一初态新建轨迹，至少5次安静重复，记录实际设备、线程数及显存口径。
- 性能候选比较保持N/h/物理参数、诊断模式和graph模式相同。每次实际PCG调用包括retry，未调用不填零；同时报告每步累计工作。

### 4.3 预算manifest

8A先校准后冻结anchored-close/tet两类预算；8H另冻结自由抓取。字段包括：基准代码/依赖/设备、资产/配置哈希、N/h/S/frame_dt、时段、物理误差参考与容差、正常收敛/残差、每调用PCG64/200、完整solve/step p50/p95、wall/sim、尾延迟允许回退、峰值内存及setup摊销时长、重复数和噪声口径。

未有数据的性能百分比/内存/时间误差阈值保持PENDING，不在本文编造。正式性能验收前先冻结；若细h参考观测尚未稳定则保持时间步选型未通过，继续现状h，不自动采用0.25ms为真解或粗h为生产默认。

## 5. PR-8B：自有质量矩阵kernel

当前公共`eval_mass_matrix`会清零H、更新world spatial inertia，再调用一个thread/articulation的累加kernel。优化路径保留相同FK、Jacobian与spatial inertia计算；在monolithic内直接调用现有惯量更新kernel和新M kernel，绝不先调用完整reference mass再覆盖。

线程按`(dof_i,dof_j)`并行，输出：

\[
M_{ij}=\sum_{\text{link }b}\sum_{k=0}^{5}\sum_{l=0}^{5}J_{bki}I_{bkl}J_{blj}.
\]

第一版线程内保持原link/k/l顺序及每link部分和，唯一写回对应元素；padding/无效范围显式清理，防A→B→A旧值。不用atomic，不做另一套惯量/COM约定，不靠镜像掩盖非对称误差。scratch共用`J/body_I_s/M`，没有逐candidate分配。

False保留原公共调用；True固定私有路径。current/trial/final/rollback、tiny与q-only均走`_eval_passive_dynamics`同一分派，不能为了整手性能冻结M一整步。

开发测试建议`test_monolithic_owned_mass_matrix.py`：混合revolute/prismatic、fixed link、分支树、惯量偏移、不同姿态/qd、零外部力、多次重放、非法选项。验证M、动能/广义惯性、actor R/K、global parity、原事务与分配。以完整mass路径及整步实测决定推荐；3–5倍kernel或15%–20%整步旧估算不作为SLA。

## 6. PR-8C：PCG必需控制与可选诊断分离

### 6.1 Buffer与结果

保留`_alpha/_beta/_r_z/_old_r_z/_products/_status/_true_norms`等算法scratch。每个主机决策边界使用小型必需状态包，至少包含状态码及该边界需要的标量/标志；整数状态保持明确表示，不把浮点位解释为状态。RHS范数一次线性solve固定，device缓存分母，不每轮重复回读。

Diagnostic另有预分配定长摘要：`min_p_ap/min_r_z/initial_guess_norm/recursive_true_residual_gap`及有效性。所有观察点按旧实现，solve/retry开始重置，返回集中导出。Production不分配/启动/写入/回读这些可选统计；结果对应字段使用原NaN/None，不写0或旧值。两模式保留`status/generation/iterations/true_residual_checks/residual_replacements/rho/rho_q/rho_x`及warm-start语义。

生产模式不关闭joint/contact最终力或全部Stats。必需状态包与可选摘要是不同职责，不为关闭诊断删掉曲率、正性、非有限、真实残差、stagnation或失败处理。

### 6.2 每轮调度与边界

1. 实际operator matvec得到Ap，归约pᵀAp并计算alpha/状态。读取必要状态；失败则跳过解更新。
2. 更新y/r并归约递推范数。在device计算“周期/末轮/提前候选收敛”检查标志，主机获取紧凑结果决定是否进入真实残差分支。
3. 检查分支以实际operator重算真实残差/global-q-x比例；主机仍决定成功、stagnation、replacement。检查未通过后的replacement仍重启方向，不沿用旧共轭。
4. 继续时应用固定预条件器并计算rᵀz/正性；保持正性检查在依赖更新之前。正常轮计算beta并更新方向，replacement轮按旧逻辑p=z。退出前保持真实残差后验与原失败优先级。

紧凑打包仅合并同一依赖边界的数据。设备stream依赖/归约与host等待分开测；无“零同步”承诺。普通继续轮当前8次源码回读可作调用审计基线，不能当8个全设备barrier或预定收益。

### 6.3 验证

建议`test_monolithic_pcg_modes.py`，对照旧路径与新diagnostic，再对照production；同归约算法时检查解/状态/迭代/检查/replacement一致及原容差。归约融合若改变判定边界导致差异，必须定位并证明原检查语义，不能一律以“浮点误差”豁免。

覆盖0轮、warm start、max-iteration、stagnation、tiny/非正曲率、非法预条件、NaN、stale generation、lambda retry、失败诊断、A→B→A。预热后审计无新增iteration分配；production无可选诊断kernel/buffer，diagnostic试探阶段不修改已提交物理状态。

## 7. PR-8D：分段CUDA Graph设计

### 7.1 Capture边界

| 图段ID（复用清单） | device计算 | 紧随其后的图外决策 |
|---|---|---|
| `pcg_direction` | scaling + BSR matvec + pᵀAp + alpha/status | 曲率/非有限失败 |
| `pcg_residual` | y/r更新 + 递推范数 + 是否需check标志 | 是否检查真实残差 |
| `pcg_true_residual` | 实际Ahat matvec、真实残差及global/q/x归约；diagnostic可选gap摘要 | 收敛、stagnation、replacement和退出 |
| `pcg_precondition` | apply + rᵀz/正性；通过后独立方向更新段或受状态保护的后续kernel | 正性失败、普通/replacement分支；不得提前更新方向 |

第一版只捕获固定计算链；Python循环、停止判断、BSR前缀重建、Newton/line-search/retry、history/State交换与最终发布均在图外。清单其余稳定段先测收益，不要求全部捕获。完整solver.step/整帧捕获不属P2出口。

### 7.2 图缓存与参数寿命

缓存由linear workspace拥有，有限数量变体按实际需要建立，不跨solver共享。键至少检查device/stream、执行模式、算子/预条件种类、buffer地址、dtype/shape、BSR布局/launch尺寸。图与factor使用不同有效标志：数值factor重建不等于重建图，图可复用也不等于factor有效。

矩阵values原地址更新且布局稳定时可重放；offset/column/values重分配、nnz/launch约束改变或预条件层次改变时图失效。dt/lambda、generation/token等捕获按值参数逐项登记；动态量使用稳定device buffer或已验证节点参数更新。跨retry必须使用新lambda，即使所有指针未变。固定alpha/beta算法buffer与kernel按值常量不得混淆。

先完成依赖/JIT预热，在业务物理状态之外准备图；capture准备和预热不能推进State/history。捕获后清理必要PCG scratch，初始化状态再执行。图内发生错误后，后续可能改状态的kernel必须被跳过或status保护；禁止为了图更大延迟原检查。

仅使用已验证capture-safe的BSR matvec与内存行为；builder继续图外。不以退回全容量排序换capture。库版本、源码修改和实际依赖记录在manifest，不能仅凭“支持GPU”判断capture可用。[CUDA官方编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)为运行时能力参考，目标Warp实际路径必须本地实测。

### 7.3 验证与清单维护

建议`test_monolithic_pcg_graph.py`，CUDA实际capture/replay多轮、不同lambda/RHS/values、地址/布局变更、普通/check/replacement轮、初始收敛、异常、stale generation、连续步及in-place。先验证单段与非graph一致，再整次solve/连续轨迹。CPU只验证非graph及非法开关，不填CUDA PASS。

清单状态为OUTSIDE_GRAPH → IMPLEMENTED_UNVERIFIED → VALIDATED；分别记录correctness/performance结果。捕获成功但无收益允许撤回并留证据，P2仍须交付分段开发/实验结论。计入capture/instantiate、重建、图内存、replay、边界等待与完整solve/step；短生命图不能只报replay微基准。

## 8. PR-8E/F/G：软体预条件器研究与准入

### 8.1 共用矩阵与SPD契约

从同一实际candidate提取已尺度化、含lambda的软体主块`Ahat_xx`，包括材料、质量与contact normal/tangent的xx项。q侧保留现有dense自块；完整q-x始终在global matvec中。预条件器只近似逆，不修改R/K或引入新的物理力。

一次PCG内固定线性SPD apply；每次从零纠正开始固定工作量，不把上次V-cycle输出当初值，不使用随residual变化的内迭代停止。assembly/scaling/lambda变化在下一次solve前更新数值并作factor校验。非SPD失败沿用regularization/失败路径，不取绝对值或任意clipping。

### 8.2 多节点局部块

构造期从tet邻接和动态节点表生成确定性的非重叠4/8节点分组，节点xyz不拆开；孤立/尾部小块保留全部动态DoF。分组与topology绑定，数值块`B_p=R_p*Ahat_xx*R_p^T`每次所需generation更新，Cholesky后应用：

\[
P_x^{-1}=\sum_p R_p^T B_p^{-1}R_p.
\]

分组不改变global行列顺序、contact key或物理节点编号。非重叠且覆盖所有动态DoF时，各SPD主子块形成SPD块对角预条件。以完整Ahat_xx提取避免consistent mass或接触遗漏；不是把孤立tet矩阵拼起来。

每个apply预分配局部向量/三角解scratch，支持原LinearOperator的alias与alpha/beta语义。先测试4/8节点与现有3×3；重叠Schwarz只有证据支持才另写加权SPD设计，不自动纳入8G。

### 8.3 Aggregation AMG-PCG

以弹性smoothed aggregation为研究首选，节点xyz共同聚合，平移/旋转候选按动态DoF限制并与尺度化一致。若物理位移候选为B，尺度化未知量的候选为`S_x^(-1)*B`；消除线性相关列。质量项和Dirichlet存在时称近零空间候选，不声称六个精确零模态。[PETSc GAMG](https://petsc.org/release/manualpages/PC/PCGAMG/)强调弹性系统的block size、坐标/近零空间输入。

研究采用Galerkin粗层`A_(l+1)=P_l^T*A_l*P_l`，固定V-cycle、适当的对称前后平滑与SPD粗解。配置必须使完整apply满足PCG契约，不能只检查每个粗矩阵SPD。[hypre BoomerAMG](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html)要求CG搭配对称平滑，GPU支持的选项需逐项核查。

层次结构/插值的构建与复用、粗矩阵/对角/谱界等数值更新分别记录；几何大转动、接触或lambda变化后的效果通过连续矩阵实验验证。复用规则在正式试验前冻结，不能运行失败后自动调整层数/平滑次数。层次重建在graph外；V-cycle固定buffer时才进入8D清单的候选扩展。

先用离线参考实现验证算法；外部库不是当前Newton依赖。获益后8G冻结后端、可选依赖/许可证与失败行为。不为了1600～2600量级的x自由度提前自研通用AMG，CPU迭代下降不等于CUDA总耗时收益。

### 8.4 两级准入与正式集成

研究先完成冻结矩阵/RHS的正确性与成本筛选；随后用实验性独立solver连续轨迹统计真实更新/重建成本。8G正式集成只能在这一总账GO后进行。每条路线单独决策，不比较各自最佳但不同h/N的运行。

总成本包括setup、格式转换/传输、数值factor/粗层更新、全部apply和PCG检查/retry、整步与固定时长轨迹、内存及尾延迟。至少5次安静重复，收益超过测量噪声且满足8A冻结门槛。以8C/8D后的相同运行模式作公平基准；旧同步开销减少不全归功于AMG。

GO记录适用设备/规模/阶段与预算；8G新增专用CPU/CUDA、q-only、generation、allocation、graph组合与事务测试，并重跑完整轨迹。NO_GO不接入production，旧actor-block继续可用。不会因研究NO_GO阻止P2出口，但缺失实验结论仍是未完成研究。

## 9. PR-8H/I/J：物理出口与最后一批优化

### 9.1 自由球校准与完整场景（8H）

复用完整手、carriage和解析支撑，不修改URDF/CSV命名映射；新增控制配置按joint name与q/qd/target offset寻址。25mm三档拓扑/密度/单位/SDF哈希独立验证。自由球构造时所有节点动态，从初态就关闭anchor；不能从anchored-close状态直接解锁并沿用history冒充自由抓取。

控制复用prepare→close→hold→lift→高位hold→release→observe的连续函数，当前9s时间表仅是探索起点；只在物理步末采样Control，手与软球不覆盖State。物理场景校准允许预先声明的姿态/球心/时序扫描，每组新建solver，保留失败；正式fixture不允许在线改参或跳步。

25mm、材料、所选h的接触刚度包络在三档球双压板试验重新确认。沿用预声明扫描`5e5/1e6/2e6/5e6/1e7/2e7/5e7/1e8 N/m^3`、mu=0.5、两倍承载余量、平衡误差<=5%、穿透<=2mm、min(detF)>=0.1、>=99%正常收敛及原残差；按实际25mm球质量定义载荷。三档全部通过的最低已测值才是新离散下限，抓取初始值`max(1e7,新下限)`。20mm的2e6旧下限不可直接复用；扫描无有效包络就报告阻塞。

冻结：至少两不同手指的有效承载力/持续时间口径、相对滑移参考frame、去刚体运动的压缩量、卸载恢复、release观测窗口/残留力、joint跟踪/速度/限位/effort容差以及性能目标。支撑平面与手指力分账；无支撑接触且所有节点动态时才累计Lift/高位Hold指标。

自由抓取actor-block预算筛查在正式G6前完成：Close/Hold/Lift每次实际PCG调用含retry统计p95，无调用不记零；>64或只有低于物理下限才能满足要求，登记8I触发。已知S=1/2粗h研究超预算保留，不因最终不用该h就把它改写成未触发；正式出口只对冻结支持包络做结论。

### 9.2 条件normal patch Schur（8I）

触发语义继承PRD ADR-003，不能由“soft AMG似乎足够”取消actor-block基准触发证据。未触发必须给有效多指/三档预算证据；触发后实现、SPD/容量与性能验证成为必做。若无法关闭，P2整体出口仍未完成，不能用性能候选NO_GO豁免。

采用**不含contact**的尺度化actor base`P0`（dense q与particle 3×3，含lambda）；正权重NORMAL factors构成`Ghat=G*S`。精确小系统参考为：

\[
P=P_0+\widehat G^T W\widehat G,\quad
S_c=W^{-1}+\widehat G P_0^{-1}\widehat G^T,
\]
\[
P^{-1}=P_0^{-1}-P_0^{-1}\widehat G^T S_c^{-1}\widehat G P_0^{-1}.
\]

normal-only预条件允许忽略切向近似；global operator与原actor-block对照仍包含全部normal/tangent。禁止用已含normal自块的base再加同一normal造成重复计入。首版不与8G的AMG/大块组成未经证明的混合Schur。

按link及boundary adjacency划分有限、非重叠的normal-row patch，零权重省略，负/非有限权重失败；patch/tet/contact静态容量用host int64校验再分配，不能截断。每次assembly/lambda更新重建必要数值；trial/history不参与持久patch状态提交。

**设计关闭项 SCHUR-SPD-01：** PRD/P1给出了精确Woodbury及patch间块对角近似方向，但normal-row不重叠不代表q/节点支撑不重叠。直接把`S_c`跨patch块删除，再代入上述减法，不能自动推出SPD。8I必须先提交多patch共享q/节点的反例测试、最终近似公式与SPD证明/对应ADR，再写production apply；不得把“每patch Cholesky成功”当完整apply SPD证明。本文冻结这一开发前置关卡，不把尚未证明的近似标成可直接实现。实际矩阵只在tiny oracle densify，不构造无界dense全接触Schur。

最小反例：`P0=1`、两个patch各一行`G1=G2=1`、`W1=W2=1`，则精确`Sc=[[2,1],[1,2]]`；只保留两个对角patch后，上式近似apply为`1-1/2-1/2=0`，失去严格正定性，而精确逆为`1/3`。该反例说明必须审查完整近似，不能仅验证局部factor。

通过该关卡后验证：单patch与多patch定义各自的dense inverse-apply、任意向量线性/对称正二次型、inactive/negative W、溢出/非法重叠、lambda refactor、generation/alias、normal/tangent分离、共享生命周期；性能必须含setup/factor/apply及全solve/step，最终重跑8K。

### 9.3 最后一批计算优化（8J）

8B～8I实际代码/NO_GO/NOT_TRIGGERED关闭后重新profile，完成固定candidate测量。仅对仍显著的Smith应力/Hessian、完整12×12 PSD投影、fixed-node/triplet写入、owner/global重复工作或contact geometry/residual/factor选择改动。

优先并行度、复用中间值和减少写入；保持Smith响应、PSD投影定义/容差、质量和接触公式，不使用Kim替换或物理近似冒充优化。每个命中热点单独小提交，CPU/CUDA raw子项/production parity与状态/history回归后测至少5次完整成本。无显著热点/收益记录NO_GO或DEFERRED，最后才进入8K正式运行。

## 10. 验证矩阵与总成本门槛

### 10.1 分层验证编号

本表V-P2编号只用于追踪，不替代原G0～G7。

| 验证ID | 范围/PR | 必须证明 | 设备 |
|---|---|---|---|
| V-P2-01 | 8A测量 | 覆盖/计数、时间单位、真实载荷、固定candidate、索引/源码可追溯 | CPU工具；CUDA真实窗口 |
| V-P2-02 | 8B质量矩阵 | 同candidate M/能量/R/K、惯量frame、A→B→A、分配与事务 | CPU/CUDA |
| V-P2-03 | 8C PCG | diagnostic/production必需结果与失败检查、无可选统计工作 | CPU/CUDA |
| V-P2-04 | 8D graph | 真capture/replay、参数/布局失效、检查边界、图内存及摊销 | CUDA；CPU非graph |
| V-P2-05 | 8E多节点块 | 完整动态覆盖、主子块、SPD/dense apply、总成本GO/NO_GO | CPU/CUDA分别决策 |
| V-P2-06 | 8F AMG | 候选空间/粗层/固定线性SPD周期、层次复用、总成本GO/NO_GO | CPU/CUDA分别决策 |
| V-P2-07 | 8G运行时 | 获准路径、q-only、lambda/generation、alias/分配、选项组合与连续轨迹 | 获准设备；原CPU/CUDA回归 |
| V-P2-08 | 8H物理校准 | 25mm/目标h三档包络、两指/支撑分账、阈值预冻结、Schur触发 | CUDA真实资产；CPU解析/控制 |
| V-P2-09 | 8I条件Schur | SCHUR-SPD-01、dense/PSD/容量/失败、物理/成本 | CPU/CUDA组件、CUDA抓取 |
| V-P2-10 | 8J末批优化 | 原物理/投影/candidate/装配一致、最新基线完整成本 | CPU/CUDA |
| V-P2-11 | 8K最终出口 | 原G6/剩余G7、模式组合、原G0–G5/G1H/G2H与兼容性 | CUDA E2E；CPU组件 |
| V-P2-12 | 8L证据 | 最终代码/资产/阈值绑定、实现/失败/延后清单完整 | 静态核验 |

### 10.2 误差与生命周期

沿用PRD/P1及实际冻结测试中更严格的门槛，不由本文重置：owner/global与sparse/dense matvec通常CPU<=1e-5、CUDA<=5e-5；tiny direct解<=1e-4；PCG真实global/q/x<=1e-4；scaled operator对称误差<=1e-4。各fixture原绝对/相对、近零量及FD阈值仍以冻结记录为准。新增预条件的inverse-apply使用相同设备门槛；SPD需算法论证与数值测试，随机向量通过不是一般性证明。

仅对exact子项/raw Jacobian做FD，平滑区至少5个步长扫描；kink按原单侧/广义导数契约。完整projected K不要求等于FD(R)。更改preconditioner允许PCG迭代变化，纯诊断/图执行改造原则上保持算法分支；差异须具体归因并按原门槛复验。

所有影响shared step的实现覆盖：current/trial、A→B→A、rejected activation、regularization retry、初始即收敛、soft stop、hard rollback、in-place、异常、State copy与joint/contact发布故障、stale generation、重复update_contacts与history epoch。缓存不能使失败/回滚态发布来自rejected trial的力。

### 10.3 G6/G7最终验收

- 全22 DoF + carriage + 25mm全动态软球；至少两不同手指持续承载，支撑反力不计入。Hold/Lift不掉落，COM上升>=carriage命令80%，高位1s Hold相对滑移<=10mm。
- min(detF)>=0.1、穿透按冻结2mm初始门槛、>=99%正常收敛、原global/q/x及有限性通过；hard rollback/异常停止并保留失败，不外推完整轨迹FPS。
- 独立solver friction-off只关闭接触mu，joint friction/资产/材料/控制/h/N相同；至少违反不掉落、COM提升或滑移一项，不能只观察摩擦力为零。
- Release观察重力下落、解除夹持与失效history清零，去刚体运动后有超过噪声的压缩与恢复；按8H冻结窗口/容差逐项判定。
- 三档网格报告nodes/tets/faces、动态/固定数、静态pair/factor/triplet容量、active与实际nnz、overflow、完整轨迹各类内存、Close/Hold/Lift PCG含retry、step/solve p50/p95及wall/sim。
- 最终代码/fixture与所有GO优化组合一致；保留全默认参考组合，验证单项开关和最终组合。graph与diagnostic/production两维测试；只为已获准预条件增加组合，不制造未实现后端笛卡尔积。

### 10.4 测试命令约定

当前可执行的范围入口：`uv run --extra dev -m newton.tests -k monolithic`；项目既定兼容检查按仓库AGENTS/指南及既有验收命令执行，GPU/资产缺失必须标NOT_RUN而非PASS。新模块名（如`test_monolithic_pcg_modes.py`）为开发建议，尚不存在时不能写成已执行命令。

每PR实际提交前执行`uvx pre-commit run -a`，保留完整日志；bug/测量修复先证明原实现失败，再验证修复。数值回归按变更范围逐步执行，8K最终运行完整monolithic与既定兼容；此架构提交仅静态文档/链接/哈希/依赖检查，不冒充数值测试。

## 11. 证据、开放项与阶段出口

### 11.1 证据所有权

每PR独立目录`agents/integration/artifacts/p2/pr8x-*`及独立报告，manifest绑定：实际代码提交/dirty diff、PRD/P1/P2计划/本架构快照、运行脚本和依赖、设备/线程/工具、URDF/mesh/SDF/CSV/映射/球网格/fixture、所有执行模式、输入参数、阈值版本、命令、traces/曲线/截图及失败点。

报告引用已提交的实现hash；若只新增记录提交，分别写实现提交与记录提交。源码/配置变更后受影响运行失效，不能仅换manifest的hash。旧P1/G1/G2/诊断快照保持原样；索引不包含自身/hash文件，不漏NPZ，不把CUDA pool reserve高水位标作全部显存。

当前交付为设计草案，开发状态统一NOT_STARTED，8G/8I为CONDITIONAL_NOT_STARTED。仅在完成对应开发与验证后写PASS/GO；NO_GO、NOT_TRIGGERED、DEFERRED、NOT_REQUIRED、NOT_RUN分别表达不同原因。继续现有integration分支本地可审查提交；本轮不推送，后续远程同步按届时授权执行。

### 11.2 尚须冻结/关闭的设计项

| ID | 项目 | 关闭位置 | 不阻止的独立工作 |
|---|---|---|---|
| OPEN-01 | P2-0测量缺口、时间误差及耗时/尾延迟/内存预算 | 8A；自由抓取增补8H | 本架构评审与测量修复 |
| OPEN-02 | graph最终接口、目标Warp实际capture支持/图缓存细节 | 8D单段反例与benchmark | 8B、8C及离线矩阵研究 |
| OPEN-03 | 大块/AMG实际总收益、后端/许可/设备规模范围 | 8E/F→8G | PCG/质量矩阵与抓取探索 |
| OPEN-04 | 25mm/目标h校准、两指姿态及形变/release/速度阈值 | 8H | anchored-close性能线 |
| SCHUR-SPD-01 | 多patch近似逆的SPD公式与证明 | 8I开发前ADR/数学关卡 | 未触发场景的其他优化；8H触发筛查 |
| OPEN-05 | 最后compute批的实际热点 | 8J前新profile | 所有前序PR |

### 11.3 出口判定

性能线关闭要求：8A预算证据有效，8B/C/D有实际开发/验证结论，8E/F各有总成本GO/NO_GO、GO路线8G验证通过，8J有执行或有证据的延期结论；推荐配置有相同质量下总耗时收益。若未达冻结性能目标，明确性能出口未通过，不以研究已做完代替收益。

P2整体出口还要求8H校准、8I触发项关闭、8K G6/剩余G7与既有组件/兼容全部通过、8L索引/报告齐全。仅性能成功或仅anchored-close稳定不能宣称V0.2 Grasp MVP完成。必做项实际阻塞就保留未完成；完整step graph、未获准AMG、多余材料替换、自碰撞/遥操/SuperDex不新增为出口要求。

## 附录A：输入绑定

本文输入文件以共享工作区为根；仓库路径以`worktrees/monolithic-integration`为根。以下哈希冻结本设计所读内容，live计划后续更新不改变本版输入。

| 输入 | SHA-256 |
|---|---|
| `dev/monothlic/newton_monolithic_solver_prd_zh.md` | `da2107a04d45227ffcb929fbca5b586c677dbcfd56be445e7eb136f479968f77` |
| `dev/monothlic/newton_monolithic_solver_p1_architecture_zh.md` | `04938ec649e7e16b523c2ca0b2d5a14e4020de4f4418e52c9ee851c024d17f17` |
| `dev/monothlic/newton_monolithic_solver_p2_plan_zh.md` | `08dc5dc4086cab2628aec201344deb5cb9109600f24073eafdf29112181c0475` |
| `agents/integration/P2_0_DIAGNOSTICS_REPORT.md` | `99bc6c9ef4866b157c6723b9ba74f88411cacc5d42725950539cf40068b4431c` |
| `agents/integration/P2_CUDA_GRAPH_INVENTORY.md` | `f4c05a1adc63ab0a78cb6c6844b2eb4c7ff53c54887c50441176c252c8a7b04b` |
| `worktrees/monolithic-integration/scripts/monolithic_reference/fixtures/p2_owned_mass_matrix_plan_v1.json` | `078d882b9d08003ae57f88f742c1ea4b9075e1cb0d2a2c22717cdaed9e27b053` |
| `worktrees/monolithic-integration/scripts/monolithic_reference/fixtures/p2_pcg_diagnostics_plan_v1.json` | `032877cd4c97f089a4b24cd5ececa0f9ae85ae2f21c5c64631510420a0540a95` |
| `worktrees/monolithic-integration/scripts/monolithic_reference/fixtures/p2_cuda_graph_inventory_v1.json` | `d9de475a927a7fb6353c7855aeb5511c641a4e50de5af7b9d7d4c67afec640cb` |
| `worktrees/monolithic-integration/scripts/monolithic_reference/fixtures/p2_preconditioner_study_v1.json` | `0ba9a90d174c32e31cffbfd2168c1fb276c5222a406d46f4541493b6de32c37a` |
| `worktrees/monolithic-integration/scripts/monolithic_reference/fixtures/p2_remaining_compute_plan_v1.json` | `0cdf8ae45da327971e3dc7a9c3b9a08d45fad62159fbc97e458cf651d07bf454` |
