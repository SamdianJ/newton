> Historical development record: commands for retired BE, q-only and x-only
> drivers below are not active entry points. See [current entry points and
> retirement notes](legacy_retirement.md). Frozen evidence and assets remain unchanged.

# PR8E/F 预条件器研究

**当前状态：P2 8E/8F 已收尾，8G 为 NOT_APPLICABLE。** 当前 MAS 未展现明显端到端提升，默认保留 actor_block；SA-AMG 转入 P3 考虑，GPU 未测。见[收尾决定](preconditioner_closeout_p2_zh.md)。以下门槛与命令保留为研究协议，不代表 P2 仍等待性能准入。

本工具不改变 solver 默认值，不接入公共预条件器选项。候选为 actor_block、local_block_4/8、mas_16/32；研究范围是完整已尺度化、含 lambda 的 A_xx，q 侧保持 dense，完整 q-x 在全局算子中。

`uv run --no-sync -m scripts.monolithic_reference.preconditioner_study --snapshot <export.npz> --output <新目录>` 可重复指定快照；省略快照运行具有几何、非均匀尺度与 q-x 耦合的合成 SPD 用例。`--linear-absolute-tolerance 0` 仅用于独立压力对照。输入使用现有 NPZ/JSON 哈希认证，结果保留向量、逐轮误差、独立真实残差与异常，不改变验收阈值。

8E 使用 FP64 研究 PCG，每轮检查独立真实残差；这不是原生 PCG 策略或 GPU 性能结果。局部块按确定性邻接遍历；MAS 根据参考位置 Morton 排序，每域 16/32 节点，粗聚合按域内拓扑连通分量拆分，最多三层，不能继续缩减时停止。层次固定；粗空间 R 使用 inverse scaling，把物理常量位移转换到 scaled 坐标。粗层块为 R Ahat_xx R^T，lambda 由投影进入，不再额外添加。

每个块用独立 FP64 Cholesky 求解。细层覆盖全部动态未知量，块 SPD 使相加后的 inverse apply SPD；有限精度仍须验证，不允许 clipping 或静默回退。研究结论须分开报告数值正确性、CPU 成本和 GPU 未测状态。

先完成先导测量，再提交评审冻结时间/尾延迟/显存预算。正式性能验证及 8G 在该节点之后；无候选通过时不改变默认。新版本记录见 fixtures/p2_preconditioner_study_v2.json，历史 v1 不改写。整套 nightly 由用户手动执行。

## PR8F 实验入口

- `preconditioner_pilot --matrix-list <NPZ路径数组.json> --candidate mas_32 --output <新目录>`：实际原生 production graph，记录层次、转换与初始 factor、数值更新、apply、solve、捕获、独立误差、显存数组容量；diagnostic 收敛观测单独运行，不计入性能。`--mode diagnostic` 可直接测 diagnostic。
- `preconditioner_pilot_batch --kind matrix --matrix-list <路径数组.json> --output <新目录>`：默认五个候选、三轮独立进程，奇偶轮反转候选顺序；每个作业保留失败后继续，最终返回失败汇总退出码。
- `preconditioner_trajectory --candidate mas_32 --kind tet --refinement 6 --output <新目录>`：现有完整 36 秒 tet 协议、owned mass matrix、production graph；`--kind sharpa --refinement 2/3` 使用完整 prepare/close/hold。`--smoke` 只验证短程，不可作为性能证据。
- `preconditioner_pilot_batch --kind trajectory --candidates actor_block mas_32 --output <新目录>`：tet5/6 与 Sharpa r2/r3 的三轮交替先导。参数只作用于研究入口，没有公共 solver 参数。

以上模块均用 `uv run --no-sync -m scripts.monolithic_reference.<模块名>` 调用。候选、固定层次指纹与实际构造配置保存在 `preconditioner-options.json`；原有执行配置和源码快照保留。轨迹 step 时间包含 factor、capture、retry 和事务路径，构造成本另外记录；末态误差仍是待标定对照，不能认证物理真解。

局部块采用 Warp tile Cholesky 三角解，MAS 采用 Cholesky 构造显式逆后 tile 矩阵乘法。为满足既有 oracle 容差，投影与 Cholesky/求逆使用 FP64；MAS 的已构造逆及 apply 保留 FP32，局部块三角解使用 FP64 后转回 FP32。96×96 逆按列块构造以满足设备 shared-memory 容量。纯 FP32 构建在病态合成用例上的失败记录保留，不放宽容差。全部 device scratch 在构造时分配；单次 PCG 不更新层次或数值。研究因子更新目前直接发射 kernels，未捕获该段，成本必须计入；PCG 的已有条件 graph 则复用数值更新后的原地址。结构或绑定变化须构造新对象；非法主元/非有限值沿用现有失败状态，不 clipping、不切换预条件器。q-only 使用原 dense q 分支。

## SA-AMG 独立对照

使用隔离的 `uv run --no-sync --with pyamg==5.3.0 --with numpy==2.5.0 --with scipy==1.17.1 -m scripts.monolithic_reference.preconditioner_pilot --candidate sa_amg_cpu --matrix-list <路径数组.json> --output <新目录>`。PyAMG 不写入项目依赖，实际 NumPy/SciPy/PyAMG 版本与整个依赖清单随每次源码证据冻结；需要复现既有结果时按该清单锁定研究环境。

动态节点 xyz 作为一个 BSR 节点；物理平移/旋转经 inverse scaling 后 SVD 去除相关列。固定 SA 插值、Galerkin 粗矩阵、一次对称 block GS 前后平滑、Cholesky 粗解，apply 总从零修正开始。平滑的对角块也用 Cholesky 显式构造逆，拒绝奇异主元。完整构建与固定物理插值的数值更新分别计时；尺度改变时同步变换首层 P/R，重新建立各层 A、平滑逆和粗层因子。每次 PCG 的周期固定。

SA-AMG 的 PCG 成本是包含每轮独立观测的 CPU FP64 研究成本，不能与原生 GPU 时间直接比较；GPU 状态始终为 `NOT_MEASURED`。MAS 结果不能替代 AMG GPU 的独立评审。

## v2 阶段门槛

8E 的冻结矩阵数值正确性与 8F 的 GPU/CPU 先导分别交付。主容差 rtol=1e-4、atol=1e-6 不变；64 为既有迭代 p95 预算，200 为硬上限。额外 atol=0 结果单独目录保留。真实残差未通过的结果，即使 native status=SUCCESS，也不得计为数值通过。

先导完成后需评审并填写最低轨迹总耗时收益、允许 solve/step p95 回退、显存增量和适用场景。此时预算仍为未冻结，不宣布性能 GO 或实测准入 NO_GO。冻结后另取至少五次独立完整轨迹重复，再进行最终源码的正式验证。

8G 仅对正式通过候选实施：显式 `soft_preconditioner`，默认 `actor_block`，同步 runner/实际构造/元数据，保留 generation、lambda、graph、q-only、history、retry、rollback 与最终发布回归。无候选通过时标记 `NOT_APPLICABLE`，AMG GPU 未测与实测 NO_GO 分开记录。当前不实现此条件分支，不自动执行 nightly 或推送。
