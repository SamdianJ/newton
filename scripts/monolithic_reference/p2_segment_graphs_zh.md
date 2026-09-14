# P8D 稳定计算段

`SolverMonolithic(..., pcg_mode="production", use_cuda_graph=True)` 在已有完整 PCG 图之外启用以下六个设备段。默认仍为 diagnostic / graph=false，CPU 保持非图路径。该选项是运行时执行配置，不是模型或 USD 物理参数。

| 段 | 图内 | 图外保留 |
|---|---|---|
| articulation | FK、Jacobian、reference/owned mass、gravity、Coriolis | model/state 校验；joint term 状态检查 |
| tet_residual | trial 残差清零、惯性、弹性、finite 检查 kernel | 参数与物理 identity 校验；detF/finite 读回与 line search |
| norms | scaling/copy、global/q/x 范数归约 | finite 检查、RMS 转换、非线性收敛判定 |
| assembly_initialize | 固定缓冲清零、固定 triplet 前缀恢复 | generation/token 更新、descriptor 建立、BSR builder |
| scaling | 动态对角复制、BSR 对角提取、尺度构建 | generation/输入检查、状态读回与有效标志 |
| factor | q/x 预条件矩阵构建、Cholesky | generation/scaling/lambda 校验、pivot 状态读回、retry |

不捕获 tet 切线、碰撞/接触求值、State/history 交换、最终发布或完整 step。不改变数值 kernel、残差阈值、regularization 顺序与检查边界。归约仍按现有浮点容差比较，不承诺逐位一致。

每个 solver 独占一个私有缓存，每段最多四组绑定，按近期使用淘汰，总计最多 24 张稳定段图，PCG 图独立。缓存键包含 stream、按值标量/模式以及数组地址、shape、stride、dtype、device，持有数组和 stream 的强引用。accepted/trial 两个真实缓冲及 final scratch 分别缓存，交换角色无需重捕获；数值 generation 更新本身也不重捕获。输入值原地址更新被下一次 replay 读取。dt、lambda、epsilon、guard、模式或绑定变化建立对应图，容量有界；频繁变参导致的淘汰/重建须计入总成本。

冷路径先执行一次可覆盖的设备计算来完成 JIT，再 capture 并 replay。所选段只覆盖派生缓存、残差或装配 scratch；预热不推进物理时间、积分或提交 history，也不包含 append 操作。token 的数值更新和 stale-view 检查仍由主机/原有生产 kernel 管理，不把旧 token 捕获进清零图。不得从已有外层 capture 中调用完整 solver；其图外主机检查仍需要读回。

tet/Sharpa 的 compact.json 记录 `segment_graphs`：每段 captures/replays/resident_graphs/evictions，以及 capture_host_seconds。后者包含冷路径预热、capture 和首次提交的主机时间，不是纯 GPU capture 或纯 replay 时间；完整 solver 计时包含实际完成成本。resident_graphs 是数量上限统计，不代表驱动图对象精确显存字节数，性能验证另测显存使用。

分层 profiler 不向捕获段内插入同步；启用稳定图时，预条件器的 q/x kernel 明细不再单独采样，零 calls 表示无独立 kernel 计时，实际开销包含在 preconditioner_setup_factor 中。

验证入口：`uv run --no-sync -m unittest newton.tests.test_monolithic_segment_graphs`；该模块也加入 dev/night 的单元回归。覆盖变化输入/dt/guard/lambda、owned/reference 动力学、候选/缓冲与 stream 重绑定、缓存上限、旧 token、固定前缀、非有限/pivot 失败、接触 history/retry/rollback 和冷图 profiling。新清单为 `fixtures/p2_cuda_graph_inventory_v3.json`；v1/v2 保留为历史快照。

性能对照需在相同 production PCG graph、atol=1e-6 下只切换新增稳定段，使用独立进程、相同场景、预热和交替次序。开发测量可以在专用 harness 中关闭私有 `_segment_graphs.enabled`，不是新增公共配置。至少五次安静重复，报告完整 step 与捕获/缓存/内存；单纯 capture 成功不等于性能准入。nightly 仍仅 diagnostic 和 production graph 两种仿真模式，不自动执行。
