# P2 验证：PR-8C2 masked smart PCG

当前 production 唯一实现为 `masked_smart_restart_v2`；diagnostic 为 `dynamic_smart_restart_v1`。
两者共用 global/q/x 分块 gap 判据，阈值默认 0.1。默认仍为 diagnostic，CUDA graph 默认关闭。
固定 production 检查点为 0、每 10 槽、预算末尾；不保留 stagnation、近零曲率启发式、递推候选额外检查或失败后额外真残差。
旧 always-replace production、旧专用测试和旧 baseline checkout 调度已删除，历史数据仅作离线追溯。

## 日常开发验证

在本仓库根目录执行：

```bash
uv run --no-sync -m scripts.monolithic_reference.validate_p2 --suite dev
```

只运行小规模 CPU/CUDA 模块：owned mass、共享线性/diagnostic restart、masked PCG、完整 graph、事务、runner 与恢复机制。
新增 graph 测试使用实际 Warp capture/replay，并对成功、失败、warm start 的热路径设置读回/同步/分配 trap。
同策略 dense 主机 oracle 仅用于小矩阵测试，不是运行时 fallback；原动态 diagnostic 不要求与固定调度迭代数相同。

短 CUDA trace（仅正确性审计，不是性能测量）：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output /tmp/pcg-trace \
  uv run --no-sync --with nvtx python -m scripts.monolithic_reference.probe_p2_8c2
```

`nvtx` 仅为这个专项探测的可选工具依赖，不新增 Newton 运行时依赖。

## 用户手动夜间验证

先预览，仅生成计划、不导入 Warp、不启动子作业：

```bash
uv run --no-sync -m scripts.monolithic_reference.validate_p2 --suite night --list
```

确认运行窗口后，使用全新输出目录：

```bash
uv run --no-sync -m scripts.monolithic_reference.validate_p2 \
  --suite night --output /absolute/path/night-p28c2
```

当前矩阵先运行 monolithic 单元回归，然后串行运行：

| 场景 | 当前源码变体 | 重复 |
|---|---|---|
| Sharpa r3/r2 | diagnostic、owned diagnostic、production、production graph、owned production、owned production graph | 各 5 次 |
| tet r5 | diagnostic、production、production graph | 各 5 次 |
| tet r3/r4/r6 | diagnostic、production、production graph | 各 1 次 |

nightly 默认无人值守：单个测试组或仿真任务以非零状态退出后，保留日志/输出并继续后续独立任务；单项结果文件缺失或格式错误同样记录为任务失败。不会在同一轮自动反复重试。源码冻结检查失败、CUDA 启动预检失败、主机进程启动/I/O 异常和用户中断仍立即停止。

全部任务尝试结束后，`summary.md` 汇总成功任务、未达标指标、失败原因及日志链接，`progress.json` 的 `failed_jobs` 列出未解决的任务失败：

| 退出码 | 最终状态 | 含义 |
|---|---|---|
| 0 | COMPLETED | 所有任务正常结束且现有筛查通过；正式性能仍需评审 |
| 1 | COMPLETED_WITH_FAILURES | 存在任务退出失败或结果无法读取；其他独立任务已继续执行 |
| 2 | COMPLETED_WITH_OPEN_GATES | 无任务执行失败，但仍有数值或 PCG 预算未通过 |

任务失败和指标未达标同时发生时返回 1，两类问题都保留在汇总中。

所有作业使用同一个干净的当前源码工作树；不传 `--baseline-repo`，不执行旧 checkout。旧 51 作业协议不再适用。

同一源码/依赖/策略恢复：

```bash
uv run --no-sync -m scripts.monolithic_reference.validate_p2 \
  --suite night --output /absolute/path/night-p28c2 --resume
```

源码、依赖、policy 或计划改变时必须使用新输出目录。旧 policy 在任何子进程之前拒绝；日志/索引/原始证据哈希也会校验。
中断、失败、旧日志和未完成输出保留；`--resume` 跳过已完成且证据校验通过的任务，只重试失败/未执行任务。重试前归档该任务旧输出，追加新日志；恢复成功后不再计为未解决失败，历史失败记录继续保留。已完成但指标未达标的任务不会自动重跑，也不会因其他任务恢复成功而变为通过。

工作流标识为 `p2-8c2-masked-smart-v2-unattended`（数值 PCG policy 不变）；先前版本的运行不能直接 resume，请使用新输出目录。源码修复或依赖变更后也必须使用新目录，不能混合版本续跑。

单项/短冒烟示例：

```bash
uv run --no-sync -m scripts.monolithic_reference.run_p2_8bc \
  --kind tet --refinement 3 --pcg-mode production --use-cuda-graph \
  --smoke --output /absolute/path/tiny-p28c2
```

省略 `--use-cuda-graph` 为相同 fixed smart 策略的非图执行。与 diagnostic 比较是策略对照；production 图开/关才隔离图执行方式。

## 结果与保留门槛

每次 PCG 调用（含 retry）记录 policy、有效 iterations、scheduled_slots、验收/最后检查/首次失败槽位、检查/replacement 数、output_valid、最后检查 ratios 与新鲜度。
新增 `prepare_seconds`、`capture_seconds`、`graph_rebuilt` 标识图外准备和首次图建立成本；`capture_seconds` 包含捕获及 Warp 第一次 launch 的延迟 instantiate/upload 和主机提交时间，GPU 执行计入完整 solve/step。热路径审计先在区间外完成一次 replay/collect 预热。measurement schema 升为 v3，不改旧证据。

当前 graph 在绑定地址/容量、稀疏结构、stream 和配置不变时跨数值 generation 复用；每个新 generation 在准备阶段读取并验证 CSR offsets/columns 摘要。结构或存储改变重建，旧 generation 请求、过期 scaling/factors 仍拒绝。RHS、矩阵值、尺度、已重算因子和 lambda 使用固定设备缓冲更新。

支持 CUDA 条件图时，PCG 每个真实残差检查批次判断是否继续；零 RHS 在初始检查后跳过全部迭代，WAIT_CHECK 会继续到下一检查点，末尾不足一批按实际剩余预算执行。CPU、非图及不支持条件图的 CUDA 保留固定 masked 调度。`scheduled_slots` 在条件图下为实际调度的槽数（含当前批次内的 masked 尾部），固定调度下仍为完整预算。最大有效迭代预算及原生数值检查保持。

CUDA 最多 2048 个标量的点积使用单 CTA 的 float32 tiled 归约，合并清零与求和；更大系统保留原并行归约，避免把大向量串行压到单 CTA。归约顺序与 v1 的 atomic add 不同，不保证逐位相同。非有限数仍由现有 alpha/gamma guard 拒绝。主机绑定验证、结构摘要读取、模块预热及分配均在热路径外；完整调用成本必须包含准备阶段。

开发正确性和 capture 通过不代表推荐启用或性能通过。PCG p95<=64/max<=200、同物理质量、固定尾部成本、图重建/内存/摊销、匹配设备状态重复计时、8A 精度与预算、near-static、G6/G7 均保持原门槛。
本轮开发不执行完整 nightly，完整性能和轨迹验收由用户手动运行。
