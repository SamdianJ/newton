# P2 验证：PR-8C2 masked smart PCG

当前 production 唯一实现为 `masked_smart_restart_v1`；diagnostic 为 `dynamic_smart_restart_v1`。
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

所有作业使用同一个干净的当前源码工作树；不传 `--baseline-repo`，不执行旧 checkout。旧 51 作业协议不再适用。

同一源码/依赖/策略恢复：

```bash
uv run --no-sync -m scripts.monolithic_reference.validate_p2 \
  --suite night --output /absolute/path/night-p28c2 --resume
```

源码、依赖、policy 或计划改变时必须使用新输出目录。旧 policy 在任何子进程之前拒绝；日志/索引/原始证据哈希也会校验。
中断、失败、旧日志和未完成输出保留；`--resume` 不把旧求解器结果计作新策略通过。

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

当前 graph 只在同一 assembly generation、绑定地址、stream 和配置内复用；跨 generation 保守重建。绑定校验的主机读回与分配发生在准备边界，不能被归入零读回热路径。长期拓扑缓存、图内存/摊销和性能优化由 8D 继续负责。

开发正确性和 capture 通过不代表推荐启用或性能通过。PCG p95<=64/max<=200、同物理质量、固定尾部成本、图重建/内存/摊销、匹配设备状态重复计时、8A 精度与预算、near-static、G6/G7 均保持原门槛。
本轮开发不执行完整 nightly，完整性能和轨迹验收由用户手动运行。
