# PR-01 补充：机器人 FP32 性能对照

2026-11-02。状态：PREPARED，正式成对测量 NOT_RUN。

用户要求增加 Franka、Sharpa 的 FP32 性能与 baseline 对照，并明确选择现有冻结窗口、三组成对测量。baseline 为集成主干 `8fa80ba7ce3fd7ce2483f2ec098e66d410bc434e`；新版为本 worktree 的 PR-01 实现（solver 代码与 `22209715` 相同，实际运行 HEAD 写入 receipt）。不改 solver、冻结输入或物理门槛。

| 场景 | case id | 恢复点 | 连续运行步数 |
| --- | --- | --- | --- |
| Franka 抓取 | franka-grasp | 1900 | 500 |
| Franka 释放 | franka-release | 4150 | 300 |
| Sharpa 闭合 | sharpa-close-fp32 | 3240 | 270 |
| Sharpa 抬升 | sharpa-lift-fp32 | 7000 | 200 |

Franka 沿用 quick 的 FP64 快照导入为 FP32；Sharpa 使用现有 FP32 快照。场景、控制程序、SDF、资源容量及发布策略全部复用 `quick_scenes` / `FrozenRobotFactory`。它们是实际机器人场景的固定窗口，不能外推完整轨迹或端到端应用吞吐。

## 测量合同

1. 同一 GPU 串行，独立进程顺序为旧→新、新→旧、旧→新；每个进程运行四个窗口，失败不阻止收集其他窗口或版本。
2. 每个窗口完整预热一次，再恢复同一快照和控制时钟，完整测量一次。初建、编译与首次 capture 排除；测量期间重复 capture 保留为实际成本。
3. 同步前排空已有设备工作，计时覆盖 `solver.step()` 至设备完成。包含入口校验、H2、最终 Stats 和原有诊断；排除控制器、快照恢复、离线接触测量及 JSON 输出。不另启 profiler。
4. 保留逐步耗时、publication、Newton/PCG/scheduled slots/trial/retry 及 capture 数；Stopped 与 Converged 分开。各轮均按原 complete、hard failure、Stopped、穿透、det F、抓取门槛验收。
5. 比较前核对导入路径、两版 HEAD、同一 harness hash、冻结 manifest/snapshot、物理 digest、配置、模型规模与环境；报告逐步工作量是否一致。工作量或轨迹分叉时只报告实际窗口性能，不宣称纯执行加速。
6. 汇总每窗口、每机器人总 solver 时间、逐步 median/P95、三组成对比例；物理结果和性能结论分别记录，不新增未经批准的速度 PASS 阈值。

入口（从受验 worktree 导入其 solver；跨版本调度使用同一已提交 harness）：

```bash
uv run --no-sync python -B -m scripts.monolithic_reference.benchmark_robot_windows \
  --precision fp32 --device cuda:0 --output <new-result-dir>
```

结果根目录：`dev/monolithic/test/results/P9/P9a/PR-01/robot-fp32-20261102-01/`。包含成对调度脚本、每版每轮 receipt/log、比较摘要、环境/输入/源码身份与哈希索引。

## 工具验证

新增功能测试验证计时只包含 solver（排除恢复、控制和 recorder），预热后回到相同输入，硬失败不能通过，Stopped 保持独立计数。新测试与原 quick manifest/gates/runner 的受影响检查共 8 项通过。Fit 和 Standards 独立静态审查通过，Requirements 计时/状态路径审查未发现阻塞项。正式运行前执行 pre-commit 并提交工具；最终结果另补文档。
