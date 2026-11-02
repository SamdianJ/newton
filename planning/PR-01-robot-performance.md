# PR-01 补充：机器人 FP32 性能对照

2026-11-02。状态：COMPLETE；四窗口原物理门槛通过，性能退化登记在 `P9a-PERF-01`。

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

新增功能测试验证计时只包含 solver（排除恢复、控制和 recorder），预热后回到相同输入，硬失败不能通过，Stopped 保持独立计数。新测试与原 quick manifest/gates/runner 的受影响检查共 8 项通过。Fit 和 Standards 独立静态审查通过，Requirements 计时/状态路径审查未发现阻塞项。pre-commit 通过后提交工具为 `7f20e48b`，正式测量期间保持源码不变。

## 正式结果

baseline：`8fa80ba7ce3fd7ce2483f2ec098e66d410bc434e`；新版实际运行：`7f20e48b6ada37c995d11b82a43bae45ce835ab8`（新增工具/测试/计划，solver 同 `22209715`）。共 6 个独立进程，24 个测量窗口及各自的完整预热窗口；全部通过原物理门槛，窗口内没有硬失败。六次运行的源码前后不变、导入 worktree 正确，同一 harness、快照/输入 manifest、控制程序、物理 digest、配置、模型规模、GPU UUID/driver、Python/NumPy/Warp 身份均通过核对。

同一 RTX 5070 Ti Laptop GPU、driver 570.211.01、Warp 1.17.0。两版每轮逐步的 publication、Newton、PCG、scheduled slots、trial、regularization retry 完全一致；每版三轮工作量也相同。因此本次变慢没有伴随求解次数增加；其他执行成本的占比仍须进一步归因。

| 窗口 | 步数 | baseline 总时间中位数 s | 新版总时间中位数 s | 总时间变化 | baseline step median/P95 ms | 新版 step median/P95 ms |
| --- | --- | --- | --- | --- | --- | --- |
| sharpa-close-fp32 | 270 | 12.913 | 14.835 | +14.89% | 27.455/233.313 | 28.281/307.660 |
| sharpa-lift-fp32 | 200 | 8.344 | 8.837 | +5.91% | 37.150/48.800 | 38.951/50.242 |
| franka-grasp | 500 | 15.439 | 16.087 | +4.20% | 33.655/36.367 | 35.474/38.260 |
| franka-release | 300 | 11.657 | 12.271 | +5.26% | 36.327/53.164 | 38.451/55.194 |

总时间先在每轮内对窗口所有 step 的 solver 时间求和，再取三轮中位数；逐步 median/P95 合并三轮测量步。正号表示更慢。下面每个机器人的总时间先在每轮内相加两个窗口，再取三轮中位数，因此不必等于上表两行中位数相加。

| 机器人选定窗口 | 每轮步数 | baseline 总时间中位数 s | 新版总时间中位数 s | 变化 | 三组成对耗时比（新/旧） |
| --- | --- | --- | --- | --- | --- |
| franka | 800 | 27.096 | 28.476 | +5.10% | 1.0624, 1.0523, 1.0384 |
| sharpa | 470 | 21.257 | 23.672 | +11.36% | 1.1577, 1.1088, 1.1206 |

两场景的软体均为 343 个粒子、1296 个 tet，模型均有 55 个 shape。Franka 模型为 9 bodies、9 个 joint DoF；Sharpa 为 34 bodies、23 个 joint DoF。

| 窗口 | 每轮 Converged / Stopped | Newton / PCG / trial / retry（每版每轮相同） | 测量期间 capture：旧→新（每轮相同） |
| --- | --- | --- | --- |
| sharpa-close-fp32 | 251 / 19 | 476 / 12570 / 2488 / 188 | 0→2055 |
| sharpa-lift-fp32 | 197 / 3 | 443 / 5430 / 682 / 20 | 0→233 |
| franka-grasp | 500 / 0 | 766 / 12750 / 766 / 0 | 0→0 |
| franka-release | 300 / 0 | 673 / 10470 / 678 / 0 | 0→1 |

Sharpa 的 19/3 个 Stopped 分别在冻结门槛 29/6 以内，两版一致；没有把它们记成 Converged。捕获计数使用既有 `profiling.capture_count`（solver segment、contact、BSR 等缓存）；不另行统计 PCG 内部图，所有重捕获的实际执行时间仍在 step 计时内。

## 新增机制证据与处置

新版 Sharpa 的反复捕获集中在 `problem_candidate`。首轮 close 的最终 segment 统计（预热＋测量）为 captures=4113、evictions=4107、resident=6；lift 为 captures=470、evictions=464、resident=6。测量段的覆盖缓存捕获增量分别为 2055 和 233，三轮完全相同；baseline 均为 0。原始记录见 `capture-analysis.json`。

[problem.py](../newton/_src/solvers/monolithic/problem.py) 的 candidate 将 `(float(alpha), dt)` 放入图参数键，[segment_graph.py](../newton/_src/solvers/monolithic/segment_graph.py) 同时按实际 buffers/stream 绑定，并只保留每段 6 个 key。线搜索步长档位与状态绑定组合导致驻留集合反复淘汰，是本次已定位到的具体问题；尚未把全部耗时差异归因于这一项。Franka 仅出现 0/1 次测量期 capture，其约 5% 的退化仍需单独归因。

归入 **P9a-PERF-01 / PR-06 的 L2/L3**：除 dt 外，应核对 candidate 的 alpha 数值参数与绑定复用，避免用无限扩容缓存掩盖问题。可提炼为组件机制测试：固定小型 q+x 的状态/方向，执行多档 alpha 及 accepted/trial 角色切换，复现超过 6 个 key 的捕获/淘汰，并用独立 candidate 几何检查结果；该提炼尚未实施。现有 Sharpa close/lift quick 窗口已经覆盖实际触发场景，无须新增或改写冻结快照/门槛。

本任务仅增加测试与对照证据，未顺带修改 solver。H1 的接口/正确性结论保持，整体性能资格仍 OPEN；四窗口结果不代表完整 Franka/Sharpa 轨迹的性能资格。

## 结果与复现

- `series.py` / `series.json`：三组成对顺序、完整 argv/cwd、source HEAD、harness hash、进程退出码与 wall time。
- `pair-{1,2,3}-{base,current}/receipt.json`：原始逐步数据、门槛、环境/来源/输入身份和 graph 统计。
- `summarize.py` / `comparison.json` / `comparison.md`：身份核对、逐步工作量比较及统计。
- `components.log`、`precommit-tool-final.log`、`precommit-final.log`：8 项组件检查、工具提交前与最终文档的 pre-commit 检查。
- 首次外部 runpy 启动器多传了脚本路径，在 argparse 阶段退出，未执行仿真；保留在 `invalid-launch/`，未计入测量。修正 argv 转发并核对帮助入口后重新执行全部六次。

既有 FP64 quick 结果未重跑；本轮 solver/冻结 quick 路径未修改，受验范围是用户选定的 FP32 性能补充。没有运行完整机器人轨迹、推送或合并。
