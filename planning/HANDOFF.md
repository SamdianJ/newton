# P9a 集成交接

2026-11-04。用户已授权将 P9a 合并到项目集成主干；阶段结论为 **ACCEPTED_WITH_OPEN_ITEM**。

## 合并身份

- 目标：`worktrees/Monolithic-refactor` / `SamiulJ/monolithic-refactor`。
- 来源：`worktrees/Monolithic-p9a` / `SamiulJ/monolithic-p9a`。
- 原主干：`8fa80ba7ce3fd7ce2483f2ec098e66d410bc434e`。
- 实现及验收记录快进至：`0a093f7e4a396a803698273c57c6c03c8162b250`，29 个提交，无分叉或冲突。
- 合并后的树与来源完全一致：`eeacb3737a023c6b5d47d18eaef294316b423d1c`。随后仅提交本交接及集成状态文档，再同步主干；最终指针见下述 verification 回执。
- 操作证据：workspace `dev/monolithic/test/results/P9/P9a/integration/20261104-01/merge.json`；最终检查为同目录 `verification.json`。未推送远程，阶段 branch/worktree 保留。

## 已交付与验收

统一 actor-owned problem/类型设备 phase，删除旧双路径；固定绑定 PCG 复用与失败恢复；收敛/后验摘要融合；材料批处理；Stats 快照；绑定注册及 dt/alpha 数值缓冲。PR-07 修复非连续能量、cable owner tangent 重绑及 suite 四类机制回归。

完整受验源码为 `51ceabe7`（运行源码等同 `8775416f`），到集成来源 HEAD 仅有 planning 文档变化。合并没有产生代码差异，复用原验收证据，不为快进合并重复运行完整 suite/E2E。

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 修复定向组件 | CPU 10 / CUDA 9 PASS | `PR-07/20261104-fixes-01/` |
| 两精度 quick | CLEAN，60 gate PASS | `PR-07/quick-8775416f/` |
| 完整 suite | CLEAN，1409 gate PASS、74 SKIPPED、0 重试 | `PR-07/local-20261104-01/suite/` |
| 完整两精度 stage | KNOWN_ONLY，47/47 作业；77 PASS、22 KNOWN_FAIL、1 UNEXPECTED_PASS | `PR-07/local-20261104-01/stage/` |
| 独立审查 | Fit: Fits；Requirements 无新增实质 finding；Standards PASS | [PR-07](PR-07.md) |
| 性能确认 | 三组成对及独立 profiling 完成，用户决定暂缓修复 | [性能报告](PR-07-performance.md) |

上表路径以 workspace `dev/monolithic/test/results/P9/P9a/` 为根。suite 25 个非强制精度诊断 FAIL 与 stage 2 个 grasp diagnostic FAIL 保留；KNOWN_ONLY 不是完整 stage PASS。

## 后续跟踪

- `P9a-PERF-01`：KNOWN / DEFERRED。Sharpa close 三组 +6.45% / +2.35% / +7.08%，19 个 stopped 步骤的相同工作量成本增加 14%～20%；重复绑定和 graph key 为候选原因，用户明确暂不修复。q+x/trees 历史退化未稳定复现，继续监测。
- L3：七档合成缓存抖动仍在；所测真实窗口无重捕获，保留现策略。
- OBS01：历史一次 checkpoint 逐位差异未复现；OBS02：防护已验证，Warp 依赖内部归因继续跟踪。
- 既有物理资格：Sharpa FP64 stopped、FP32 cable cantilever、FP32 tet cantilever-8 沿用已批准登记；FP32 tet vibration bias 保留跨 GPU 差异跟踪。具体处置见 [PR-07](PR-07.md#最终人工处置)。

## 环境与后续测量

继续复用共享 `.venv` 和资产，使用 `uv run --no-sync`，不对共享环境执行 `uv sync`。主干已前进，`Monolithic-refactor` 不再是 `8fa80ba7` 的原始性能 baseline；后续对照须从该固定 SHA 建独立 checkout，并核对实际 import/source 身份。旧证据仍绑定原提交，不因主干路径前进而改写。
