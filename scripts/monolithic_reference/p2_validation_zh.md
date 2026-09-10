# P2 开发验证与夜间验证

日常开发使用小规模回归；完整轨迹与五次性能重复由用户夜间运行。
轻量通过表示可以继续开发，不表示性能或 P2 最终出口已经通过。

## 日常开发

在 integration 工作树执行：

```bash
uv run --no-sync -m scripts.monolithic_reference.validate_p2
```

默认 `--suite dev`，输出到新的 `/tmp/monolithic-p2-dev-<时间>/`。
允许验证未提交修改，要求 CUDA 可用；CPU 和 CUDA 用例在同一个测试进程中执行。
复用已有小 fixture，不运行完整 Sharpa/tet 轨迹，不做五次性能重复。

| 范围 | 保留的检查 |
|---|---|
| 质量矩阵 | reference/owned 的 M、惯量/能量、R/K、姿态重放、分配与共享生命周期 |
| 线性系统/PCG | dense 对照、真实残差、warm start、正性/曲率、NaN、stagnation、replacement、generation/lambda |
| 两项执行开关 | 四种组合、必需结果、production 无可选统计、q-only、回滚、in-place、发布和接触 history |
| 测量工具 | 参数传递、输入绑定，以及验证脚本的范围、恢复与 PCG 预算反例 |

目标为预热缓存下几分钟以内；冷 JIT 首次运行可能更长，不把运行时间作为数值门槛。
2026-09-10 在当前机器实跑92项CPU/CUDA用例通过：测试69.953s，作业墙钟72.7s，包含部分首次JIT。
修改期间先运行受影响模块，例如：

```bash
uv run --no-sync -m unittest newton.tests.test_monolithic_pcg_modes
```

提交前执行轻量入口及仓库规定的完整 `uvx pre-commit run -a`。
仅文档等低影响修改按变更范围检查，不自动触发仿真。
后续 P2 功能应把对应的小规模反例加入 `validate_p2.py` 的 `DEV_MODULES`；长轨迹留在 night。

## 夜间完整回归与性能数据

先提交待验证代码，确保 candidate 与 reference 工作树干净。reference 应是评审确认的比较起点；
当前 PR-8B/8C 使用 `../monolithic-p28-baseline`，提交为 `ef6c4577`。
所有作业记录实际源码提交、依赖、命令及输出；运行期间保持这两个工作树不变。

只查看计划，不运行 GPU：

```bash
uv run --no-sync -m scripts.monolithic_reference.validate_p2 --suite night --list
```

夜间启动（请为每份新源码选择新的输出目录）：

```bash
nohup uv run --no-sync -m scripts.monolithic_reference.validate_p2 \
  --suite night --baseline-repo ../monolithic-p28-baseline \
  --output ../../agents/integration/artifacts/p2/night-pr8bc \
  > /tmp/p2-night.log 2>&1 &
```

night 串行运行全部 `test_*monolithic*.py` 回归，然后运行以下 50 条完整 CUDA 轨迹。
它覆盖当前 P2/8B/8C 的回归和测量矩阵；不是整个 Newton 项目的所有测试，也不替代 8K 的正式 G6/G7。

| 负载 | 变体 | 重复数 | 轨迹数 |
|---|---|---:|---:|
| Sharpa r3 | 旧 baseline、新 diagnostic、owned、production | 各 5 | 20 |
| Sharpa r2 | 新 diagnostic、owned | 各 5 | 10 |
| tet r5 | 旧 baseline、production | 各 5 | 10 |
| tet r3/r4/r6 | 旧 baseline、production | 各 1 | 6 |
| 补充控制 | tet r5 diagnostic、r2 production、r3/r2 combined | 各 1 | 4 |

完整时长仍为 Sharpa 4.5s、tet 36s；N/h、100 步独立预热及实际逐次 PCG 记录沿用原协议。
8B 比较 owned 与新 diagnostic；8C 比较 production 与旧 baseline；combined 单次只验证组合。
CPU 只做正确性回归，不新增 CPU 性能测试。不要同时运行其他 GPU 测试；脚本不修改电源设置。
规模、模式或收敛机制变化时再更新此矩阵，不能用单次冒烟代替最终五次重复。

## 进度、恢复与判定

查看 `/tmp/p2-night.log` 或输出目录的 `progress.json`；每个作业有独立日志，轨迹在 `runs/`。
完成后阅读 `summary.md`，其中逐作业列出耗时、PCG p95 和数值筛查结果。
`plan.json` 记录冻结的命令与来源。终端中运行时可用 Ctrl-C 停止；中断现场会保留。

用相同源码、依赖和参数恢复：

```bash
uv run --no-sync -m scripts.monolithic_reference.validate_p2 --suite night \
  --baseline-repo ../monolithic-p28-baseline \
  --output ../../agents/integration/artifacts/p2/night-pr8bc --resume
```

恢复前校验已完成日志与逐文件证据哈希，成功作业跳过。
未完成/失败目录移入 `attempts/` 后再试，旧日志保留；来源或依赖变化会拒绝混用，要求新的输出目录。
回归或作业执行失败即停止队列；已有 PCG 预算失败作为结果保留，不用重跑筛掉。

| 退出码 | 含义 |
|---|---|
| 0 | 所有作业完成，已实现的自动筛查未发现失败；不自动授予性能准入 |
| 1 | 前置条件、测试或作业执行失败，查看日志后处理 |
| 2 | 全部作业完成，但正常收敛/残差/有限性/detF 或 PCG64/200 等自动筛查仍有未通过项 |

命令行参数错误也遵循 argparse 的标准退出码2；应结合 `progress.json` 的状态区分，不能只凭退出码认定完成。

报告按实际 PCG 调用计算 p95，包含 retry；单步累计迭代数另有原始记录。
tet 当前已知 p95 超 64，因此夜间出现 `COMPLETED_WITH_OPEN_GATES`/退出码 2 不等于脚本崩溃。
尾延迟、内存、相同质量下的重复收益和实际频率仍需评审；8A 预算草案不因此自动冻结。
固定节点完整轨迹对照、时间步误差、near-static、专项 profile 及 G6/G7 保留各自验收要求。

已保留的 PR-8B/8C 开发期证据不因新脚本而作废，也不自动导入成新版本夜间结果。
其部分完整轨迹和设备频率变化记录见 `agents/integration/artifacts/p2/pr8b-8c/`。
