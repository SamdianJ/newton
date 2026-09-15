# Tet cube Close debug checkpoint — 2026-09-15

本分支 `debug/cube-close-residual-20260915` 保存当前诊断代码、实验候选与失败证据，基于 `SamiulJ/monolithic-integration` 的 `60f8cd82f5ec6fab5027588c23ee968569f68fa8`。**完整 Close 尚未修复；本分支不是可合并的修复结论，也没有通过抓取或性能验收。**

## 场景与当前结论

- 40 mm、全动态 cube r2：343 节点、1296 tet。手从侧面推压，物体可能沿支撑面滑走；这个旧姿态只用于定位 Close，等待新的上方抓取轨迹。
- 只执行 Prepare/Close，最多 2.5 s / 2500 步；不推进 Hold/Lift/Release。
- 主要实验使用 production graph、actor_block、dt=1 ms、normal stiffness=1e7、mu=0.5；Newton 更新预算 10，原最终 global/q/x 收敛标准不变。
- 接触连续性候选为私有、默认关闭的 cubic SDF 和 smooth history 开关。`combined` 研究入口显式打开两者；仍需重新校准，不能称为已验证生产方案。
- SPD/GN 装配不包含完整残差 Jacobian，实际 trial 残差可能沿 PCG 方向上升。法向、摩擦贡献随失败状态变化；不能统一归为摩擦或 PCG 精度不足。
- 一次放行上升步、完整 FD Jacobian 修正、提前介入及阻尼都未完成失败物理步。按 global/q/x 目标加权的组合候选使 q 达标，但 global/x 仍超标。

最新加权候选的**最终残差 / 原验收目标**如下，三项必须都 ≤1：

|位置 / 物理步|global|q|x|结果|
|---|---:|---:|---:|---|
|18 / 1293|1.592|0.564|1.490|10 次更新后未收敛，回滚|
|24 / 1395|1.834|0.092|1.717|10 次更新后未收敛，回滚|

研究入口拒绝把干预后的 `NONLINEAR_MAX_ITERATIONS` soft-stop 当作成功输出。检查了回滚状态及 committed friction history；同一运行中的重放分支使用相同的物理输入和控制。独立运行的失败步数可能不同，不跨运行拼接状态或推断性能收益。

## 代码与报告导航

|内容|入口|
|---|---|
|接触连续性候选与冻结输入测试|[contact_continuity_zh.md](scripts/monolithic_reference/contact_continuity_zh.md)|
|法向/摩擦分解、完整 FD Jacobian oracle|[close_direction_zh.md](scripts/monolithic_reference/close_direction_zh.md)|
|平衡误差及物理量级|[close_balance_zh.md](scripts/monolithic_reference/close_balance_zh.md)|
|全部残差项、能量与导数完整性|[close_terms_zh.md](scripts/monolithic_reference/close_terms_zh.md)|
|步长验收与 SPD/GN 匹配、单次放行反事实|[close_merit_zh.md](scripts/monolithic_reference/close_merit_zh.md)|
|最新修复尝试、失败与后续方向|[close_repair_zh.md](scripts/monolithic_reference/close_repair_zh.md)|
|保存到 fork 的结果与输入|[close_debug_20260915](scripts/monolithic_reference/fixtures/close_debug_20260915)|

保留历史报告中的当时结论；最新修复状态以 v5 为准。完整 FD/dense 计算只在研究入口中使用，没有替换默认 production PCG，也没有新增运行时 SciPy 依赖。

## 复现

从本分支仓库根目录执行。`task_root` 指向已有的项目工作区；手 URDF/mesh、contact SDF、CSV 轨迹与绑定校准文件位于仓库外，需要已有资产环境。cube fixture 已随源码保留。源资产及控制文件的哈希可在证据包各运行的 `manifest.json` 查看。研究环境需要 NumPy、Warp、SciPy 和 threadpoolctl；依赖版本以原始运行依赖快照为准，不为复现更新项目锁文件。

```bash
task_root=/home/lightwheel/Desktop/newton/SamiulJ
uv run --no-sync -m scripts.monolithic_reference.run_close_repair \
  --asset-dir "$task_root/assets/left_sharpa_wave" \
  --contact-dir "$task_root/agents/integration/artifacts/p1/pr6d-assets/hand" \
  --trajectory "$task_root/assets/trajectory/Ball_catch - Left_hand_motion_rad.csv" \
  --calibration "$task_root/agents/integration/artifacts/p2/pr8hij-resume-20260914/calibration/calibration.json" \
  --position 24 --goal-weighted \
  --output ./outputs/close-debug-24-weighted
```

输出目录必须不存在。位置 18 使用 `--position 18` 和另一个输出目录。不带 `--goal-weighted` 会比较原重试失败后介入、步初介入和步初阻尼三个分支。研究失败返回非零，这是应保留的诊断结果；不能因此改阈值或把失败轨迹计作完整性能样本。

查看原 Close 场景的 GL 入口为 `scripts.monolithic_reference.view_contact_continuity`；参数与 cube example 共用，可运行 `uv run --no-sync -m scripts.monolithic_reference.view_contact_continuity --help`。它在失败或 Close 结束后停止步进，保留窗口供检查；不会自动启用 FD 修复。

## 证据保存范围与后续工作

证据包包含主要运行的 `repair.json`、物理步输入 NPZ、实际参数、资产 manifest 和原始证据索引；`bundle.json` 为随分支保存的文件逐项记录来源及 SHA256。原始 `evidence-index.json` 同时列出了未放入 Git 的大矩阵、日志和源码快照，不表示这些文件也在 fork 中。

完整原始目录仍保留在工作区 `agents/integration/artifacts/p2/close-repair-20260915-*`，包含 FD 矩阵、修正向量、完整 trace、源码与依赖快照。随分支保存的输入 NPZ 不是独立 solver 序列化格式，重放仍需要上述资产、时间步及 history 生命周期设置。

下一步定位加权后 q 已达标、x 仍停滞的状态：核验 FD 尺度与 FP32 可表示精度，再区分非线性曲率、接触分支及搜索目标折中。当前结果不证明平衡解不存在，也不支持把未通过候选加入默认 solver。

未执行 nightly；此前修复研究专项与源码保护测试共 7 项通过。此次 checkpoint 的完整定向检查结果记录在 [checkpoint_validation.json](scripts/monolithic_reference/fixtures/close_debug_20260915/checkpoint_validation.json)。
