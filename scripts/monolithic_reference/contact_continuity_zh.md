# Cube Close 接触连续性研究 v2

2026-09-15。基线 HEAD：`60f8cd82`；每次实验的实际源码、未提交 diff 和依赖均保存于 `source.zip/source.json`。

**状态：局部连续性修复已实现，完整 Close 尚未通过，不切换默认策略。**

本记录承接 `agents/integration/artifacts/p2/cube-close-fresh-20260914/SUPERDEX_REPAIR_PLAN_v1_zh.md`，保留 v1 和原始失败输入。不是 8H fixture 冻结、8I 关闭或 G6/G7 通过的证据。

## 实现与适用范围

- 原三线性 SDF 和 45° history 规则仍是默认。研究入口用私有 `_cubic_sdf`、`_smooth_history` 选择候选，没有新增公共 solver 选项或材料/USD 属性。
- C1 候选使用共享的规则网格节点与 Catmull–Rom 标量插值，距离和梯度由同一个函数计算。粗细网格节点按已有 canonical voxel 读取规则导出。
- 首轮原始 SDF 的 11 个独立场中，4 个不能通过三次插值有效域覆盖检测范围的保守认证。运行时原型向外扩展 **16 格**，外部节点取旧标量 SDF 的边界延拓，再统一做 C1 重建；没有在有效接触域回退旧法线。
- 构建时验证有限值，并认证最外层有效 cell 的距离下界大于完整检测范围（包括粒子半径、shape margin 和 gap）。域外返回“不生成接触”的内部哨兵，不把它当作物理距离/法线使用。法向力有效区域的连续性是本实验的要求，未声称域外哨兵是全空间 C1 SDF。
- 三次重建不保证 1-Lipschitz，因此关闭该候选 mesh 的原 centroid-distance 剔除。AABB 使用整个扩展场的包围盒；测试比较启用/关闭 AABB 的接触集合、法线与位置。
- 数据在构造时预分配，运行中不更新。更换字段数组、底层存储或实验配置会拒绝旧绑定；正常数值组装与 PCG graph 路径继续复用现有缓冲区。
- history 候选在 **40°—45°** 区间平滑衰减，45°及以上清零。40°是本轮研究配置，尚未通过静态承载/滑移标定成为默认值。新阈值加入 history 配置身份和失效检查。
- history 每个 trial 都从同一份 committed 数据计算，不在拒步时累计衰减；原 Coulomb 上限、return mapping、最终提交与 rollback 保持。`0≤w≤1` 与单位法线投影使转运不增加旧弹簧的二次能量，损失计入原 transport energy 分账；这不是整个积分器的能量证明。

保持 actor_block、h=1 ms、原材料/质量/刚度、rtol/atol、64/200 预算、global/q/x 目标、Newton/line-search/retry 与事务验收。研究候选改变接触离散或摩擦记忆，不属于纯性能优化。

### 边界下界

一维 Catmull–Rom 相对相同节点线性插值的偏差不超过 `max|Δ²v|/8`；其插值权重绝对值和不超过 `1.25`。逐轴展开得到三维保守界：

`E = (Mx + 1.25 My + 1.25² Mz)/8`。

以最外层有效 cell 的角点最小值减去 E，作为该层距离下界。运行时原型使用全场二阶差分最大值；研究脚本也提供更紧的局部 stencil 界。该界只针对共享 canonical 节点的三次/三线性插值差异，**不是相对真实三角网格距离的误差证明**。

## 证据

证据目录位于工作区的 `agents/integration/artifacts/p2/`。

- `contact-continuity-20260915-final-geometry/`：最终导出/查询函数的全资产抽样与域检查。
- `contact-continuity-20260915-final-runs/`：相同最终数值源码的独立进程轨迹，含命令、stdout、GPU 记录、完整失败统计、Newton 事件和终止状态探针。
- `contact-continuity-20260915-terminal24-v2/`：首次成功的双侧方向导数探针。
- 早期 `local/domain/boundary/assets/winding/ablation` 目录保留，不能混为同一份冻结运行。`assets` 的法线符号 oracle 与资产 winding 规则不一致，由 `winding` 及最终目录更正；未把这些符号差异判为 solver 缺陷。
- 第一版 terminal24 探针错误地传入负 alpha，被原 trial 接口拒绝；v2 改为相反方向、正 alpha。接口与求解器的 alpha 规则没有放宽，失败记录保留。

局部固定输入：

|项目|原实现|连续候选|
|---|---:|---:|
|位置24约10 nm查询变化的法线转角|27.1477°|约0.00182°|
|局部729点相对三角网格的最大距离误差|0.0883 mm|0.0595 mm|
|候选局部零等值面至三角网格的最大偏移|—|0.0526 mm|
|位置18的45°转运跳变|存在|固定输入回归通过|

11 个 SDF 的确定性近表面抽样共 45,056 点，使用与资产一致的 winding 符号 oracle；具体误差、符号与域结果以最终 `results.json` 为准。抽样验证沿用资产已有 0.4 mm 误差限，不把它扩充成新的抓取精度验收。

运行时原型额外常驻规则网格约 **103.6 MiB**。目前有追踪开销，且部分早期实验与其他 GPU 检查重叠；所有轨迹明确标记 `performance_sample=false`，不能用于提速结论。

最终数值源码的完整轨迹尝试（六个独立进程，全部在 Close 停止）：

|位置|候选|模式|已接受步数|失败步|原因|
|---|---|---|---:|---:|---|
|24|基线|production graph|1090|1091|stagnation|
|24|仅 C1 SDF|production graph|1394|1395|regularization exhausted|
|24|仅平滑 history|production graph|1090|1091|stagnation|
|24|合并|production graph|1435|1436|regularization exhausted|
|18|合并|production graph|1292|1293|regularization exhausted|
|24|合并|diagnostic|1394|1395|regularization exhausted|

各运行的四个核心文件 hash 与最终工作树一致；完整命令、源码和失败返回码保留。历史运行曾在其他步数失败，不把不同版本/运行的失败点拼接成同一条轨迹。推进步数增加不等于完成抓取。

### 新的拒步证据

最终合并候选的独立终止状态探针中：

- alpha=0 的 current/trial 尺度化残差差异约 `3.6e-11`。
- 对 `F=||S r||²/(2N)`，位置24 production 在 `h=1/8` 时的装配预测为 `-1.026e-10`，独立差分为 `+2.479e-11`。
- 位置18 production 的相应结果为 `-1.233e-11` / `+3.761e-12`；位置24 diagnostic 为 `-2.008e-11` / `+6.030e-12`。
- 三个探针在七档双侧差分下均为装配预测下降、真实 merit 上升；接触项为正、非接触项抵消一部分。最小扰动的差分波动增加，未把它当作高精度 Jacobian 标定。
- 探针同时保存实际 FP32 位移，并用实际位移计算装配预测，避免把不可表示的小位移误认为有效下降。

这支持当前近似切线给出了真实残差 merit 的上升方向。尚未把差异完整分摊到法线变化、摩擦载荷变化、history 权重和其他接触几何导数，不能声称完整 Jacobian 根因分析已经结束。

## 验证与下一道关卡

回归覆盖独立 FP64 Hermite oracle、标量梯度差分、三轴 cell 连续性、非法 stencil/数值/梯度、真实失败快照、转运能量与客观性、current/trial 一致性、字段绑定、配置失效、AABB、active history rollback、最终发布及 State-copy 失败，以及现有 graph/step/cube 回归。结果摘要与完整日志保存在证据根目录的收尾报告。

当前 **不进入三档抓取冻结、摩擦负对照正式验收或性能准入**：Close 仍失败，新接触规则的压板/承载校准仍待重新验证。旧校准在研究入口仅用于保持原物理配置，候选元数据明确标为 `REVALIDATION_REQUIRED`。

下一步应在这些冻结终止状态上拆分真实接触方向导数与现有 PSD/GN 近似的差异，提出与 SPD/PCG 和当前残差 merit 兼容的接触线性化/下降方向方案，然后复验；不能直接把可能非对称或不定的完整 Jacobian 放进 PCG。预算和残差门槛保持原值。

## 复现

在 `worktrees/monolithic-integration` 下运行局部回归（无需外部手资产）：

```bash
uv run --no-sync -m unittest newton.tests.test_monolithic_contact_continuity -v
```

当前仅诊断 Prepare/Close，最多运行到 2.5 秒（2500 步），不推进 Hold/Lift。原九秒抓取计划暂停，等待用户重新录制从上方接近的手部轨迹；当前侧推场景不作为抓取验收 fixture。

研究命令使用已存在的资产路径，输出目录必须是新目录：

```bash
task_root=/home/lightwheel/Desktop/newton/SamiulJ
uv run --no-sync -m scripts.monolithic_reference.run_contact_continuity \
  --asset-dir "$task_root/assets/left_sharpa_wave" \
  --contact-dir "$task_root/agents/integration/artifacts/p1/pr6d-assets/hand" \
  --trajectory "$task_root/assets/trajectory/Ball_catch - Left_hand_motion_rad.csv" \
  --calibration "$task_root/agents/integration/artifacts/p2/pr8hij-resume-20260914/calibration/calibration.json" \
  --variant combined --position 24 --pcg-mode production --steps 2500 --probe-failure \
  --output "$task_root/agents/integration/artifacts/p2/contact-continuity-manual-$(date +%Y%m%d-%H%M%S)"
```

`--variant` 支持 `baseline/sdf/history/combined`；失败保存证据并返回非零。批量结果保留每个退出码，失败不计为完整轨迹。没有自动运行 nightly 或推送。

追加 `--probe-decomposition` 可执行终止状态的法向/切向残差分解和两档步长的完整有限差分 Jacobian oracle（研究环境需要 SciPy 和 threadpoolctl）。这些反事实试探不接受物理步，不改变 committed history，也不替换生产 PCG。结果解释见 [Close 方向诊断 v1](close_direction_zh.md)。

`view_contact_continuity` 的 GL 入口同样在 Close 末尾或失败回滚后停止推进，保留窗口供检查，关闭窗口时保存最后状态。

追加 `--probe-balance`（要求 `--probe-failure`）可审查惯性/材料/接触/驱动分账、分项能量及实际增量；与 `--probe-decomposition` 合用时还记录完整 FD 局部修正的物理量级。最新结论及修复优先级见 [平衡与验收审查 v2](close_balance_zh.md)，不能把分项能量直接当作全系统势能。
