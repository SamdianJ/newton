# PCG 综合基准

`benchmark_pcg.py` 用同一份 K、S、RHS、λ 和 actor 预条件器比较当前原生 diagnostic、production；CUDA 上额外比较 production graph。每种模式绑定后的矩阵、RHS、尺度及 Cholesky 因子做相同哈希检查。基准不修改求解器，不复制旧 PCG 作为替代实现。

## 直接运行

在 monolithic-integration 仓库根目录执行：

```bash
env -u VIRTUAL_ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync -m scripts.monolithic_reference.benchmark_pcg \
  --device cuda:0 --repeats 10 --warmups 3 \
  --output "$HOME/Desktop/newton/SamiulJ/pcg-benchmark-$(date +%Y%m%d-%H%M%S)"
```

默认执行六个固定种子的 SPD 系统：良态、耦合及尺度变化、病态、q-only、x-only、零 RHS。病态用例允许达到预算或停滞；它们用于暴露限制，不能把程序执行完成当作全部求解收敛。SciPy 用于离线参考解，Matplotlib 用于输出图，不新增 Newton 运行时依赖。

使用 `--snapshot` 追加已有 `export_candidate()` 输出的 NPZ/同名 JSON，可多次传入。工作区现有快照例如：

```text
--snapshot ../../agents/integration/artifacts/p2/pr8a-measurement/diagnostic/sharpa-matrix-cuda0-r3/matrices/candidate-3.5.npz
--snapshot ../../agents/integration/artifacts/p2/pr8a-measurement/diagnostic/tet-matrix-cuda0-r5/matrices/loading.npz
```

历史快照只作为相同的固定线性输入；执行当前两种算法，不加载旧求解器，不恢复物理 history，不重跑完整轨迹。输入必须通过原始 fingerprint 校验。

## 收敛与真实解偏差

参考目标为 `Ahat = S K S + λ I`、`Ahat y = rhs_hat`，用**实际 float32 输入值**在 float64 中构造矩阵并由 SciPy SuperLU 直接求解，不对矩阵对称化或更换 RHS。参考解自身必须满足相对残差 <=1e-9、范数后向误差 <=1e-12；另报告 1-范数条件数估计与相对非对称性。条件数是估计值，不是严格误差上界；参考解是该线性系统的数值真值，不是非线性物理轨迹的真值。

记录以下独立指标：

- 原生退出状态、有效迭代、固定调度槽位、检查次数、restart 次数与失败位置。
- float64 复算 global/q/x 真实残差；分母沿用当前 solver 的 `max(||rhs||_block, floor_block)`，门槛仍为 1e-4。
- `||y-y_ref||/||y_ref||`、q/x 分块偏差、物理增量 `delta=S*y` 的相对及绝对偏差。
- 1-范数后向误差、SPD 能量范数相对偏差。零参考解且零误差记为 0，非有限量在 JSON 中记为 null。

“原生 SUCCESS”和“float64 独立残差通过”分别统计；float32 matvec/归约与 float64 目标矩阵有舍入差异，可能出现原生通过而独立检查略超门槛。保留这种差异，不改原容差，不把小残差等同于小解误差。

收敛曲线来自额外的一次原生非图求解：只在解更新后读出 y/r，不写 solver 缓冲、不改变检查频率或退出逻辑。曲线同时保存递推残差与相对 float64 真实残差的 gap。采样含同步，**绝不使用该运行做计时**。CUDA 浮点归约可导致重复运行轨迹和迭代数略有差异；不要求 Euclidean 残差每一步单调下降。graph 使用相同 production 数值政策，图模式的最终结果单独检测。

## 效率口径

每种模式每个范围默认预热后采样 10 次，按轮次轮换/反转模式顺序，报告原始样本及 p50/p95。当前代码在两个范围中使用同一矩阵值与 RHS，并始终从零初值开始：

| 范围 | 包含成本 | 用途 |
|---|---|---|
| setup / first_solve | workspace 建立、初始因子及第一次求解；可能包含 JIT；首次绑定和捕获另记录 | JSON 中保留初次成本 |
| reuse | 同 generation/地址上的整次 PCG、主机准备、结果读取、成功后的 delta 恢复 | 观察稳定复用收益 |
| new_generation | 更新 generation、重算 actor 因子、整次 PCG、当前策略要求的重新捕获和结果恢复 | 暴露当前集成的重建成本 |

`new_generation` **不是完整 Newton step**：它不重做 BSR 装配或物理求值，只隔离数值版本更新对线性求解成本的影响。必须同时查看它和 reuse，不能只报预热图重放时间。固定调度的尾部开销和条件图实际执行的批次完整计入；图捕获/延迟 instantiate/upload/主机提交沿用当前 `capture_seconds` 口径。

当前 `masked_smart_restart_v2` 在存储/稀疏结构不变时跨 generation 复用图，新 generation 的结构摘要验证计入准备成本。支持 CUDA 条件图时每个原有检查批次后可退出；`scheduled_slots` 记录实际调度批次中的槽数，其余执行方式仍记录完整预算。旧 v1 报告的“每代重捕获”和固定 200 槽是历史数据，不回写。

每轮前后读取 GPU P-state、SM 频率、温度、功耗及利用率，存入各 case JSON；采样在计时范围外，不修改设备功率/频率。未锁频时需要评审这些记录，短样本 p95 也不是整轨迹尾延迟。完整 solver/step 性能仍由独立的 nightly 测量。

## 输出和验证

输出包含 `REPORT.md` 表格、每个系统的收敛/解误差/计时 PNG、完整 JSON、原始输入 NPZ、源码/依赖快照和 `evidence-index.json`。`*-solutions.npz` 保存参考解、每次计时求解结果和非计时迭代向量；JSON 的 `solution_key` / `iterate_key` 对应其中的数组，可独立复算偏差。查看未收敛状态与独立检查失败，不能只看耗时。

轻量验证：

```bash
uv run --no-sync python -m unittest -v newton.tests.test_monolithic_pcg_benchmark
```

只验证参考解、分块误差、快照完整性、CPU/CUDA 原生分派、曲线不改变政策、稳定图复用及 generation 重建。综合基准是单独的显式命令，不由这些单测启动完整矩阵扫描或 nightly。

## 与 Style3D / Kamino 比较

追加 `--include-external`，在同一轮中测量七种模式：原有三种，以及 Style3D、Kamino 的非图/图模式。CPU 无图模式。真实矩阵仍由相同的 `--snapshot` 参数追加。

这是 **PCG 迭代实现的共享 A/M 比较**，不是完整 cloth / rigid-body solver 性能比较。所有模式使用原来的标量 `S K S + λI` 算子和同一 actor Cholesky 预条件器，逐份检查绑定哈希，没有改用更便宜的 Jacobi 或把稀疏系统变成稠密系统。

| 实现 | 适配边界 | 原生停止条件 |
|---|---|---|
| Style3D | 继承原生 `PcgSolver.solve`、更新及归约；仅将 `step2_update_z` / `step5_update_Ap` 接到共同 A/M。标量向量零拷贝映射到 vec3，末尾不足三维补零 | 固定 200 次调度，无原生收敛状态；报告为 `FIXED_BUDGET`，迭代数包含零 alpha 更新 |
| Kamino | 原生 `CGSolver`，通过 `BatchedLinearOperator` 回调绑定共同 A/M，单 active world | 全局递推残差；每次迭代判断一次。`rtol=0`，`atol=1e-4 * min(非空块残差分母)`，使全局绝对界足以覆盖所有块，但仍可能存在递推/真实残差偏差 |

Kamino 的全局条件可能比各块独立判定更保守；不声称停止策略相同。Style3D 和 Kamino 都不新增 Monolithic 的真实残差 restart、曲率/事务诊断。统一精度判断始终由原来的 FP64 global/q/x 检查完成；不能只按原生 SUCCESS 或固定循环完成判断通过。

两种外部图模式由基准包一层 outer capture/cache：Style3D 捕获完整固定循环，Kamino 调用其原生 `capture_while` 条件循环。固定地址、维度、lambda、参数时，更新数值 generation 不重捕获；当前 Monolithic graph 则执行自己的存储/结构和数值有效性检查。外部缓存只是本基准的固定绑定适配，没有模拟外部完整 solver 的拓扑变化处理。

计时包含外部最终 delta 恢复，Kamino 另包含原生结果计数/残差读取。首次编译、捕获及暖机保存在 first_solve。`software_counts` 来自额外非计时调用，统计 Python 层 launch、CUDA `.numpy()` 读取、graph launch 及 Style3D inner 调用；**不是实际 GPU kernel 数量**，不包含底层内部 CUDA 操作。完整迭代向量和独立误差仍保存到原有 NPZ/JSON。

适配验证：

```bash
uv run --no-sync python -m unittest -v newton.tests.test_monolithic_pcg_backends
```
