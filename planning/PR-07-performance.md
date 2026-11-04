# PR-07：性能留项确认

2026-11-04。状态：**诊断完成，未修改求解器**。用户要求先确认问题，再决定是否修复。结论：Sharpa close 的 stopped 步骤存在稳定的执行成本退化，值得针对重复绑定元数据做受控优化实验；q+x/trees 的历史退化在当前版本未稳定复现，不据此修改实现。整体性能资格仍 OPEN。

## 身份与测量范围

| 项目 | 身份 |
| --- | --- |
| 当前受验提交 | `d8da652b55ae9a67b75d8e84a1949dbb9e499595`，运行源码等同修复提交 `8775416f` |
| 小场景增量 baseline | `b5868dc4c5882d368e982569dec8ee37102a76a4`，PR-05 后 |
| 机器人 baseline | `8fa80ba7ce3fd7ce2483f2ec098e66d410bc434e`，P9a 起点 |
| 结果 | workspace `dev/monolithic/test/results/P9/P9a/PR-07/performance-confirm-20261104-01/` |

沿用原 PR-06 脚本、输入和冻结门槛。小场景保持全部 20 个组合与泛型注册顺序，每组合 4 次预热、7 次计时；先做两次 current 同版本重复，再做三组 AB/BA/AB。机器人使用原四个 FP32 冻结窗口，各做完整窗口预热、恢复同一快照后计时，三组 AB/BA/AB。未重跑完整轨迹或 suite。

全部 checkout 干净且前后身份不变；小场景输入、配置、工作量和诊断哈希一致；机器人逐步 publication/Newton/PCG/trial/retry 工作量一致，预热与计时物理门槛全部通过。所测窗口 graph captures 均为 0。503 个两秒间隔 GPU 遥测样本最多观察到一个计算进程；机器人期间 GPU 时钟中位数约 2745–2752 MHz、温度 62–74°C。未锁时钟或修改电源配置，未测量 CPU 频率，不能据此排除全部主机波动。

三组复测于 13:40–13:57 JST 完成；另对 Sharpa close 的 3430、3447、3469、3500 步做 baseline/current 各一次独立 cProfile。两条普通步骤、两条 stopped 步骤的工作量与未插桩运行完全一致。插桩结果不进入性能比值。

## 小场景：当前证据不支持修复

正数表示变慢。median 是每版本汇集 21 个样本后的值，三组成对变化单独保留，避免只看汇集中位数。

| 场景 | baseline → current median ms | 汇集变化 | 三组成对变化 | 判断 |
| --- | --- | --- | --- | --- |
| q+x FP64，无导出 | 35.196 → 33.987 | -3.43% | -6.78% / -2.12% / +4.91% | 原 +5.13% 未稳定复现 |
| trees FP64，无导出 | 61.358 → 61.255 | -0.17% | +0.29% / -2.44% / -1.41% | 原 +1.49% 未稳定复现 |
| trees FP32，完整 JSON | 38.304 → 39.370 | +2.78% | -7.19% / +3.27% / -1.13% | 方向混合，不认定稳定退化 |

同版本两次进程的 median 变化依次为 +0.83%、+1.15%、+0.02%；这只是局部重复性观察，不是完整噪声上界。第三项即使汇集 median 上升，两组成对结果仍更快，不能单独用 +2.78% 支持修复。历史 PR-06 证据保留；本轮受验版本也包含 PR-07 正确性修复，未证明历史差异全由噪声或某次修复造成。

## Sharpa：close 与 stopped 成本已复现

| FP32 窗口 | baseline → current median 总秒数 | 汇集变化 | 三组成对变化 |
| --- | --- | --- | --- |
| Sharpa close | 12.806 → 13.276 | +3.67% | +6.45% / +2.35% / +7.08% |
| Sharpa lift | 8.361 → 8.043 | -3.81% | -0.20% / -5.17% / -4.91% |
| Franka grasp | 15.441 → 14.823 | -4.01% | -2.48% / -4.51% / -3.46% |
| Franka release | 11.664 → 11.197 | -4.00% | -2.80% / -5.50% / -6.26% |

Sharpa 两窗口合计汇集变化 +0.72%，三组 +3.84% / -0.62% / +2.27%，未一致复现历史合计 +2.39%。close 的三组方向一致，step P95 为 235.510 → 275.007 ms（约 +16.77%，仅为描述统计）。Franka 合计 -4.00%，三组均改善。

将相同步骤按 publication 分组，进一步缩小问题：

| 分组 | 每窗步数 | 三组总耗时变化 |
| --- | --- | --- |
| close / CONVERGED | 251 | -1.75% / -5.42% / -1.14% |
| close / STOPPED_ACCEPTED | 19 | **+18.96% / +14.18% / +19.77%** |
| lift / CONVERGED | 197 | -2.02% / -6.81% / -6.59% |
| lift / STOPPED_ACCEPTED | 3 | **+19.48% / +12.54% / +13.61%** |

close 的 19 步每窗多耗时约 0.72–1.02 秒，普通收敛步骤节省部分成本。不是迭代数、trial 数或重捕获变多；相同求解工作量在大量 trial/retry 时执行更贵。这里只确认性能问题，不改变这些 stopped 步骤的物理资格分类。

## 调用分析：重复绑定与 graph key 为主要优化候选

第 3447 步两版本均为 Newton 9、PCG 460、trial 158、regularization retry 14。第 3469 步也保持相同工作量，出现同类热点。

| 第 3447 步调用 | baseline | current |
| --- | --- | --- |
| segment graph `run` | 1176 次 | 1609 次 |
| graph key 签名生成器调用 | 95,287 次 | 387,482 次 |
| 统一 problem `_buffers` | 无此统一函数 | 1146 次 |
| problem `_storage_layout` | 无此统一函数 | 124,914 次 |
| `wp.array.numpy` | 1370 次 | 1514 次 |
| Python `wp.launch` | 2248 次 | 1040 次 |

当前 `_buffers` 的插桩 inclusive 时间约 133 ms，graph `run` 约 221 ms（baseline 约 70 ms）；这些是有 cProfile 开销的嵌套时间，不能相加或用于推算可回收的实际毫秒数。签名生成器次数也不等于独立数组数量。直接设备 kernel 提交减少，同时统一 phase 的 graph 提交和缓冲签名工作增多，不能只看某一项调用数判断整步性能。

源码与热点吻合：`problem.py::_buffers` 每次构建全量状态/历史/owner 列表并验证布局；`segment_graph.py::run` 每次为所有 buffers 重建 pointer/shape/strides/dtype/device key。candidate、difference、domain、evaluate 在大量 trial 中反复执行这些工作。此为明确的优化候选，尚未通过受控消融证明它解释全部退化或预先保证收益。

## 处置建议

- **建议进入小范围修复实验**：Sharpa close 的高 trial/retry 成本。先减少重复绑定描述和 graph signature 构建，保留合法重绑失效、无效绑定拒绝、状态角色切换和完整物理检查；以相同三组窗口复测决定是否保留实现。
- **本轮不修**：q+x FP64、trees FP64、trees FP32 JSON，当前没有稳定复现证据；保留历史观察与监测。
- **L3 保留限制**：七档合成抖动仍是既有证据；本轮真实窗口零重捕获，没有理由把扩大缓存或改变 span 策略作为此处性能修复。
- 本轮未修改 solver、测试门槛、资产、基线或 CI；整体 PERF01 与 PR-07/P9a 状态仍 OPEN / IN_PROGRESS，不把局部诊断完成记为整体性能 PASS。

`validation-summary.json`、`comparison.json`、`publication-costs.json` 和 `profile-focus.json` 分别保存验收核对、成对结果、分组成本和热点。`files.sha256` 覆盖 56 个证据文件，其 SHA256 为 `fa57fd6c9b090fb1051e631d7c24d07b54c351c0016a8bc490902c712b2ee1c5`。原始日志与首次采用“合计三组全慢”条件而未执行的 profile 记录均保留；随后按稳定复现的 close 窗口单独执行两份 profile，没有混入正式计时。
