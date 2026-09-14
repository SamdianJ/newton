# PR8E/F 预条件器研究

本工具不改变 solver 默认值，不接入公共预条件器选项。候选为 actor_block、local_block_4/8、mas_16/32；研究范围是完整已尺度化、含 lambda 的 A_xx，q 侧保持 dense，完整 q-x 在全局算子中。

`uv run --no-sync -m scripts.monolithic_reference.preconditioner_study --snapshot <export.npz> --output <新目录>` 可重复指定快照；省略快照运行具有几何、非均匀尺度与 q-x 耦合的合成 SPD 用例。`--linear-absolute-tolerance 0` 仅用于独立压力对照。输入使用现有 NPZ/JSON 哈希认证，结果保留向量、逐轮误差、独立真实残差与异常，不改变验收阈值。

8E 使用 FP64 研究 PCG，每轮检查独立真实残差；这不是原生 PCG 策略或 GPU 性能结果。局部块按确定性邻接遍历；MAS 根据参考位置 Morton 排序，每域 16/32 节点，粗聚合按域内拓扑连通分量拆分，最多三层，不能继续缩减时停止。层次固定；粗空间 R 使用 inverse scaling，把物理常量位移转换到 scaled 坐标。粗层块为 R Ahat_xx R^T，lambda 由投影进入，不再额外添加。

每个块用独立 FP64 Cholesky 求解。细层覆盖全部动态未知量，块 SPD 使相加后的 inverse apply SPD；有限精度仍须验证，不允许 clipping 或静默回退。研究结论须分开报告数值正确性、CPU 成本和 GPU 未测状态。

先完成先导测量，再提交评审冻结时间/尾延迟/显存预算。正式性能验证及 8G 在该节点之后；无候选通过时不改变默认。新版本记录见 fixtures/p2_preconditioner_study_v2.json，历史 v1 不改写。整套 nightly 由用户手动执行。
