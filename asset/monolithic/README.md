# Monolithic 保留资产与参考轨迹

2026-09-30 按用户要求清理开发产物。这里保存 **Franka 九条＋Sharpa 一条，共十条完整 FP64 参考轨迹**，不是只保留默认配置。大文件保存在本机 `data/`，不进入 Git；克隆代码不会自动获得这些资产。

- `data/reference/`：Franka 的三种 globalization × 三种 friction law，均为 production/graph、5250 步；Sharpa 为 OR＋C-IE＋production/graph、8667 步。
- `data/inputs/`：机器人网格/URDF、官方 SDF 缓存、smooth SDF 节点与资格证书、冻结控制、Close 场景参数、输入清单及比较容差。
- [references.json](references.json)：每条轨迹的来源、原回执身份、物理 digest、当前相对位置及录制格式。
- [files.sha256](files.sha256)：1086个保留文件的SHA-256，可从本目录用 `sha256sum -c files.sha256` 核对。
- [relocations.json](relocations.json)：旧位置到本目录的迁移表。原路径保留相对符号链接；文件搬迁没有重算物理身份或改写历史清单。

Franka 保持原 P3 recording 格式，回放入口使用各 case 的 `recording/` 目录。Sharpa 保持原 P5 native state chunks 格式；没有假装转换成 P3 回放格式。状态块、初末状态、模型及原始回执未修改。每个 `steps.jsonl.gz` 仅保留原有标量统计，去掉嵌套迭代细节；相邻 `log-compaction.json` 记录原/新哈希。这些精简日志不能代替原始独立数值审计数据。

两个 Sharpa geometry 目录代表原资产与重建资产的不同资格代次。旧场景仍引用原代次，因此两个都保留；不得混合证书、替换节点或静默重绑定 digest。正式 runner 的历史资产绑定问题 R1 仍未关闭。重建资产使用 `data/inputs/frozen-inputs-rebuilt.json`，不应误用原清单。

开发历史及证据压缩规则见 [development_history.md](../../scripts/monolithic_reference/development_history.md)。完整资产和轨迹文件的 SHA-256 清单在 workspace 的 `agents/integration/artifacts/history/retained-files.csv.gz`。迁移及清理只做文件完整性核对，没有启动测试、仿真或重新授予验收结论。
