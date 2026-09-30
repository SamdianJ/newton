# Sharpa 早期控制输入

这两份小型 JSON 是早期用户控制轨迹，不是仿真回放结果，也不是当前十条回归的输入。原字节从 workspace 根目录 `trajectories/` 搬入此处，哈希在上级 `files.sha256`。

- `sharpa_cube_grasp.json`：原始抓取控制。
- `sharpa_cube_grasp_approach_5mm.json`：5 mm 接近版本。

原目录的 `sharpa_cube_grasp_approach_10mm.json` 与正式 `../data/inputs/sharpa/recorded-control.json` 逐字节相同，因此只保留后者。三个旧路径均可通过上级 `relocations.json` 定位；不得自动用早期版本替换冻结控制。
