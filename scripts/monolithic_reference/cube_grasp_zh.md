# PR8H：40 mm cube 抓取入口

**当前物理校准未通过，抓取扫描和 fixture 冻结尚未开展。** 三档资产与入口可用；不能用旧球校准或修改结果中的状态绕过门槛。版本化状态见 [cube v1](fixtures/p2_cube_grasp_v1.json)。历史球入口与 anchored-close 保留，cube 不用于重写旧性能结果。

资产为每边 4/6/8 格的 Freudenthal tet cube，125/343/729 节点，质量 0.064 kg，全部动态。默认使用 actor_block、owned articulation mass 和 production graph；`--pcg-mode diagnostic` 为独立数值对照。CPU 支持资产/双压板组件，真实手部 SDF 抓取需要 CUDA。

生成/核验资产及运行三档完整接触扫描：

```bash
uv run --no-sync -m scripts.monolithic_reference.calibrate_cube_contact --build-assets
uv run --no-sync -m scripts.monolithic_reference.calibrate_cube_contact --output "$PWD/output/cube-calibration"
```

输出目录须为新目录。24个刚度/网格点逐项保存失败后继续；无三档共同通过的刚度时返回非零，`calibration.json` 保持 `FAILED_NO_PHYSICAL_ENVELOPE`。物理是否通过与逐次 PCG p95 是否超过64分开记录。未完成轨迹的 force_error 仅为部分轨迹诊断，`force_error_is_full_hold=false`，不能当稳态结果。

只有获得真实完整的 `FROZEN` cube 校准后，才可运行交互 example 或位置扫描。以下路径变量由调用者设置为实际文件/目录，`CUBE_CALIBRATION` 必须指向通过校准的 JSON：

```bash
uv run --no-sync -m newton.examples monolithic_sharpa_soft_cube \
  --asset-dir "$SHARPA_ASSET_DIR" --contact-dir "$SHARPA_CONTACT_DIR" \
  --trajectory "$SHARPA_TRAJECTORY" --calibration "$CUBE_CALIBRATION" \
  --output "$PWD/output/cube-example"

uv run --no-sync -m scripts.monolithic_reference.run_cube_grasp --scan \
  --asset-dir "$SHARPA_ASSET_DIR" --contact-dir "$SHARPA_CONTACT_DIR" \
  --trajectory "$SHARPA_TRAJECTORY" --calibration "$CUBE_CALIBRATION" \
  --output "$PWD/output/cube-position-scan"
```

example 默认中档 `cube_r2.npz`，`--cube` 可选择独立三档资产；默认900帧/9秒。扫描固定中档、production graph、摩擦开启、27个预声明偏移；每个作业独立进程，失败保存并继续，2400秒单作业超时。单次轨迹使用相同研究入口去掉 `--scan`，可指定 `--position-offset-mm X Y Z` 和 `--friction-off`，用于固定候选的三档及负对照。

研究入口保存源码/依赖、校准哈希、实际构造选项、逐步与逐PCG调用记录、最终状态、GPU频率/功耗/显存与主机RSS/HWM。计时包括全部已执行更新、capture和retry；监测记录需评审后才能作性能对照。失败或缺失阶段不补零、不外推完整耗时。两指承载的0.001N观察阈值明确标为探索值；正式承载持续时间、形变恢复、release和时间/性能门槛仍待标定，G6/G7始终不由入口自动宣布通过。

`render_sharpa_grasp --cube <asset.npz> ...` 根据cube manifest重放记录状态；`plot_sharpa_grasp`自动使用cube COM字段。README缩略图只展示初态资产，不是成功抓取证据。

8I保留历史触发与 `SCHUR-SPD-01` 前置关卡；当前压板轨迹不完整，不能据此标记自由抓取 `NOT_TRIGGERED`。8J等待8H/8I关闭后重新profile。MAS不进入生产，AMG GPU留P3；无自动nightly或推送。
