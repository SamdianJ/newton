# P2 验证（8C2 清理中间态）

旧 production 已删除，新 masked production 尚未接入。当前只支持 diagnostic smart 和 owned mass。
旧 baseline checkout 不再是执行依赖；旧 plan/progress 因 policy 与源码不同拒绝恢复。旧证据保留离线追溯。

开发门槛：`uv run --no-sync -m scripts.monolithic_reference.validate_p2 --suite dev`。
仅预览夜间计划：`uv run --no-sync -m scripts.monolithic_reference.validate_p2 --suite night --list`。
完整夜间运行由用户手动执行；没有固定旧作业数承诺。
