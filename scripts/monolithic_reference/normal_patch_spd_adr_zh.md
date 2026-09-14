# ADR：SCHUR-SPD-01 — 用支撑重叠次数构造 normal patch Schur 上界

2026-09-14。状态：数学设计和 tiny FP64 oracle 验证；production setup/factor/apply 尚未实现，8I 整体未完成。本设计依据既有历史触发推进，不以 cube 压板预算替代自由抓取触发筛查。

## 问题与选择

原方案将精确 Schur 矩阵跨 patch 项删除后代入 Woodbury 减法，无法保证完整近似逆 SPD。令 B=1、两个单行 patch 的 G=W=1，各局部 Cholesky 均成功，但完整 apply 为 1-1/2-1/2=0；三个 patch 时为 -1/2。normal-row 分区不重叠无法排除共享 q/节点。

选择在 patch 内放大**共享 base 块的 Gram 贡献**，使 patch 块对角矩阵成为精确 Schur 的上界，再执行 Woodbury 型减法。局部块仅按真实支撑重叠次数放大，无经验 damping 参数；单 patch 和支撑互不相交的多 patch 保持精确 Woodbury。

未选择将所有 patch 组成无界 dense Schur；也未选择无条件平均所有 patch 逆，后者会削弱不相交区域的修正。SPD 是准入前置正确性条件，本 ADR 不承诺迭代/性能收益。

## 定义

- B=P0：不含 contact 的尺度化 block diagonal actor base，含当前 lambda 一次。q 使用完整 dense actor 块，x 使用动态节点 3×3 块。每块须通过合法 Cholesky；不修补非法主元。
- U=W^(1/2) Ghat，Ghat=G S。只选 NORMAL factors，W>0；W=0 省略，负/非有限值失败。全局 normal/tangent 算子和当前 actor_block 对照保持原样。
- normal rows 按 link 与 boundary adjacency 分成有限、互不重叠且完整覆盖的 patch i。单个 row 只能属于一个 patch；固定节点自由度事先排除。
- a 表示 B 的一个完整块；d_a 为接触到该块的不同 patch 数。dense q 内的不同坐标仍属于同一个块，不能分别计数。按结构可能支撑计数允许保守高估，禁止低估；零块不贡献。

令 U_ia 为 patch i 限制到 base 块 a 的行列矩阵，定义：

```
S_c = I + U B^{-1} U^T
D_i = I + sum_a d_a U_ia B_a^{-1} U_ia^T
D   = blockdiag(D_i)
M^{-1} = B^{-1} - B^{-1} U^T D^{-1} U B^{-1}
```

因此 apply 可拆成 base 三角解、normal-row 限制、各有界 patch 的 Cholesky 解、转置收集与 base 解。所有参数在一次 PCG 内固定；限制和收集必须严格互为转置，禁止单向 Gauss–Seidel 式原地交叉更新。

## 完整 SPD 证明

任取 patch 空间向量 v=(v_i)。设 t_ia=B_a^{-1/2} U_ia^T v_i。对每个 base 块，最多 d_a 个 t_ia 非零，由 Cauchy–Schwarz：

```
||sum_i t_ia||^2 <= d_a sum_i ||t_ia||^2
```

对 a 求和即得 U B^{-1} U^T <= blockdiag(sum_a d_a U_ia B_a^{-1} U_ia^T)，故 D >= S_c > 0。

SPD 矩阵求逆反转 Loewner 序：0 < D^{-1} <= S_c^{-1}。左右乘 B^{-1}U^T 和 UB^{-1}，再从 B^{-1} 相减：

```
0 < (B + U^T U)^{-1} <= M^{-1} <= B^{-1}
```

左端由精确 Woodbury 得到。因此完整 apply 线性、对称、严格正定，与 patch 是否共享 q/节点无关。这个证明针对完整算子，不能用各 D_i 单独 Cholesky 成功替代。

单 patch 或各 patch 不共享 base 块时，所有有效 d_a=1、跨 patch Gram 为零，D=S_c，恢复精确逆。无活动 NORMAL rows 时 M^{-1}=B^{-1}。多个完全相同的标量共享支撑反例中，本式也恰好恢复 1/(1+m)。

证明为精确算术结论。FP32 的减法抵消、condition number、pivot、finite、r^Tz 和独立真实残差仍需 production 数值验证；不能从证明推导 GPU 误差或性能已通过。

## 实现约束与验证关卡

1. 当前提交只包含 tiny FP64 oracle（最多 64 DoF、128 rows）、反例与随机共享支撑测试；不接入 solver。oracle 的 dense 上限不是未来 production 容量配置。
2. production 先建立有界 row patch 与 base-block incidence；容量溢出必须失败，不能截断。patch 行容量、总 patch 容量、q 容量及节点支撑容量须在实际 cube 负载下确定并记录，分配乘积用 host int64 校验。
3. 每个有效 assembly/scaling/lambda 更新重算 B、U、d_a、D_i 及因子。图按地址/容量/结构区分，原地址数值更新可复用；不能沿用旧 d_a 低估新接触支撑。
4. 必须从不含 contact 的 base 构造，禁止把已包含 normal 自块的现有 actor_block 因子直接拿来再次加 normal。切向近似可省略；全局切向项不可删除。
5. GPU apply 必须验证 tiny Woodbury、共享 q/节点、尾块、零权重/q-only、非法权重/主元、容量、alias、generation/lambda/scaling、双模式/graph 与 history/rollback。限制/收集的转置关系和失败后发布沿用现有契约。
6. fixture 未冻结时不宣称完整 8I 通过；计入 setup、factor、apply、PCG、capture、retry、整步与完整轨迹成本后复验物理质量。历史触发仍然有效；性能无收益也不能直接豁免已触发的 P2 出口要求。

测试入口：`uv run --no-sync -m unittest newton.tests.test_monolithic_schur_spd`。数学/CPU oracle PASS 只关闭 SCHUR-SPD-01 的公式障碍，不代表 production 或 8I 正式验收完成。
