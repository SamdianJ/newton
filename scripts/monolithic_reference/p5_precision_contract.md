# P4 T5 precision contract

Status: T5-A in progress. This document freezes the implementation boundaries;
it does not certify FP32 or complete T5-A/B/C/D. The public solver continues to
run FP64 until all online components and lifecycle checks are connected.

Starting commit: `56d054ae7dfa11bc9eb97582c3815b4d59349798`.
Evidence: `agents/integration/artifacts/p4/precision-comparison/t5-a-v1/`.
`frozen-inputs.json` verifies 396 input hashes and freezes 92 source files.
`dtype-occurrences.json` records 725 source locations, with enclosing symbols,
before implementation edits. This lexical inventory is an audit aid; the
following data-flow classification determines what changes.

## Dtype boundaries

`real` below means FP64 by default, FP32 only for the explicit experiment.
Integer IDs, schemas, status, masks, generations and flags retain their types.

| Module / responsibility | Input → arithmetic / reduction → output | Required handling |
| --- | --- | --- |
| `articulation_state`, `coupled_state` | model FP32 or explicit source → real → real q/qd, x/v, bound copies, joint/link v/omega/V_sym/last_dt | Quantize before native validation. No hidden FP64 authoritative state. Copy/reset/rollback include every field. |
| `articulation_variational` FK/Jacobians | frozen model values, real q → real transforms/COM/rotation/Jacobians → real scratch | Single formula source; typed constants and intermediates. Display body outputs remain model FP32. |
| `_TreeKinematics.validate_inertia` | model inertia/armature, real FK → NumPy real matmul/eigvalsh → native validity | Existing `.astype(float)`, normalization, second moments and positivity checks must be audited, not categorized as logging. |
| `ArticulationObjective` | real history/targets/loads → real joint/link E/r and tangent preparation → real scratch | Cast FP32 controls once at the frozen-input boundary; B/PCG storage remains FP32. |
| `tet_precision`, `tet_material` E/r | real x/xn/vn plus model FP32 material/rest/mass/loads → real displacement, F, detF, energy/stress and atomic sums → real residual/parts/min_det | Same Kim/Smith formulas and detF guard. Native flags execute in both modes. |
| `tet_common` | model FP32 → existing construction mass validation → FP32 mass snapshots | Preserve the frozen coefficient definition, density checks and physical identity; constructor reference arithmetic is not online E/r. |
| `tet_assembly`, `linear`, `pcg_common`, `pcg_production`, `soft_preconditioner` | selected candidate / geometry → existing B32 / PCG32 → FP32 direction | Do not change linear precision or budgets. Audit generic candidate kernels and any high-precision auxiliary branch. |
| `contact_geometry`, `contact_variational` | real state/geometry and FP32 texture nodes → real shape query, normals, law quantities, E/r and sums → real contact scratch | Keep all three laws, capacity/finite/domain/normal/mask rules. No tolerance widening. |
| `smooth_sdf_builder` | frozen 128-node field → common offline FP64 coefficients → FP64 reference field | Never rebake or change boundary conditions; report build cost separately. |
| `smooth_sdf` online | selected coefficient copy, real coordinates → real phi/gradient/query scratch → real production values | Preserve independent FP64 diagnostic Hessian/oracle. Keep source hash and target dtype/coefficient hash distinct. |
| `coupled_kernels` | real residual/state, FP32 B/direction → real combine/predict/trial/posterior/force arithmetic → real scratch/state/forces | Validate actual written candidate, not a pre-cast higher-precision value. |
| `solver_monolithic` | frozen controls/forces → selected host/device arithmetic → publication | Audit load assembly, energy sums, norm/domain/force guards, effective displacement and scale casts. Freeze inputs once per transaction. |
| `variational_iteration` | real residual/reference/scale/direction → NumPy real norm/dot/sums/targets → native decisions | Remove unconditional direction `.astype(float)`; explicitly type sqrt, slope, caps, relative/absolute target arithmetic and posterior limits. Logging casts occur only after decisions. |
| `contact_forces` | published real arrays → independent read-only NumPy copies → real snapshot | Old snapshots cannot change after step/reset/load. |
| `segment_graph` | precision + buffer/physics identity → captured graph → matching dtype arrays | No hot switch, mixed-precision graph or cache reuse. |
| `aabb`, `collision`, independent reference scripts | existing independent API inputs → existing arithmetic | Not part of solver precision selection. Do not alter their public capabilities or oracle precision. |

## Frozen numerical rules

- Preserve `128 * np.finfo(float).eps` and measured E/r/slope noise constants.
  They define the original qualification envelope, not selectable machine epsilon.
  Record rounding of native thresholds separately; never enlarge them for FP32.
- Keep existing B32 formula, PCG tolerance, regularization, line search, publication,
  trajectory, SDF assets and solver budgets. No FP64 fallback or audit correction.
- Component approximation checks quantify FP32 arithmetic error against an independent
  oracle. Their test tolerances do **not** replace production acceptance thresholds
  or establish scene-level precision qualification.
- Freeze per-step FP32 controls before running either precision. Source/asset hashes
  are recorded now; sampled control-buffer hashes remain an explicit pending item.
- Actual FP32 candidates are promoted exactly for an independent FP64 audit using
  their own step-start state and history. A separate FP64 trajectory is a comparison,
  not the residual oracle. Record false convergence and unsafe native acceptance.

## Lifecycle implementation contract

The eventual public constructor accepts `SolverMonolithic.Precision` and its
`fp64`/`fp32` values, defaults to FP64 and cannot be switched after construction.
Precision participates in continuation, configuration and graph identities.
Cross-precision continuation/checkpoint restore is rejected before mutation.

Keep a separate legacy FP64 validation path: validate the original schema, all
physical/configuration digests, field dtypes, shapes, finite values, bound history
and force generations in scratch before migrating to the new identity. Do not
recompute a digest and call it validated. Save stays atomic; failed load leaves
the destination unchanged. Native FP32 dt representability/underflow and guard
arithmetic must be validated before any scratch is published.

## Experiments and commands

`frozen-inputs.json` contains eight IDs, each expanded across FP64/FP32:

- `close-18--C-EE`, `close-18--C-IE`, `close-18--C-II`;
- `close-24--C-EE`, `close-24--C-IE`, `close-24--C-II`;
- `franka-cube--C-IE`, `sharpa-recorded-10mm--C-IE`.

All use 1 ms, OR, current_iterate, CUDA production/graph. Close uses 2,500 steps;
Franka 5,250; Sharpa 8,667. The old Close driver's last_accepted configuration is
not the T5 configuration. Reuse its frozen construction and targets, not its
solver factory/configuration. Each numerical case is attempted independently,
stops on first hard failure and records complete input/trial/history evidence.

Performance repeats each of these eight cases three times per precision, serially,
from t=0 after warmup and state reinitialization. Separate build/compile/capture,
solve, recording and independent audit cost. Report failed-prefix cost without
claiming a qualified speedup. Parameterized FP64 must rerun the existing ten-case
full matrix (44 deferred cases remain deferred).

Executable current FP64 regression commands (from this worktree):

```bash
UV_PROJECT_ENVIRONMENT=../monolithic-integration/.venv uv run --no-sync python -B -m scripts.monolithic_reference.run_p4 --profile dev --list
UV_PROJECT_ENVIRONMENT=../monolithic-integration/.venv uv run --no-sync python -B -m scripts.monolithic_reference.run_p4 --profile full --list
```

The precision experiment runner and its executable commands are pending; do not
run the historical Close CLI as if it already supported this contract. T5-A cannot
be marked complete until the runner, sampled controls, complete field/kernel/host
parameterization and public construction/lifecycle identity are implemented.
