# P4 T5 precision contract

Status: T5-A complete. This document freezes the implementation boundaries;
it does not certify FP32 or complete T5-B/C/D. The public solver now accepts
FP64 (default) and experimental FP32. Completion evidence is recorded in
`agents/integration/artifacts/p4/precision-comparison/t5-a3-v1/receipt.json`.

Starting commit: `56d054ae7dfa11bc9eb97582c3815b4d59349798`.
Evidence: `agents/integration/artifacts/p4/precision-comparison/t5-a-v1/`.
`frozen-inputs.json` verifies 396 input hashes and freezes 92 source files.
`dtype-occurrences.json` records 725 source locations, with enclosing symbols,
before implementation edits. This lexical inventory is an audit aid; the
following data-flow classification determines what changes.

## Implemented component batches

- Tet E/r: native FP64/FP32, independent component checks and exact FP64 parity
  (`t5-a-v1`, commit `0476bf91`).
- Articulation: native FK/Jacobians, joint/link E/r, predictor/history, private
  state storage and host inertia checks (`t5-a2-v1`). Seven fixture-model probes
  pass on CPU/CUDA; 378 FP64 arrays match the previous implementation exactly.
  Both precision graphs coexist and replay. This does not expose a q-only solver.
- Coupled implementation: generic contact/SDF and q+x kernels, native state,
  host acceptance and posterior arithmetic; public immutable Precision enum.
  Schema 3 adds the precision field; FP64 schema-2 checkpoints pass the original
  complete validation before migration. Default FP64 physical digest is unchanged.
- Four full FP32 control streams cover the eight scenarios. The precision runner
  consumes the same frozen bytes for both precisions; it does not certify
  independent numerical accuracy or perform the T5-D timing protocol.
- Pending: systematic T5-B safety/negative-control qualification, independent
  T5-C audit/full numerical experiments and T5-D isolated performance repeats.

## Dtype boundaries

`real` below means FP64 by default, FP32 only for the explicit experiment.
Integer IDs, schemas, status, masks, generations and flags retain their types.

| Module / responsibility | Input → arithmetic / reduction → output | Required handling |
| --- | --- | --- |
| `articulation_state`, `coupled_state` | model FP32 or explicit source → real → real q/qd, x/v, bound copies, joint/link v/omega/V_sym/last_dt | Quantize before native validation. No hidden FP64 authoritative state. Copy/reset/rollback include every field. |
| `articulation_variational` FK/Jacobians | frozen model values, real q → real transforms/COM/rotation/Jacobians → real scratch | Single formula source; typed constants and intermediates. Display body outputs remain model FP32. |
| `_TreeKinematics.validate_inertia` | model inertia/armature, real FK → NumPy real matmul/eigvalsh → native validity | Existing `.astype(float)`, normalization, second moments and positivity checks must be audited, not categorized as logging. |
| `ArticulationObjective` | real history/targets/loads → real joint/link E/r and tangent preparation → real scratch | Cast FP32 controls once at the frozen-input boundary; B/PCG storage remains FP32; its existing scalar-policy exceptions are listed below. |
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

## Retained linear-policy precision

The label FP32 selects the nonlinear physical path; it does not change the
pre-existing linear solver contract. BSR, preconditioner, PCG vectors/products
and returned direction retain FP32. The following existing **decision** arithmetic
remains shared by both precisions, as required by keeping B/PCG unchanged:

- `pcg_common.residual_converged`: FP32 norms/limits are compared with an FP64
  scalar target; `block_drift`, `max_block_drift` and `decide_restart` use FP64
  gap ratios/restart thresholds.
- `linear._cache_monolithic_pcg_denominator` and
  `_pack_monolithic_pcg_check` retain the FP64 denominator and scheduled-check
  threshold. `pcg_production` retains the FP64 replacement threshold/drift.
- Tet B32 assembly retains its original host timestep coefficient setup before
  conversion to the existing kernel argument type. FP64 candidate-only tangent
  preparation branches are never selected for FP32 candidates.

These are not classified as logging. No linear module was changed. The nonlinear
physical posterior, complete residual goals, slope/acceptance arithmetic and
publication decisions introduced by P3 now follow the selected native precision.
`linear.densify_for_test` and independent reference tools remain FP64 diagnostics.
`remaining-fp64-occurrences.json` records all remaining lexical locations for review.

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
  are recorded now; sampled control-buffer hashes are in `t5-a3-v1/controls/manifest.json`.
- Actual FP32 candidates are promoted exactly for an independent FP64 audit using
  their own step-start state and history. A separate FP64 trajectory is a comparison,
  not the residual oracle. Record false convergence and unsafe native acceptance.

## Lifecycle implementation contract

The public constructor accepts `SolverMonolithic.Precision` and its
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

Executable precision commands (same environment prefix as above):

```bash
UV_PROJECT_ENVIRONMENT=../monolithic-integration/.venv uv run --no-sync python -B -m scripts.monolithic_reference.run_p5_precision --list
UV_PROJECT_ENVIRONMENT=../monolithic-integration/.venv uv run --no-sync python -B -m scripts.monolithic_reference.run_p5_precision --controls ../../agents/integration/artifacts/p4/precision-comparison/t5-a3-v1/controls --output /path/to/new-results
```

Append `--max-steps 2` for implementation smoke checks, always PARTIAL; use
`--case close-18--C-EE--fp32` (or another listed ID) for an independent case.
The numerical recorder returns COMPLETE_NOT_AUDITED on full advancement, never
PASS without the separate independent audit. It records complete state chunks,
trials/statistics, shared controls identity and first-failure scratch states.
Compile/capture are included in its solve timing, explicitly labelled; these
numbers are not the isolated T5-D performance measurements.

For independent T5-C analysis, the existing p3_articulation_oracle,
p3_tet_oracle, p3_contact_oracle and p3_smooth_sdf_oracle stay FP64 and unchanged.
Promote the actual saved native initial/history/candidate values exactly; compare
against the original thresholds using the original FP64 scale/target formulas.
Report native-vs-reference threshold rounding independently from E/r error.
The production path keeps the fixed noise constants and rounds each native
operation in its selected dtype, including sqrt, norm, dot and reductions.
The independent audit must not supply directions, corrections or continuation.

