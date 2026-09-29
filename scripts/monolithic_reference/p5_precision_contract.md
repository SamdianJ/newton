# P4 T5 precision contract

Status: T5-A and T5-B complete. FP64 remains the default; experimental FP32
fails the original numerical qualification, including observed component-level
false convergence. T5-C full-scene independent audits and T5-D performance
experiments remain pending. T5-B evidence is recorded in
`agents/integration/artifacts/p4/precision-comparison/t5-b-v1/receipt.json`.

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
- T5-B: native component/transaction checks, negative controls and the current
  ten-case FP64 full regression are complete. FP32 numerical qualification fails.
- Pending: independent T5-C full numerical audits and T5-D isolated performance repeats.

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


## T5-B validation scope

The retained precision checks are implementation and safety gates. Their component
approximation tolerances do not replace the original E/r/slope noise qualification.

| Boundary | Retained checks / reproducible evidence |
| --- | --- |
| Native trial and physical posterior | `test_monolithic_p5_safety`: actual written q/x displacements, independent E/r, actual B32 posterior, frozen original history, three laws and both dtypes on CPU/CUDA. |
| Publication and rollback | The same module injects failure after initialized/accepted candidates for strict, last_accepted and current_iterate; rejected trial coordinates contain NaN. Existing capacity/nonfinite/final-force/publication failure tests run in both precisions and both buffer arrangements. |
| Checkpoints and identity | Corrupt schema, precision, digest, history, force metadata and dtype must leave the target unchanged. `test_monolithic_p5_solver_precision` additionally checks cross-precision continuation/restore and validates the legacy FP64 digest before migration. |
| Modes and identity after warmup | `test_monolithic_p5_safety` additionally exercises all three globalizations/laws in CPU diagnostic/production and CUDA diagnostic/production/graph, then rejects the opposite precision without modifying the output. |
| Contact and history components | `test_monolithic_p5_contact_precision` checks plane/sphere/box, radial slip, release/new contact and invalid geometry. Existing native tet/articulation suites cover both materials/mass modes, seven joint fixtures and graph replay. |
| Native numerical qualification | External `native-probes.py` and `contact-probes.py` repeat evaluations 32 times, compare eightfold errors with the original envelopes and retain actual representable endpoint chords. C-EE energy differences and all-law contact residual differences are separate; C-IE/C-II are not treated as conservative energy gradients. |
| Negative controls | External `negative-controls.py` reads the two preserved PR1 failure inputs and extracts only the pure PR2 q/pose/history precision formulas. It executes no retired solver. PR1 uses an explicitly reported mass-only scale; its merit is not a replay of the retired assembled-B solver. |
| FP64 full regression | `run_p4 --profile full` runs the existing ten-case matrix. Its explicit schema-2 → schema-3 FP64 comparison validates the added precision field and schema transition; every physical field and original tolerance still applies. |

External probe scripts and source/input hashes are retained under
`agents/integration/artifacts/p4/precision-comparison/t5-b-v1/`. The coupled component oracle promotes every frozen numeric input before arithmetic;
it holds the native predictor fixed, with prediction/history checked separately
by the articulation component tests. These component probes do not implement the independent full-scene T5-C audit or the T5-D timing
protocol. The 44 deferred P4 trajectories remain deferred.
