# P4 stage results and commit policy

Complete each independently reviewable phase or subtask, run its applicable checks, and commit its code, tests, and result summary on `SamiulJ/monolithic-refactor` before starting the next phase. Reference external evidence by path and SHA-256. Keep unrelated edits, large recordings, caches, and environments out of commits. Do not push without separate authorization. A blocked or unfinished phase is not complete.

## T0a–T0c: baseline admission

These phases were completed together before the per-phase commit rule was introduced and are committed as one T0 batch. Solver implementations are unchanged. The frozen P3 source remains `39f80c48da2022187823f8218131a249807597a0` in the integration checkout.

- Freeze source/input identities, API contracts, test discovery and migration candidates.
- Add an isolated parameterized Franka baseline driver and complete recording checker.
- Pass nine Franka trajectories: 47,250 steps, 47,193 converged, 57 Stopped, no hard failures; all original grasp gates pass.
- Verify and reuse Sharpa: 8,667 steps, 8,637 converged, 30 Stopped; grasp remains uncertified.
- Pass the default driver comparison and eight temporary same-configuration repeat comparisons; freeze per-case numerical envelopes. Remove duplicate temporary trajectories after retaining reports.
- Pass seven tooling tests, CPU/CUDA three-law independent E/r/B and noise checks, and pre-commit. The 8,160 discovered baseline tests were not all executed.

T0 admission allows T1 to start. This is not refactored-solver acceptance or P4 full qualification. The other 44 original matrix cases are deferred. Pairwise envelopes are not statistical confidence bounds; a complete historical package inventory is unavailable.

External evidence is relative to the workspace root containing `worktrees/`, `dev/`, and `agents/`. Large recordings remain outside Git.

| Evidence | SHA-256 |
| --- | --- |
| `agents/integration/artifacts/p4/t0-v1/RESULTS.md` | `55e4a394f714a768b0932c6f40c510f0d4b1004f5048d52692342200d8e3e714` |
| `agents/integration/artifacts/p4/t0-v1/t0c-receipt.json` | `98099618a289c9be2d376b2d41322d3c34b89b117c0e81f28894e6d66029ae94` |
| `agents/integration/artifacts/p4/t0-v1/input-manifest.json` | `48d7d876e1d66989e85a58dc793d59297bcbd990c04bf479f336157d13200c69` |
| `agents/integration/artifacts/p4/t0-v1/comparison-tolerances.json` | `a6ac690f0e8c1764f494ea8616ca2a1e65873754c62848476f6e3abc64604d6d` |

## T1: neutral definitions and import isolation

Extract unchanged enums, immutable stats/joint options, calibrated configuration,
and the step exception into `solver_types.py`. Keep existing solver type access.
Resolve the solver and independent collision pipeline lazily; neutral definitions
and collision imports no longer load the historical solver. Move P3 exception
imports to the neutral module. No kernels or solver methods change.

Validation: 37 tests run, 36 passed, one existing CPU mesh-texture skip; the three
new boundary tests fail on the frozen baseline. AST comparison preserves eight
definitions after normalizing docstring indentation and the status annotation,
and preserves every historical solver method/kernel unchanged.
Full trajectories are deferred to the planned later gates.

Remaining: shared joint/tet dependencies belong to T1a/T1b; the transitional
parent chain and old public solver stay until T2. No source is retired yet.

Evidence: `agents/integration/artifacts/p4/t1-v1/receipt.json`, SHA-256
`a551d8e3c754cae5052384a0406630311a2d75a793725e9c2ae4cddf33239e56`.

## T1a: shared joint parameters and target snapshots

Extract scalar layout validation, model parameter freezing, allowed properties,
target mapping/snapshots and the control-validation kernel into
`joint_parameters.py`. P3 uses `JointParameters` without old evaluation buffers.
The historical workspace shares these parameters and retains its own evaluation
and publication methods. FP64 kinematics now imports only the shared validator.
No FK, target arithmetic, E/r/B, or history formula changes.

Validation: 48 tests run, 47 passed, one existing CPU texture skip. Cover CPU/CUDA
parameter and layout rejection, target snapshots, duplicate force rejection,
FK/history/configuration, independent E/r/B and coupled transactions/graphs.
AST checks confirm identical validation, constructor parameter prefix, target
snapshots and input kernel. Two initial test-only CPU shared-view mistakes were
corrected; their failure log is retained.

The component import boundary passes. The historical solver parent chain remains
explicitly deferred to T2; no source files have been retired.
Evidence: `agents/integration/artifacts/p4/t1a-v1/receipt.json`, SHA-256
`7e69db7d0563195a079c0007ea93bbcb5606f39287ccb1733ba4fde51a8ae8fe`.

## T1b: shared tet configuration, material and assembly

Split the existing tet definitions into `tet_common.py` (configuration/mass and
validation), `tet_material.py` (Kim/Smith physics and PSD projection), and
`tet_assembly.py` (production B, scatter and required scratch). P3 FP64 E/r and
coupled assembly import these components directly. Move unchanged acceptance
predicates into `variational_iteration.py`. Keep historical residual-only
code and transitional aliases until parent removal and archival.

All 38 original definitions/type aliases and the acceptance function retain
identical ASTs. Validation runs 61 tests: 60 pass and one existing CPU texture
skip. Direct CPU/CUDA tests cover eight material/mass/fixed-node combinations,
independent E/r and declared projected B, PSD and owner/scalar scatter. Existing
tests cover shared nodes, failures, checkpoint/rollback and execution graphs.
New import checks exclude both historical tet modules from shared components.

An initial oracle-input mismatch used Python gravity instead of the stored
FP32 model value; the test input was corrected without relaxing tolerances.
Full trajectories remain deferred to T2/T3 dev and T4 full.
Evidence: `agents/integration/artifacts/p4/t1b-v1/receipt.json`, SHA-256
`b15f068826bc020c861008a6fbec33401b8625f409d2dcd7f2b368f5241d33b6`.

## T1c: contact boundary and historical friction isolation

P3 contact already shares factors through linear and geometry through its own
query components. Mark historical spring/Huber and xi code as deprecated; add
no new abstraction or runtime warnings. Fresh-process imports confirm contact,
state and collision components exclude the historical solver/contact modules.
CPU/CUDA checks reject spring parameters, unsupported laws and old xi namespaces
while allowing ordinary model state initialization. Production ASTs are unchanged.

The predefined P3 and independent collision/AABB gate runs 101 tests: 95 pass,
six existing CPU texture/graph skips. Full pre-commit passes. An initial broader
run also executed two historical spring trajectory checks: CUDA physics_parity
failed (scalar error 2.96831131e-05); its fixed-baseline rerun passed. This remains
an unresolved legacy-only observation, not a claimed pre-existing failure and
not part of the planned compatibility scope. Logs retain both outcomes; no
test or tolerance is changed. Full trajectories remain deferred.

The historical parent still introduces a transitive contact import until T2.
No files are retired at this stage.
Evidence: `agents/integration/artifacts/p4/t1c-v1/receipt.json`, SHA-256
`7cdb761a9a586ed841a68f30a7d03c1ce3b2452935277c38602cc8cbb0a62959`.

## T2-1: merged q+x transaction

The complete private solver now directly inherits SolverBase. Inline the former
parent's initialization and transaction, retaining predictor/trial/history and
publication order; allocate only the q+x linear workspace. Move unchanged scalar
kernels into coupled_kernels and retain the exact linear failure mapping.

Validation: 30 tests, 29 passed, one existing CPU texture skip; full pre-commit
passes. CPU/CUDA three-law three-step comparison against 39f80c48 is bitwise
identical for all captured arrays, including identity/history. Both trees pass
final-force failure rollback injection. The new inheritance check fails on the
baseline as expected. Existing failure injections now patch the same iterate
call directly instead of the removed parent; assertions and tolerances unchanged.

Public entry and long dev trajectories remain T2-2.
Evidence: `agents/integration/artifacts/p4/t2-1-v1/receipt.json`, SHA-256
`68cd12e947d1fb0311edce5e0a2518631310f3b23e64b36aec941223c94d06e4`.
