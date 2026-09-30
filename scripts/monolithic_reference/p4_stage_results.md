# P4 stage results and commit policy

> 2026-09-30 存储整理：保留九条Franka＋一条Sharpa完整FP64参考；其余大录制和日志按用户要求清理。下文是历史记录，部分原证据路径已归档，定位见[开发历史](development_history.md)。本次不关闭R1，也不产生新的测试通过结论。

Current acceptance: [P4 acceptance record](p4_acceptance.md). Historical entries
below retain their original tested-source labels and time-specific status.
Documentation closeout is complete. R1 asset/reference binding is now implemented;
runtime validation remains pending. No tests or trajectories were rerun.

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

## T2-2: public solver, lifecycle API and complete dev regression

Expose the merged SolverMonolithic directly through the experimental namespace,
with all 23 frozen constructor parameters and defaults. Remove private factories
from the public solver; preserve the old source under solver_monolithic_legacy.py
for later audited retirement. Existing 26 methods retain their T2-1 ASTs.

Add complete-state creation/reset, scratch-validated load, atomic sibling-temp
checkpoint save, and independent read-only FP64 MonolithicContactForces snapshots.
Explicitly reject external Contacts, collision scheduling, generic reset and
model-rebinding notifications. Keep public lazy loading and independent collision.

Validation: 59 checks run, 58 pass, one existing CPU texture skip. Full pre-commit
passes. A public construction/step process loads no historical solver/components.
P3 checkpoint arrays restore exactly without schema/digest migration and resume.
Franka completes 5250 converged steps and its 500-step grasp hold; Sharpa completes
8667 steps (8637 converged, 30 Stopped, no hard failures; grasp NOT_CERTIFIED).
Both full recordings pass the frozen T0 envelopes with identical discrete labels
and identities. Franka replay checks all 5251 frames. These are dev results; the
ten-case full remains T4. No equations, budgets or tolerances change.

Required callers and fault injection sites use the public API. Contact-off checks
now solve independent NumPy equations instead of retired single-physics entries.
Extract only the original normal-loading model builder to retain input assertions.
Historical test discovery and remaining legacy aliases are deferred to T3.

An initial Sharpa invocation selected the script's unfrozen default trajectory.
It was stopped, marked ABORTED_WRONG_INPUT and excluded. The accepted run uses
`agents/integration/artifacts/p3/pr3/current-iterate-production-v1/sharpa-trajectory.json`
explicitly; its raw hash, payload, SDF identity and all initial fields match T0.
Evidence: `agents/integration/artifacts/p4/t2-2-v1/receipt.json`, SHA-256
`b78077c554f894454b0482ba314b1ce2c32b8ea024677fefb9344e9a79f0ee90`.

## T3-1 — Share smooth SDF construction

Expose build_smooth_sdf and SmoothSDFSet through the experimental namespace.
Move existing coefficient construction, source texture identity, node sampling
and frozen qualification into the library; the production builder imports no
scene or diagnostic drivers. Franka and recorded Sharpa use the public entry.
Reference drivers temporarily retain import aliases pending T3-3 retirement.
Source/check/qualification metadata does not enter the physical identity.

Validation: both scene construction identities, coefficient hashes and Franka
component reports are byte-identical before/after. Coefficient, texture identity,
node kernel and sampling function ASTs are unchanged. Qualification rejection
checks cover certificate, textures, geometry status, table size, node hashes,
coordinates and coefficient identity, with no sampling fallback. Public imports
load no reference or legacy solver modules. 23 tests: 21 pass, two existing CPU
skips. Full pre-commit passes. Independent spline/contact oracle remains unchanged.

Evidence: agents/integration/artifacts/p4/t3-1-v1/receipt.json, SHA-256
50db6d41dbdcc1582c4f022e79d822cdcbfab173b16dee353bad424ff23241f5.
T3-2 runner/scenes, T3-3 retirement and post-migration dev remain outstanding;
this component result is not full trajectory qualification.


## T3-2 — Migrate scenes and deliver dev/full runner

Move unchanged Franka/Sharpa scene, control and recording helpers into examples;
retain temporary reference import wrappers until T3-3. The supported scenes
import no diagnostic drivers. Parameterize only Franka law/globalization with
unchanged defaults. The serial process-isolated run_p4 runner shares the T0
matrix, checks frozen inputs and real module origins, verifies every recorded
field against frozen case envelopes, and labels filtered runs PARTIAL. Dev has
two complete scenes; the temporary full matrix has ten cases. Bring recorded
Sharpa replay into the repository and remove its hard-coded integration import.

Validation: 35 targeted tests pass, full pre-commit passes, and 17 extracted
definitions have identical ASTs. Dev PASS: Franka 5250 converged steps and grasp
hold pass; Sharpa 8667 steps, 8637 converged and 30 Stopped, no hard failures.
Both complete recordings pass frozen T0 envelopes with unchanged physical digest.
Real module origins (234 Franka / 227 Sharpa) point into the development tree;
none imports the six historical solver/component modules. Source hashes still
match after validation. Replay checks 5251 and 8668 frames respectively. Sharpa
grasp remains NOT_CERTIFIED; T4 full was not run.

Evidence: agents/integration/artifacts/p4/t3-2-v1/receipt.json, SHA-256
47c513cf08bc16d7b910f9e3463d3d8fc1cc1608f332ae8f51c707c9c8f4ca3f.
The T3-1 receipt now records an explicit development-tree reconstruction repeat:
initial /tmp invocation had resolved the shared integration editable install.
The corrected check confirms identical fields and coefficients with the new
builder; no acceptance envelope changed.

T3-3 dependency audit is recorded separately in t3-3-v1/dependency-audit.json and
RETIREMENT_GATE.md. No legacy retirement occurred in this extraction commit;
per-test-ID migration, archive hashes and final discovery remain required.


## T3-3a: preserve supported assertions before retirement

Migrate eight mixed test modules to the public complete q+x state and shared
joint/tet/contact components. Preserve independent FK/E/r/B, constitutive FD,
mass, PSD/scatter, input validation, checkpoint, publication and rollback
assertions. Keep pure PCG and preconditioner oracles. Extract seven unchanged
linear reference/evidence definitions from historical profiling drivers so
retained benchmarks no longer import those drivers.

The ID map preserves 177 baseline test IDs (two device-specific budget test IDs
are renamed from `budget_and_legacy` to `budget_and_last_accepted`). Document the
retired q-only/owned-mass dispatch combinations, duplicate historical residual
calls and old trajectory-profiler telemetry tail explicitly. Complete-state
rollback preserves each original buffer; tests now exercise that P3 contract.
Constructor tolerance rejection timing remains identical to the fixed baseline.
No production solver source, formula, budget or trajectory input changes.

Validation: 196 tests, 195 passed and one existing CPU graph skip. Re-run the
joint-contract and tet modules after the final test-name/singleton-loop cleanup.
Seven extracted definitions have identical ASTs; independent imports of all
11 exercised test modules resolve 217 local modules and load none of the six
legacy solver modules. Full pre-commit passes. No full trajectories are rerun
for this test/helper-only batch; T4 acceptance is still pending.

Preserve 13 originals with SHA-256 and source commit in the external archive;
all active files remain present. Discovery improves from 7,954 IDs / 22 import
errors to 8,105 IDs / 8 import errors. Discovery is not execution, and this is
not a repository-wide test pass. Remaining historical tests/drivers, mixed
x-only coverage, examples scheduling and physical retirement still require
T3-3b mapping and verification. Do not declare T3-3 complete from this batch.

Evidence: `agents/integration/artifacts/p4/t3-3-v1/receipt-migration.json`.
SHA-256: `69ccfc7767b9ebdb9fef09114a203a65dc7ff001249585a44bbc88148bfbdf2a`. Final follow-up: 44 tests passed.


## T3-3b: graph coverage and evidence import chains

Preserve the independent segment cache and linear assembly/scaling/factor tests,
using complete public q+x states for transaction checks. Verify changing buffer
values, scalar parameters, allocation bindings, streams, bounded graph retention,
stale assembly tokens and pivot failure. Keep PCG-only versus optional linear
segment comparisons; this does not enable old segment dispatch in production.
Map all ten historical graph test IDs: seven migrate, three old BE-dispatch /
mutable-residual-guard / cold-profiler-specific tests retire after snapshots.
Current joint inertia and tet guard oracles remain and are run in this batch.

Point cube tooling at the previously extracted evidence helpers; move the
unchanged process RSS helper out of the old profiler. Cube calibration, asset
and scan tests can now import without the obsolete profiler chain. This is not
qualification of the old GraspCase solver path. No production code or frozen
input changes; no source files or research assets are deleted.

Validation: 116 tests, 113 passed, three skips (two optional real-asset tests
without their environment variables and the existing CPU mesh-texture skip).
The graph module imports no six legacy solver modules. The RSS helper and
assembly-token test retain identical ASTs. Six pre-edit sources have archived
snapshots and SHA-256 records. Full pre-commit passes.

Discovery: 8,118 IDs and five remaining import errors, down from eight. Remaining
modules are test_monolithic_p2_8bc_runner, test_monolithic_p2_measurement,
test_solver_monolithic, test_solver_monolithic_calibration and
test_solver_monolithic_release_profile. T3-3 remains incomplete: their supported
assertions must be mapped before retirement, then source/driver archival and
actual scheduling cleanup can finish. Discovery is not a full execution result;
no long trajectory qualification is claimed for this test/helper batch.

Evidence: `agents/integration/artifacts/p4/t3-3b-v1/receipt.json`.
SHA-256: `afd45d55398a9f0cd484d13f883d0d6745637e1f65df917dae044a8c410655ae`.


## T3-3c: retire five historical test modules by explicit scope decision

The user explicitly confirmed that the five remaining import-failing test
modules guard retired development functionality and should themselves retire.
Snapshot all five original files before removal, record SHA-256 and their 79
fixed-baseline test IDs, and remove their active test files. This supersedes the
earlier plan to migrate their individual assertions; no old solver compatibility
symbols are restored. Keep independent supported checks migrated in prior batches.

Remove the two explicit registrations from the historical P2 development list
and replace the obsolete README measurement-test command with the archive note.
A repository-wide search including hidden configuration found no importing
modules or public exports; other references are chronological stage evidence.
No production solver, scenario, asset or Newton-wide nightly changes.

Validation: full unittest discovery has zero import errors (8,113 IDs); 20 P2 scheduling
and formal P4 runner tests pass, as does full pre-commit. Discovery does not
imply that all retained tests pass. Legacy source/driver and example scheduling
cleanup still remain, so T3-3 overall and T4 acceptance are not complete.

Evidence: `agents/integration/artifacts/p4/t3-3c-v1/retirement-manifest.json`
and `agents/integration/artifacts/p4/t3-3c-v1/receipt.json`.
Receipt SHA-256: `b3f8971ca1a73b9f5377046e1ebdc534a19dcab244438abeac5e154d4e166c9c`.


## T3-3d: audit and retire remaining deprecated test coverage

Retire 19 whole historical test modules and obsolete cases in nine mixed
modules after source snapshots, hashes, direct/transitive dependency and
scheduling review. Subjects include BE/q-only/x-only implementations, xi
spring friction, old calibration, diagnostics and P2 workflow. Remove the
old normal-loading example registration and three obsolete P2 test entries.
Keep independent collision/AABB, asset, shared linear/math and oracle checks,
and current API rejection contracts. Move the unchanged generic capacity
overflow assertion into the linear suite. No production source is changed.

The initial retained-suite run exposed 12 old calibration-constructor errors
in the support module; archive/retire that harness coverage and keep its two
independent integration oracles. Preserve the failed run as evidence.
Final retained run: 408 tests, 401 passed, seven existing platform/optional
dependency skips; the subsequently restored capacity test separately passes.
Real Sharpa assets were enabled. These tests load none of the six legacy
solver modules. Discovery: 7,844 IDs, zero import errors; 269 IDs retired and
two relocated/renamed, with no unrelated Newton test IDs removed.

Archive: legacy-archive/39f80c48/t3-3d-input/9ab8853d (35 original inputs).
Evidence: agents/integration/artifacts/p4/t3-3d-v1, including per-ID mapping.
Legacy source/driver retirement remains pending. This batch does not complete
T3-3 overall or claim T4/full trajectory acceptance.

Full pre-commit passes. Receipt: `t3-3d-v1/receipt.json`.
SHA-256: `0cc92736e958292c68a2c058cb4f257a69f91fa6254029eb83b9e9af1b690ac6`.


## T4: FP64 numerical acceptance on 9ff616c5

Run the complete temporary ten-case matrix, unchanged inputs, budgets and
frozen tolerances, serially on CUDA production/graph. All ten pass original
scene gates and same-configuration frozen-envelope comparisons. Franka:
47,193 converged plus 57 Stopped over nine complete grasps; Sharpa: 8,637
converged plus 30 Stopped, with zero-acceptance publication only at step 3470.
Total: 55,917 steps, no hard failures. Sharpa grasp remains NOT_CERTIFIED;
the other 44 long cases remain deferred. No production code is changed.

The default two full-run recordings supply dev qualification by receipt hash
without duplicate storage. Replays validate 5,251/8,668 frames. The explicit
retained suite runs 409 tests: 402 pass, seven existing skips (six CPU
CUDA-feature restrictions and one optional research pyamg dependency).
An additional 27-case law/globalization/CUDA-mode matrix passes three actual
q+x steps per case, including real graph captures and replays. Checkpoint,
force ownership, rejection and rollback contracts pass in the component suite.

All case runtime origins and hashes bind to this development tree; no old
solver is loaded. Exact SDF/source/coefficient identity, actual configuration
and physical digest match the respective baseline. Full discovery finds
7,844 IDs with zero import errors. Full source pre-commit passes.

Report equivalence, dev and temporary full numerical qualification as PASS.
T3 legacy source/driver physical retirement is still pending: this is not
completion of all P4 cleanup, and T5 has not started. The scene runner's
PASS is combined with separate component and lifecycle evidence here.

Evidence: `agents/integration/artifacts/p4/t4-v1/RESULTS.md` and `receipt.json`.
Receipt SHA-256: `811dff9bf95b3e936f579ab21242e6f234f1d827c226edb82f16053af748aa79`.


## T3 closeout after T4 numerical acceptance

Archive sources at 19b43e89 with original paths and SHA-256 before removing
54 Python files: six legacy solver modules, eleven old examples/helpers and
37 exclusive calibration, profiling and diagnostic drivers. Retire four old
usage guides and remove their navigation/example advertisements. No fixture,
SDF asset, certificate, trajectory or research result is removed or rewritten.
Historical Markdown and frozen source-hash metadata remain explicitly historical.

Preserve normal-loading model/control recipes and independent envelope math,
removing their BE runners. Close controls retain the original arithmetic and
use current joint mapping; independent matrix pilots use linear_evidence and
reject the retired trajectory mode. Remove unused q-only initialization from
the articulation fixture. Extract the unchanged clamp model into coupled
fixtures: cube diagnostics no longer construct and discard an old solver.
The original constructor failure is reproduced on CPU/CUDA; two added tests
advance the supported cube through two real steps without claiming a full run.
Their initial output-guard failure was test harness configuration, not a solve
failure; the driver prints its result, so these tests allow expected output.

Validation: 17 extraction parity checks pass. Final retained suite: 411 tests,
404 passed and seven existing skips; actual Sharpa assets enabled. Discovery:
7,846 IDs, zero import errors, no removed IDs and two added cube checks. All
seven retired examples disappear from repository example discovery. There are
no remaining static imports of the retired modules. Frozen input hashes pass.

T4's complete ten-case recordings retain their original tested-source labels.
All 236 runtime files loaded by those cases still exist; 235 are byte-identical
and solver_types differs only in an archive-reference docstring. Executable AST
is unchanged. Long trajectories are not rerun for this removal/helper batch;
current components and lifecycle checks run again after removal. No numerical
qualification or tolerance is widened. This closes T3 source/driver retirement;
T4 numerical evidence is preserved with explicit runtime binding. T5 has not
started. No production arithmetic, state schema, physical inputs or budgets change.

Evidence: `agents/integration/artifacts/p4/t3-closeout-v1/`, including
`retirement-manifest.json`, `extraction-parity.json`, `t4-runtime-binding.json`,
`final-components.json` and `discovery.json`.

Full pre-commit passes. Receipt: `t3-closeout-v1/receipt.json`.
SHA-256: `35d1f880b3bbb8cdbbbe374aa47fd20c7ed260555ab2afcab36603d5e55e661d`.


## T5-A initial batch: freeze boundaries and specialize tet E/r

T5 starts from 56d054ae. Freeze 396 input hashes, 92 source hashes, 725 dtype/
host-decision locations and eight scene definitions. Record implementation,
noise, lifecycle and experiment boundaries in p5_precision_contract.md.

Specialize tet E/r and material energy/stress from one formula source for native
FP64/FP32. Keep FP64 default, existing FP32 model inputs and B/PCG arithmetic;
reject mixed coordinates and unrepresentable native dt/guard before scratch
mutation. Rename the internal objective without retaining a deprecated alias.
The public solver remains FP64-only pending the rest of the online path.

New tests first fail for the absent objective. Final retained suite: 416 tests,
409 passed, seven unchanged skips, five IDs added and none removed. Both dtypes
pass independent tet oracle/guard checks on CPU/CUDA and graph replay on CUDA.
Sixteen FP64 baseline fixture cases produce 64 exactly matching arrays.
Full pre-commit passes. No long trajectories run in this batch; prior T4 evidence
retains its tested source. Component tolerances do not widen production gates.

T5-A remains in progress: FK/history, contact/SDF, host acceptance, solver state/
identity, sampled controls and the scene runner are pending. T5-B/C/D have not
started. Do not interpret this component batch as full FP32 qualification.

Evidence: agents/integration/artifacts/p4/precision-comparison/t5-a-v1/.
Receipt SHA-256: `240b58105076c489fcaaa82571dce1082faf77070ea0e11eb4ff804d7b5a9952`.


## T5-A second batch: native articulation components

Specialize shared FK/COM/Jacobians, joint/link E/r and tangent preparation,
predictor and integration history using the same formulas for FP64/FP32.
Private articulation state storage follows the kinematics dtype; standard model/
display buffers and B/PCG remain unchanged. Select NumPy dtype for normalization,
second moments, effective inertia and eigenvalue checks. Reject unrepresentable
dt before reset/objective writes; distinguish FP32 component identity while
preserving the original FP64 digest and schema.

New tests first reject the unsupported dtype option. Seven original fixture
models exercise component probes on CPU/CUDA, independent E/r/FK/Jacobian/history
math, dtype rejection, copy/reset and coexisting precision graphs. Final retained
suite: 421 run, 414 passed, seven unchanged skips; five added IDs, none removed,
no legacy modules loaded. Fourteen FP64 probes produce 378 arrays exactly equal
to the pre-edit capture at 0476bf91. Full pre-commit passes. No long trajectories
run and no production acceptance/noise threshold changes.

T5-A remains partial: contact/SDF, complete coupled state/step, host acceptance,
public precision/lifecycle migration, sampled scene controls and runner remain.
T5-B/C/D have not started; public SolverMonolithic still runs FP64 only.
Evidence: agents/integration/artifacts/p4/precision-comparison/t5-a2-v1/.
Receipt SHA-256: `ede3c854ab417ab729398c5f9b036b1acc1db3234bfd9b4d63cdce069cb35b9e`.

## T5-A completion: selectable nonlinear precision

Complete the public immutable Precision enum, native contact/SDF and q+x
state/history/forces, native host acceptance/posterior decisions, precision-bound
state/configuration/graphs, and validated schema-2 FP64 checkpoint migration.
Default FP64 physical digests remain unchanged. Common offline FP64 SDF
coefficients are converted once for FP32, with their reference identity retained.
Four frozen FP32 control streams cover the eight scenarios in both precisions;
the serial runner binds imported sources and preserves first-failure state.

Final suite: 426 run, 419 passed, seven unchanged skips; five added IDs, none
removed, no retired solver modules loaded. Full pre-commit passes. The pre-edit
FP64 step/checkpoint yields 26 exactly equal physical/history arrays. Default
Franka/Sharpa physical digests match the original frozen configurations.

All 16 two-step scene checks completed: eight FP64 cases each converge twice;
FP32 Franka publishes two Stopped states; six FP32 Close cases and FP32 recorded
Sharpa reject the first step with complete atomic rollback. Short runs remain
PARTIAL; numerical rejections remain FAIL. Actual FP32 composed quaternion norm
error reaches 1.1920928955078125e-7 against the unchanged 1e-10 geometry guard.
No noise/guard widening, fallback or trajectory repair is introduced.

T5-A is complete; T5-B/C/D and full FP64 scene requalification remain pending.
FP32 is not numerically qualified. B/PCG arrays/directions retain FP32 and its
existing scalar convergence/restart policy retains FP64 in both precisions;
the precision contract explicitly lists these preserved decision operations.
The numerical recorder is not the independent audit or isolated timing runner.

Evidence: agents/integration/artifacts/p4/precision-comparison/t5-a3-v1/.
Receipt SHA-256: `cc53b662c5e7c6762cf030e123fd32d2eba1ce7a7fce937e3d713c8a02753f38`.


## T5-B: component safety and FP64 requalification

Keep production solver sources, formulas, defaults, budgets and noise gates
unchanged. Add CPU/CUDA tests for both native precisions, actual trial writes,
physical B32 posterior, publication policies and rejected trials, atomic rollback,
corrupt checkpoints, contact branches and mixed-precision rejection after graph
warmup. The component oracle now promotes all frozen numeric inputs to FP64
before arithmetic; a regression test rejects the previous mixed scalar arithmetic.
The full runner explicitly validates the schema-2 to schema-3 FP64 transition;
all physical fields and original frozen tolerances remain checked.

Retained suite: 439 run, 432 passed,
7 unchanged skips; 13 added IDs and none removed. All 27 original
FP64 law/globalization/CUDA-mode cases pass. Full repository discovery, current
recording replay and imported-source bindings are checked separately; this is not
an assertion that every discovered Newton test was executed.

All ten fresh FP64 full cases pass: 55,917 steps, 55,830
converged, 87 Stopped and zero hard failures.
Franka preserves its grasp gates; Sharpa completes its trajectory with grasp
success still uncertified. Current dev uses the matching two current full cases.
SDF identities, qualification labels and physical/configuration digests remain
unchanged. The 44 deferred long cases remain deferred.

The original 32-repeat/eightfold E/r/slope envelopes are retained. FP64 component
probes pass; FP32 probes fail numerical qualification. Published small-system
FP32 SUCCESS disagrees with the independent frozen-predictor residual gates in
3 CPU and 0 CUDA law cases. Complete native
component snapshots and block goals are retained. These are numerical failures,
not xfails or permission to loosen gates; no oracle correction or fallback is
introduced. The coupled component oracle freezes native predictors; articulation
prediction/history is checked separately. This does not substitute for T5-C's
independent full-scene history audit.

The two original PR1 coordinate-failure inputs and 72 PR2 q/pose/history probes
reproduce their precision negative controls without executing retired solvers.
PR1 uses an explicitly identified mass-only scale, not the retired solver's
assembled-B merit. T5-B is complete; FP32 remains unqualified and T5-C/D remain
pending. Full pre-commit evidence and commit binding accompany the stage receipt.

Evidence: `agents/integration/artifacts/p4/precision-comparison/t5-b-v1/`.
Component receipt SHA-256: `ab2b672c60e31895ec5c79c0acea0f70cbad68b6bc7ebf673f6f9a1ee15f63a3`.
Full matrix summary SHA-256: `7719e76c957798bdf802a924dd1ed5d931c027144d6502092489b5a6e1d1add1`.


## T5-C: ten existing configurations and independent FP32 audit

Apply the user's scope change: nine Franka globalization/law combinations and
one Sharpa OR/C-IE, production/graph. Stop supplemental Close runs and retain
those files as OUT_OF_SCOPE. Reuse all ten T5-B FP64 trajectories after checking
production source hashes, exact initial states, configuration digests and
control schedules. No new FP64 trajectory is launched for these ten cases.
Franka controls are regenerated from the recorded pose plan and unchanged
controller source; Sharpa controls match the stored per-frame arrays directly.

Reuse the user's two matching FP32 runs and execute the other eight Franka
configurations. Every case terminates at its first hard failure: Franka on step
3 or 4, Sharpa on step 1. Across the ten original native runs there are zero
CONVERGED, 22 Stopped publications and ten verified atomic rollbacks.

Add a read-only observer and offline NumPy/SciPy audit. Preserve native step-start
histories, every evaluated candidate/trial, B32 directions/scales and actual
publication. Recompute FK, link/joint prediction/history, tet and contact E/r,
force/wrench, physical posterior and acceptance from the instrumented run's own
inputs. Qualify the audit with retained small-system math, energy differences,
independent natural splines and an end-to-end successful transaction test.
The instrumentation and plain runs share exact initial states and classifications,
but can differ numerically; report these differences without an enlarged gate.
Do not claim the supplemental observations reproduce the old FP32 run bitwise.

All ten first failures coincide with native composed-quaternion norm error above
the original 1e-10 guard (Franka 1.1920928955078125e-7; Sharpa
5.960464477539063e-8); independent geometry remains valid. Audit 1,024 candidate
observations and 259 physical postchecks. There are 51 native-accepted trials
rejected by independent acceptance recomputation. No scene-level false convergence
is observed because no scene step reports convergence. Stopped remains
unconverged. Physical safety, convergence agreement, prefix trajectory differences
and qualification are reported separately; FP32 qualification is FAIL.

Only same-time published prefixes are compared to FP64. Failed rolled-back frames
are retained separately, with no fabricated continuation or time shift. These
millisecond prefixes do not establish grasp accuracy, sustained contact quality
or speed. Sharpa grasp remains uncertified. T5-D is not performed.

Retained tests: 443 run, 436 pass, seven unchanged skips; four new IDs and none
removed. Full pre-commit and source bindings accompany the final receipt.
Production solver code, equations, numerical gates, controls and defaults remain
unchanged. Evidence: `agents/integration/artifacts/p4/precision-comparison/t5-c-v1/`.

## FP32 quaternion guard adjustment — 2026-09-29

FP32 unit-quaternion norm tolerance is now 1e-6; FP64 remains 1e-10.
Residual/acceptance tolerances, budgets and publication rules are unchanged.
FP32 configuration identity includes the new guard. Prior FP32 checkpoints
are rejected atomically. FP64 configuration fields remain unchanged.
Historical audit archives without guard metadata retain their original 1e-10.

Regression was RED on both CPU and CUDA before the fix (four failing subcases).
After the fix, contact/solver precision tests (7), audit/runner/checkpoint tests
(7, including one repeated checkpoint test), and prior-identity checkpoint test
(1) all passed. Full pre-commit passed.

All ten FP32 production/graph configurations completed 10 steps each:
2 CONVERGED, 98 STOPPED, 0 HARD_FAILURE. Every receipt is PARTIAL.
The two native convergence reports are from Sharpa and have not been independently audited.
This clears the earlier quaternion failures at steps 1–4, but does not establish
full-trajectory qualification. No FP64 trajectories or Close cases were run.
Prior T5-C evidence is retained unchanged. T5-D remains unperformed.

Evidence: agents/integration/artifacts/p4/precision-comparison/fp32-quaternion-v1/.

## Sharpa deleted asset rebuild — 2026-09-29

Rebuilt the hand assets with prepare_sharpa_contact at 964da0d0 using the
unchanged source URDF/meshes and integration environment. All 11 meshes pass
the 128-resolution geometry checks; four CPU/CUDA asset tests pass.

Compared with the 101 frozen paths: 45 match byte-for-byte, 38 differ and
18 historical cache files were not produced by the current preparation recipe.
The two marching-cubes repairs (left_MP and left_thumb_MC) do not reproduce
original mesh bytes. Cache files also contain new creation timestamps.
The original manifest was recovered exactly from an embedded historical copy,
and is saved separately; its old timings/results are not assigned to new assets.
No complete original mesh backup was found in the searched local artifacts.

A real model built from the new assets has different texture identity and is
rejected by the old qualified smooth SDF certificate. Do not overwrite frozen
hashes or use that certificate for new textures. Rebuilt assets remain in this
report directory's hand/ folder; original baseline manifests are unchanged.

To resume Sharpa, regenerate and qualify smooth SDF against these new textures,
then establish a new Sharpa FP64 reference before comparing FP32. This requires
a change to the prior frozen-asset/baseline scope. Franka assets are unchanged;
its runner is currently also blocked by the shared all-input validation.
No long trajectories have been rerun.

Evidence: agents/integration/artifacts/p4/sharpa-asset-recovery-v1/.

## Rebuilt Sharpa qualification and runner binding — 2026-09-29

User authorized reconstruction, new smooth SDF certification and one replacement
Sharpa FP64 reference. Restore the rebuilt hand at its runtime path, preserving
old frozen manifests. Eleven fields pass both C1/C2 geometry checks and all
independent oracle, continuity and expanded frozen-probe checks. Create a new
409-file manifest and rebind unchanged control bytes with parent provenance.

Add --input-manifest to the precision runner; preserve fail-closed hash checks,
record the selected manifest and resolve its qualified SDF directory. Propagate
the manifest into subprocesses and observer replay. Two new tests fail before
the change; eight runner/audit tests pass afterward. Production code is unchanged.
The single Sharpa FP64 baseline is pending; no Franka FP64 trajectories rerun.
Evidence: agents/integration/artifacts/p4/sharpa-asset-recovery-v1/.

## Rebuilt Sharpa FP64 reference completed — 2026-09-29

User-authorized continuation after deleted Sharpa hand assets were rebuilt.
Implementation commit: 4a93478b. Production solver code and numerical gates
are unchanged in this continuation.

- Rebuilt hand assets restored at p1/pr6d-assets/hand; old manifests preserved.
- Eleven C1/C2 fields pass new geometry, independent oracle, continuity and
  expanded frozen-query certification. Current source textures bind successfully.
- New 409-file frozen-inputs.json includes rebuilt hand assets, qualified SDF
  and unchanged remaining inputs. Control streams are byte-identical to T5-A3;
  controls/manifest.json records the explicit rebind and parent provenance.
- Runner --input-manifest selects this generation and propagates it to workers
  and observer replay. Historical inputs still fail closed. Two new regressions
  fail before implementation; eight runner/audit tests pass afterward.
- New Sharpa FP64 OR/C-IE production/graph reference completes 8667 steps:
  8637 CONVERGED, 30 STOPPED, zero hard failures. Complete finite states,
  histories, force generations and independent tet determinants pass validation;
  minimum det F = 0.5900113022643424. Final checkpoint restores exactly;
  old-asset checkpoint is rejected without modifying the target state.
- Initial state arrays equal the old Sharpa reference except config digest.
  This is a new physical input generation, not a claim of old-asset equivalence.
- Nine Franka FP64 references are reused with checked receipts/control hashes;
  zero new Franka trajectory steps. fp64-reference-index.json identifies all ten.
- Sharpa FP32 ten-step smoke completes without hard failure (PARTIAL).
  Full FP32 qualification and grasp success remain unclaimed; no full independent
  residual audit of this new FP64 trajectory was performed.

Use bash run-fp32.sh from this directory to run all ten FP32 trajectories on
this generation. It creates a fresh timestamped output directory and never
reruns FP64. The original failed run and historical T5-C evidence remain intact.

Full pre-commit passed. Integration is unchanged at 39f80c48. No push.

Evidence: agents/integration/artifacts/p4/sharpa-asset-recovery-v1/REQUALIFICATION.md.

## Explicit FP32 tolerance experiment implementation — 2026-09-29

Add opt-in fp32_absolute_tolerance to SolverMonolithic and the precision runner.
The current experiment sets global/q/x absolute scaled RMS residual goals to
1e-6, preserving relative goals, acceptance predicates, geometry safety, budgets
and publication. Defaults and their checkpoint identities stay unchanged;
override values participate in checkpoint identity and FP64 rejects the option.
Derived posterior caps track the selected goals as before.

Regression reproduced before implementation; 20 interface/precision/contact
checks pass after implementation. All ten production/graph configurations
complete ten-step smoke prefixes without hard failure. Full trajectories are
pending. Evidence: agents/integration/artifacts/p4/precision-comparison/fp32-relaxed-v1/.

## FP32 relaxed full trajectory experiment completed

At cd9a84c4, all ten FP32 production/graph cases complete their original lengths
with explicit absolute global/q/x goals of 1e-6. Total 55,917 steps: 8,624 native
convergence reports, 47,293 Stopped, zero hard failures. Franka's nine cases have
four converged steps and 47,246 Stopped; Sharpa has 8,620 converged and 47 Stopped.
Trial acceptance, relative goals, physical gates, budgets and defaults unchanged.

All stored states are finite; history bindings and force validity/generations
pass. Same-time full trajectory comparison uses nine retained Franka references
and the rebuilt-asset Sharpa reference. Max particle coordinate-component error:
Franka 0.237–0.878 mm across cases; Sharpa 0.294 mm. No FP64 reruns.
Complete advancement is not convergence or accuracy certification. Independent
full residual auditing and grasp certification remain unperformed; qualification
is NOT_AUDITED. Stopped is never counted as converged. T5-D remains unperformed.

Evidence: agents/integration/artifacts/p4/precision-comparison/fp32-relaxed-v1/RESULTS.md,
analysis.json and receipt.json. Twenty affected tests passed before execution.

## T5-D default Franka measurement harness — 2026-09-29

User narrowed T5-D to Franka OR/C-IE production/graph only. Compare default
FP64 with the explicit FP32 absolute-goal 1e-6 experiment, three fresh processes
per precision, full 5250 steps. No other scenes or new full recordings.
The harness verifies frozen inputs/controls, warms 20 steps, restores the exact
initial checkpoint, synchronizes solve boundaries, and retains final state and
lightweight statistics. Warmup does not reset the solver's monotone generation;
final force generation is checked against that offset. Compilation/cache loads,
SDF construction/conversion, warm restore, solve, final I/O and graph counters
are separately observable. Tests reproduce the absent harness before addition;
runner unit checks and 25-step FP32 smoke pass, complete pre-commit passes.
Smoke reveals objective graph recapture on every measured step; capture-free
steady-state cost must not be claimed or obtained by subtracting async host
capture timings. Formal measurements are pending. No production code changed.
Evidence: agents/integration/artifacts/p4/precision-comparison/t5-d-franka-v1/.

## T5-D scoped Franka experiment completed — 2026-09-30

Measured e7e18416: Franka OR/C-IE production/graph, three fresh processes per
precision, 5250 steps each. All 31,500 steps published without hard failure.
FP64 solve seconds: 430.241 / 251.724 / 251.975 (median 251.975), every step
converged. FP32 with explicit absolute goals 1e-6: 642.944 / 573.110 / 583.445
(median 583.445), respectively 0 / 1 / 0 converged steps; all others Stopped.
The observed FP32 wall cost is 2.32 times FP64, not a qualified speedup ratio.
CUDA pool peaks including construction are about 527 / 594 MiB. FP32 retains
FP64 SDF reference coefficients and their identity copy, so its total memory
is not halved. Objective graph recapture persists after warmup in both modes.

Source, asset, control, config, SDF and initial-state identities agree across
repetitions. Maximum pairwise final particle-coordinate difference is 2.48e-12 m
for FP64 and 0.342 mm for FP32. FP32 iteration paths also differ; root cause
has not been isolated. First-round clocks/timing differed markedly; cold compile
is separately recorded and cannot explain all solve-time variation. Preserve
all samples; do not present this as a fixed-clock hardware benchmark.

Prior full recordings supply separate position/force comparisons, not numerical
qualification of the new performance trajectories. Complete independent FP32
residual auditing remains unperformed. Default FP64 and production code are
unchanged. Only the user-scoped T5-D cost experiment is complete, not FP32
qualification or a renewed ten-case full regression. Eight runner/summary tests
and complete pre-commit pass; integration remains unchanged; no push.

Evidence: agents/integration/artifacts/p4/precision-comparison/t5-d-franka-v1/RESULTS.md,
performance-analysis.json, trajectory-analysis.json and receipt.json.

## P4 acceptance documentation closeout — 2026-09-30

At the user's request, reuse existing long trajectories and phase receipts; do
not run tests again. Consolidate 27 phase/experiment receipts in an evidence
index, preserving tested-source labels and SHA-256. Deliver p4_acceptance.md,
a machine-readable final receipt and snapshots of the four external documents.
Update the plan checklist, regression plan, Source Map and HANDOFF.

Historical FP64 equivalence/dev/full remain PASS at their recorded versions.
T5 experimentation is delivered; original FP32 qualification remains FAIL and
the relaxed experiment remains NOT_AUDITED. Current run_p4 and the Sharpa scene
still bind old assets (R1 OPEN, not waived). This documentation-only request does
not fix or waive R1 and does not certify a new current-source full run. No solver,
runner, test, asset or trajectory is changed. No simulations or test discovery
are executed. Static documentation checks are recorded separately.

Evidence: agents/integration/artifacts/p4/final-acceptance-v1/RESULTS.md,
evidence-index.json, receipt.json and commit-binding.json. No push.

## R1: retained runner and rebuilt Sharpa binding (2026-09-30)

Formal dev/full now reads the rebuilt manifest and the ten retained FP64 references directly from `asset/monolithic`. Both scene builders use retained inputs; the precision runner defaults to the rebuilt manifest. Original manifests remain byte-identical and historical observation fallback remains on the original generation.

Verify input, reference receipt/chunk and control identities before stepping; reject changed physical configuration or constructed controls. Read native Sharpa blocks without rewriting the stored reference, aligning the initial state, publication labels and frozen control frames. Keep Franka's frozen envelopes; use exact comparison for rebuilt Sharpa because no new repeatability envelope was qualified.

File integrity: 409 frozen inputs and all ten retained references passed. The rebuilt manifest SHA-256 is `d66d567014570ed7275116cbdcf5c52e302c71969c51554ea25e5118e3e875fb`. Evidence: `agents/integration/artifacts/history/runner-asset-binding-v1/file-integrity.json`. Added focused regression checks, **NOT_RUN** per user instruction; no GPU construction, simulation, unit tests or discovery. Static pre-commit checks are separate from numerical acceptance. R1 implementation is complete; runtime acceptance is pending, and no historical PASS is rebound to this commit.

## Archived handoff banners (2026-09-30)

These dated notices were moved from the former workspace handoff. They describe
their original source versions, not acceptance of the current checkout. Original
bytes and paths are retained in the workspace consolidation archive. Retired
evidence links can be resolved through the history archives and relocation index.

> **R1后续修复（2026-09-30）：** runner/场景现直接绑定worktree的重建Sharpa清单、资格目录和十条FP64参考，原生Sharpa记录只读适配。409项输入和十参考文件身份核对通过；Franka沿用冻结包络，重建Sharpa严格比较。代码修复已完成，运行验收待执行；按要求未重跑测试或轨迹，不能据此宣称full通过。下文旧OPEN描述为历史状态。

> 2026-09-30 存储整理：保留Franka九条＋Sharpa一条完整FP64参考，关键资产位于 `worktrees/Monolithic-refactor/asset/monolithic/`。其余大轨迹、日志与大JSON已按用户要求清理；小型原证据压入 `agents/integration/artifacts/history/evidence.tar.gz`。详见[轻量开发历史](development_history.md)。未重跑测试，R1未关闭；旧证据链接请按历史原路径在压缩包内定位。

> **P4验收文件收口（2026-09-30）：** 按用户要求复用既有长轨迹和阶段回执，不再运行测试。FP64重构/历史dev/full已有通过证据，T5实验已交付；FP32未获数值资格，默认仍FP64。总验收报告和27项回执索引已整理。正式dev/full的旧Sharpa资产绑定为开放项R1，未修复、未豁免，因此本次是文档交付完成，不是当前入口无保留PASS。见[总验收报告](p4_acceptance.md)。

下方阶段回执中的“待开始/未执行”描述保留其历史时点；当前结论以总验收报告为准。本批没有新的数值测试结果，也没有将旧结果重标为当前源码通过。


> **T5-D收口实验完成（2026-09-30）：** 仅Franka OR＋C-IE＋production/graph，两种精度各三次5250步，共31500步、0硬失败。FP64默认：三次均5250步收敛，求解中位数251.975s；FP32显式绝对阈值1e-6：三次收敛0/1/0步，中位数583.445s，耗时约为FP64的2.32倍。CUDA池峰值约527/594MiB。各次源码、输入一致，同精度三次初态一致；FP32重复终态粒子坐标最大分量差0.342mm，根因未定位。首轮环境频率变化、持续graph重捕获及精度质量不同均记录，不宣称同精度加速或FP32资格通过。未改生产代码、默认配置或扩大场景；受测源码 `e7e18416`。见[报告](../../../../agents/integration/artifacts/p4/precision-comparison/t5-d-franka-v1/RESULTS.md)。


> **FP32放宽绝对收敛阈值实验完成：** 实现cd9a84c4，显式global/q/x绝对阈值1e-6，相对阈值、三接受判据、几何检查、预算及默认配置未改。十条完整55917步：8624步原生报告收敛、47293 Stopped、0硬失败。Franka九条仅4步收敛；Sharpa8620步收敛/47 Stopped。全状态有限性、历史和力代次检查通过；粒子坐标最大分量差Franka约0.237–0.878mm、Sharpa0.294mm。未重跑FP64；原生COMPLETE_NOT_AUDITED保留，完整独立残差审计及抓取认证尚未执行。见[报告](../../../../agents/integration/artifacts/p4/precision-comparison/fp32-relaxed-v1/RESULTS.md)。

> **Sharpa重建资产重新认证与FP64对照完成（2026-09-29）：** 实现提交4a93478b；11组C1/C2几何、独立导数、连续性、扩展探针全部通过。新409文件清单绑定新资产，控制轨迹字节不变。Sharpa OR/C-IE production/graph完整8667步：8637收敛、30 Stopped、0硬失败；有限性、历史/力代次、独立tet行列式、checkpoint精确恢复和旧资产身份拒绝均通过。Franka九条复用，未新增推进；Sharpa FP32十步短验证无硬失败，完整FP32尚未认证。后续使用sharpa-asset-recovery-v1/run-fp32.sh，旧命令默认旧冻结清单不能用于本轮。见[报告](../../../../agents/integration/artifacts/p4/sharpa-asset-recovery-v1/REQUALIFICATION.md)。

> **误删资产恢复（2026-09-29）：** 重建Sharpa手部资产完成，128分辨率11/11通过、CPU/CUDA检查4/4通过。原101路径中45哈希一致、38不同、18历史缓存未由现配方生成。两项marching-cubes修复不能逐位重建，实际模型纹理身份与旧smooth SDF证书不符；旧证书正确拒绝绑定。新资产保存在p4/sharpa-asset-recovery-v1/hand，未替换冻结基线。继续Sharpa需要重新认证smooth SDF及建立新的Sharpa FP64对照；Franka资产未变。见[恢复报告](../../../../agents/integration/artifacts/p4/sharpa-asset-recovery-v1/RESULTS.md)。

> **FP32四元数门槛调整（2026-09-29）：** 按用户确认，单位四元数范数容差改为FP32 1e-6、FP64保持1e-10；残差、接受条件、预算不变。新FP32配置身份拒绝旧checkpoint，失败原子性有测试。十个production/graph配置各10步均推进完成：2步原生报告收敛（Sharpa，尚未独立审计）、98 Stopped、0硬失败，全部PARTIAL；解除原第1–4步几何阻塞，不构成完整轨迹认证。未重跑FP64或Close。历史T5-C结论保留，T5-D未执行。见[报告](../../../../agents/integration/artifacts/p4/precision-comparison/fp32-quaternion-v1/RESULTS.md)。

> **P4 T5-C完成，阶段提交 `304dfded`（未push）：** 按用户调整，仅对照已有十条配置（Franka三接受条件×三law、Sharpa OR＋C-IE），production/graph。十条FP64经初态、配置、控制与源码核对后复用T5-B，未重新推进这些长轨迹；Close补跑已停止并记OUT_OF_SCOPE。FP32复用用户两条、补跑八条，合计0收敛、22 Stopped、10次几何硬失败且完整回滚。Franka首失败在第3/4步，Sharpa第1步；组合四元数范数误差分别约1.19e-7/5.96e-8，超过原1e-10门槛，独立几何仍有效。只读补录和独立离线审计覆盖1024次候选与259次方向后验，发现51次原生接受而独立判据拒绝。原生与插桩结果可有舍入差异，各自保存，不冒充逐位复现。443项检查436通过/7既有跳过，新增4个ID无丢失；discovery 7878项/0错误，pre-commit通过。**FP32资格FAIL；没有完成抓取轨迹，不能据此前缀评价完整抓取或加速。** 生产公式/门槛/默认FP64不变，T5-D未执行。见[报告](../../../../agents/integration/artifacts/p4/precision-comparison/t5-c-v1/RESULTS.md)、[回执](../../../../agents/integration/artifacts/p4/precision-comparison/t5-c-v1/receipt.json)。

> **T5-C范围调整（用户确认，历史记录）：** 取消Close对照；仅比较已有T5-B十条配置：Franka三接受条件×三law共九条，以及Sharpa OR＋C-IE一条，均为production/graph。FP64经身份核对复用T5-B十条，不重新推进长轨迹。FP32复用用户已运行的两条匹配结果，其余八条补跑；必要时只读补录失败前短前缀。已停止Close补跑，已生成数据保留为OUT_OF_SCOPE，不计入本轮验收。T5-C尚未完成，T5-D尚未执行。


> **P4 T5-B完成，阶段提交 `96a9466f`（未push）：** 生产 solver、方程、默认FP64、预算和数值门槛均未改。双精度真实trial、发布/回滚、checkpoint损坏拒绝、接触及graph身份检查完成；独立组件oracle先提升冻结数值输入至FP64。439项检查432通过/7既有跳过，新增13个ID无丢失。新FP64 full十条全部通过：55,917步，55,830收敛、87 Stopped、0硬失败；dev复用其中默认两条，27组小矩阵与两场景重放通过。SDF/物理身份保持，discovery7874项/0导入错误、pre-commit通过。**FP32原数值包络未通过，CPU小型耦合三law观察到3个组件假收敛案例；相同CUDA fixture为0，仍不构成FP32资格认证。** 保留原生快照和负对照，不放宽门槛、不加fallback。Sharpa抓取仍未认证，44条延期长轨迹未执行；T5-C完整独立历史审计及T5-D性能实验待做。开发树与integration干净。见[完成报告](../../../../agents/integration/artifacts/p4/precision-comparison/t5-b-v1/RESULTS.md)、[受验源码回执](../../../../agents/integration/artifacts/p4/precision-comparison/t5-b-v1/receipt.json)。

> **P4 T5-A完成，阶段提交 `4866e1c2`（未push）：** 公开不可热切换的 `SolverMonolithic.Precision` 支持默认FP64与实验FP32；接触/SDF、完整q+x状态/历史/力、主机接受和物理后验随精度选择。Schema 3增加精度身份，旧Schema 2 FP64检查点先按原完整契约验证再迁移；跨精度续算/恢复拒绝。冻结四套完整FP32控制流并交付16-case串行runner。426项检查419通过/7既有跳过，新增5项无丢失，pre-commit通过；FP64单步26个物理/历史数组逐值一致，Franka/Sharpa物理digest不变。16条两步检查中：8条FP64均收敛；FP32 Franka为2步Stopped，其余7条首步因几何检查拒绝且逐数组验证回滚。FP32组合四元数范数误差最高 `1.19e-7`，原门槛 `1e-10` 未放宽。**T5-A完成不等于FP32数值资格通过；T5-B/C/D及新FP64完整场景验收仍待执行。** B/PCG数组/方向保持FP32，既有线性标量阈值/重启检查在两模式保留FP64，已明确列入dtype边界。见[完成报告](../../../../agents/integration/artifacts/p4/precision-comparison/t5-a3-v1/RESULTS.md)、[精度契约](p5_precision_contract.md)。

> **P4 T5-A第二批关节组件完成，提交 `56b4c77c`，T5-A仍在进行：** FK/COM/Jacobian、关节/link E/r、predictor及积分历史采用同一公式源的FP64/FP32特化，私有关节状态浮点字段随精度分配；主机归一化、惯性和特征值检查也使用对应NumPy精度。FP64原schema/摘要不变，FP32组件身份独立；公开solver仍为FP64-only，B/PCG及接受门槛不变。七组fixture模型的CPU/CUDA组件探针与独立oracle通过，两精度graph同时存活并正确重放；14组FP64探针的378个数组与改动前逐值一致。最终421项检查414通过/7既有跳过，新增5个ID无丢失，修正测试未使用变量后全量复验及pre-commit通过。本批未跑长轨迹；接触/SDF、完整耦合状态/步进、主机接受判断、公开precision/生命周期迁移、逐步控制哈希及runner仍待完成，T5-B/C/D未开始。未push。见[本批报告](../../../../agents/integration/artifacts/p4/precision-comparison/t5-a2-v1/RESULTS.md)。

> **P4 T5已启动，T5-A首批组件提交 `0476bf91`：** 冻结396个输入哈希、92个源文件及725处dtype/主机决策位置；tet E/r和材料能量/应力采用同一公式源的FP64/FP32特化，默认FP64，B/PCG及生产门槛不变。416项组件检查409通过/7既有跳过，新增5个ID、无丢失；16组FP64 fixture共64个数组与固定基线逐值一致，双精度CUDA graph重放和pre-commit通过。公开solver仍为FP64-only；T5-A的FK/history、contact/SDF、主机接受判断、状态/身份、逐步控制哈希及runner尚未完成，T5-B/C/D未开始。本批未跑长轨迹，不把旧T4标签沿用为修改后验收。证据见[本批报告](../../../../agents/integration/artifacts/p4/precision-comparison/t5-a-v1/RESULTS.md)，实现边界见[精度契约](p5_precision_contract.md)。未push。

> **P4 T3退役收尾完成：** `56d054ae`已快照/哈希归档78份原文件，移除54个Python文件（6个旧solver、11个旧示例/辅助、37个专属驱动）和4篇旧指南；无Legacy导出，独立collision、oracle、资产及共享模型/控制保留。修正cube验证驱动先构造旧solver的问题，新增CPU/CUDA两步检查。17项提取一致性通过，最终411项检查404通过/7既有跳过，discovery7846项/0错误且原ID无丢失，pre-commit通过。活动代码无已退役模块导入，旧示例/调度和推荐命令已清理；历史文档及冻结来源哈希保留并标注。T4原十条录制保留原受验版本；236个实际运行文件均保留、可执行AST一致，本批未重跑长轨迹。T3已闭环，T4数值结论有来源绑定，下一阶段T5尚未启动。生产方程/预算/资产不变，开发树与integration干净，未push。见[收尾报告](../../../../agents/integration/artifacts/p4/t3-closeout-v1/RESULTS.md)和[逐文件清单](../../../../agents/integration/artifacts/p4/t3-closeout-v1/retirement-manifest.json)。

> **2026-09-29 P4 T4数值验收完成：** `19b43e89`记录受验源码`9ff616c5`的结果，仅提交验收记录，生产代码不变。重构等价性、dev和临时十项full均PASS；55,917步中55,830收敛、87 Stopped、0硬失败。Franka九条均通过原抓取指标；Sharpa为8637收敛＋30 Stopped，抓取仍未认证。409项保留检查402通过/7既有跳过，27组三law×三接受条件×三CUDA模式全部通过；checkpoint/力/回滚/graph、5251/8668帧重放、SDF与物理身份一致性和pre-commit通过。discovery7844项/0错误。dev引用本轮full默认两条，未重复录制；新增证据约2.3GB。T3旧源码/诊断驱动物理退役仍未收尾，T5未启动，另44条长轨迹延期。开发树与integration干净，未push。见[验收报告](../../../../agents/integration/artifacts/p4/t4-v1/RESULTS.md)和[回执](../../../../agents/integration/artifacts/p4/t4-v1/receipt.json)。

> **2026-09-29 P4 T3-3d完成：** `9ff616c5`按用户要求审计并退役deprecated测试：19个整文件、9个混合文件中的旧BE/xi摩擦/校准/诊断守护；35份原文件与SHA-256已归档，269个ID退役、2个迁移/重命名。保留当前q+x、共享数学、独立collision/AABB、资产和oracle检查，以及当前API拒绝旧输入的边界。408项回归401通过/7既有跳过，随后容量断言单独1项通过；真实Sharpa资产检查已执行。discovery为7844项/0错误，完整pre-commit通过。生产实现未改，开发树与integration干净，未push。T3-3整体仍有旧源码/驱动退役工作，T4 full尚未执行。见[测试ID映射](../../../../agents/integration/artifacts/p4/t3-3d-v1/test-id-mapping.json)和[回执](../../../../agents/integration/artifacts/p4/t3-3d-v1/receipt.json)。

> **2026-09-29 P4 T3-3c完成：** 用户明确确认剩余5个测试守护的开发内容已退役，因此按整文件退役，不再逐条迁移其断言。`9ab8853d`在保存原文件、SHA-256与79个基线测试ID后移除5个活动测试，清理P2调度的2处引用和旧文档命令。全仓discovery为8113项、0导入错误；20项调度/runner测试和完整pre-commit通过。这里只确认导入与定向检查，不称全仓测试通过。T3-3整体仍有旧源码/诊断驱动和示例调度清理，T4 full尚未执行。生产求解逻辑未改，开发树与integration干净，未push。见[退役清单](../../../../agents/integration/artifacts/p4/t3-3c-v1/retirement-manifest.json)和[回执](../../../../agents/integration/artifacts/p4/t3-3c-v1/receipt.json)。

> **2026-09-29 P4 T3-3b完成，T3-3整体仍未完成：** `8bd6797e`迁移独立graph缓存/线性装配与公开q+x事务测试，10个旧graph测试ID中7个迁移、3个旧路径专属测试先快照后退役；cube工具改用共享证据函数，RSS辅助定义AST不变。116项检查113通过、3跳过（2项未设置真实资产环境变量、1既有CPU texture限制），完整pre-commit通过。discovery目前8118项/5个导入错误，剩余P2 runner/measurement、旧基础事务、calibration和release profiler；不能据此声称全仓通过或旧GraspCase可运行。6份原源码及SHA-256已归档，生产逻辑/冻结输入不变，未删除源码文件。后续继续剩余断言映射、源码/驱动退役及实际调度清理；T4 full未执行，integration干净，未push。见[T3-3b回执](../../../../agents/integration/artifacts/p4/t3-3b-v1/receipt.json)。

> **2026-09-29 P4 T3-3a完成，T3-3整体仍未完成：** `c71854b6`将8个混合测试文件中的有效断言迁至公开q+x与共享组件，177个测试ID完成映射；从旧诊断驱动抽出7个AST一致的线性辅助定义。196项检查195通过、1既有CPU graph跳过；最终命名/循环清理后44项复验通过，完整pre-commit通过。11个测试模块独立导入均来自开发树且不加载六个旧solver模块。13份原文件及SHA-256已归档，尚未删除活动源码；discovery由7954项/22个导入错误变为8105项/8个导入错误，不能视为全仓测试通过。下一步T3-3b继续剩余混合断言、历史入口和示例调度迁移，依赖闭合后归档删除。T4 full未执行，production公式/预算未改，integration干净，未push。见[T3-3a回执](../../../../agents/integration/artifacts/p4/t3-3-v1/receipt-migration.json)。

> **2026-09-28 P4 T3-1/T3-2完成，T3尚未整体完成：** `c3da2755`统一smooth SDF构建；`836fbb40`迁移Franka/Sharpa场景、录制辅助并交付正式dev/full runner。35项T3-2定向检查与完整pre-commit通过；Franka5250步全部收敛并通过抓取保持，Sharpa8667步为8637收敛＋30 Stopped、无硬失败，抓取仍未认证。两场景完整状态录制均通过T0冻结容差、物理digest一致；重放检查5251/8668帧。实际源码哈希已核对至提交，开发树和integration均干净，未push。下一子任务T3-3：有效断言及测试ID迁移、快照/哈希归档、discovery与实际调度清理；尚未删除历史源码。T4十项full尚未执行。见[T3-2回执](../../../../agents/integration/artifacts/p4/t3-2-v1/receipt.json)与[退役前审计](../../../../agents/integration/artifacts/p4/t3-3-v1/RETIREMENT_GATE.md)。


> **2026-09-28 P4 T3进行中：** T3-1统一smooth SDF构建已提交`c3da2755`，公开`build_smooth_sdf`与`SmoothSDFSet`；Franka采样、Sharpa冻结资格两路径已收口，身份与系数哈希精确一致。23项检查21通过、2既有CPU跳过。T3-2正在迁移场景和正式dev/full runner；T3-3退役迁移及最终dev/full尚未完成。构建对照发现/tmp启动会解析共享环境integration，已显式从开发树复验一致，见T3-1 receipt的来源说明。


> **2026-09-28 P4 T2完成（T2-2提交`24ff69fd`，未push）：** T2-1 `66dbb0a5`合并q+x事务并去掉q-only重复分配；T2-2公开SolverMonolithic直接继承SolverBase，提供完整状态、原子checkpoint和独立只读力快照，拒绝外部Contacts/调度/通用reset/模型重绑定。59项组件检查58通过、1既有跳过；完整pre-commit通过。Franka5250步全部收敛并通过抓取保持；Sharpa8667步8637收敛＋30 Stopped、无硬失败，抓取仍未认证。两场景完整录制通过T0冻结容差、标签/身份一致；公开构造及步进无旧solver/组件导入。下一阶段T3统一smooth SDF、场景/runner及测试退役，T4十项full尚未执行。见[T2回执](../../../../agents/integration/artifacts/p4/t2-2-v1/receipt.json)。

> **Sharpa输入注意：** dev必须显式传`--trajectory agents/integration/artifacts/p3/pr3/current-iterate-production-v1/sharpa-trajectory.json`（实际执行使用绝对路径）。脚本默认`trajectories/sharpa_cube_grasp.json`不是T0冻结输入；本轮一次误选已停止并标记排除，随后用正确输入完整重跑。下方进行中陈述为历史记录。


> **P4 T2进行中：** T2-1已提交`66dbb0a5`，完整q+x直接继承SolverBase，移除q-only重复分配；30项检查29通过/1既有跳过，CPU/CUDA三law三步全状态与固定基线逐位一致，原子回滚通过。T2-2公开入口/状态/checkpoint/力快照已实现，59项检查58通过/1既有跳过，完整pre-commit通过；Franka/Sharpa完整dev正在串行执行，未判定T2完成。证据位于`agents/integration/artifacts/p4/t2-1-v1/`与`t2-2-v1/`。


> **2026-09-28 P4 T1c完成（提交`b1bb31fb`，未push）：** P3接触组件已独立，补齐旧law/弹簧参数/xi历史拒绝与导入边界测试；仅修改生产模块说明，AST保持。组件门槛101项：95通过、6项既有CPU纹理/graph跳过，完整pre-commit通过。额外旧摩擦轨迹CUDA测试失败而固定基线复测通过，差异未定位、如实保留，不计入预定P3兼容范围；未放宽容差。旧父类传递依赖留T2，下一阶段T2合并求解事务。见[回执](../../../../agents/integration/artifacts/p4/t1c-v1/receipt.json)。


> **P4 T1b完成并提交：** `20200995`提取tet_common（质量/配置/验证）、tet_material（Kim/Smith/PSD）和tet_assembly（生产B与scatter），P3 E/r与B直接依赖共享模块；接受条件移入variational_iteration。38个定义/类型别名及acceptance AST未变。61项测试60通过、1原有CPU纹理跳过；新CPU/CUDA独立E/r/B覆盖两材料×两质量×固定/自由，完整pre-commit通过。测试源码与提交哈希一致，两个工作树干净，未push。初次新oracle重力输入精度不一致已修正测试输入，未放宽容差。旧tet residual-only入口及父类过渡引用保留到T2/T3，未删源码、未重跑长轨迹。下一阶段T1c contact边界清理。见[T1b回执](../../../../agents/integration/artifacts/p4/t1b-v1/receipt.json)。

> **P4 T1/T1a完成并逐阶段提交：** T1 `0841a923`提取中性类型/统计/校准配置/异常，并拆开solver与collision的按需导入；37项定向测试36通过、1原有CPU纹理跳过。T1a `0ffbcd97`提取joint_parameters，P3关节组件直接依赖共享布局/参数/目标检查，旧求值/发布留旧模块；48项测试47通过、1原有跳过。两阶段完整pre-commit与AST保持检查通过，测试源码哈希与提交一致。下一阶段T1b tet共享提取；父类间接旧依赖仍按计划留到T2，尚未归档删除源码、切换正式API或运行重构后长轨迹。两工作树干净，未push。证据见[T1回执](../../../../agents/integration/artifacts/p4/t1-v1/receipt.json)、[T1a回执](../../../../agents/integration/artifacts/p4/t1a-v1/receipt.json)。下方“solver未改/下一批T1”为历史状态。

> **P4 T0已补交：** `3e79a049`（`SamiulJ/monolithic-refactor`），包含基线工具、7项工具测试及仓库内阶段结果/提交规则。完整pre-commit通过，已验收工具哈希保持，两个工作树干净，未push。后续每阶段完成并验证后立即commit。见[提交回执](../../../../agents/integration/artifacts/p4/t0-v1/commit-receipt.json)。

> **P4提交规则更新（用户确认）：** 每阶段/独立子任务完成并验证后立即本地commit，再进入下一阶段；提交代码、测试和结果摘要，外部大证据按路径/哈希引用，不自动push。已完成T0a–T0c合并补交一次；后续逐阶段提交。

> **2026-09-28 P4 T0准入完成：** 九条Franka完整5250步全部通过，共47,193收敛＋57 Stopped、无硬失败；默认Sharpa8667步核验复用，抓取仍未认证。八项非默认临时同配置重跑校准通过，发布序列/身份一致，比较容差冻结；临时重复轨迹已清理，永久新增九条约1.9GiB。工具7测试、CPU/CUDA三law独立E/r/B和噪声检查、完整pre-commit通过；924源码/配置与396输入再次核验，integration保持干净。T0a/T0b/T0c完成，下一批T1；未修改solver、未commit/push。见[准入报告](../../../../agents/integration/artifacts/p4/t0-v1/RESULTS.md)。下方T0进行中陈述保留为历史。


> **2026-09-28 P4 T0开发进行中：** `Monolithic-refactor`新增私有基线驱动器、窄构造适配器、完整录制核验工具及7项工具测试；尚未修改solver。已冻结924份源码/配置与396份输入，实际发现8160项测试（728项Monolithic/导入相关），API契约/参数/依赖清单见[工作记录](../../../../agents/integration/artifacts/p4/t0-v1/API_CONTRACT.md)。Sharpa完整8668帧核验及27字段初态/配置/SDF身份复现通过，未推进新的Sharpa物理步。默认Franka5250步全部收敛、抓取通过，全状态对照通过冻结噪声包络；T0b已通过，其余八条Franka串行收集中，T0c尚未准入；不能据此进入T1。比较界限在查看新轨迹差异前依据已有P3记录冻结，非默认组合的非零轨迹容差仍待独立校准。

> **2026-09-28 P4临时full矩阵缩为10项（用户决定）：** CUDA production/graph下，Franka三接受条件×三law九条＋Sharpa默认OR/C-IE一条；总55,917步。T0计划新增九条Franka（含默认驱动对照），Sharpa核对身份/完整性后复用现有基线。原54项中的44条额外长轨迹延期，三执行模式组件测试和逐case门槛保留。T4以及T5后的FP64 full按10项验收，T5独立实验不变；不能用原P3 Sharpa替代重构后验收。见[P4 v11](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_plan_zh.md)和[回归v4](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_regression_plan_zh.md)。此条发布时尚未实现runner或开始新录制；最新进展见顶部。

> **2026-09-28 旧日志已按用户要求永久删除：** 在上一轮压缩之后，用户明确要求全部清掉旧日志及PR5虚拟环境。本轮删除2134份历史原始/压缩日志，约5.89 GiB；PR5环境与编译缓存此前已删除，本轮确认不存在。当前P3发布基线、提交记录、SDF geometry节点/证书、Franka缓存及P4证据保留，320份文件哈希核验通过。历史日志不再可从本地恢复，上一条归档恢复说明已失效；需要日志的旧脚本不能再按原命令直接复核。摘要报告、NPZ录制和源码保留，未重跑实验或修改solver。见[删除记录](../../../../agents/integration/artifacts/history/storage-cleanup/cleanup-20260928-delete-logs/README.md)。

> **2026-09-28 P4启动与空间清理：** 已从固定提交`39f80c48`创建`worktrees/Monolithic-refactor`（分支`SamiulJ/monolithic-refactor`），两工作树均干净；尚未实现基线runner或修改solver，T0a未完成。用户要求先清理agents空间：302份旧P2/P3大JSONL日志已逐文件解压哈希验证后无损压缩，移除原始副本；旧PR5虚拟环境/编译缓存已保存清单后删除。日志节省约6.46 GiB，当前P3基线录制、SDF资格/节点、资产缓存和P4证据保持。历史读取工具需要时先恢复原路径；详见[清理与恢复说明](../../../../agents/integration/artifacts/history/storage-cleanup/cleanup-20260928/README.md)及[核验结果](../../../../agents/integration/artifacts/history/storage-cleanup/cleanup-20260928/verification.json)。历史文档中“尚未创建worktree”的陈述已由本条更新。

> **2026-09-28 P4 Source Map已准备：** [代码导航](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_source_map_zh.md)绑定已推送Fork的P3基线`39f80c48`，覆盖当前公开旧入口→私有coupled工厂→逐级单继承链、真实q+x时间步、FP64状态/历史、物理/线性/graph组件、仍存旧依赖及P4迁移落点，并列出两个抓取案例、外部资产和测试/CI入口。已按源码符号、导入和链接静态核对；仅文档与索引交付，未修改源码、未重跑仿真、未创建refactor worktree。新agent先读P4计划与本图，再开展T0。

> **2026-09-28 P3基线已提交，P4开发worktree已明确：** `SamiulJ/monolithic-integration`新提交为`39f80c48da2022187823f8218131a249807597a0`（Finalize P3 grasp and publication baseline），包含已验收零接受发布策略和Franka/Sharpa抓取验证交付。920项源码哈希与验收版本一致，提交前后19项定向测试及完整pre-commit通过；integration工作树干净，未push。P4将在待创建的`SamiulJ/monolithic-refactor`分支、`/home/lightwheel/Desktop/newton/SamiulJ/worktrees/Monolithic-refactor`路径上，以该固定提交为基线开展；当前未创建、未开始重构、未删除旧worktree。外部资产/轨迹仍须T0单独冻结。见[P4 v8 §2](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_plan_zh.md)及[提交记录](../../../../agents/integration/artifacts/p3/pr3/baseline-commit-v1/README.md)。下方未提交状态为历史记录。

> **2026-09-28 P4 v7回归重设，尚未实施：** 按用户要求，T1a/T1b只验证组件直接依赖，T2后检查整个P3旧依赖链；Legacy及q-only/tet-only等过渡入口不再守护，不要求可跑。新回归以完整P3为中心：dev默认OR＋C-IE＋CUDA production/graph，含完整Franka/Sharpa；full为3接受条件×3law×3CUDA模式×2场景＝54条E2E，加P3组件/安全覆盖，替代旧Monolithic nightly。Sharpa仍不认证抓取成功，未跑组合不预记PASS。仅更新计划、回归设计与展开矩阵，未实现runner、未执行测试。见[P4 v7](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_plan_zh.md)及[新回归方案](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_regression_plan_zh.md)。

> **2026-09-28 P4计划v6：q-only退出范围已确认，其余过渡产物已盘点。** 用户明确不再保留q-only独立运行能力；新生产入口限定既有q+x，保留共享FK/目标/积分历史，PR2纯关节轨迹不再列为新入口认证。已列出tet-only、旧joint-diagnostic、私有coupled工厂/状态入口、单物理分支、实验覆盖与研究原型，并区分已批准退出和仅候选。tet-only历史入口尚未批准删除；PCG diagnostic、独立oracle等不随入口退役。仅更新规划，未改源码。见[P4计划v6](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_plan_zh.md)及[过渡产物清单](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_transition_inventory_zh.md)。

> **2026-09-28 P4计划v5新增T1c，尚未实施：** 按用户要求，旧切向弹簧历史摩擦、`tangential_stiffness`及xi历史协议列为deprecated，Legacy仍保留复现。已核对旧`contact.py`与P3 `ContactObjective`在几何、法向导数、law、精度及发布所有权上的差异；P3组件不直接导入旧contact，共享接触装配已在`linear.py`，不强行新增共享层或转换旧物理/状态。T1c位于T1b之后、T2之前；本轮仅规划，未修改代码或执行实验。见[P4计划§3.6与T1c](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_plan_zh.md)。

> **2026-09-28 P4计划v4新增T1b，尚未实施：** 用户批准tet共享提取与旧模块弃用方案。T1a之后、T2之前迁出配置/校验、材料和P3实际使用的FP32 B装配，并把通用acceptance迁至共享迭代工具；旧`tet.py`保留deprecated兼容入口。PR1的`tet_variational.py`与P3的`tet_precision.py`保留各自求值协议，数值公式、质量模型、精度和预算不变。T5范围不变。本轮仅更新计划与交接，未修改代码、未执行实验。见[P4计划§3.5与T1b](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_plan_zh.md)。

> **2026-09-28 P4计划v3新增T1a，尚未实施：** 按用户要求，在T1与T2之间抽出articulation共享布局校验、关节参数和目标快照，迁移新旧调用方；旧动力学及关节项求值留在`articulation.py`并标deprecated。新路径不得直接或间接依赖旧模块，共享逻辑不发弃用警告；旧文件保留，数值行为不变。T5精度对照范围不变。本轮仅更新计划与交接，未修改求解代码或执行实验。见[P4计划§3.4与T1a](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_plan_zh.md)。

> **2026-09-28 P4计划v2新增FP32/FP64对照，尚未实施：** 用户要求将P3当前FP64在线状态/计算/检查做成构造期可选FP32/FP64；B/PCG仍FP32，默认仍FP64。先T0–T4完成等价重构，再T5参数化并独立测精度/性能；包含FP64独立重算、完整Franka/Sharpa与Close、同工作量三次串行计时，失败不放宽门槛、不把提前终止当加速。仅更新计划，未修改代码、未运行实验、未commit/push。见[P4计划§7](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_plan_zh.md)。

> **2026-09-27 P4重构计划已建立，尚未实施：** 新P3统一入口拟命名`SolverMonolithic`，旧实现保留为`SolverMonolithicLegacy`；新实现直接继承SolverBase，通过组件组合支持既有q-only/q+x，不再对外区分articulation-only/coupled。保持P3已验收物理、精度、预算与current_iterate规则，PR1暂保留内部路径；先T0冻结dirty基线/API，再隔离Legacy、统一实现、迁移调用方和完整验收。本轮只写文档，未改代码或弃用标记，未commit/push。见[P4计划](../../../../dev/monolithic/P4/newton_monolithic_solver_p4_plan_zh.md)。

> **2026-09-27 零接受发布已接入核心production路径：** 新增私有显式`publish_policy="current_iterate"`，沿用配置身份/检查点保护；Franka和Sharpa入口直接选择，删除实例发布覆盖，strict/last_accepted旧行为不变。完整复跑Franka5250步全收敛，Sharpa8667步（8637收敛＋30Stopped，仅3470零接受），均无回滚；13919帧与原逐步标签核对通过。19定向测试通过；720广范围回归699通过/17原跳过/4既有校准错误，无新增失败；pre-commit与源码身份核对通过。P3已验收状态保持，未commit/push。见[生产接入报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_current_iterate_production_zh.md)。

> **2026-09-27 P3 已验收（用户确认范围）：** 用户已检查Franka回放，确认以Franka成功抓取和冻结Sharpa案例完成P3验收；明确采用允许零次Newton更新被接受时发布当前有效迭代点的规则，Stopped不算收敛，最终安全检查/回滚保留。Sharpa未成功抬升及其他未认证项转为已知限制/后续事项，不再阻塞本次验收。实例显式启用策略，核心默认未改，当前抓取文件尚未提交，未push。详见[P3验收报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_acceptance_zh.md)。

> **2026-09-27 Sharpa基准冻结＋Franka抓取示例完成：** 用户锁定Sharpa当前10mm/零接受发布结果，原完整证据保持，新增哈希清单与代码快照；姿态原因仍为未验证假设。新增Franka FR3动态7臂关节＋2指、原40mm软体cube；用户批准独立双指对称PD近似，mimic不扩展。18mm闭合首轮滑落保留，14mm闭合最终完整5250步全部收敛，无Stopped/回滚；最大COM升71.610mm，0.5s保持最小升70.686mm、地面支撑0N，放回/松开完成。5251帧、920源码/64资产、预算/状态核验通过，31测试通过/1原有跳过，pre-commit通过，示例/重放/截图齐备。核心solver默认未改，无commit/push。见[报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_franka_grasp_zh.md)。

> **2026-09-27 10mm抓取零接受Stopped对照完成：** 用户批准仅诊断取消“至少一次accepted”门槛，保留最终检查与原10/8/200。CUDA production＋graph/OR/C-IE从初态完成8667步：8637收敛＋30 Stopped，无回滚；仅3470步零接受发布预测初值，3475恢复收敛，未发布失败trial。cube最大COM升2.127mm、终态地面承重99.988%，仍未成功抓起。8668完整帧、发布generation、预算、916源码身份及非有限最终状态回滚核对通过，4项轨迹测试通过。含记录711.64s（12.18步/s），纯solve612.83s。生产默认未改，无commit/push。见[报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_recorded_grasp_zero_accepted_zh.md)与[重放](../../../../agents/integration/artifacts/p3/pr3/recorded-grasp-zero-accepted-v1/README.md)。

> **2026-09-27 10mm抓取第3470步失败诊断完成：** 原10/8/200预算重现零接受回滚，直接触发为食指弹性指腹与soft face61/key4950的硬朝向mask重新启用，增加约4.079e-6能量；C²距离/梯度在此连续，graph开关对照一致。切换alpha≈0.005994，小于原最小试探1/128；仅诊断增至12次回溯后，3470–3474共5步Stopped，3475–3480共6步Converged，无回滚。此为11步局部对照，未重跑完整抓取、未改变生产预算/接触规则。916源码身份及完整回滚核对通过；原失败保留，无commit/push。见[诊断报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_recorded_grasp_10mm_diagnosis_zh.md)。

> **2026-09-27 抓取初始下降5mm对照完成：** 经用户批准加入0.5s张手下降，后续闭手/抬升整体下移5mm，物理和production＋graph/OR/C-IE配置保持。8667步均发布，8666收敛＋1 Stopped（6.490s耗尽Newton10，下一步恢复），无回滚。cube最大COM上升1.601mm，高于无下降的0.916mm，但终态地面仍承重99.981%，未成功Lift。8668完整帧、916源码/191资产身份和15项测试通过；原输入及旧证据保留，生产solver未改。见[报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_recorded_grasp_approach_zh.md)及[重放/对照](../../../../agents/integration/artifacts/p3/pr3/recorded-grasp-approach-v1/README.md)。本轮未commit/push。

> **2026-09-27 新录制cube抓取首轮完成：** 基线8be8c9a7，按用户确认将40mm r2全动态软体cube置于(0,0,0.02)m、地面z=0，采用录制掌部初态与闭手—抬升目标。仅CUDA production＋graph、ArmijoOrResNorm、C-IE，沿用C²/mask/last-accepted。8167步全部收敛，零Stopped/回滚；但cube最大COM上升仅0.916mm、终态水平移22.748mm，地面仍承重99.951%，未实现有效Lift。8168完整帧及916源码/191资产核验通过，14项相关测试通过；已有生产solver未改。原文件缺少接近/高位Hold/Release；三网格/friction-off/连续精度仍未认证，不称5.3.3完成。见[首轮报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_recorded_grasp_zh.md)及[重放与证据](../../../../agents/integration/artifacts/p3/pr3/recorded-grasp-v1/README.md)。仅新增轨迹诊断/测试，本轮未commit/push。

> **2026-09-27 PR3生产候选接入与工程验收完成（按用户批准范围）：** C²＋朝向mask＋ArmijoOrResNorm＋仅最后accepted的Stopped发布已进入私有production路径；旧严格/guard默认保留，三law显式选择。18/24×三law×三执行模式共18组均从t=0完成2500步，合计44,739 Converged＋261 Stopped，无硬失败/回滚；主矩阵15006完整帧与全矩阵控制/预算/发布/源码核验通过。709项同进程回归688通过／17原有跳过／4既有校准错误，无新增未解释回归，AABB旧波动本轮未复现；tracked和untracked pre-commit及diff检查通过。913正式Python源码与191资产已核对；默认策略、物理、1ms、门槛及10/8/200预算不变。T5按新发布契约、T6按已知例外范围收口；严格逐步收敛、连续精度和一般接触覆盖仍未认证，旧失败记录不改写。详见[接入与验收报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_production_zh.md)及[可复现/重放证据](../../../../agents/integration/artifacts/p3/pr3/production-v1/README.md)。HEAD e83ca2b1，未commit/push。

> **2026-09-26 两个零接受失败的方向／能量判据诊断完成：** 18 C-EE/1345与24 C-IE/1419的原ResNorm均精确回滚；通过线性后验的两个方向各八次试探均E下降／残差上升。实际导数分解定位B省略的SDF曲率及C-IE摩擦n/N变化，同B FP64直接解不改变方向符号；更强正则存在残差下降方向但未通过原后验。Armijo与OR均越过原失败，四个CUDA production＋graph的100步窗口全部完成，Converged/Stopped分别92/8、91/9、74/26、93/7，无回滚；24 C-IE OR首步6次更新收敛，CEE OR未显示同样优势。C-IE能量导数与r·p不一致，此为私有启发式对照，非生产支持或物理认证；未重跑完整Close，T5仍未认证、T6未启动。1592源码／191资产和原预算门槛保持，生产默认未改，未commit/push。见[结果与适用边界](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_stopped_globalization_zh.md)及[证据核对](../../../../agents/integration/artifacts/p3/pr3/stopped-globalization-review-v1/README.md)。

> **2026-09-26 Stopped规则从初态Close与录制完成：** C²＋alignment mask＋最后accepted发布规则下，六组CUDA production＋graph串行从t=0尝试，4/6走完2.5s；18 C-EE与24 C-IE分别在1345/1419步零接受更新而回滚，末发布时间为1.344/1.418s。共12,762发布步＝12,696 Converged＋66 Stopped，连六初态保存12,768帧、每帧27完整状态字段；全帧/首末/回滚核验与六组末帧重放渲染通过。已提供时间轴拖动/逐帧/速度控制的重放脚本，原记录FP64，轨迹块约426.89MiB。1592仓库文件、191资产、Git index与旧adapter/policy未改；完整推进不代替物理精度及T5验收，T6未启动，生产默认不变，未commit/push。见[结果报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_stopped_close_zh.md)及[重放命令与证据](../../../../agents/integration/artifacts/p3/pr3/stopped-close-review-v1/README.md)。

> **2026-09-26 Stopped发布隔离诊断完成：** 在C²＋alignment mask的六个首失败完整输入上，以CUDA production＋graph最多各推进100步；六个首失败均发布最后accepted，之后均恢复过Converged。四组完成100步窗口，18 C-EE/1345和24 C-IE/1419因零接受更新再次完整回滚。合计448发布步＝405 Converged＋43 Stopped，最长连续Stopped为8，已发布最大global残差比约1417；只能证明部分阻塞可越过，累计物理精度与完整Close仍未认证。完整历史恢复、配置身份、NaN/退化tet/原guard反例及回滚检查通过。1592仓库文件、191资产和Git index未改，生产默认与门槛/预算不变，T5未认证、T6未启动，未commit/push。见[诊断报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_stopped_accept_diagnostic_zh.md)及[证据核对](../../../../agents/integration/artifacts/p3/pr3/stopped-accept-review-v1/README.md)。

> **2026-09-26 朝向与收敛根因双agent诊断完成：** 15个保存状态的432个软体面均朝外，无翻面；18主例由采样点经过原mesh凸棱换面造成同向，24主例为粗软体面跨过弯曲rigid表面，3求积点在mesh外但91个诊断点中19个在内部，存在已证实的局部重叠。另5个失败输入精确重放：3个慢收缩、2个真实残差上升方向，主要缺项为法向SDF曲率及摩擦n/N状态导数；保留原正则化的同B FP64直接解对照5/5结局不变。18/C-IE硬mask有限跳变仍独立存在。建议分别设计接触覆盖/连续选择与更一致的求解方向，本轮未实施新策略。1592仓库文件、191资产保持不变，生产默认三线性＋原guard不变，T5未认证、T6未启动，未commit/push。见[根因与后续方案](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_contact_root_causes_zh.md)及[证据汇总](../../../../agents/integration/artifacts/p3/pr3/failure-root-cause-v1/README.md)。

> **2026-09-26 SuperDex式朝向mask隔离诊断完成：** 以C²、阈值0、不淡出、按law取步初／当前碰撞体方向，屏蔽整项E/r/B。六个原首失败输入4/6单步通过；六条CUDA production＋graph从t=0尝试仍0/6完成，共8077成功发布步。18/C-EE与C-II推进至1302，18/C-IE仍1232；24/C-EE推进至1438、C-IE至1414、C-II仍1395。阻塞分为Newton预算与接受条件／线性后验；18/C-IE实测硬mask在近乎相同坐标下造成约0.0118N局部残差跳变。诊断位于独立证据目录，1592仓库文件及191原资产未改，默认三线性＋原guard保持，T5未认证、T6未启动，未commit/push。见[诊断报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_alignment_mask_diagnostic_zh.md)和[证据／复现](../../../../agents/integration/artifacts/p3/pr3/alignment-mask-v1/README.md)。

> **2026-09-26 C¹/C² SDF 对照完成，暂无合格候选：** 两条私有FP64路线均通过11资产几何不劣化、独立oracle、连续性、CPU/CUDA、接触E/r/B与原噪声界检查；各通过14/18当前失败输入和6/6较早径向失败输入。两候选×六场景×三次共36次串行Close尝试完成，三轮失败步一致，每候选每轮0/6完成：位置18三law在1232步、位置24/C-EE与C-IE在1412步触发原朝向guard，位置24/C-II在1395步耗尽Newton10。47,454个成功步的控制／后验／发布及全部失败回滚核对通过；两候选均未获得复核资格，不选择生产候选。700项同进程回归678通过／17跳过／5个既有失败，无新增失败；完整pre-commit通过，191资产及912项正式Python源码核对一致。默认三线性、guard、物理、1ms、停止门槛及预算不变，T5未认证、T6未启动，未commit/push。见[结果报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_smooth_sdf_results_zh.md)、[计划回执](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_smooth_sdf_plan_zh.md)及[可复现证据](../../../../agents/integration/artifacts/p3/pr3/smooth-sdf-v1/README.md)。

> **2026-09-24 径向修复后的两类阻塞已完成双 agent 诊断，T5 仍未通过：** 14 个 regularization 失败输入均精确复现，试探跨越三线性 SDF 单元后梯度/接触残差跳变；18 C-EE 格面两侧距离几乎连续，而法向残差发生有限跳跃。外部增至24次回溯仍在原Newton10预算内失败，不能据此只增加预算。另4个第1232步失败由合法步初的速度预测跨同一SDF格面，使key5355朝向由合法转为非法；没有NaN、面退化或盒外查询错误。仅换合法初始iterate仍卡同一域边界。此整体候选朝向guard来自PR3设计，区别于SuperDex的接触项alignment mask；SDF采样/外推规则未改。本轮仅诊断、生产源码与物理/门槛/预算不变，没有重跑完整Close或提交。下一步需设计一致的较光滑SDF表示，并独立评估几何合法域/接触选择语义；未证明这两项设计必然使整轨迹通过。见[残差与后验诊断](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_t5_resnorm_diagnosis_zh.md)、[几何合法域诊断](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_t5_geometry_diagnosis_zh.md)及[证据汇总](../../../../agents/integration/artifacts/p3/pr3/t5-followup-v2/README.md)。下方历史“根因待定位”应以本次诊断的明确范围更新，T6未启动。

> **2026-09-24 摩擦径向切线修复已实施，完整 Close 仍未通过：** 用户批准将等权切向近似改为冻结 n/N 的径向/横向两非负因子，保持势/残差、1 ms、物理参数、停止门槛和迭代预算不变。实际FP32 PCG的六个旧失败状态×三执行模式全部2次更新通过；新测试补齐零滑移投影消减与平滑边界。117项P3定向测试为113通过/4预期跳过；685项广范围回归为666通过/15跳过/4个既有校准错误，无新增失败，历史AABB波动本轮未复现；pre-commit与最终交付核对通过。原T5矩阵全部重跑：18条正式分支累计19,523成功步、0/18完成；新首失败为18 C-EE第1087步、18 C-IE production第1209步（diagnostic第1232步）、18 C-II第1232步、24 C-EE/C-IE第994步、24 C-II第990步。后续阻塞为ResNorm/线性后验拒绝或接触几何合法域拒绝，不能继续用原“径向慢收缩”概括。全部失败回滚、控制/发布核对通过；历史重置单步4/6通过。详见[修复报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_t5_radial_fix_zh.md)及[新证据](../../../../agents/integration/artifacts/p3/pr3/t5-radial-fix-v2/README.md)，完整回归和文件哈希已归档。T6未启动，未commit/push；此前诊断报告与旧T5回执保留历史范围。

> **2026-09-24 T5 双 agent 调查完成，正式 Close 仍未通过：** [Newton 诊断](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_t5_diagnosis_zh.md) 精确重放六组首失败状态；主两例分项差分与六组单步干预表明，冻结摩擦系数的 fitted 切线遗漏滑移径向导数，造成过大的径向刚度和慢收缩。主两例将同一 B 改用直接解仍失败；补径向项并用独立直接解时，六组均在原门槛/10 次预算内用 2 次更新收敛。这不是生产 FP32 PCG/graph 或完整 Close 验收；未修改 Newton 生产源码、参数、门槛或预算。[superDex 场景](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_t5_superdex_zh.md) 已导入两位置、全部 2,500 步目标控制、原初态与 33 个动态 link 惯量；原生公式仍有明确差异，两组均在首步因线性求解给出零方向而 STOPPED，且原生状态未回滚，不能宣称同方程或整段收敛对照。建议下一步单独设计两非负切向因子的生产实现与回归，不据单步结果放行 T6。原 1,585 项源码与十八组初态的独立核对见 [汇总入口](../../../../agents/integration/artifacts/p3/pr3/t5-followup-v1/README.md)。未 commit/push。

> **2026-09-23 PR3 T5 已尝试，完整 Close 未通过：** 六组旧步初诊断加十八条从 t=0 开始的正式分支全部执行至成功终点或首失败；正式分支 0/18 完成 2.5 s。三个 CUDA 执行模式的首失败步一致：位置 18 三 law 均为 812，位置 24 的 C-EE/C-IE 为 796、C-II 为 795，均耗尽 Newton 10 次。14,451 个成功步的冻结控制及发布 generation 一致；所有失败完整回滚。没有放宽物理/精度/步长/预算，生产 solver 未修改。113 项 P3 定向测试为 109 通过/4 跳过；681 项广范围回归为 662 通过/15 跳过/4 个既有校准失败，无新增失败，AABB 本轮通过但历史波动保留；完整 pre-commit 通过，1,585 项源码及 191 项原资产核对通过。见 [T5 报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_t5_zh.md) 和 [证据](../../../../agents/integration/artifacts/p3/pr3/t5-v1/manifest.json)。当前应定位固定预算下的外层收敛速度，T6 未启动；HEAD e83ca2b1，未 commit/push。下方 T4 的“下一任务 T5”为历史提示，不能据 T4 通过宣称整手 Close 通过。

> **2026-09-23 PR3 T4 完成（保留已知例外）：** C-EE/C-IE/C-II 的共同 ResNorm 关卡、三档 cube 校准、独立同离散参考与动态时间细化通过。165 组矩阵完成，参考 142,800 步全部成功，生产路径累计 213,481 个成功步；4 条纯 Armijo 失败保留，完整回滚核对通过。局部 96 组、tiny/cube 噪声 162/54 组通过；111 项定向检查 107 通过/4 跳过，679 项广范围回归 659 通过/15 跳过/5 既有失败，无新增失败；完整 pre-commit 通过。191 项原资产不变。见 [T4 回执](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_t4_zh.md) 与 [证据](../../../../agents/integration/artifacts/p3/pr3/t4-v2/manifest.json)。下一任务为 T5：三个候选各在 18/24 位置完整 Close，并补 CUDA 模式复核；本轮未启动 T5，未选默认 law，未 commit/push。HEAD 仍为 e83ca2b1；连续接触精度未认证，SuperDex 原参考映射仍阻塞。下方 T3/T2 的“下一任务”为历史状态。

> **2026-09-23 PR3 T3 完成：** 私有 FP64 q/x 联合状态、完整 E/r/B、PR2 共享迭代、真实稀疏线性后验、原子发布/回滚及分段 graph 已实现。78 项定向检查为 74 通过/4 预期跳过；667 项广范围回归为 647 通过/15 跳过/5 个既有失败，无新增失败；完整 pre-commit 通过。三 law、五种设备/执行配置共 45 个 tiny 步全部成功，18 组联合噪声资格通过。见 [T3 回执](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_t3_zh.md) 与 [证据](../../../../agents/integration/artifacts/p3/pr3/t3-v1/manifest.json)。下一任务 T4 物理校准/离散参考与时间细化；T5 Close 尚未执行。191 项原资产不变，HEAD 仍为 e83ca2b1，未 commit/push。下方 T2“下一任务 T3”为历史状态。

> **2026-09-23 PR3 T2 完成：** 已实现私有 FP64 全配对接触几何/E/r、三law的FP32 fitted B与直接FP64 tet E/r，旧PR1/PR2入口和SDF距离外推规则未改。组件、实际trial差分、噪声及CUDA mesh资格通过；651项同进程回归为633通过、14跳过、4个既有校准接口错误，无新增失败；CUDA AABB此次通过，既有波动问题继续保留。完整pre-commit通过。见 [T2回执](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_t2_zh.md) 与 [证据](../../../../agents/integration/artifacts/p3/pr3/t2-final-v1/manifest.json)。下一任务T3联合状态/求解/graph尚未开始；初态全查询不代表Close认证。HEAD仍为e83ca2b1，新增6个未跟踪文件，未commit/push。

> **2026-09-23 T0/T1 收口 v2：** 保留既有 SDF clamp＋外推距离规则，补齐其 FP64 raw gradient、联合装配精度、实际 BSR/PCG 稀疏后验及设备状态原型。T0 PASS_INPUT_FREEZE、T1 PASS_DESIGN_PROTOTYPE，可以进入 T2；生产实现和轨迹尚未认证，本轮未开始 T2。见 [收口报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_design_validation_v2_zh.md) 与 [证据](../../../../agents/integration/artifacts/p3/pr3/t0-t1-v2/manifest.json)。Newton 工作树仍为 e83ca2b1 且干净，无 commit/push。

> **2026-09-23 SDF 查询规则勘误：** 现有生产规则未修改，盒外点采用 clamp＋距离外推，不会仅因盒外而失败。重新对18/24各34,992个key调用原查询器，全部返回有限距离及非零有限gradient；各33,696只是盒外数量，不是失败数。此前“第一步前阻断”的生产含义撤回，T1仍有其他未验证项。见 [复核与勘误](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_sdf_rule_confirmation_zh.md)。下文旧严格盒域诊断及剔除建议为被更正的历史解释，不能据此要求改动现有SDF规则。

> **2026-09-23 PR3 T0/T1 实际验证：** T0 输入冻结完成；81 组接触方向差分、13 项不变量、CPU/CUDA 局部 FP64/mesh 原型及 96 项已有组件契约通过。T1 尚未关闭：v2 全对查询并域外失败的组合，在 18/24 初态各有 33,696 个 mesh key 域外，第一步前即阻断；深内部 SDF 跨层级跳变也已记录。未擅自引入 cull/外推，未进入 T2。见 [设计验证回执](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_design_validation_zh.md) 与 [证据 manifest](../../../../agents/integration/artifacts/p3/pr3/t0-t1-v1/manifest.json)。本次全部脚本/证据位于 worktree 外，生产源码和默认值不变，分支仍为 e83ca2b1 且干净，未 commit/push。下方“尚未运行 T1”为历史规划状态。

> **2026-09-23 PR3 具体实施方案 v2：** 已按用户四项选择收口：本轮截至完整Close，三种law显式选择，v_reg=0.01m/s，首轮不加对齐淡出。计划拆为T0–T6，补齐tet residual直接FP64装配、mesh插值/raw gradient、全对候选、联合停止/后验和原子提交。Close固定18/24两个位置从0至2.5s；新抓取与默认选择留后续。见 [实施计划v2](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_plan_zh.md)、[接触设计v2](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_contact_design_zh.md) 与 [规划回执](../../../../agents/integration/artifacts/p3/pr3/planning-v2/manifest.json)。状态PLANNED_NOT_IMPLEMENTED：仅更新文档，T1数值验证尚未运行，Newton工作树仍为e83ca2b1且干净；未commit/push。

> **2026-09-22 PR3 规划与设计启动：** 基于已提交 PR2 `e83ca2b1`，新增 [PR3 开发计划](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_plan_zh.md) 和 [接触数学与状态设计](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr3_contact_design_zh.md)。状态 DESIGN_DRAFT；下一入口 PR3-01 为接触 FP64 几何/映射、冻结量、联合停止配置和状态事务的设计验证。只比较 C-EE/C-IE/C-II，共同对照先用 ResNorm；PR2 已知例外与连续精度失败保留。完整 Close 后的抓取仍等待确认的新轨迹。本轮仅文档，未修改 solver、未运行仿真、未提交/推送。下方旧阶段状态保留历史含义。

> **2026-09-22 PR2 已本地提交：** `e83ca2b106ed1ebc34c73304be755420d20e12c5`（`SamiulJ/monolithic-integration`），21个文件、5110行新增；工作树干净，未push。完整pre-commit通过，939项源码/fixture哈希及5项配置哈希与验收快照一致。保留纯Armijo已知例外及正式1ms连续精度失败结论，不代表原完整物理目标全部通过。外部文档与诊断证据保留在工作区，不包含在该Git提交中。见 [提交回执](../../../../agents/integration/artifacts/p3/pr2/implementation/commit-checks-v1/receipt.json)。下列“未提交”均为历史状态。

> **2026-09-22 PR2验收收口：** 已按Armijo已知例外补齐其余配置。当前四个停止目标×0.1下，57组小系统和11条整手共152,800生产步全部SUCCESS，严格同离散比较、原整手表现与两设备动态q/qd细化通过；纯Armijo按用户决定不再追加实验/修复。正式1ms连续物理仍35/49通过，链条/驱动失败保留；八子步仅驱动场景已认证。55项组件及pre-commit、635项广范围回归证据经同源码哈希核对复用（619通过、11跳过、5既有失败）。**本轮离散验收及交付收口，原完整物理目标并非全部PASS。** 无源码修改、无commit/push。见 [验收报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_acceptance_zh.md) 与 [回执](../../../../agents/integration/artifacts/p3/pr2/implementation/acceptance-v1/receipt.json)。

> **2026-09-22 最新决定：** 用户要求将纯Armijo近收敛斜率精度问题记为已知问题并先跳过（`PR2-KNOWN-ARMIJO-01`）。不再等待Armijo设计确认，不继续该问题的设计/修复；保留现有停止系数候选和四个CPU场景失败证据。此例外不计为三模式验收通过，其他未完成矩阵及链条0.125ms噪声资格等缺口仍单列。驱动限位控制保持八子步通过结论保留。见 [已知问题记录](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_production_policy_zh.md#已知问题-pr2-known-armijo-01) 和 [决定回执](../../../../agents/integration/artifacts/p3/pr2/implementation/production-policy-known-issues-v1/decision.json)。本次仅更新文档，未修改solver、未commit/push。

> **2026-09-22 生产策略验证更新：** 用户批准推进停止精度生产化及1ms控制保持下八子步设计验证。articulation工作树已加入原四个停止系数×0.1的候选默认值，PR1/tet共享配置未改；但CPU三模式小系统仅17/21通过，纯Armijo在四个场景被2.84e-14固定斜率噪声下限阻塞，**候选尚未通过生产推广验收**。当时已请求用户确认Armijo设计；最新决定将其记为已知问题并暂缓处理，待确认状态已取消。独立OR验证中，驱动限位1ms起点控制保持＋8×0.125ms在CPU/CUDA各9600步通过原连续门槛：q≈0.003423rad、qd≈0.049000rad/s，速度余量约2%；串行求解成本约单步7.7～7.9倍。链条0.125ms校准超过冻结噪声界，其子步认证未推进；整手/全场景八子步未认证。55项组件/诊断及完整pre-commit通过；广范围635项中619通过、11跳过、5个旧失败仍保留，与四个Armijo轨迹失败分列。正式步长/公共API未变，未commit/push。见 [生产策略与设计验证报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_production_policy_zh.md)。下列已完成诊断和旧默认描述是历史状态。

> **2026-09-22 停止精度与连续步长诊断完成：** 用户批准的两条诊断已在CPU/CUDA通过。Sharpa诊断实例s=0.1、1/0.5/0.25ms共56,000生产步全部SUCCESS；q/qd信噪比约101/77，COM/旋转也通过原20倍及细化比例要求，未运行s=0.01。严格参考28,000根通过敏感性检查。驱动限位0.125ms双设备各9,600步通过原连续门槛，最大q≈0.003407rad、qd≈0.047636rad/s；连续参考输出全部原生比较时刻，未运行0.0625ms。51项组件/诊断测试及完整pre-commit通过；广范围631项中615通过、11跳过、5个既有失败用例保留。**生产默认s=1、正式1ms基线、物理参数与预算均未改**；旧默认的细化/连续验收结论没有翻转。下一步是依据诊断制定生产策略，不自动部署。见 [诊断报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_precision_diagnostics_zh.md) 与 [回执](../../../../agents/integration/artifacts/p3/pr2/implementation/precision-diagnostics-v1/receipt.json)。无commit/push。

> **2026-09-22 PR2 审查修复与模式补证完成：** 活动 KINEMATIC 拒绝（保留合法固定根）、实际状态及最终候选的有效惯量检查已修复；CPU/CUDA 46 项组件/契约测试与完整 pre-commit 通过。共同 OR 模式补齐 CPU production、CUDA diagnostic、CUDA production graph-off：21 组小系统 21,600 步、3 组整手 12,000 步全部 SUCCESS，同离散及原整手表现门槛通过。广范围626项：610通过、11跳过、5个既有失败用例再次失败；AABB原HEAD隔离再现，根因仍未确定。连续精度失败和细化未认证保留，无阈值/预算变更，无commit/push。见 [本轮修复报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_review_fixes_zh.md) 与 [回执](../../../../agents/integration/artifacts/p3/pr2/implementation/review-fixes-v1/receipt.json)。下一步数值实验建议见 [精度后续审查](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_accuracy_followup_review_zh.md)，尚未执行新的停止精度/步长设计。

> **2026-09-22 第 2、3 项收口：** 用户授权保留细化未认证后，已修复最终残差发布复核、muscle/模型计数变更拒绝和有效惯性检查，补齐 40 项组件/契约测试。212 项定向混合回归与完整 pre-commit 通过；广范围 620 项中 604 通过、11 跳过、5 项失败均在原 7d69a9c9 基线隔离复现（旧校准接口 4 项、CUDA AABB 1 项）。冻结源码后重跑小系统 43,200 步与整手 72,000 步，实际记录轨迹与修复前一致；原连续精度失败、细化未认证和旧回归问题保留。见 [收口报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_closeout_zh.md) 与 [交付回执](../../../../agents/integration/artifacts/p3/pr2/implementation/final-v1/receipt.json)。本轮检查与交付完成，不是 PR2 全部验收 PASS；工作树未提交／推送。

> **2026-09-22 整手独立参考更新：** 1/0.5/0.25 ms 共 28,000 个独立高精度根全部收敛，10/10 组整手同离散比较通过。但 q/qd 细化信号仅为噪声界约 5.54/6.09 倍，低于原 20 倍条件，细化仍未认证。用户要求按顺序收口 1/2/3；第 1 项证据已完成，退出条件未通过，已询问是否保留该缺口继续第 2、3 项。未改门槛或生产源码。见 [独立参考与认证报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_sharpa_certification_zh.md)。

> **2026-09-22 整手诊断完成：** 保留原连续精度目标后，Sharpa 1 ms 双设备三模式共 24,000 步全部 SUCCESS、原整手表现指标通过；共同模式 0.5/0.25 ms 双设备再完成 48,000 步，亦全部 SUCCESS。细化比例约 0.177–0.356，但独立整手 oracle／噪声界未认证，不冒充连续准确度或完整细化验收。见 [整手诊断报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_sharpa_diagnostic_zh.md)。小系统连续物理仍为 30/42，原失败保留；PR2 未完成，无提交／推送。

> **2026-09-22 最新决定：** 用户批准保留原连续精度目标，分别记录离散实现验证和 1 ms 连续精度验收，并继续 Sharpa 整手诊断。已解除诊断推进的阶段阻断；未放宽阈值，既有连续物理失败保留，PR2 未完成。见 [门槛评估](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_continuous_threshold_assessment_zh.md) 和开发方案顶部修订。下方“等待取舍澄清”为此前状态。

> **2026-09-22 最新实施进展：** 用户已批准同离散角度5e-4 rad、角速度1e-3 rad/s（fixture v3）。七小系统×双设备×三模式42/42同离散检查通过；连续物理检查30/42通过，双连杆和驱动系统在1ms下仍超原连续门槛。独立精确离散解和时间细化确认主要为离散误差，已发出新的取舍澄清；未进入Sharpa正式轨迹验收。详见 [连续误差分析](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_continuous_error_zh.md)。停止准则未改，工作树仍未提交。

> **2026-09-22 后续决定：** 用户允许先放宽同离散角度验收到5e-4 rad，已版本化为fixture v2；停止准则未改。继续推进状态/事务检查，新旧定向测试105项通过。角速度仍有独立失败，已另行询问，未自动放宽或宣称PR2验收通过。见 [最新实施记录](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_implementation_zh.md)。

> **2026-09-22 PR2实施更新：** 已开始私有FP64 FK/状态、独立oracle、完整E/r/B及BSR/PCG接入初版，未commit/push。新旧混合进程101项定向测试通过，但PR2尚未完成。自由旋转CPU/CUDA三模式6000步均SUCCESS，却超过同离散q/qd误差门槛；已保存证据并暂停依赖该停止准则的正式验收，等待用户是否允许新增PR2内部停止精度设计。详见 [实施进度与冲突](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_implementation_zh.md)。当前工作树有未提交实现；下方“干净/未实现”是历史记录。

> **2026-09-21 PR2完整规划更新：** [开发方案v2](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_plan_zh.md) 已将工作拆为7阶段，补齐 [完整力项与B公式](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_formulations_zh.md)、私有FP64状态/FK接口、事务/graph、混合精度后验检查和 [7个小系统fixture及门槛](../../../../agents/integration/artifacts/p3/pr2/planning-v2/fixtures-v1.json)。新增27项独立公式检查通过，尚无生产实现或轨迹验收。下一步从PR2-01设计收口/PR2-02独立oracle与FP64运动学状态开始，按阶段验证后推进。Newton分支仍为7d69a9c9且干净；本轮仅更新外部规划与证据，未提交/推送。旧规划文档已保存到planning-v2/previous-docs，PR2-01原manifest中的旧文档hash可按该目录index追溯。

> **2026-09-21 PR2设计验证更新：** 首轮62项数学检查通过，真实CPU/CUDA候选的FP32坐标表示检查发现无法达到原门槛。用户已批准追加精度设计，72组对照及CPU/CUDA最小原型支持本轮采用私有FP64 q/qd、FK和积分历史/物理验收，B/PCG先保留FP32。见 [设计验证](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_design_validation_zh.md) 和 [精度ADR](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_precision_adr_zh.md)。PR2-01整体尚未关闭、PR2-02完整力项尚未完成，生产实现未开始。所有原型在worktree外；Newton分支仍为7d69a9c9，未commit/push。

> 更新：2026-09-20。P2 已中止；**P3-PR1 私有tet-only已实现并通过验收；PR2 已更新计划与积分状态方案，尚未实现；PR3 未开始**。
> **当前干净开发基线：`7d69a9c9abe5c6157fa9f005bcaba5818ee3d3d0` / `SamiulJ/monolithic-integration`。** 用户要求清除debug修改后，已撤回旧Close实验代码、专属脚本/测试/数据；12个未提交研究文件归档到worktree外，5个P3历史规划文件纳入Git。PR1的17个提交文件保持不变。完整pre-commit及30项PR1专项、149项旧路径/碰撞/接触回归通过，`git status --porcelain`为空。见 [清理核验](../../../../agents/integration/artifacts/p3/debug-cleanup-20260920/verification.json)。未推送，PR2从此基线继续。
> PR1提交 `bd45ecf6bee8199b8fbfaaf2aa11a20f9f72f439` 和回迁记录保留： [提交核验](../../../../agents/integration/artifacts/p3/pr1/commit-checks/commit.json)、[分支迁移核验](../../../../agents/integration/artifacts/p3/pr1/branch-migration/verification.json)。原debug分支仍指向bd45ecf6，旧报告/原始证据未删；被移出的未跟踪研究位于 `agents/integration/artifacts/p3/debug-cleanup-20260920/untracked-backup/`。历史报告中的“未提交”、原分支名或17个未跟踪文件均是当时状态。
> PR2 最新决定：保留 q/qd 接口，增加随状态复制、提交和回滚的 joint/link 积分 buffer，保留旋转速度对称分量；先验证 FP32。见 [PR2执行计划](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_plan_zh.md) 与 [积分状态ADR](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr2_state_adr_zh.md)。不再使用对话中省略对称分量的 predictor 提案。
> PR1 后续审查发现的 P2 级诊断采样时间问题已修复，见 [修复与验证记录](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr1_p2_fix_zh.md)；原 v6 验收快照保留，修复源码另行绑定。
> 完整结论见 [PR1验收报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr1_acceptance_zh.md)。当前进度见 [PR1 进度报告](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr1_progress_zh.md)、[执行计划](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr1_plan_zh.md) 与 [用户批准的精度 ADR](../../../../dev/monolithic/P3/newton_monolithic_solver_p3_pr1_precision_adr_zh.md)。下文工作区清单与第9节保留2026-09-19交接时的历史记录，不代表当前无代码修改。
> 本文是接手导航，不替代下列规划权威。交接后的新用户决定优先；不要依据历史报告中的“下一步”恢复 P2。
