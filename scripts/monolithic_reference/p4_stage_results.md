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
