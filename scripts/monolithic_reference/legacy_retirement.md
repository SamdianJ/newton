# Current Monolithic entry points and retired drivers

The supported solver is `newton.solvers.experimental.monolithic.SolverMonolithic`
with complete q+x state. Use `run_p4 --profile dev|full` for frozen scene matrices
and `--list` to inspect their exact expansion. The scene runner does not replace
component, lifecycle or replay qualification. Formal acceptance evidence is in
`agents/integration/artifacts/p4/t4-v1/` under the shared workspace.

P4 retires the old BE, q-only, tet-only and spring/xi solver implementations,
their exclusive examples, P1/P2 calibration and profiling drivers, and the old
`validate_p2` scheduler. Their former commands are no longer runnable interfaces.
No Legacy export or compatibility wrapper is supplied. The independent collision
pipeline and current solver rejection contracts remain supported.

Original sources are archived outside the active tree, preserving relative paths:

`agents/integration/artifacts/p4/legacy-archive/39f80c48/closeout/19b43e89/`

The exact deletion/migration list, SHA-256 values, source commit and validation
records are in `agents/integration/artifacts/p4/t3-closeout-v1/`. Earlier retired
tests have separate manifests in the same P4 artifact directory. Git history
also retains every removed file. Snapshots are historical evidence, without a
promise that they can still run.

Shared control/model construction used by current P3 diagnostics remains:

- `verify_p3_close`: frozen Close schedule; joint mapping comes from the current
  Sharpa scene. Close/SDF verification and the independent SDF oracle remain.
- `normal_loading`: model/input validation and the unchanged control recipe only.
  `internal_envelope` retains independent math and frozen envelope validation;
  their old BE runners are removed.
- `p3_coupled_fixtures`: clamp model construction without creating an old solver.
- `linear_evidence`: shared source/matrix evidence for independent benchmarks.
- `preconditioner_pilot_batch --kind matrix`: retained matrix research. The old
  BE trajectory mode is rejected by argument parsing.

The isolated baseline adapters and offline reference harness remain tied to
explicit historical source identities; they are not alternate current solver
entry points. Frozen JSON source hashes, recorded results, fixtures and research
assets are not rewritten or deleted to make old provenance look current.
Historical development Markdown retains its original commands as records,
explicitly labeled inactive where their drivers have retired.
