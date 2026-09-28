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
