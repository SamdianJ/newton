# Default Franka validation — 2026-09-30

Run the user-requested complete Franka grasp on source `558d958b79a973282760d3432f0a5857d266a370` after the
module split and renaming. No solver parameters, tolerances or reference data
were changed. This validates one default scene, not the entire matrix.

| Check | Result |
| --- | --- |
| Configuration | FP64, ArmijoOrResNorm, C-IE, production PCG, CUDA graph |
| Device | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Published steps | 5250 / 5250 |
| Converged / Stopped / hard failure | 5250 / 0 / 0 |
| Grasp hold | PASS; 500 steps (0.5 s), minimum COM rise 70.685910 mm |
| Maximum ground force during hold | 0.0 N |
| Maximum penetration | 1.795431 mm |
| Minimum tet det(F) | 0.509159323 |
| Recorded frames and complete state | PASS; 5251 frames including initial state |
| Reference comparison | PASS_FROZEN_ENVELOPE; all compared fields within original bounds |
| Maximum particle-position difference | 1.648475800e-11 m |
| Solve time / scene elapsed time | 296.178 s / 341.703 s |

The single case is **PASS**. The filtered dev matrix is **PARTIAL**, as intended:
Sharpa and the other eight Franka configurations were not run. Component suites
were not run. A single timing observation is not a performance qualification.

Command, from the development worktree:

```bash
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=../monolithic-integration/.venv \
  uv run --no-sync python -B -m scripts.monolithic_reference.run_regression \
  --profile dev --case franka-cube--ArmijoOrResNorm--C-IE--production--graph1 \
  --output /home/lightwheel/Desktop/newton/SamiulJ/agents/integration/artifacts/validation/franka-default-558d958b-20260930
```

The workspace output contains `RESULTS.md`, `summary.json`, the case receipt and
its process log. Receipt SHA-256: `fb6349a656142ac3e48fd6603456cd6a9368d2b1cf0226729127c6d08eb0e799`.
The receipt binds source hashes, imported module origins, frozen input and
reference identities, full per-field errors, environment and physical metrics.
The successful new recording was removed by the default retention policy after
comparison; the ten retained references remain unchanged.
