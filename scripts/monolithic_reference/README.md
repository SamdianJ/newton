<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Monolithic offline reference harness

The PR-5 harness runs separate Newton and pinned SuperDex interpreters against
an exact clean source SHA, validates JSON-lines records, and reports observed
differences with explicit mapping limits. SuperDex is an offline dependency only;
normal Newton imports do not load it. The measured adapter subset is the separate
`reference_no_contact_draft_v1.json` articulated/tet scene. It is **DRAFT**, not a
normal-loading envelope or P0/V0.1 acceptance result. The original PR-0 tiny
manifest remains unchanged.

From the Newton repository root:

```bash
uv run python -m scripts.monolithic_reference.cli validate scripts/monolithic_reference/fixtures/tiny_draft_v1.json
uv run python -m scripts.monolithic_reference.cli inspect-worktree /path/to/superdex-reference --expected-sha 54ae749a042709e897cad12da66822c71bcd1b97
```

The tiny fixture contains the frozen programmatic geometry, material and initial
state specification. Its **comparison manifest is DRAFT**: benchmark timing,
shape/contact/control parameters, calibration, support and response envelopes
remain `null`. A draft hash identifies inputs; it certifies no numerical result.
The stored material values are float32; mass and geometric inputs use SI units.
The link `xform` inputs are builder transforms before forward kinematics.

`manifest.schema.json` specifies the complete frozen V0.1 contract. The standard
library validator implements only the keywords used in this checked-in schema.
Draft validation permits omitted fields and `null` pending values. Populated
fields receive schema type/range checks and basic node/topology length and index
checks; full cross-field coherence is checked only for frozen inputs. Identity,
repository SHAs, all unit declarations and
mapping entries must always be complete. Frozen validation requires every schema
field, rejects `null`, and rejects blocking `UNMAPPED` entries. Completing fields
requires separately reviewed measurement evidence; schema validation cannot
certify that evidence. V0.2 friction/grasp fields require a new schema version.

Frozen coherence checks cover joint/state/drive dimensions, world-anchored link
trees, valid transforms, positive tet volumes and exact manifold boundary sets,
dynamic mass/material validity, fixed-node velocities, fixed P1Q3 slots and the
reference-area weight rule, and ordered benchmark stages and intervals. These
checks certify input consistency, not successful physics execution or calibration.
Unit-vector/quaternion checks allow `1e-6` squared-norm storage roundoff; this is
an input representation tolerance, not a measured solver acceptance threshold.

Supported shape spellings are `sphere`, `box`, `capsule`, `cylinder`, `cone`,
`infinite_plane` and `volume_sdf`. `dimensions_m` holds Newton primitive dimensions
(sphere radius first, box half-extents, axial primitive radius/half-height), or
volume dimensions. Unbaked volume runtime scales must be positive and exactly
uniform after float32 storage. A baked volume may retain positive nonuniform
scale provenance; its scale has already been applied and adapters must not apply
it again. Analytic dimensions are not subject to the uniform-volume-scale rule.

Canonical hashes use compact sorted-key UTF-8 JSON with no NaN/Infinity or
duplicate keys. All fields, including status, mappings and repository SHAs,
contribute to the hash. Arrays retain their order; changing a physics input
invalidates old evidence. Exact full SHA and Git cleanliness checks include
tracked, untracked and submodule changes; ignored build outputs are permitted.

`step_record.schema.json` is the JSON-lines output contract. It keeps
physical force/wrench, generalized physical force and residual contributions in
separate fields. Residual/generalized q components use N*m for revolute and N for
prismatic coordinates; particle residuals use N. World wrenches list force [N]
then torque [N*m]. `actual_parameters_json` must contain the adapter's actual
effective parameters, and metadata records both SHAs, the manifest hash, build,
device, hardware and timing. Unavailable fields are `null` with exact explanations in `unavailable_fields`;
missing native linear-iteration, rollback and residual APIs are never represented
as zero. Comparisons reject source/hash/time/dimension mismatches and do not
create an equality gate from observed differences.

Mapped comparison must interpret `INTENTIONALLY_DIFFERENT` laws explicitly:
Newton V0.1 quadratic hinge and SuperDex PolyReLU are not a trajectory equality
gate. `UNMAPPED` entries list the conclusions they block. Calibration, a normal
response envelope and an audited adapter are later P0 work.

## Running the DRAFT adapters

The manifest must contain the exact clean Newton and SuperDex source SHAs to be
executed. To run a later Newton commit, copy the DRAFT manifest, update only its
`repositories.newton_sha`, and retain its new hash with both outputs. Do not use
this operation to imply frozen provenance. The harness rejects existing output
directories, dirty source trees and failed/invalid output publication.

```bash
uv run python -m scripts.monolithic_reference.cli run superdex common.json native-output --worktree /path/to/pinned-superdex --python /path/to/native-venv/bin/python --build-type Release-fp32
uv run python -m scripts.monolithic_reference.cli run newton common.json newton-output --worktree /path/to/clean-newton --python /path/to/newton-venv/bin/python --build-type python-fp32 --device cpu
uv run python -m scripts.monolithic_reference.cli compare common.json newton-output/records.jsonl native-output/records.jsonl comparison.json
```

Comparison exit code 2 with `status=BLOCKED` preserves the explicitly unmapped
normal-loading conclusions even when both no-contact runs completed. This is
expected for the checked-in DRAFT. Native construction uses noncolliding tet
geometry as an inertia carrier with explicit rigid mass/COM/inertia; shapeless
native links ignore supplied mass. Both adapters consume the same rest mesh,
masses, initial q/qd/x/v, gravity and timestep. The native adapter reads back mass,
inertia and initial q/qd, explicitly disables articulation/soft contact, and
checks contact count and physical force remain zero.

The pinned public `NeoHookeanMaterialParams` selects **Smith** log-stabilized
energy. Its internal constants are mu_hat=4*mu/3 and lambda_hat=lambda+5*mu/6;
Newton's frozen PR2 material is **Kim no-log**, using lambda_tilde=mu+lambda.
Matching public E/nu does not make these energies equal. Energy law and Newton GN
versus native spectral Hessian projection are separate
`INTENTIONALLY_DIFFERENT` mappings. No exact constitutive or trajectory equality
conclusion follows from this harness. Native default nonlinear tolerances are
recorded as observed; they are not tuned to match Newton.

Native dynamic articulated links reject an infinite plane because it has no
surface mesh (`mochi_rigid.cpp`, `InitRigidActor_Dynamic`). Consequently the
separately owned `normal_loading.py` plane experiment is **UNMAPPED** here.
Replacing it with a finite box requires a new common fixture and fresh evidence.
No native residual/generalized-force measurement is inferred by negating a
physical force. No native rollback or soft-commit state is inferred from status.

## Reproducing the pinned native build and public probes

At SuperDex SHA `54ae749a042709e897cad12da66822c71bcd1b97`, the local default
GNU 11 compiler was rejected by the upstream compiler check. The unmodified
source builds with the installed Clang 20 toolchain:

```bash
CC=/usr/bin/clang-20 CXX=/usr/bin/clang++-20 UV_PROJECT_ENVIRONMENT=/path/to/native-venv CMAKE_BUILD_PARALLEL_LEVEL=4 uv sync --extra core --locked
UV_PROJECT_ENVIRONMENT=/path/to/native-venv uv run --no-sync python /path/to/newton/scripts/monolithic_reference/probe_superdex_public.py public-probe.json
```

Run the second command from the pinned reference root. The probe independently
checks a 90-degree rotation plus translation of native local nodes, shapeless
versus geometry-carried mass, and the dynamic plane error. It modifies no source.
The wrapper validates exact substep count, the 1-based step/time sequence,
manifest-sized state/residual/force arrays, effective dt and requested device/build
before publishing results. Comparisons include separate link translation and
quaternion-sign-invariant rotation differences plus per-step status and aggregate
convergence, soft-commit, rollback, nonfinite-status and unavailable-field counts.
A BLOCKED mapping never suppresses these observations. Invalid nonfinite numeric
records are rejected.

Native soft inertia uses a **consistent mass matrix** (full density*Nf*Ng
quadrature), whereas Newton uses lumped particle mass. This is an explicit
`INTENTIONALLY_DIFFERENT` mapping, separate from matching total mass and geometric
COM weights. The native lumpedMass cache is not evidence of lumped dynamics.
Manifest node masses must match rest-volume/density weights even if their total
is correct; native COM uses independently computed geometric weights.

Adapter records contain native binary hash/path, precision, hardware, worker
count and effective parameters. Native total timing excludes offline queries;
Newton total timing surrounds a synchronized solver step. Neither includes a
comparable calibrated stage breakdown, and these tiny timings are not C5
performance evidence.

## Newton release profiling

`profile_release.py` is the opt-in Newton-only C5 harness. It measures the
versioned single-finger fixture and a level-3 refined tet mesh on CPU/CUDA,
compares P1Q3 with the existing adaptive-face narrow phase, and records
synchronized p50/p95 timings for collision, contact, BSR construction,
preconditioner factorization, PCG, and the whole step. It also runs offline
diagonal-preconditioner and disabled-contact-cross-block negative controls,
brackets the fixture-specific contact-stiffness lower bound, and audits native,
process/driver, and Python device allocations. Component timers overlap their
enclosing current/trial timers and must not be summed.

```bash
uv run python -m scripts.monolithic_reference.profile_release \
  --output /new/output/directory --device cpu --device cuda:0
```

The profiler rejects an existing output directory and publishes a source hash,
Newton commit, worktree-cleanliness bit, raw samples, and an artifact SHA-256.
Its negative controls are process-local instrumentation and do not alter the
production solver configuration. SuperDex is deliberately not executed and is
not a V0.1 exit gate; its constitutive/mass-model alignment is deferred until
after V0.2.
