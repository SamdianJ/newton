<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Monolithic offline reference scaffold

PR-0 provides input validation, canonical hashes, explicit mapping status, an
output schema and a clean-worktree guard. It does not run Newton or SuperDex
physics. `run_adapter()` and `compare_runs()` raise `NotImplementedError` without
creating results. No runtime imports or dependencies on SuperDex are added.

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

`step_record.schema.json` is the future JSON-lines output contract. It keeps
physical force/wrench, generalized physical force and residual contributions in
separate fields. Residual/generalized q components use N*m for revolute and N for
prismatic coordinates; particle residuals use N. World wrenches list force [N]
then torque [N*m]. `actual_parameters_json` must contain the adapter's actual
effective parameters, and metadata records both SHAs, the manifest hash, build,
device, hardware and timing. The scaffold ships no step records or adapter output.

Mapped comparison must interpret `INTENTIONALLY_DIFFERENT` laws explicitly:
Newton V0.1 quadratic hinge and SuperDex PolyReLU are not a trajectory equality
gate. `UNMAPPED` entries list the conclusions they block. Calibration, a normal
response envelope and an audited adapter are later P0 work.
