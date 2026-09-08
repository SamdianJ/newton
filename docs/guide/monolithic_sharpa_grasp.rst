.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Sharpa soft-ball integration
============================

.. experimental::

   ``monolithic_sharpa_soft_ball`` is a contact experiment using
   :mod:`newton.solvers.experimental.monolithic`. Completing the stage script
   does not certify G6/G7 or a successful friction-supported grasp.

The default ``anchored-close`` experiment places the 20 mm soft sphere in
front of the palm at ``(0.045, -0.005, 0.085) m``. A palm-facing cap is fixed
in world coordinates before constructing the solver; all remaining nodes
can deform. The cap selects reference nodes with local ``x <= -0.7 R``:
5, 13 and 29 fixed nodes for the coarse, medium and fine meshes respectively.
These artificial constraints test contact stability, not free grasping.

The base URDF retains its 22 finger degrees of freedom and fixed root.
This experiment has no carriage or support plane. Gravity remains enabled,
and hand self-collision remains disabled. Fixed nodes use the solver's
existing Dirichlet treatment; the example never overwrites returned states.

Prepare the verified hand collision assets as described in
:doc:`monolithic_sharpa_assets`. Then run from the repository root::

   uv run --extra examples -m newton.examples monolithic_sharpa_soft_ball \
     --asset-dir /path/to/left_sharpa_wave \
     --contact-dir /path/to/prepared-contact-assets \
     --trajectory '/path/to/Ball_catch - Left_hand_motion_rad.csv' \
     --calibration scripts/monolithic_reference/fixtures/sharpa_contact_calibration_v1.json

Use ``--viewer null`` for headless simulation, or ``--headless`` for offscreen
OpenGL rendering. The default is 1 ms per physical step and ten physical steps
per displayed frame. The default schedule is 4.5 simulated seconds:
prepare 0.5 s, close 2 s, then hold 2 s (450 displayed frames).

The CSV's original two-second closure and two-second hold are preserved. Targets are sampled at physical-step end and frozen inside each
solver step. A rollback stops the example without advancing physical time.

The default ball is ``ball_r2.npz``. Select ``ball_r1.npz`` or ``ball_r3.npz``
with ``--ball`` to inspect other mesh resolutions. All use Smith material and
consistent mass. These are simulation parameters, not identified hardware or
material properties. ``--friction-off`` creates an independent run with zero
contact friction; joint friction remains enabled.

The output directory contains the manifest, actual state and target traces,
per-finger and support forces, history diagnostics, convergence results, and
per-call PCG iterations. Forces are world-space forces on the rigid body;
negate them to obtain forces on the sphere. Palm and finger forces are
recorded separately. Stage snapshots record actual simulated positions.
The trace checks fixed positions and zero fixed velocities at every step.
Step, collision, PCG and remaining step time are reported separately;
rendering and trace serialization are outside solver step timings.

Plot the output with::

   uv run --extra examples python scripts/monolithic_reference/plot_sharpa_grasp.py \
     --run output/monolithic-sharpa-soft-ball \
     --calibration scripts/monolithic_reference/fixtures/sharpa_contact_calibration_v1.json

The frozen pre-integration calibration uses a double-plane clamp and three
sphere meshes. The lowest passing sampled stiffness is ``2e6 N/m³``;
the example retains ``1e7 N/m³``. The PCG iteration p95 budget is 64 per
actual solve, evaluated separately by stage, including retries. The existing
200-iteration hard limit and residual criteria remain unchanged.

Collision uses two conservative AABB gates followed by the existing centroid
and P1Q3 SDF queries. It has no BVH and retains the full static pair table,
stable history keys, and allocation capacities. ``--disable-aabb`` selects
the internal full-table oracle. AABB reduces query work; it does not reduce
the static global assembly capacity or guarantee a frame-rate improvement.

Full hand SDF simulation requires CUDA. CPU verification covers analytic
collision, controls and articulation components; CPU mesh SDF is not required.
Final placement/control calibration, friction-off negative acceptance, G6/G7
and any triggered patch-Schur work belong to PR-7B.

With ``--test``, anchored closure requires at least 99% normal convergence,
the unchanged residual criteria, penetration at most 2 mm, min(detF) at least
0.1, unchanged anchors, and finger force at least 1 mN on 90% of hold steps.
A completed schedule with no sustained finger loading does not pass.
PCG iteration p95 above 64 is reported separately as an efficiency trigger.

The earlier exploratory nine-second grasp experiment remains available with
``--experiment grasp --num-frames 900``. Its original all-dynamic ball,
carriage, support and unsuccessful initial placement remain available for
reproduction. Its outcomes are separate from anchored closure.

For a sequential resolution comparison, use the same asset, contact,
trajectory and calibration arguments with::

   uv run --extra examples scripts/monolithic_reference/run_sharpa_anchored.py \
     --asset-dir /path/to/left_sharpa_wave \
     --contact-dir /path/to/prepared-contact-assets \
     --trajectory '/path/to/Ball_catch - Left_hand_motion_rad.csv' \
     --calibration scripts/monolithic_reference/fixtures/sharpa_contact_calibration_v1.json \
     --refinements 1 2 3 --output output/anchored-resolution

This runner executes the complete closure for each mesh, then profiles a
separate 20-step continuous hold tail. The instrumented tail synchronizes
individual stages; overlapping stage timings must not be added. Formal
state snapshots remain at the end of the 4.5-second trajectory.
