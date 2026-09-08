.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Sharpa soft-ball integration
============================

.. experimental::

   ``monolithic_sharpa_soft_ball`` is an exploratory PR-7A example using
   :mod:`newton.solvers.experimental.monolithic`. Completing the stage script
   does not certify G6/G7 or a successful friction-supported grasp.

In the initial measured configuration the ball rolls away during preparation
and remains supported during lift. The nine-second script converges, but this
configuration does not grasp the ball. Placement and timing need calibration
in PR-7B; the default parameters deliberately preserve this observed result.

The scene retains the base URDF's 22 finger degrees of freedom and adds one
vertical prismatic joint. A fixed support plane and the hand belong to the
same articulation. The 20 mm-radius tetrahedral sphere has no fixed nodes.
Gravity is enabled. Hand self-collision remains disabled.

Prepare the verified hand collision assets as described in
:doc:`monolithic_sharpa_assets`. Then run from the repository root::

   uv run --extra examples -m newton.examples monolithic_sharpa_soft_ball \
     --asset-dir /path/to/left_sharpa_wave \
     --contact-dir /path/to/prepared-contact-assets \
     --trajectory '/path/to/Ball_catch - Left_hand_motion_rad.csv' \
     --calibration scripts/monolithic_reference/fixtures/sharpa_contact_calibration_v1.json

Use ``--viewer null`` for headless simulation, or ``--headless`` for offscreen
OpenGL rendering. The default is 1 ms per physical step and ten physical steps
per displayed frame. The complete schedule is nine simulated seconds:
prepare 0.5 s, close 2 s, hold 2 s, lift 1 s by 50 mm, hold high 1 s,
release 2 s, and observe 0.5 s.

The CSV's original two-second closure and two-second hold are preserved.
Release reverses the closure segment. The carriage uses a smooth quintic
trajectory. Targets are sampled at physical-step end and frozen inside each
solver step. A rollback stops the example without advancing physical time.

The default ball is ``ball_r2.npz``. Select ``ball_r1.npz`` or ``ball_r3.npz``
with ``--ball`` to inspect other mesh resolutions. All use Smith material and
consistent mass. These are simulation parameters, not identified hardware or
material properties. ``--friction-off`` creates an independent run with zero
contact friction; joint friction remains enabled.

The output directory contains the manifest, actual state and target traces,
per-finger and support forces, history diagnostics, convergence results, and
per-call PCG iterations. Forces are world-space forces on the rigid body;
negate them to obtain forces on the sphere. Support reactions are excluded
from finger support. Stage snapshots record actual simulated positions.

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
