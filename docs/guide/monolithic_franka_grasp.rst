.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Experimental monolithic Franka grasp
===================================

Run this example from the repository checkout with a CUDA device:

.. code-block:: bash

   uv run --no-sync python -m newton.examples monolithic_franka_grasp

Record the complete physical trajectory to a new directory:

.. code-block:: bash

   uv run --no-sync python -m newton.examples monolithic_franka_grasp \
     --viewer null --test --output /tmp/franka-grasp

Replay the recording without running the physics solver, then click **Play**:

.. code-block:: bash

   uv run --no-sync python -m scripts.monolithic_reference.replay_franka_grasp /tmp/franka-grasp

The first run builds and caches the Franka collision SDFs. The robot uses the
asset revision pinned by this checkout. ``--asset-cache`` selects the SDF cache
directory. The output includes the URDF/asset/source hashes, controller settings,
step statistics and complete integration states. Existing output directories
are rejected.

Scene and controller
--------------------

The fixed-base FR3 has seven dynamic arm joints and two dynamic finger joints.
The cube is the existing P3 40 mm r2 tetrahedral fixture: 343 dynamic nodes,
1296 tetrahedra, density 1000 kg/m³, Young's modulus 10000 Pa and Poisson ratio
0.3. Its initial center is (0.45, 0, 0.02) m, on the z=0 support plane.

The two-finger mimic constraint is intentionally replaced by independent PD
actuation with equal targets. This approved example approximation preserves
finger mass, inertia and contact response, but permits asymmetric finger motion
under asymmetric loads. It is not an exact model of the mechanical coupling.

Arm PD gains are 20000 N·m/rad and 200 N·m·s/rad. Finger gains are 2000 N/m and
10 N·s/m; imported effort limits are retained. Each finger opens to 40 mm and
closes toward 14 mm. Targets are cubic smoothstep segments in joint space,
with matching analytic target velocities. IK computes the two downward TCP
poses before simulation; it does not overwrite the dynamic state each frame.

The 5.25 s sequence settles, approaches, closes, lifts by a commanded 75 mm,
holds for 0.5 s, lowers, releases, and retracts. The solver takes 1 ms steps,
with ten physical steps per displayed frame. ``--test`` requires completion
and, over the final 500 steps of the hold, ground support below 5% of the cube
weight and a COM rise of at least a quarter of the cube's smallest edge; it
does not certify general physical accuracy. ``--precision fp32`` runs the
example in FP32.

Solver scope
------------

The example uses production PCG with CUDA graph segments, ArmijoOrResNorm,
C-IE contact, C² mesh SDFs, and the existing orientation mask. The URDF finger
colliders remain analytic boxes. It uses the P3 material, consistent mass,
friction coefficient 0.5 and normal stiffness 1e7. Newton/backtracking/PCG
budgets remain 10/8/200.

The example explicitly selects the solver's private ``publish_policy="current_iterate"``
strategy. It allows an eligible Stopped status to publish the current iterate
even when no Newton trial was accepted. The initial iterate
is normally the velocity predictor; rejected trial states are never published.
Final finite-value, geometry, inertia and complete-state checks remain active.
Zero-update publications are recorded separately from convergence. The global
solver default remains ``strict``; ``last_accepted`` still requires at least one
accepted update. The policy is included in checkpoint identity: restoring across
policies or changing the policy on a live solver is rejected. The example does
not override solver methods.

The simulation includes rigid-soft and ground-soft contacts. Rigid-rigid
collisions and robot self-collision are outside this example. New Franka mesh
SDFs have node-reproduction and derivative checks; full independent geometric
accuracy qualification is not claimed.
