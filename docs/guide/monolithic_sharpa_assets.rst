.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Sharpa contact and soft-ball assets
==================================

.. experimental::

   These tools and fixture helpers are internal PR-6D diagnostics for
   :mod:`newton.solvers.experimental.monolithic`. They may change without notice.

PR-6D extends the collision-free Sharpa joint fixture with the base URDF's
26 collision meshes and a fully dynamic, 20 mm radius tetrahedral ball.
The source hand assets are supplied separately; the tools never substitute
visual meshes, convex hulls, or another hand version. CPU checks cover import,
volume quality and component dynamics. Mesh SDF queries require CUDA.

Generate volume assets offline, or load the three committed NumPy assets::

   uv run --extra examples -m scripts.monolithic_reference.prepare_soft_ball --output /path/to/balls

The deterministic spherified cube uses a conforming six-tet subdivision per
cell. Refinements 1/2/3 contain 27/125/343 nodes and 48/384/1296 tets. Boundary
and interior resolution increase together. Relative volume errors are about
23.4%, 6.5%, and 3.0%; the coarse ball is a geometric contrast, not a canonical
grasp discretization. Newton loads the complete volume from ``.npz`` without
a tetrahedralizer. The display/contact boundary comes from its tetrahedra.
Density is explicitly 1000 kg/m³, Young's modulus 10000 Pa, Poisson ratio 0.3,
and damping zero. These are simulation calibration parameters, not measured
hardware/material properties. The canonical component uses Smith/consistent;
Kim/lumped free-fall regression is also checked.

Prepare derived collision meshes, SDF caches, and a measured support manifest::

   uv run --extra examples -m scripts.monolithic_reference.prepare_sharpa_contact --asset-dir /path/to/left_sharpa_wave --output /path/to/hand-contact

This audits all collision references. Non-manifold source meshes receive
separate derived assets with the repair recipe, hashes, volume change and both
directed vertex-to-surface deviations. This sampled geometric metric is not a
continuous Hausdorff bound. The original URDF and meshes remain unchanged.
SDF bakes use float32 textures and explicit padding, band and sign settings.
The 32/64/128 resolution comparison retains failed results. Runtime loading
requires the selected resolution to pass every mesh's distance/sign/finite
nonzero-gradient checks; raw trilinear gradients need not have unit length.
The collision path normalizes them when producing physical normals.

Run a short whole-hand contact check::

   uv run --extra examples -m scripts.monolithic_reference.run_sharpa_assets --asset-dir /path/to/left_sharpa_wave --contact-dir /path/to/hand-contact --output /path/to/contact-check

The diagnostic holds all 22 joint targets at their initial values and lets
the free ball approach a finger for 20 ms, with gravity enabled. It uses the
ordinary coupled solver, joint PD/limits/friction, smoothed normal contact
and contact friction. It records actual forces, deformation after removing
rigid translation and rotation, slip speed, penetration, convergence and step cost. The latter
is a short component timing, not a steady-state performance claim. A rollback
stops the run without advancing physical time. This is not the closure
trajectory, a grasp demonstration, or a self-collision check.

Run the opt-in real-asset checks::

   NEWTON_SHARPA_ASSET_DIR=/path/to/left_sharpa_wave NEWTON_SHARPA_CONTACT_DIR=/path/to/hand-contact uv run --extra dev --extra examples -m unittest newton.tests.test_monolithic_sharpa_assets newton.tests.test_monolithic_soft_ball -q

The source-dependent tests explicitly skip when these paths are absent.
Independent FK checks cover all links and several poses; CUDA checks cover
all shape descriptors and transformed queries. The existing collision-free
``monolithic_sharpa_close`` example keeps its previous behavior.

PR-7 still owns two-level AABB acceleration, lift carriage and the complete
Close–Hold–Lift–Release fixture, friction-off negative control, and G6/G7.
G5 asset evidence does not close its separate AABB-equivalence requirement.
