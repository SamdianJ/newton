.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _monolithic-g2h:

Monolithic tet comparison (G2H)
=======================================

G2H compares Kim/Smith material and lumped/consistent mass through the same
monolithic stepping path. Run the four configurations side by side:

.. code-block:: bash

   uv run --extra examples -m newton.examples monolithic_tet_compare
   uv run --extra examples -m newton.examples monolithic_tet_compare --direction axial
   OPENBLAS_NUM_THREADS=1 uv run --extra examples -m newton.examples monolithic_tet_compare --viewer null --device cpu --test --output /tmp/g2h

Rows are Kim/lumped, Smith/lumped, Kim/consistent, Smith/consistent, in blue,
orange, green and purple. Geometry and displacements use the same scale.
The default is transverse loading; ``--direction axial`` uses axial traction.
Each displayed frame advances 100 ms in five 20 ms substeps. The final frame
is held after the complete four-second action. This example is experimental
and does not certify contact, a real material or soft-ball grasping.

Fixture and parameter authority
---------------------------------------

The cantilever is 1.2 x 0.3 x 0.3 m, with E=10000 Pa, nu=0.3 and density
1000 kg/m³. One end is fixed with zero velocity. Gravity and all collision
participation are disabled. A separate unforced revolute articulation with
no shapes satisfies the current coupled-model scope; it does not support or
load the soft body. This is not a new general soft-only solver mode.

The medium mesh has 6 x 2 x 2 cells, 120 tetrahedra and 63 particles. Coarse
and fine variants have 15 and 405 tets. Grid topology is generated locally.
The fixture explicitly replaces ``add_soft_grid``'s uniform nodal cell masses
with density-derived tet row sums, preserving the fixed-node mask. Consistent
and lumped cases therefore share physical density and total mass authority.
Changing mass mode changes the discrete operator, not the material density.

End-face external loads are distributed by triangle rest area, with a maximum
of 10 N axially or 0.5 N transversely. Loading, hold, unloading and free response
last one second each; ramps use a cubic smoothstep. No material damping or
moving fixed boundary is introduced. All four modes use identical inputs and
separate solver instances. Targets are external nodal forces frozen per step.

These are calibrated simulation inputs, not identified hardware/material
properties. An initial smaller fixture produced false-looking convergence with
negligible motion under the existing absolute residual floor. Scaling the
fixture's lengths by 10, forces by 100 and times by 10 preserves the dimensionless
loading while resolving the response at the original solver tolerances.
The small fixture is retained only as calibration history.

Frozen acceptance and independent reference
---------------------------------------------------

The fixture and gates are defined in ``monolithic_tet_compare.py`` and hashed
in each manifest. Formal runs do not change them on failure:

* All states finite, fixed nodes unchanged, no rollback, at least 99% normally
  converged steps, and existing global/q/x residual gates. The original
  ``detF >= 0.2`` guard remains active.
* Peak tip displacement at least 2 mm times the load scale. A stationary body
  cannot pass merely because the solver reports convergence.
* Maximum tip-vector error relative to the peak independently computed linear
  reference displacement at most 3%. The reference assembles small-strain
  isotropic P1 stiffness and the selected full mass matrix in NumPy float64,
  then solves the same backward-Euler load schedule. It is an approximation
  for these small strains, not an exact oracle for arbitrary large deformation.
* CPU/CUDA tip traces differ by at most 50 micrometres for identical cases.
* Canonical Smith/consistent transverse dt refinement compares 20 and 10 ms
  against 5 ms at matching timestamps. Mesh refinement compares 15 and 120
  tets against 405 tets. The refined error must be at most 80% of the coarse
  error. This demonstrates a trend over the measured envelope, not asymptotic
  convergence or resolution sufficiency for grasp contact.

Run the full 16-case matrix on each device, including a quarter-load linear
probe for each mode and the canonical refinements:

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 uv run --extra examples -m scripts.monolithic_reference.run_g2h --device cpu --output /tmp/g2h-cpu
   OPENBLAS_NUM_THREADS=1 uv run --extra examples -m scripts.monolithic_reference.run_g2h --device cuda:0 --output /tmp/g2h-cuda
   uv run --extra examples -m scripts.monolithic_reference.run_g2h --compare /tmp/g2h-cpu /tmp/g2h-cuda
   uv run --extra examples -m scripts.monolithic_reference.plot_g2h --results /tmp/g2h-cpu --output /tmp/g2h-plots

``--calibration`` records exploratory results and never reports formal PASS.
G2's exact energy/force/FD, projected PSD, mass/gravity/Dirichlet, owner/global,
scale, dense and lifecycle checks remain separate from G2H. Different material
or mass modes need not produce equal trajectories or a minimum visible difference.

Outputs and limits
---------------------------------------

Each case writes a manifest, per-step trace and summary. Measurements include
actual/reference tip displacement, applied force, volume and minimum detF,
elastic energy, kinetic energy ``0.5*v.T*M*v``, backward-Euler external work,
convergence ratios, iteration counts and measured step time. Energies refer to
the returned state and each case's actual material/mass law. Numerical decay
under backward Euler is not a measurement of material damping or exact energy
conservation. Correctness-run timings are not performance certification.

The four-second trace includes post-unloading free response; its one-second
free window is insufficient to identify a complete vibration period or decay
rate. Those measurements are not certified by this gate. This test validates
material/mass integration; contact friction/history, SDF, soft-ball preparation,
Sharpa grasping and production performance remain separate work.
