.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _monolithic-tet-response:

Monolithic material and vibration demonstration
================================================

Run focused comparisons with the experimental monolithic solver:

.. code-block:: bash

   uv run --extra examples -m newton.examples monolithic_tet_response
   uv run --extra examples -m newton.examples monolithic_tet_response --experiment mass
   uv run --extra examples -m newton.examples monolithic_tet_response --experiment gravity

The material experiment stretches identical 120-tet beams with 300 N axial
traction, smoothly loading for four seconds, holding for four seconds, then
unloading for four seconds. Both use consistent mass; blue is Kim and orange
is Smith. The simulation uses the existing coupled stepping path and an
independent unforced articulation, with fixed left nodes, zero gravity and no
collision. The beam is 1.2 x 0.3 x 0.3 m, E=10000 Pa, nu=0.3, density=1000
kg/m³. These are demonstration inputs, not measured material properties.

The mass experiment fixes the material to Smith and compares lumped (blue)
with consistent (orange) on the same 15-tet mesh. Both start with identical
``u_x=0.04*sin(pi*x/(2*L))`` metres and zero velocity, then evolve without
external force for twelve seconds. The coarse mesh makes discretization effects
observable; it is not a recommendation for production mesh resolution.

Grey lines show the undeformed mesh and dark lines show the current mesh.
The sidebar reports physical tip displacement and plots both traces on the
same axes ranges. Open the viewer's plots for additional logged tip histories.
Material deformation displays at 1x. Mass deformation displays at **5x** by
default, explicitly labelled in the sidebar; ``--display-scale 1`` shows its
true geometric scale. This transforms rendering buffers only. All outputs,
period estimates and physics remain at 1x.

Each frame advances 40 ms: material uses two 20 ms steps, mass four 10 ms steps.
The example runs 300 frames, then holds its final state. These are fixed,
verified timestep presets. An additional 5 ms vibration probe triggered the
solver's small-step/unresolved-residual rollback guard; timestep convergence
is not certified for this demonstration, and solver tolerances were not relaxed.
Three positive peaks above 1 mm are required before
reporting a period; the estimate averages intervals between parabolically
interpolated maxima. It is the period of this discrete, finite-amplitude
response, not an exact infinitesimal natural frequency.

Typical CPU results are about 496/470 mm maximum extension for Kim/Smith,
and 1.475/1.436 s vibration periods for lumped/consistent. Eight positive peaks
are captured in each default vibration run. Decay is primarily a property of
backward-Euler integration here; no material damping has been added. Neither
comparison establishes which material law matches a real object better.

Record and plot physical curves
--------------------------------

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 uv run --extra examples -m newton.examples monolithic_tet_response --viewer null --device cpu --test --output /tmp/tet-response
   OPENBLAS_NUM_THREADS=1 uv run --extra examples -m newton.examples monolithic_tet_response --experiment mass --viewer null --device cpu --test --output /tmp/tet-response
   uv run --extra examples -m scripts.monolithic_reference.plot_tet_response --results /tmp/tet-response --output /tmp/tet-response-plots

CUDA uses the same commands with ``--device cuda:0``. Each case writes its
manifest, physical trace and summary under ``material/Kim``, ``material/Smith``,
``mass/lumped`` or ``mass/consistent``. Overlay plots include physical tip
displacement, applied force or peak envelope, energy and volume. The simulation
stops on rollback; its time is not advanced past the rejected step.

``--test`` requires complete duration, finite states, fixed nodes unchanged,
at least 99% normal convergence, original residual/detF safety gates, and a
measurable response. The material demonstration requires at least 300 mm
extension and 5 mm peak separation; the vibration demonstration requires four
positive peaks and at least 1% period separation. These checks serve this
specific demonstration. They do not replace or modify the frozen G2H
small-strain reference and refinement gates.

The solver implementation and frozen G2/G2H inputs are unchanged. Contact,
soft-ball grasping, material identification and performance certification are
outside this example.

Gravity and mesh resolution
----------------------------

``--experiment gravity`` compares 15, 120 and 405 tetrahedra (16, 63 and 160
nodes), coloured blue, orange and green. All three use the same 1.2 x 0.3 x
0.3 m geometry, Smith material, consistent mass, E=10000 Pa and nu=0.3.
This preset uses density **10 kg/m³** in all three meshes, for a total beam
mass of 1.08 kg. It is a lightweight simulation fixture; density differs from
the material/vibration presets, and no hardware properties are implied.

Gravity ramps smoothly from zero to 9.81 m/s² downward over two seconds and
then remains constant until 36 seconds. This is actual model gravity, not a
tip load or a replacement external-force field. Body loads use the complete
consistent-mass row sums before fixed-node elimination. External nodal forces
remain zero. Every mesh fixes the same left face; its separate articulation
does not support or load the beam. Display is 1x and tip displacement is
positive downward. Each of 300 display frames advances 120 ms in six 20 ms
steps. Material and vibration preset timing remains unchanged.

The final two-second window is checked separately for near-static behaviour:
tip range below 0.1 mm and maximum nodal speed below 1 mm/s, in addition to
the original residual/safety checks and at least 99% normally converged steps.
The example requires at least 50 mm response. A beam still oscillating cannot
pass the near-static comparison simply because nonlinear solves converged.

Typical CPU tail-mean downward deflections are about **104.4, 202.2 and
258.2 mm**. Independently assembled small-strain static FEM references for
the same respective meshes give **104.8, 205.6 and 265.8 mm**. These references
use NumPy float64 stiffness and the full consistent gravity load, with fixed
rows/columns eliminated after constructing that load.

The material parameters have not changed with refinement. The coarse mesh
exhibits lower self-weight compliance in this test: the observed bending
response depends strongly on spatial discretization. The matching trend in
the independent linear calculation supports this interpretation. The fine
mesh itself is not continuum truth: the medium mesh still differs from it by
about 22%, so these three meshes do not establish a mesh-independent result.
The nonlinear tail mean also need not equal a small-strain linear reference.

The output ``gravity/resolution.json`` records tail motion, mean deflection,
self-weight compliance (deflection / total beam weight), and difference from
the finest tested mesh. Compliance is a distributed self-weight response
measure, **not** a fitted Young's modulus or tip-load spring stiffness.
No timestep-convergence or real-material identification claim is made.

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 uv run --extra examples -m newton.examples monolithic_tet_response --experiment gravity --viewer null --device cpu --test --output /tmp/tet-gravity
   uv run --extra examples -m scripts.monolithic_reference.plot_tet_response --results /tmp/tet-gravity --output /tmp/tet-gravity-plots

Use ``--device cuda:0`` for the same CUDA fixture. The gravity plot overlays
all three physical tip histories, their linear static references, nonlinear
tail means versus tet count, elastic/kinetic energy and volume change.

Extended refinement and computation cost
----------------------------------------

Select additional meshes with ``--gravity-refinements``. The integers are
strictly increasing subdivision factors from 1 to 8; each factor r creates
``15*r^3`` tetrahedra and ``(3*r+1)*(r+1)^2`` nodes. Geometry, material,
density, gravity, timestep and the near-static checks remain identical.
The default three-mesh view is retained for interactive responsiveness.
The r<=8 limit belongs to this demonstration fixture, not to the solver.

.. code-block:: bash

   uv run --extra examples -m newton.examples monolithic_tet_response --experiment gravity --gravity-refinements 2 4 6 8

For quantitative comparison, run each mesh separately without a viewer.
The following sweep uses factors 1, 2, 3, 4, 6 and 8 (15 through 7680 tets).
Run CPU and CUDA sequentially, with no other simulation or GPU workload.

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --extra examples -m scripts.monolithic_reference.run_gravity_resolution --device cpu --output /tmp/gravity-sweep-cpu
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --extra examples -m scripts.monolithic_reference.run_gravity_resolution --device cuda:0 --output /tmp/gravity-sweep-cuda
   uv run --extra examples -m scripts.monolithic_reference.plot_gravity_resolution --results /tmp/gravity-sweep-cpu /tmp/gravity-sweep-cuda --output /tmp/gravity-sweep.png

Each mesh first runs a separate 100-step warmup trajectory to exercise loaded
solver paths. Each of two measured repetitions starts with a fresh solver and
State and executes the full 36-second trajectory. Construction and the dense
NumPy static reference solve are timed separately and excluded from stepping
cost. File writing and rendering are also excluded.

``solver_ms`` is wall time around ``solver.step`` with device synchronization
at its boundaries, including the solver's own host orchestration and internal
transfers. ``step_ms`` additionally includes force/gravity input setup and
physical measurements. ``measurement_ms`` includes State downloads and the
NumPy energy/geometry diagnostics. In particular, the existing dense reference
mass multiplication is diagnostic overhead, not a solver kernel. These
headless timings cannot be substituted for interactive viewer FPS.

Outputs retain per-step timing and physical traces, mean/median/p95 latency,
nonlinear and PCG iteration counts, setup time, and separate ramp, settling
and tail statistics. Simulated seconds per wall second use the **mean** step
cost; a value below one is slower than real time. Plot error bars show the
range of the two repetition means, not a statistical confidence interval.

Convergence plots compare verified tail means, show successive relative
changes and use the finest verified mesh only as a finite-resolution
reference. A failed or unsettled repetition excludes that mesh from the
comparison and causes a nonzero runner exit; it cannot serve as a reference.
Decreasing differences indicate a refinement trend, not proof of convergence
to a continuum solution or a changed physical material modulus. The step
size is fixed at 20 ms throughout; no temporal error estimate is implied.

The plot also retains complete dynamic runs that fail the near-static gate:
these tail means are marked with x and bars spanning plus/minus the full
tail range. They illustrate the observed response but are not accepted
static solutions. Their timings remain useful for the same complete
36-second dynamic workload. The cost-versus-refinement panel uses the
independent linear static reference difference, explicitly labelled; it
does not turn an unsettled nonlinear response into a verified static error.

Measured refinement trend
-------------------------

On a Ryzen 9 8945HX / RTX 5070 Ti Laptop with the protocol above, the
complete two-repeat sweep gave the following mean solver costs (median of
the two run means). These are physical-step costs, not viewer frame times.

.. list-table:: Six-mesh sweep (2026-09-08)
   :header-rows: 1

   * - Tets
     - Linear static tip [mm]
     - CPU solver [ms/step]
     - CUDA solver [ms/step]
   * - 15
     - 104.818
     - 5.50
     - 9.36
   * - 120
     - 205.621
     - 11.48
     - 14.82
   * - 405
     - 265.808
     - 24.12
     - 20.29
   * - 960
     - 296.427
     - 48.33
     - 28.40
   * - 3240
     - 324.475
     - 286.41
     - 143.80
   * - 7680
     - 335.976
     - 630.22
     - 184.83

The last two linear static references differ by 3.42%. This supports a
spatial refinement trend but does not establish a mesh-independent result.
The nonlinear runs all complete with 100% normal solver convergence;
near-static checks pass only r1/r2/r3/r4 on CPU and r1/r2/r3 on CUDA.
At r8, the final two-second tip range is about 21.4 mm on CPU and 23.0 mm on
CUDA, so nonlinear static convergence has **not** been demonstrated. The
runner therefore exits nonzero while retaining all measurements.

Mean PCG iterations rise from about 10 to 403 per step. Iteration count
is an important profiling target, while assembly/residual and other
solver work also contribute: r8 steps with zero PCG iterations still cost
about 181 ms on CPU and 7 ms on CUDA. These observations do not provide
a kernel-level attribution or a solver complexity bound.

P2 deformable profiling study
--------------------------------------

The independent P2-0B-T study ran ``--experiment gravity
--gravity-refinements 5`` on CPU and CUDA: 1875 tets, 576 nodes and 36
fixed nodes, with r3/r4/r6 controls. Each displayed frame advances six
20 ms physical steps. Loading-phase PCG iteration count dominates the
cost; prefix BSR is not the r5 bottleneck. r5/r6 remain not near-static.
The original quality checks were not relaxed. Records are in
``scripts/monolithic_reference/fixtures/p2_deformable_profile_plan_v1.json``
and ``agents/integration/P2_0_DIAGNOSTICS_REPORT.md``. Example and solver
defaults are unchanged.

The P2-0 record audit identifies incomplete measurement coverage. Sharpa
current timings cover only the first step of each profiling window; material
comparisons evolve from common initial positions instead of replaying fixed
candidates. Quiet Sharpa repetitions and loaded tet timing repetitions remain
incomplete. The original compact FPS field is a real-time factor; corrected
units are published separately without replacing measurements. See the
validation gaps in ``p2_0_diagnostics_v1.json`` and report section 10.
These findings do not change production defaults or certify P2-0 completion.
