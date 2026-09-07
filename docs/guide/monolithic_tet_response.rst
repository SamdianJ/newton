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
