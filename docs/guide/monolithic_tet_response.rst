.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _monolithic-tet-response:

Monolithic material and vibration demonstration
================================================

Run two focused comparisons with the experimental monolithic solver:

.. code-block:: bash

   uv run --extra examples -m newton.examples monolithic_tet_response
   uv run --extra examples -m newton.examples monolithic_tet_response --experiment mass

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
