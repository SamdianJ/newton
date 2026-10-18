.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Experimental monolithic cables
==============================

.. experimental::

   ``SolverMonolithic.CableInput`` and its state/control namespaces are
   experimental. Open, independent chains support finite axial stiffness,
   anisotropic bending and natural curvature/twist. Closed loops, branches,
   connectors, self-collision and soft-soft contacts are outside this interface.
   Cable surfaces contact supported articulation, free rigid and static shapes.
   Cloth can share the solve; see :doc:`monolithic_cloth`.

Supply ordered Model particle IDs, reference directors and material angles
through ``cable_inputs``. A director is perpendicular to its edge and specifies
the orientation of the material bending tensor. Dynamic Model masses must equal
the reference line-density lumping. A cable needs no mechanical tree or tet;
several separate cables may share a global solve. Create state and controls with
``solver.create_state(dt=...)`` and ``solver.create_control()``.

``control._monolithic_cable.target_position`` prescribes fixed cable vertices;
``target_theta`` prescribes fixed edge angles. Positions and theta are independent
boundary conditions. For a cantilever clamp, fix the first two vertices and the
first material angle. Control torques act only on physically free angles.
Vertex forces use ``state.particle_f``. Targets and loads are frozen per step;
resetting state never changes the natural shape.

Each step transports reference directors from the committed state to the
candidate tangents. Reference twist follows its committed unwrapped branch;
material angles are never reduced modulo a full turn. Rejected steps preserve
coordinates, directors, angles, velocities and force-publication history.
Checkpoint continuation preserves this chart and validates material identity.

Surface contact and spin
------------------------

Each edge samples its circular surface using two axial Gauss points and 64
angular points. The two open ends additionally sample disks with two Gauss rings
in the area coordinate. Areas use reference edge lengths and the supplied
radius. Interior edge junctions add no caps or spherical joints. This discrete
surface is not a continuous capsule union; narrow contact patches and sharply
bent junctions need a resolution assessment.

Surface positions and analytic Jacobians include xyz, transported directors and
theta. Friction therefore applies spin torque as well as nodal forces. Common
spin is retained whenever material contact can constrain it; temporary null
modes are checked against both material and contact tangents. Radius and bending
stiffness remain independent inputs.

``solver.get_contact_forces(state).cable_torque`` is an independent, read-only
array of generalized contact torque [N m], in input cable/edge order, including
fixed-angle reactions. It shares the force validity flag, publication generation,
checkpoint and rollback transaction with the existing force fields. This is a
chart-coordinate torque: for a general transported chart, world torque balance
uses the full xyz/theta virtual-rotation map rather than only nodal lever arms.
Material checkpoint schema 6 is required for this history.

Target types, normal selection, energy-acceptance opt-in and capacity behavior
are described in :doc:`monolithic_cloth`. CPU/CUDA short-contact qualification
and graph coverage do not establish long contact-trajectory or performance
qualification, or remove the existing localized-helix acceptance limitations.

Numerical contract
------------------

The nonlinear energy and residual use the selected native precision. The
position and massless material-angle coordinates share one Newton direction,
line search and publication transaction. There is no separate theta equilibration
loop and no theta inertia. The full strain-Jacobian Gauss-Newton matrix includes
position/angle cross terms; only Dirichlet constraints and the specified permanent
isotropic gauge are eliminated when contact is absent. A temporary common
angle mode is rechecked at each assembly and released when bending or contact
constrains it.

B and PCG vectors remain FP32. Per-cable band Cholesky uses interleaved vertex
and angle rows only in the preconditioner; cross-object entries remain in the
global operator. The fixed angle unit scale is the inverse median rest-edge
length. Cancellation-prone sparse products accumulate in FP64 before rounding
to FP32. FP32 correction solves accumulate into a native delivered direction,
which must pass a separate physical posterior check. A rounded matrix that
loses positive definiteness can retry with diagonal regularization; the original
energy and residual remain the convergence criteria. Nonmaterial numerical
policies are unchanged.

The residual-norm predicate normalizes every row by its own tolerance, so
position roundoff cannot hide unresolved torque.

Every cable position and free-angle row must satisfy the solver's per-row force
[N] and torque [N m] tolerance described in :doc:`monolithic`. A common-angle
null mode additionally requires the net free-angle torque to stay within the
absolute torque target. Prescribed boundary reactions are excluded from free equilibrium.
These accuracy requirements can reject FP32 native runs or poorly conditioned
inputs. Stopped publication is never analytical convergence. The material
nonlinear driver uses host acceptance; CUDA graph mode captures the linear
solve and supported segments, not the entire material step.
Band factorization and triangular solves are sequential within each chain;
long chains can be slower on CUDA than CPU. This experimental path has no
accelerator performance qualification yet.
Use CPU for the full localized-helix analytical matrix. CUDA coverage includes
the full cantilever and component checks; CPU/CUDA trajectory equivalence during
helical buckling has not been qualified.

Analytical examples
-------------------

The cantilever applies a small transverse tip force to each principal bending
axis. The independent Euler--Bernoulli reference is
``w(s)=F*s^2*(3L-s)/(6B)``. The first edge is fixed, so the clamp discretization
has a first-order mesh error that must converge; it is not removed by fitting
an effective length.

.. code-block:: console

   python -m newton.examples monolithic_cable_cantilever --viewer null --test
   python -m newton.examples monolithic_cable_cantilever --axis 2 --viewer null --test

The localized helical example rotates one end by 27 turns, then shortens the
naturally straight isotropic rod symmetrically from both ends. Its reference is the long-rod Kirchhoff
localized solution from `van der Heijden and Thompson (2000)
<https://doi.org/10.1023/A:1008310425967>`_, as used in
`Discrete Elastic Rods (2008), section 9.1
<https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf>`_. The reference depends only
on prescribed end rotation, shortening and material constants. Centerline,
tangent envelope, torque and twist/writhe balance are checked. Only arbitrary
transverse phase is removed; localization width, tilt and torque are not fitted.
Finite-end tails and finite-EA errors are reported separately. Every EA case
retains all single-case analytical and physical gates. The low-to-baseline EA
shape change is an error-decay sample. The baseline-to-high EA shape change
must stay below 1% and be smaller than half the low-to-baseline change.

.. code-block:: console

   python -m newton.examples monolithic_cable_helix --segments 720
   python -m scripts.monolithic_reference.run_cable_matrix cantilever --output /tmp/cantilever-matrix
   python -m scripts.monolithic_reference.run_cable_matrix localized_helix --output /tmp/helix-matrix

The full recipes and gates are in
``scripts/monolithic_reference/cable_recipes.json``. The helix uses the classical
9.29 m length and 0.3 m shortening, with all rigidities and mass scaled by 1e-6;
this preserves geometry and characteristic times. Coordinates are centered to
reduce roundoff without changing the physical experiment. Each matrix includes
three meshes, finite-EA comparisons and fresh-solver checkpoint continuation.
Receipts retain source hashes, execution mode, failures and physical metrics.
A partial run is not PASS. The viewers overlay the reference in orange and the
simulation in blue; cantilever display magnification changes only the drawing.

Long benchmark runs save a progress checkpoint every 100 converged steps.
Repeat the same single-case or matrix command with ``--resume`` after an
interruption. Resume requires identical sources, recipe and execution settings;
a failed experiment cannot be resumed into a passing acceptance result.

Current qualification
---------------------

The cantilever matrix and its full CUDA production/graph runs pass. For the
localized helix, the complete CPU 360/720-segment runs pass the physical gates;
centerline errors decrease from 9.9902% to 2.7523% (observed order 1.86).
This is evidence consistent with approximately second-order convergence over
one refinement, not a completed three-mesh qualification.

The 720-segment tangent-envelope error is 3.2850%, above the original 3% fine-mesh
limit. Its torque error is 0.3092% and writhe error is 0.005306 turns. This
experimental delivery is accepted with a documented concession. The screenshot
uses this completed 720-segment simulation. The interactive command above does
not certify the strict analytical test: adding ``--test`` still enforces the
original limits and rejects that envelope error.

The 1440-segment baseline/low-EA runs are incomplete; the high-EA run fails at
step 2001 with regularization exhausted and position residual 1.217e-9 N above
its 1e-9 N limit. Fine-mesh robustness, the remaining refinement comparison,
and the cross-EA shape/decay gates remain open. Their original recipes and
failures are retained. The full matrix command still requests those experiments
and must not be interpreted as a completed PASS. No STOP acceptance or change
to the physical thresholds is used to grant the concession.
