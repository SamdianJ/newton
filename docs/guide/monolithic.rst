.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _monolithic-normal-loading:

Monolithic Normal Loading
=========================

.. experimental::

   The ``newton.solvers.experimental.monolithic`` module, including
   ``SolverMonolithic`` and ``MonolithicCollisionPipeline``, may change without
   the normal deprecation period. The normal-loading example uses frictionless normal
   contact between one articulated rigid actor and one connected tetrahedral
   soft body in a single world.

Run the example
---------------

The example compresses a tetrahedron against its fixed apex using a dynamic
infinite plane attached to a world-anchored prismatic joint. All geometry is
built locally; no downloaded assets, SuperDex installation or reference scripts
are needed.

.. code-block:: bash

   uv run --extra examples -m newton.examples monolithic_normal_loading
   uv run --extra examples -m newton.examples monolithic_normal_loading --viewer null --device cpu --num-frames 100 --test
   uv run --extra examples -m newton.examples monolithic_normal_loading --viewer null --device cuda:0 --num-frames 100 --test

Each display frame advances ten substeps of 0.001 s. The full test runs for
1 s: free approach ends at 0.2 s, loading ends at 0.7 s, and the remaining time
holds the command. Longer interactive runs continue holding the final target.
The test checks every substep for convergence, fixed-apex preservation,
penetration below 1 mm and determinant of the deformation gradient above 0.2.
Its final check requires at least 5 mm of deformation measured relative to the
fixed apex. This example is a regression and usage demonstration; the release
harness separately checks the frozen force-displacement envelope.

The plane mass is 0.1 kg, the tetrahedron density is 1000 kg/m³, and the initial
vertices are ``(-0.02, -0.015, 0)``, ``(0.02, -0.015, 0)``, ``(0, 0.03, 0)`` and
``(0, 0, 0.03)`` m. The last vertex is fixed. Material parameters are
``mu = 3846.15380859375`` Pa and ``lambda = 5769.23095703125`` Pa, with zero
material damping. Contact uses a 1 mm particle radius, a 2 mm candidate gap and
normal stiffness ``1e7`` N/m³. The gap controls candidate discovery; it is not
an allowed penetration depth.

Use the public API
------------------

.. code-block:: python

   import newton
   from newton.solvers.experimental.monolithic import (
       MonolithicCollisionPipeline,
       SolverMonolithic,
   )

   # Build the supported articulation and tetrahedral mesh first.
   model = builder.finalize()
   model.request_contact_attributes("force")
   state = model.state()
   newton.eval_fk(model, state.joint_q, state.joint_qd, state)
   control = model.control()
   collision = MonolithicCollisionPipeline(model, soft_contact_gap=0.002)
   solver = SolverMonolithic(
       model, collision_pipeline=collision, contact_stiffness=1.0e7
   )
   # Set external control.joint_f once before each implicit substep.
   solver.step(state, state, control, None, 0.001)
   stats = solver.last_stats

The example computes an external PD force with gains 2000 N/m and 30 N·s/m.
That force remains fixed during the nonlinear solve; model actuator gains are
zero. The solver owns contact generation at accepted and trial states, so do
not pass default ``CollisionPipeline`` contacts to ``step``. In-place stepping
is supported. Production numerical defaults are used without overrides.

``last_stats`` exposes convergence, rollback, nonlinear and linear iteration
counts, current-scale convergence ratios and physical guards. A nonlinear
iteration limit can commit a safe last candidate with ``converged=False``;
a hard failure rolls back the state. Check these fields instead of assuming
that returning from ``step`` means convergence. Contact force publication
requires requesting the force attribute before constructing the solver.

Optional implicit joint terms
-----------------------------

See :ref:`monolithic-p1-contract` for the versioned P1 input, factor/history,
joint and Sharpa fixture contracts, including the still-disabled capabilities.

The external-force example above is unchanged. To enable implicit joint PD,
physical joint limits and regularized joint friction, pass
``SolverMonolithic.JointTerms`` when constructing the solver:

.. code-block:: python

   # Example widths for two DOFs ordered [revolute, prismatic].
   terms = SolverMonolithic.JointTerms(
       implicit_pd=True,
       limits=True,
       friction=True,
       limit_width=(0.02, 0.002),             # rad, m
       friction_velocity_scale=(0.01, 0.005), # rad/s, m/s
   )
   solver = SolverMonolithic(
       model, collision_pipeline=collision, contact_stiffness=1.0e7,
       joint_terms=terms,
   )

Configure Model joint gains, lower/upper bounds, limit gains, nonnegative
friction magnitudes and positive effort/velocity limits before construction.
Each driven DOF requires ``JointTargetMode.POSITION_VELOCITY``; either PD gain
may be zero. Parameters are copied into the solver: rebuild after changing
Model parameter values. The widths above illustrate units, not calibrated
settings for arbitrary assets.

Set ``control.joint_target_q`` and ``control.joint_target_qd`` before each
step, using ``model.joint_target_q_start`` to index position targets. Initialize
targets explicitly and generate a speed-limited trajectory outside the solver.
Set ``joint_f=0`` on driven DOFs: supplying both PD and an external drive on
those DOFs is rejected before the state transaction. Undriven DOFs retain the
external frozen-force path. Position/velocity targets are copied once per
physical step; each nonlinear trial recomputes forces at its candidate state.

The effort limit clamps the sum of elastic and damping PD forces. The PD
position tangent is ``kp + kd/dt`` inside the bound and zero when saturated,
including the boundary. Limit forces act only outside lower/upper bounds;
the smoothly activated limit damper only opposes motion further outside.
Limits are penalties with finite compliance, not post-step coordinate clamps.
Joint friction uses ``-f*v/sqrt(v*v + v_eps*v_eps)``; it dissipates energy but
does not implement an exact static-friction lock at zero velocity. Limit and
friction forces are independent of the PD effort bound.

Returned-state ``last_stats.joint_pd_force``, ``joint_limit_force`` and
``joint_friction_force`` contain per-DOF physical forces (N*m or N), with
saturation flags, damping/friction power and ``joint_force_generation``.
They are recomputed for the state actually returned, including rollback;
rejected trials do not publish their diagnostics. Without ``joint_terms`` the
fields remain absent/None and the generation is -1.

The G1 component tests cover CPU and CUDA. They do not certify the Sharpa
hand-closing action (G1H) or soft-object grasping. G1H has a separate fixture below.

Sharpa joint closure diagnostic
-------------------------------

The internal G1H example uses the base 22-DoF Sharpa URDF and a time-stamped
four-second CSV closure, with no particles, contacts or SDF. Supply the local
asset directory and trajectory explicitly; these external assets are not bundled:

.. code-block:: bash

    python -m newton.examples monolithic_sharpa_close \
        --asset-dir /path/to/left_sharpa_wave --trajectory /path/to/motion.csv \
        --device cuda:0 --viewer null --test --output /path/to/results

Use ``--device cpu`` for the matching CPU fixture, or omit ``--viewer null``
for visualization. The fixture uses 1 ms physical steps, implicit PD, joint
limits and nonzero joint friction. Its gains are simulation calibration
parameters, not identified hardware properties. The signed CSV adapter defines
an URDF closure; it does not assert equivalence to the source CAD coordinates.

The example's private joint diagnostic factory is an internal test entry, not a
supported general rigid-only mode. It reuses the production nonlinear and linear
solver and state transactions. Particle/tet/contact metrics are not applicable
(and use NaN in solver stats), not successful measurements. The normal public
constructor continues to require a connected tet body.

Joint candidates accumulate a displacement relative to the beginning of the
step. BE velocities are recovered from that displacement before absolute
positions are rounded into float32 State storage. This preserves sub-ULP motion
at nonzero joint angles; differencing two stored absolute positions can lose it.
The change applies to the shared candidate path, including rigid/tet steps.

Outputs include the asset/parameter manifest, step trace and per-joint summary.
A rollback stops the trajectory without advancing simulated time. This action
has no self-collision and does not certify soft-ball grasping or mesh contact.

Current support boundaries
--------------------------

The default material is Kim stable Neo-Hookean without a logarithmic term,
with Newton lumped particle mass and first-order backward Euler integration.
PR-6A also supports Smith log-stabilized material and consistent P1 mass;
see :ref:`monolithic-p1-contract` and :ref:`monolithic-g2h`. Contact uses
fixed P1Q3 boundary quadrature, with a quadratic hinge by default. PR-6B adds
opt-in PolyReLU and elastic contact friction with transactional history; see
:ref:`monolithic-contact-friction`. Full grasping acceptance remains incomplete.
Strict SuperDex matching is deferred until after V0.2.

The example's validation applies to the exact geometry, material, timestep and
loading conditions above. Separate frozen C4 fixtures define additional
measured support. Merely accepting an analytic shape type does not establish
accuracy for arbitrary curved meshes or narrow features: a feature can lie
between all fixed quadrature samples. The 2 mm narrow-feature failure remains
outside the measured support envelope.

Supported joints are world-anchored tree joints of type fixed, revolute or
prismatic, with zero armature and independent passive damping. Joint friction,
limit gains and target gains must also be zero unless their corresponding
``joint_terms`` capability is explicitly enabled. Moving Dirichlet nodes, loop closures, mimic joints, multiple actors,
multi-world batching and CUDA graph capture are unsupported. The tetrahedral
surface must exactly match the derived boundary. Fixed-node input velocities
must be zero. Finite planes and ellipsoids are rejected; this example uses an
infinite plane. Volume SDFs have additional provenance and scale restrictions.

CPU and CUDA correctness are supported. Python orchestration and explicit
assembly are current implementation limits; this tiny example does not imply
GPU speedup or a large-scene performance guarantee.

