.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _monolithic-normal-loading:

Monolithic Normal Loading
=========================

.. experimental::

   The ``newton.solvers.experimental.monolithic`` module, including
   ``SolverMonolithic`` and ``MonolithicCollisionPipeline``, may change without
   the normal deprecation period. The current scope is frictionless normal
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

Current support boundaries
--------------------------

The material is Kim stable Neo-Hookean without a logarithmic term, with Newton
lumped particle mass and first-order backward Euler integration. Contact uses
fixed P1Q3 boundary quadrature and a quadratic hinge. There is no friction,
contact history or grasping support. Smith material and consistent mass belong
to later development; strict SuperDex matching is deferred until after V0.2.

The example's validation applies to the exact geometry, material, timestep and
loading conditions above. Separate frozen C4 fixtures define additional
measured support. Merely accepting an analytic shape type does not establish
accuracy for arbitrary curved meshes or narrow features: a feature can lie
between all fixed quadrature samples. The 2 mm narrow-feature failure remains
outside the measured support envelope.

Supported joints are world-anchored tree joints of type fixed, revolute or
prismatic, with zero armature, damping, friction, limit gains and actuator
gains. Moving Dirichlet nodes, loop closures, mimic joints, multiple actors,
multi-world batching and CUDA graph capture are unsupported. The tetrahedral
surface must exactly match the derived boundary. Fixed-node input velocities
must be zero. Finite planes and ellipsoids are rejected; this example uses an
infinite plane. Volume SDFs have additional provenance and scale restrictions.

CPU and CUDA correctness are supported. Python orchestration and explicit
assembly are current implementation limits; this tiny example does not imply
GPU speedup or a large-scene performance guarantee.
