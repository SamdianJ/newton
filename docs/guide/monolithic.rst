.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _monolithic-normal-loading:

Monolithic Coupled Solver
=========================

.. experimental::

   :mod:`newton.solvers.experimental.monolithic` supports one articulation and
   one connected tetrahedral body. Its interfaces may change without the normal
   deprecation period. The old BE, q-only and tet-only solver entry points and
   their examples have been retired.

Use the public API
------------------

Build the supported model and supply its rest density, then create the complete
solver state. A plain ``model.state()`` can initialize this state but cannot be
used directly for continuation.

.. code-block:: python

   from newton.solvers.experimental.monolithic import SolverMonolithic

   solver = SolverMonolithic(
       model,
       contact_model="C-EE",
       globalization="ResNorm",
       friction_velocity_scale=0.01,
       tet_rest_density=rest_density,
   )
   state = solver.create_state(dt=0.001, source=model.state())
   control = model.control()
   solver.step(state, state, control, dt=0.001)
   forces = solver.get_contact_forces(state)
   solver.save_checkpoint(state, "checkpoint.npz")
   solver.load_checkpoint(state, "checkpoint.npz")

The default solver uses FP64 nonlinear arithmetic, strict publication, diagnostic
PCG and no CUDA graph.
Force results are independent read-only NumPy snapshots; check ``force_valid``
and ``force_generation`` before using them. Stopped publication under an opt-in
production policy is not convergence. Reset with ``reset_state(state, dt=...)``.
Checkpoint restoration validates complete state and configuration identity before
replacing the destination state.

The experimental constructor parameter ``precision`` accepts
``SolverMonolithic.Precision.FP64`` / ``FP32`` or ``"fp64"`` / ``"fp32"``.
It cannot change after construction. FP32 selects native state, history,
contact forces, E/r, posterior checks and host acceptance arithmetic; B/PCG
arrays and directions remain FP32 in both modes. Existing PCG scalar threshold
and restart checks retain their original FP64 arithmetic in both modes.
FP32 numerical qualification is pending: unchanged
physical guards and noise envelopes may reject a candidate.

The precision-tagged state uses schema 3. Cross-precision continuation and
checkpoint loads are rejected. Schema-2 FP64 checkpoints are validated against
the original complete field schema and physical digest in scratch before being
upgraded; failed loads preserve the destination. A plain initialization source
is a new start, not a conversion of integration history.

Smooth SDF construction still produces common offline FP64 coefficients.
The FP32 solver derives a separately identified coefficient copy and retains
validation of the reference source. Force snapshots use the selected precision.

For mesh contact, attach the official mesh SDF before model finalization and use
``build_smooth_sdf(model)`` for the derived C2 field. A qualified frozen node
source can instead be supplied through ``qualified_directory``; failed
qualification is rejected without falling back to resampling.

``MonolithicCollisionPipeline`` remains independently available, but the coupled
solver does not accept external Contacts or collision scheduling configuration.

Examples and validation
-----------------------

See :doc:`monolithic_franka_grasp` for the complete Franka demonstration and
:doc:`monolithic_sharpa_assets` for asset preparation. Sharpa's recorded scenario
checks complete advancement and state safety; successful grasping is not certified.

The repository's scene matrix uses explicit frozen external assets:

.. code-block:: bash

   uv run --no-sync python -m scripts.monolithic_reference.run_p4 --profile full --list
   uv run --no-sync python -m scripts.monolithic_reference.run_p4 --profile dev --output /path/to/new-output

The scene runner's result must be combined with component, lifecycle and replay
checks for stage qualification. Retired driver snapshots, hashes and migration
notes are documented in ``scripts/monolithic_reference/legacy_retirement.md``.

The precision experiment runner enumerates eight scenes in both precisions::

   uv run --no-sync python -m scripts.monolithic_reference.run_p5_precision --list
   uv run --no-sync python -m scripts.monolithic_reference.run_p5_precision --freeze-controls --output /path/to/controls
   uv run --no-sync python -m scripts.monolithic_reference.run_p5_precision --controls /path/to/controls --output /path/to/results

Use ``--max-steps 2`` for a smoke run. A complete run reports
``COMPLETE_NOT_AUDITED`` until the independent T5 numerical audit is performed;
short runs report ``PARTIAL`` and hard failures retain their first-failure state.
This runner does not yet perform the separate T5 performance experiment.
