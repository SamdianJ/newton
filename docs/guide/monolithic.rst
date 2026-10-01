.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _monolithic-normal-loading:

Monolithic Coupled Solver
=========================

.. experimental::

   :mod:`newton.solvers.experimental.monolithic` supports one articulation and
   one connected tetrahedral body, with optional single-body world-root FREE
   rigid actors. Its interfaces may change without the normal
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

The precision-tagged state uses schema 4, including actor ownership, layout
identity and native rigid continuation fields. Cross-layout and cross-precision
checkpoint loads are rejected. Schema-2 FP64 and schema-3 checkpoints are
validated against their original complete fields and physical digest in scratch
before migration to a matching canonical articulation-plus-tet model; failed
loads preserve the destination. A plain initialization source
is a new start, not a conversion of integration history.

FREE actors use native world COM position, rotation, velocity and integration
history. ``body_q`` and ``body_qd`` are derived FP32 displays. The FREE coordinate
and velocity segments remain synchronized through their Model mappings; FREE
velocity is COM linear/angular velocity in the parent joint frame.
Each rigid translation and rotation block must pass its own convergence gate.

Rigid contact is not yet supported. Disable particle collision on rigid shapes
and filter all allowed shape pairs involving rigid actors before finalization;
otherwise construction fails even when the shapes are initially separated.
Existing articulation-tet and tet-static contact remains available.
FREE PD, limit gains, friction, damping and armature must be zero.
Rigid ``body_f`` and FREE ``control.joint_f`` each contribute a world COM wrench
[N, N·m] once. Convert off-center loads to COM wrenches before calling ``step``.
Nonzero rigid torque with Armijo or ArmijoOrResNorm requires explicit
``allow_nonconservative_energy=True``; ResNorm supports that torque directly.

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

The repository's scene matrix uses the retained local files under
``asset/monolithic/data`` (not included in Git). The current default is the
recertified Sharpa generation in ``inputs/frozen-inputs-rebuilt.json``.
Historical manifest hashes remain unchanged; recorded paths are resolved through
``asset/monolithic/relocations.json`` rather than requiring old artifact links.

.. code-block:: bash

   uv run --no-sync python -m scripts.monolithic_reference.run_regression --profile nightly --list
   uv run --no-sync python -m scripts.monolithic_reference.run_regression --profile nightly --output /path/to/new-output

The nightly profile contains nine Franka configurations and one Sharpa configuration
and is triggered manually for overnight regression. The full profile selects the
same ten cases; dev selects the default configuration of each scene and still runs
two complete trajectories. Routine development and ordinary CI use functional
component and small integration tests instead. The Monolithic project policy is
in the workspace file ``dev/monolithic/project.md``; repository ``AGENTS.md``
links to that project-specific policy. The runner binds each case
to ``asset/monolithic/references.json``, verifies retained file hashes and frozen
controls, and rejects a changed physical configuration before stepping. Franka
uses its existing frozen comparison envelopes. The rebuilt Sharpa reference uses
native state chunks, read without conversion on disk; comparison includes the
initial frame, complete history, publication labels and frozen control frames.
Sharpa requires exact equality because no repeatability envelope has been
qualified for the rebuilt generation. Missing or differing data produces FAIL;
filtered profiles remain PARTIAL.

After completion, review ``summary.json``, ``RESULTS.md`` and per-case receipts.
By default, successful new recordings are removed after their summaries and
comparison results are saved; failed recordings remain for diagnosis. Pass
``--retain-recordings all`` to keep all new recordings. Existing reference
trajectories are never modified by this retention policy.

The scene runner's result must be combined with component, lifecycle and replay
checks for stage qualification. Retired driver snapshots, hashes and migration
notes are documented in the workspace file
``dev/monolithic/P4/script_records/legacy_retirement.md``.

The precision experiment runner uses the same ten configurations, defaults to
FP32, and accepts ``--precision both``. The retained controls already match the
rebuilt manifest; they do not need to be generated again::

   uv run --no-sync python -m scripts.monolithic_reference.run_precision --list
   uv run --no-sync python -m scripts.monolithic_reference.run_precision --controls asset/monolithic/data/inputs/controls --output /path/to/results

Use ``--max-steps 2`` for a smoke run. A complete run reports
``COMPLETE_NOT_AUDITED`` until the independent T5 numerical audit is performed;
short runs report ``PARTIAL`` and hard failures retain their first-failure state.
This runner does not yet perform the separate T5 performance experiment.
