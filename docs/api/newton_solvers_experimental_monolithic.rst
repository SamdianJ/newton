.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers.experimental.monolithic
======================================

Experimental implicit articulated rigid/tet solver with P1Q3 contact.

.. experimental::

The solver advances one supported articulation and one connected tet body with
explicit C-EE, C-IE or C-II contact, scaled PCG and transactional publication.
The collision pipeline can also detect tet-face contacts independently.

State and migration
-------------------

Construct ``SolverMonolithic(model, contact_model=..., globalization=...,
friction_velocity_scale=..., ...)`` directly. The experimental q-only, tet-only
and private coupled factories are no longer supported. Constructor options are
explicit; historical spring friction and ``tangential_stiffness`` are rejected.
The defaults remain strict publication, diagnostic PCG and CUDA graphs disabled.
C-IE/C-II or masked contact with Armijo/ArmijoOrResNorm require the explicit
``allow_nonconservative_energy=True`` opt-in.

Use ``create_state(dt=..., source=None)`` for complete FP64 coordinates and P3
integration history. An ordinary ``model.state()`` can supply initial coordinates
and velocities via ``source``; it cannot be passed directly to ``step`` for
continuation. ``reset_state(state, dt=...)`` rebuilds history and clears published
forces. Generic ``reset`` is rejected because history initialization needs dt.

``save_checkpoint(state, path)`` preserves the P3 NPZ schema and appends ``.npz``
when absent. It writes a sibling temporary file and replaces the destination
after success. ``load_checkpoint(state, path)`` validates in scratch before
replacing the target. Both return None. Incompatible configurations or edited
bound models are rejected; rebuild the solver after model changes.

``get_contact_forces(state)`` returns an independent read-only FP64 NumPy snapshot.
Later stepping, resetting or loading does not modify previously returned arrays.
Fresh/reset states have invalid zero forces; published Stopped states can have
valid forces without satisfying equilibrium. Hard failures retain the original
state and its force generation. See :class:`MonolithicContactForces` for units.

``supports_collision_pipeline`` is False and ``contacts`` returns None. External
Contacts, contact updates and collision scheduling are rejected. Use
:class:`MonolithicCollisionPipeline` separately when independent queries are needed.

.. py:module:: newton.solvers.experimental.monolithic
.. currentmodule:: newton.solvers.experimental.monolithic

.. rubric:: Classes

.. autoclass:: MonolithicCollisionPipeline

.. autoclass:: SolverMonolithic

.. autoclass:: MonolithicContactForces

.. autoclass:: SmoothSDFSet

.. autofunction:: build_smooth_sdf

Smooth SDF construction
-----------------------

Pass ``build_smooth_sdf(model)`` to the solver's ``smooth_sdf`` argument. It
samples existing attached mesh SDF textures and constructs FP64 C2 B-splines;
it does not bake meshes again. ``representation="c1-hermite"`` selects Hermite
interpolation. Sampling performs node and gradient checks, without certifying
geometry. ``qualified_directory=...`` instead requires frozen nodes, matching
source textures, geometry certificates and exactly reproduced coefficients.
Failed qualification raises and never falls back to sampling.

The returned ``SmoothSDFSet.build_metadata`` records the source, checks and
qualification separately from its physical ``identity`` and ``digest``.
Coefficient buffers and descriptors remain bound for the solver lifetime;
changes require rebuilding the fields and solver.

Repository regression runner
----------------------------

From the development checkout, list the frozen profiles without loading CUDA::

    uv run --no-sync python -m scripts.monolithic_reference.run_regression --profile dev --list
    uv run --no-sync python -m scripts.monolithic_reference.run_regression --profile full --list

Run with ``--output <new-directory>``. Each case executes serially in its own
process and binds source paths/hashes, frozen inputs and comparison tolerances.
Dev includes complete default Franka and recorded Sharpa trajectories. The
current full profile has nine Franka law/globalization combinations and one
Sharpa OR/C-IE case, all CUDA production with graphs. ``--case <id>`` filters
within the chosen profile and reports PARTIAL instead of profile qualification.
Missing frozen external inputs block execution; the runner does not fetch or
replace them. Sharpa completion does not certify grasp success.

Use ``scripts.monolithic_reference.replay_franka_grasp`` or
``scripts.monolithic_reference.replay_sharpa_grasp`` with a recording directory
and ``--check-only`` to verify display frames without simulating again.
