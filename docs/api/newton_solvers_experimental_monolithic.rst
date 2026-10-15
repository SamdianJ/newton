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

Material input and continuation (experimental)
---------------------------------------------

``SolverMonolithic.ClothInput`` and ``SolverMonolithic.CableInput`` describe
independently owned shell meshes and open, anisotropic DER chains. Pass tuples
through the keyword-only ``cloth_inputs`` and ``cable_inputs`` arguments.
This interface is under development: material scenes provide construction,
layout, control and continuation storage only. Their ``step()`` explicitly
raises ``NotImplementedError`` until material assembly is connected. The
existing articulation/rigid/tet stepping path is unchanged.

Cloth inputs select Model triangle rows, bending coefficients ``(alpha, beta)``,
thickness [m] and surface density [kg/m²]. Cable inputs select ordered particle
IDs, ``EA`` [N], a symmetric positive definite 2x2 ``bending`` tensor [N m²],
``GJ`` [N m²], radius [m], linear density [kg/m], unwrapped ``rest_theta`` [rad],
unit perpendicular ``rest_director`` vectors and boolean ``fixed_theta`` flags.
Dynamic particle masses must match reference density lumping. Multiple separate
objects are supported; each input must describe one connected object. Shared
particles, invalid topology and conflicting native forces are rejected.
Inputs are copied at construction; later caller edits do not alter the solver.

Material scenes may omit both the mechanical tree and tet. They use version-five
checkpoint history containing theta, directors and the unwrapped reference twist.
``create_state(dt=...)`` initializes the reference configuration. Explicit
``source`` initialization with cables requires compatible material history; an
ordinary ``model.state()`` can initialize cloth-only scenes but is insufficient
for cable orientation. Resetting integration
history preserves the frozen natural shape. Older checkpoints cannot supply
missing material history.

``create_control()`` retains ordinary joint controls and owns a
``_monolithic_cable`` namespace with ``target_theta`` [rad] and ``torque`` [N m]
arrays. Targets apply only to physically fixed theta; torques apply only to
physically free theta. This control namespace is experimental. No theta inertia
or angular velocity is allocated. USD authoring/import support is separate and
is not provided by these Python inputs.

.. rubric:: Other classes

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
