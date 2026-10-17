.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _monolithic-normal-loading:

Monolithic Coupled Solver
=========================

.. experimental::

   :mod:`newton.solvers.experimental.monolithic` supports one articulation and
   one connected tetrahedral body, with optional single-body world-root FREE
   rigid actors. Independent cloth/open cables and mixed material scenes support
   contact with articulation, free rigid and static shapes; see
   :doc:`monolithic_cloth` and :doc:`monolithic_cable`. Its interfaces may change
   without the normal deprecation period. The old BE, q-only and tet-only solver entry points and
   their examples have been retired.

For scenes without cloth/cable, spatial contact is the default. Candidates refresh at each Newton boundary and
remain frozen during its trials, using zero additional law margin [m]. This
approximation can omit contacts reached by a trial outside the frozen bounds;
it does not provide continuous collision detection or automatic correction.
The step-start convergence reference still uses complete contact enumeration.

Candidate capacity defaults to the model's complete static sample upper bound;
``contact_factor_capacity=None`` reserves three times that bound. These are
fixed capacities, not active contact counts. Explicitly reducing the factor
budget can discard whole contacts; overflow is reported and taints the result.
Existing checkpoints from the previous complete-contact default have a different
configuration identity and cannot be loaded into the new default. Use the offline
legacy recording tools to inspect those records, or start a new spatial run.

Material scenes use complete contact enumeration in the deterministic pool.
Their capacity overflow rolls back the step without publishing partial forces.
This separate material policy is described in :doc:`monolithic_cloth`.

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

Model topology, static physics and geometry (including SDF buffers and texels),
and solver policies must remain unchanged for the solver's lifetime. Changes
require a new solver and compatible state initialization. Runtime modification
of these values is unsupported and is not detected by stepping or checkpoint
operations. Construction validates the model and records its configuration
identity; loading a checkpoint still validates that recorded identity and the
complete state. There is no hot-update or model-validation mode.

Control targets and joint forces, state external forces, gravity and ``dt`` are
step inputs. Update them between calls; they must remain stable within a step.
Monolithic reads gravity each step without a model-change notification;
``notify_model_changed`` continues to reject model updates. Supported force,
control, finite-value and state-history checks remain active. To reposition
coordinates externally, use ``reset_state`` before continuation.

The experimental constructor parameter ``precision`` accepts
``SolverMonolithic.Precision.FP64`` / ``FP32`` or ``"fp64"`` / ``"fp32"``.
It cannot change after construction. FP32 selects native state, history,
contact forces, E/r, posterior checks and host acceptance arithmetic; B/PCG
arrays remain FP32 in both modes. Nonmaterial delivered directions remain
FP32; material scenes accumulate direction corrections in native precision. Existing PCG scalar threshold
and restart checks retain their original FP64 arithmetic in both modes.
Global scalar B assembly orders duplicate contributions by row, column and
unsigned FP32 value bits, then sums them with a fixed pairwise FP32 tree on CPU
and CUDA. This removes insertion-order dependence for identical triplet values;
it does not guarantee bitwise determinism of residuals, forces or full trajectories.
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

Rigid contact uses Model-selected shape pairs and fixed directed surface queries.
Articulation-rigid and rigid-rigid pairs sample both surfaces at half area;
rigid-static pairs sample the rigid surface at full area. Rigid-tet contact
samples tet faces when the rigid shape enables particle collision. Both dynamic
endpoints contribute to the coupled residual, tangent and published COM wrench.
Box, sphere and mesh sources are supported; mesh targets require CUDA and a
bound C2 SDF. New plane targets must be infinite and static. Changing collision
flags, pair storage or participating geometry requires rebuilding the solver.
Mixed-contact numerical qualification remains partial. Some Armijo cases stop
at the existing descent-slope noise floor while a rigid rotation residual still
exceeds its target. Strict publication rolls back those steps; they are a known
limitation, not converged results. Existing numerical and publication gates
apply. Articulation-tet and tet-static contact keep their original sampling and
behavior.
FREE PD, limit gains, friction, damping and armature must be zero.
Rigid ``body_f`` and FREE ``control.joint_f`` each contribute a world COM wrench
[N, N·m] once. Convert off-center loads to COM wrenches before calling ``step``.
Nonzero rigid torque with Armijo or ArmijoOrResNorm requires explicit
``allow_nonconservative_energy=True``; ResNorm supports that torque directly.

Smooth SDF construction still produces common offline FP64 coefficients.
The FP32 solver derives a separately identified coefficient copy and retains
construction-time validation of the reference source. Both fields must remain
unchanged during simulation. Force snapshots use the selected precision.

For mesh contact, attach the official mesh SDF before model finalization and use
``build_smooth_sdf(model)`` for the derived C2 field. A qualified frozen node
source can instead be supplied through ``qualified_directory``; failed
qualification is rejected without falling back to resampling.

The coupled solver evaluates contact internally and does not accept external
Contacts or collision scheduling configuration.

Examples and validation
-----------------------

The experimental repository runner ``scripts.monolithic_reference.run_actor_coupling``
uses one prismatic pusher, two FREE boxes and a connected tet with a fixed face.
Its versioned input is ``scripts/monolithic_reference/fixtures/actor_coupling_v1.json``:
zero gravity, 1 ms timesteps, an 8 mm quintic loading/hold/unloading target,
FP64 C-IE/ArmijoOrResNorm and strict production/graph execution on CUDA.
Run it explicitly from the checkout::

   uv run --no-sync python -B -m scripts.monolithic_reference.run_actor_coupling --output /path/to/new-output

This command advances 2000 primary steps, restores the complete schema-4
checkpoint at step 1000 for another 1000 steps, then advances 2000 steps with
rigids created before the articulation. It authenticates each Model and Actor
layout before comparing native state, integration history and forces by physical
role. There is no cross-layout checkpoint migration. The old recording bridge
continues to accept only canonical articulation-plus-tet inputs.
Recordings contain both analytic targets and the actual FP32 Control targets;
control frames must match exactly across continuations and creation orders.

Every step must converge; Stopped and hard failures stop the dependent runs.
The final 100 hold steps require all three interfaces to carry at least 0.1 N,
tet displacement of at least 2 mm and relative force variation of at most 5%.
Native state, determinant, penetration, per-actor residual and reaction balance
guards also apply. Known Armijo limitations do not waive these gates.
The recording audit recomputes box gaps, tet determinants, native/display
bindings and force balance. It uses the promoted Model rest matrices and
normalized joint anchors; device-specific BVH caches are excluded from physical
Model identity. Dense cross-block diagnostics are intended for this small
acceptance scene, not performance measurement.

Review the batch ``summary.json``, ``RESULTS.md`` and per-case ``receipt.json``.
``--retain-recordings failures`` is the default: successful recordings are removed
only after all audits and comparisons pass; failed batches retain recordings.
``--retain-recordings all`` preserves all new recordings. Short component tests
do not establish complete hold qualification. This runner does not launch the
nightly matrix; both complete scene qualification and nightly regression are
manual gates, and stage qualification remains partial until they pass.

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
