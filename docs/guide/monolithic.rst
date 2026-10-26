.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _monolithic-normal-loading:

Monolithic Coupled Solver
=========================

.. experimental::

   :mod:`newton.solvers.experimental.monolithic` supports one world with
   independent fixed-base articulations, at most one connected tetrahedral body, single-body
   world-root FREE rigid actors and independent cloth/open cables. Any of these
   may be absent, but the scene needs at least one dynamic degree of freedom.
   Every scene uses the same step pipeline and contact path; see
   :doc:`monolithic_cloth` and :doc:`monolithic_cable` for material inputs. Its
   interfaces may change without the normal deprecation period. The old BE, q-only and tet-only solver entry points and
   their examples have been retired.

Each articulation retains its own joint block and force/torque convergence groups
inside a shared Newton solve. Trees may differ in topology and DoF count.
FREE roots with descendant links and same-tree self-contact are unsupported.
Shape contact between distinct articulations projects onto both trees and
retains their cross blocks in the shared sparse operator. Fixed links can be
zero-Jacobian endpoints. Articulation contacts with FREE actors and materials
retain their existing paths.

Every scene, including cloth/cable scenes, uses spatial contact. Candidates
refresh at each Newton boundary and remain frozen during its trials. This
approximation can omit contacts reached by a trial outside the frozen bounds;
it does not provide continuous collision detection or automatic correction.
The step-start convergence reference still uses complete contact enumeration.

``contact_candidate_capacity=None`` reserves the model's complete static sample
upper bound, and ``contact_factor_capacity=None`` reserves three times that
bound. These are fixed capacities, not active contact counts. Explicitly
reducing either budget can discard whole contacts; candidate or factor overflow
fails the step and rolls it back without publishing partial forces.

Use the public API
------------------

Build the supported model and supply its rest density, then create the complete
solver state. A plain ``model.state()`` can initialize this state but cannot be
used directly for continuation.

.. code-block:: python

   from newton.solvers.experimental.monolithic import SolverMonolithic

   solver = SolverMonolithic(
       model,
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
PCG and no CUDA graph. Both PCG modes run the same solver; the diagnostic mode
also records every checkpoint's true residual norms and cannot be captured in a
CUDA graph. PCG has no stagnation exit: it iterates until its tolerance or
``linear_max_iterations``. Constructor options are validated before the Model is
inspected. Regularized direction retries use the same schedule for every scene,
starting without regularization.

FREE rigid actors predict translation and rotation from their current velocities;
this changes the Newton starting point, not the discrete inertial equation.
ArmijoOrResNorm permits either energy or residual progress. Within a physical
step, each accepting branch must also preserve its best value reached so far,
up to the existing rounding allowance, to prevent alternating regressions.
The other measure may still increase; C-IE energy acceptance remains heuristic.

``alignment_policy="guard"`` rejects penetrating samples whose surface normals
point in the same direction. Robot surface contact can instead explicitly use
``"mask"`` to exclude each such complete contact term, including both endpoint
forces. This existing one-sided sampling model retains invalid-geometry checks
and does not provide continuous collision or continuous normals at sharp edges.

A step converges when every residual row lies within its own tolerance: the
largest of an absolute target (1e-11 N for force rows, 1e-13 N m for torque
rows), 1e-6 times the largest step-start residual of the same block and unit,
and four times the precision's machine epsilon times the row's rounding
magnitude (its summed absolute contributions plus the tangent-weighted
coordinate magnitudes). Rows are never pooled into a norm, so a small local
load is resolved regardless of model size. ``Stats.convergence_ratio*`` report
each block's largest residual-to-tolerance ratio, and
``Stats.variational["convergence_groups"]`` counts the rows that passed only
within their rounding allowance. With FP32 absolute coordinates that
allowance can exceed small physical loads: a coordinate cannot move by less
than its unit in the last place. Each Newton direction reduces the scaled
residual by ``linear_tolerance`` or reaches half of every row tolerance.
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
checkpoint loads are rejected. The configuration identity covers every option
that changes the solution, the resolved contact resources and the physical,
material and SDF digests; PCG mode and CUDA graph use are excluded. CPU and
CUDA resolve the same contact batch size, so checkpoints move between devices.
Only the current schema loads: older schemas are not migrated, and checkpoints
written before the current configuration version are rejected; start a new
run. Failed loads preserve the
destination. A plain initialization source is a new start, not a conversion of
integration history.

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
Mixed-contact numerical qualification remains partial. Existing numerical and
publication gates apply. Articulation-tet and tet-static contact keep their original sampling and
behavior.
FREE PD, limit gains, friction, damping and armature must be zero.
Rigid ``body_f`` and FREE ``control.joint_f`` each contribute a world COM wrench
[N, N·m] once. Convert off-center loads to COM wrenches before calling ``step``.
Nonzero rigid torque is accepted through the residual-norm predicate; its
energy acceptance is reported as heuristic.

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
guards also apply.
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

The nightly profile runs the production Franka and recorded Sharpa FP64
trajectories (C-IE, ArmijoOrResNorm, CUDA production graphs) and the 64-box
rigid collision scene; it is triggered manually for overnight regression. The
dev and full profiles run the two robot trajectories. Routine development and
ordinary CI use functional component and small integration tests instead. The Monolithic project policy is
in the workspace file ``dev/monolithic/project.md``; repository ``AGENTS.md``
links to that project-specific policy. The runner binds each case
to ``asset/monolithic/references.json``, verifies retained file hashes and frozen
controls, and rejects a changed physical configuration before stepping. Robot
cases qualify on the frozen physical plausibility gates in
``scripts/monolithic_reference/fixtures/physical_gates_v2.json``: no hard
failures, at most 1% Stopped steps, sampled penetration of at most 2 mm, a
minimum tet determinant of 0.2 and, for Franka, a successful grasp. A grasp
succeeds when, over the final 500 steps at the top of the lift, every published
step keeps the ground support within 5% of the cube weight and the cube center
of mass at least a quarter of its smallest edge above rest. Sharpa reports the
same grasp result as a diagnostic: its recorded hand closure does not lift the
cube. Comparison with the
retained reference trajectories is reported as a diagnostic only; new results
need not match them. Missing frozen inputs produce FAIL; filtered profiles
remain PARTIAL. Pass ``--precision fp32`` to run every case in FP32: the cases
qualify on the same physical gates and complete-candidate contact audit, skip
the comparison with the FP64 reference trajectories, and their receipts record
the precision. The material, cable, cloth and tet runners accept the same
option.

After completion, review ``summary.json``, ``RESULTS.md`` and per-case receipts.
By default, successful new recordings are removed after their summaries and
comparison results are saved; failed recordings remain for diagnosis. Pass
``--retain-recordings all`` to keep all new recordings. Existing reference
trajectories are never modified by this retention policy.

The scene runner's result must be combined with component, lifecycle and replay
checks for stage qualification. Retired driver snapshots, hashes and migration
notes are documented in the workspace file
``dev/monolithic/P4/script_records/legacy_retirement.md``.

The precision experiment runner uses the same two robot configurations,
defaults to FP32, and accepts ``--precision both``. The retained controls already match the
rebuilt manifest; they do not need to be generated again::

   uv run --no-sync python -m scripts.monolithic_reference.run_precision --list
   uv run --no-sync python -m scripts.monolithic_reference.run_precision --controls asset/monolithic/data/inputs/controls --output /path/to/results

Use ``--max-steps 2`` for a smoke run. A complete run reports
``COMPLETE_NOT_AUDITED`` until the independent T5 numerical audit is performed;
short runs report ``PARTIAL`` and hard failures retain their first-failure state.
This runner does not yet perform the separate T5 performance experiment.
