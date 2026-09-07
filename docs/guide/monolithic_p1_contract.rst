.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _monolithic-p1-contract:

Monolithic P1 contract
==============================

This supplement freezes ``monolithic-p1-contract/v1`` against the implemented
PR-6C joints and PR-6D hand adapter at base commit
``299f43ceab1d6647f12a4c13dab6100893878be1``. It also reserves the interfaces
for the remaining P1 work. A reserved interface does not enable its physics.
The original design identifier remains ``monolithic-p1-interface/v1-draft``;
this supplement does not rewrite that historical design or acceptance evidence.

.. experimental::

   ``SolverMonolithic`` and its nested ``JointTerms``, ``MaterialModel`` and
   ``MassMode`` types in ``newton.solvers.experimental.monolithic`` are
   experimental. The P1 keyword arguments and the internal diagnostic entry
   described here may change without the normal deprecation period.

Design provenance
-------------------------

The source documents are external workspace inputs, not files installed with
Newton. Their SHA-256 identities are:

* ``newton_monolithic_solver_prd_zh.md``:
  ``d9ce8206114f15c512d65b2c523fa398c28a2b6e4d2494ccdb058be2c52158c7``.
* ``newton_monolithic_solver_p1_architecture_zh.md``:
  ``145fc87c219193b76127cd078c9fdffa48ab0a533bd697a4563393832064e0b8``.

The contract implements the staged enablement required by architecture
sections 4, 8, 10, 13 and 14. PR-6A/B implementations must satisfy the reserved
contracts below before accepting their options. Historical G1/G1H reports
remain bound to their original code; this supplement is not a new grasp gate.

Constructor and capability boundary
-------------------------------------------

``step(state_in, state_out, control, contacts, dt)`` is unchanged. ``contacts``
must be ``None``; the solver owns collision. The ordinary constructor requires
one world-anchored articulation and one connected tetrahedral body. The private
``SolverMonolithic._create_joint_diagnostic`` entry is restricted to one
articulation, zero particles/tets and disabled collision participation.

.. list-table:: P1 keyword schema
   :header-rows: 1
   :widths: 24 30 46

   * - Keyword
     - Default / units
     - Current validation and future enablement
   * - ``material_model``
     - ``MaterialModel.KIM_STABLE_NO_LOG``
     - Enum or exact string ``kim_stable_no_log``. The reserved
       ``smith_log_stabilized`` value raises ``ValueError`` until PR-6A/G2.
   * - ``mass_mode``
     - ``MassMode.LUMPED``
     - Enum or exact string ``lumped``. Reserved ``consistent`` raises
       ``ValueError`` until PR-6A/G2.
   * - ``tet_rest_density``
     - ``None``; kg/m³
     - Non-None rejected now. Consistent mass will require positive finite
       float32 ``(tet_count,)`` values on the model device, copied privately.
       Density is authoritative; lumped particle mass is not substituted.
   * - ``joint_terms``
     - ``None``
     - Implemented opt-in ``JointTerms``; absent/disabled capabilities retain
       the original nonzero Model-attribute rejection rules.
   * - ``normal_smoothing_width``
     - ``0``; m
     - Nonnegative finite float32; positive values rejected until PR-6B/G3.
       Zero retains the V0.1 quadratic hinge. Canonical V0.2 requires positive
       PolyReLU smoothing and a candidate gap covering its support.
   * - ``friction_coefficient``
     - ``0``; dimensionless
     - Nonnegative finite float32; positive values rejected until PR-6B/G3-G4.
       Future coefficient is explicit and global, not shape-material mixing.
   * - ``tangential_stiffness``
     - ``None``; N/m³
     - Non-None rejected now. Contact friction will require a positive finite
       value, multiplied by rest-area quadrature weight.

Boolean values are not physical scalars. Unknown modes and pending options
fail before solver layout/workspace allocation. Both constructor paths share
this check. Numerical tolerances, regularization and preconditioner selection
remain private and retain the existing calibrated values.

Construction and step identity
--------------------------------------

Model topology, coordinate mapping, enabled gains/limits/friction and per-DoF
smoothing values are construction inputs. Joint workspaces copy numerical
parameters; replacing their source arrays is rejected. In-place parameter
edits do not reconfigure the copied parameters: rebuild the solver. Existing
articulation, tet and collision scope/array-identity checks remain in force.

Linear assembly identity contains ``step_index``, ``nonlinear_iteration``,
``contact_generation``, ``assembly_sequence``, ``config_generation`` and
``history_epoch``. Factorization/apply/solve must use the sealed assembly's
identity. Changing either new field makes an old assembly stale. The new fields
default to zero for existing callers. They are workspace-local counters, not
portable asset hashes. Current solvers cannot reconfigure and have no history,
so both counters remain zero. A future history commit must advance the epoch;
it must not be simulated by incrementing it during current/trial evaluation.

Portable future history identity separately contains lowercase SHA-256 strings
``topology_sha256`` and ``config_sha256`` plus these two nonnegative integer
counters. Hash inputs must include the derived boundary/ordered static pairs,
joint/link mapping, source and derived asset hashes, material/mass/density
authority, contact parameters and transport settings. Device pointers are
runtime identity only and cannot stand in for a portable content hash.
The history identity and buffer types are defined now; hash construction,
history allocation and transactional publication remain disabled with PR-6B.

PR-6C joint contract
----------------------------

``JointTerms`` has three boolean switches: ``implicit_pd``, ``limits`` and
``friction``. ``limit_width`` and ``friction_velocity_scale`` must supply one
positive finite value per DoF when their term is enabled, and must be ``None``
otherwise. Scalar broadcasting is not supported. Revolute units are rad,
rad/s and N·m; prismatic units are m, m/s and N. Gain units follow force per
coordinate and force per coordinate velocity.

Authority is ``Model.joint_target_ke/kd``, ``joint_limit_lower/upper/ke/kd``,
``joint_effort_limit``, ``joint_velocity_limit`` and ``joint_friction``.
Driven DoFs have at least one positive PD gain, require
``JointTargetMode.POSITION_VELOCITY`` and positive finite effort/velocity limits.
Zero-gain DoFs retain external-force support. Unsupported target modes,
armature, independent passive damping, mimic and general model actuators remain
rejected. Position target mapping uses ``joint_target_q_start``; the currently
supported scalar-joint layout requires it to match ``joint_q_start``. This is
not support for arbitrary coordinate layouts or additional joint types.

Each physical step validates and copies Control targets before the state
transaction. Driven DoFs require ``joint_f == 0`` to reject double driving;
undriven external forces remain frozen. Current/trial/final evaluate the same
physical terms at their own candidate state. No candidate evaluation allocates
device storage or copies complete arrays to the host. Target generation,
position range and target speed limiting belong to the caller; state q/qd
are never clipped after solving.

All contributions use ``R = -Q``. The workspace stores separate force,
residual and position tangent arrays of shape ``(nq, 3)``, ordered
PD, limit, friction. Current tangents contribute to both dense-q owner and
global scalar triplets. Trial forces are recomputed while the current sealed
matrix, scale and preconditioner remain frozen.

* PD clips the sum of elastic and damping output to the effort bound. Its
  tangent is ``kp + kd/h`` strictly inside the bound and zero outside/on the
  boundary. Its increment potential is a differentiation oracle, not motor
  energy consumption.
* Limits use double-sided penalty springs and smooth activation of damping
  only for continuing outward velocity. Returning motion is not damped by
  the limit. Finite penalty compliance permits small violations.
* Friction is ``-f*v/sqrt(v*v + v_eps*v_eps)`` with nonpositive mechanical
  power and nonnegative position tangent. It has no exact zero-speed lock.

The shared candidate stores per-step joint displacement and recovers BE
velocity before rounding absolute q to float32 State. This applies to coupled
and q-only stepping. Absolute-coordinate offline probes retain their existing
entry; no second hand integrator or relaxed residual gate is introduced.

Final force diagnostics belong to the actually returned State and step
generation. Safe soft stops commit the last accepted candidate; hard failure
rolls back; rejected trials never publish. Failed final publication must not
leave a valid cache generation. The diagnostic q-only path uses the same
transactions/BSR/PCG/global and q gates. x/tet/contact metrics are unmeasured
NaNs in Stats and ``NOT_APPLICABLE`` in reports; corresponding counts are zero.

Contact factor and history contract
-------------------------------------------

Private scalar factor storage keeps ``gq``, ``gx_columns``, ``gx_values`` and
``weights`` and now adds ``kind`` and ``candidate_tid`` integer arrays. Closed
tags are ``NONE=0``, ``NORMAL=1``, ``TANGENT=2``. Current production emits only
NORMAL; TANGENT is reserved. Only slots below the active factor count are valid.
Existing Contacts record schema and public attributes are unchanged.

The fixed key is ``candidate_tid = 3*static_pair + quadrature_slot``; the pair
identifies derived boundary face and rigid shape/link. The validated
``soft_contact_tids[candidate_tid]`` map points to atomically allocated records.
Factors recover this key from the map, never from record order. Reordering
records together with the map must preserve keyed factors and physical R/K.

The reserved history buffer has ``valid: int32[C]``, ``xi_local: vec3[C]``
and ``normal_local: vec3[C]``. ``xi_local`` is elastic tangential displacement
in metres in the rigid body's local frame; the normal is dimensionless. PR-6B
must own two buffers and apply this state machine:

1. At step start, bind committed history to topology/config identity and epoch.
   Committed history is read-only throughout current and rejected/accepted trials.
2. Rotate old history into the candidate frame and project onto the current
   tangent plane without increasing its length. Missing/new contacts start
   at zero. Contact loss or a normal rotation outside the frozen supported
   angle resets pending history and records the reason.
3. Evaluate final forces and pending history for the actual return candidate.
   Use the inverse final body rotation to store displacement and normal locally.
4. Commit State, history and force cache together, incrementing the epoch once
   for success or safe soft stop. Hard rollback/final failure discards pending
   data without advancing it. Idempotent force publication does not advance it.

A new scene, initial condition or friction-off comparison uses a new solver.
No checkpoint API or multi-trajectory history reuse is provided. Rotation/dt
support and reset-angle values remain pending G3/G4 calibration; no arbitrary
finite-rotation objectivity claim is made. Tangential forces on rigid and soft
sides will use the same soft quadrature point for moment balance. Production
PSD tangents need not equal the full friction residual Jacobian. Future
normal-patch preconditioning filters NORMAL tags; actor blocks use all factors.

For q scalar articulation DoFs, n dynamic particles, T tets and P static pairs,
the frozen conservative capacity formulas are:

.. code-block:: text

   contacts = history_slots_per_buffer = C = 3*P
   scalar_contact_factors = C                  (normal only)
                         = 3*C                (normal + two tangents)
   internal_mat33_triplets <= n + 16*T
   global_scalar_triplets <= q*q + 9*(n + 16*T)
                             + scalar_contact_factors*(q + 9)^2

``MonolithicLinearCapacities.for_p1`` evaluates this bound with host integers
and validates int32 capacity before allocation. It does not enable friction
or replace the current exact sparsity pattern. The current contact workspace
still requires normal-only capacity C. PR-6B must switch the producer and
consumer together, allocate both C-slot history buffers, check complete device
byte budgets before allocation, and reject overflow rather than grow or compact
runtime pair tables. The byte-budget check for that future storage is pending;
an int32 count check alone is not proof that a scene fits GPU memory.

PR-6D asset and G1H evidence contract
---------------------------------------------

The supported source is ``left_sharpa_wave.urdf`` with 33 links,
22 revolute and 10 fixed joints. Import preserves 33 bodies, 22 DoFs and
54 shapes, resolves and hashes every mesh, fixes the root, records inertia
corrections and explicitly copies URDF lower/upper/effort/velocity values.
All shape collision participation flags are disabled for G1H; no SDF is built.
Gains, limit penalties and joint friction come from simulation calibration.

The fixture schema remains ``sharpa-g1h/v1``. Required top-level fields are
``schema``, ``parameter_source``, ``gravity``, ``dt``, ``duration``, ``joints``
and ``gates``. Unknown, missing and duplicate JSON fields are rejected before
asset import. Gravity is three finite SI components; duration is 4 s; dt is
positive and divides 10 ms. The frozen formal fixture uses 1 ms, hence ten
substeps per displayed frame. A different valid fixture is not automatically
covered by its historical acceptance report.

``joints`` is keyed by the exact 22 URDF names and requires ``target_ke``,
``target_kd``, ``limit_ke``, ``limit_kd``, ``friction``, ``limit_width`` and
``friction_velocity_scale``. Gains/friction are nonnegative finite values;
widths/smoothing velocities are strictly positive. ``gates`` requires
``moving_threshold``, ``max_tracking_error``, ``final_tracking_error``,
``hold_drift``, ``hold_speed``, ``completion_min``, ``limit_violation``,
``velocity_ratio``, ``effort_ratio`` and ``converged_fraction``. All are positive;
completion is at most one and convergence is in [0.99, 1]. Thresholds must be
frozen before formal runs, not adjusted from failed results.

The BOM CSV has 201 finite frames, 22 unique named channels and strictly
increasing timestamps from 0 to 4 s. Mapping rows contain source channel,
URDF name, sign (+1/-1) and finite zero offset [rad]. Both source and target
names are one-to-one; duplicate rows are rejected. Position targets must fit
URDF limits without clipping; segment slopes must fit URDF velocity limits.
Targets use piecewise linear interpolation and the selected segment's slope,
sampled at each physical step end. At time zero and after the last frame,
target velocity is zero. The final second must be a hold; the supplied asset
trajectory additionally holds for approximately the final two seconds.

Four fingers follow MCP/PIP/DIP names; thumb source MCP maps to CMC, PIP to
MCP, DIP to IP, and pinky CMC is separate. The explicit mapping in
``newton/examples/softbody/sharpa_close.py`` defines an adapted URDF closure,
not a reconstruction of the original software's full coordinate conventions.
New assets/mappings require axis/FK and open/intermediate/closed pose checks
before G1H can be reported as passed.

New example manifests carry ``schema=sharpa-g1h-manifest/v1`` and
``contract=monolithic-p1-contract/v1`` in addition to URDF/mesh/CSV/fixture
hashes, mapping, ordered joint names, body/shape counts, mass, inertia corrections,
warnings, collision status, parameters, device and scope. Acceptance records
must additionally bind code, design, environment and actual trace/curve hashes.
Existing unversioned historical manifests remain untouched.

G1H initializes q from the mapped first frame and qd/target velocity to zero;
later steps update Control only. Rollback does not advance trajectory time
and stops the run. Per-joint tracking, hold drift/speed, completion for moving
joints, absolute errors for holding joints, actual effort/saturation, limit
violation and friction power accompany global/q residual and finite-state
checks. At least 99% of substeps must normally converge. Visual self-intersection
does not fail this joint gate and does not establish valid grasp contact.

Remaining asset schema for full PR-6D must include source/derived SDF cache
identity and scale/resolution, soft-ball connectivity/rest quality/density and
material/mass authority, carriage and initial-condition provenance, contact
parameters, numerical/capacity/performance budgets, and Close/Hold/Lift/Release
plus friction-off metrics. Unknown values remain pending. CPU mesh SDF and
same-asset CPU grasp are not required; CUDA mesh SDF and CUDA grasp are required.

Verification ownership and completion
---------------------------------------------

* G0 currently checks reserved-mode rejection, joint input validation,
  generation identity, capacity bounds, stable factor keys and Sharpa fixture
  input. Runtime material/density/history hashing, history transactions and
  complete memory budgets remain tied to future PR-6A/B/G0-G4 work.
* G1 and G1H have separate historical CPU/CUDA acceptance. Their tests remain
  regression requirements for changes to this contract. G1H cannot replace
  G1 saturation/limit/friction component oracles.
* G2-G4 physics/history, the SDF/soft-ball portion of PR-6D/G5, G6 grasp and
  G7 scale/performance are not completed by this contract subcommit.

Portable contract tests live in ``test_solver_monolithic_p1_contract`` and
``test_sharpa_trajectory``; CPU/CUDA joint, contact and q-only lifecycle tests
exercise the corresponding production paths. Future enablement must add its
own physical acceptance evidence and preserve the existing tolerances.
