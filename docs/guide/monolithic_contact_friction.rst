.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _monolithic-contact-friction:

Contact friction comparison
===========================

Run the press, drag, hold and lift example with::

    uv run --extra examples -m newton.examples monolithic_contact_friction --device cuda:0
    uv run --extra examples -m newton.examples monolithic_contact_friction --device cpu --viewer null --test --output /tmp/friction-cpu

Blue is contact friction on; orange is off. The bottom of each soft block is
fixed. The two prismatic plate joints use implicit PD, and both simulations
use Smith material and consistent mass. Geometry and all other inputs are
identical. The demonstration shows sticking, sliding, elastic recovery and
clearing history on separation. It does not certify a grasp or self-collision.

The frozen simulation parameters are in
``newton/examples/softbody/monolithic_contact_friction.py``. They describe an
80-tet, 80 x 40 x 40 mm block with E=10 kPa and density 1000 kg/m³; they are
calibration parameters, not measured properties. The physical step is 2 ms,
with 10 steps per displayed frame and a four-second trajectory. Deformation
is displayed at its actual scale. Without ``--device`` Warp selects its
preferred device; pass the flag explicitly for reproducible comparisons.

Each output directory contains the fixture manifest, summary, per-step JSONL
trace and final State arrays. Plot the force, displacement, slip and history
curves with::

    uv run --extra examples -m scripts.monolithic_reference.plot_contact_friction /tmp/friction-cpu --output /tmp/friction.png

Experimental contact options
----------------------------

The existing ``SolverMonolithic`` constructor accepts
``normal_smoothing_width`` [m], ``friction_coefficient`` [dimensionless] and
``tangential_stiffness`` [N/m³]. These parameters are experimental and may
change without the normal deprecation period. Defaults ``0, 0, None`` retain
quadratic normal contact without friction. Positive smoothing requires the
pipeline detection gap to cover the entire smoothing half-width. Positive
friction requires positive tangential stiffness; both stiffnesses multiply
the reference-area quadrature weight once. Shape material coefficients are
not mixed into this explicit scene-wide coefficient.

Friction uses an elastic sticking displacement capped by mu times the actual
sample normal force. The production operator uses a symmetric PSD local
approximation, not the generally nonsymmetric full friction Jacobian.
Tangential force and rigid COM moment use the same soft sample point.

History belongs to one continuous trajectory and is indexed by static
candidate key, independently of record order. Trial evaluations read the
step-start history. State, final force and history commit together; hard
rollback preserves the previous epoch. Safe nonlinear iteration exhaustion
commits its accepted State/history once and remains marked unconverged.
``update_contacts`` republishes cached forces without integrating history.
Create a new solver when resetting a scene or running a friction-off control.

History rotates with the body and projects onto the current tangent plane;
a normal change beyond the private 45-degree threshold resets it. This
threshold is a reset rule, not a claim of arbitrary finite-rotation
objectivity. Component tests verify BE rotation refinement at 1 rad/s with
5, 10 and 20 ms steps, and the normal-reset boundary separately.

``solver.last_stats.contact_history`` reports final force magnitudes, relative
slip, sticking/sliding sample counts, matching/reset reasons, elastic energy,
friction work, radial-return dissipation and the history epoch. Elastic
friction can release stored energy: instantaneous friction power need not
always be negative. Fixed-load work/storage/return audits are separate from
transport, separation and changing-normal-load energy changes. The q-only
joint diagnostic reports this field as None.
