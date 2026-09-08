.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Sharpa soft-ball integration
============================

.. experimental::

   ``monolithic_sharpa_soft_ball`` is a contact experiment using
   :mod:`newton.solvers.experimental.monolithic`. Completing the stage script
   does not certify G6/G7 or a successful friction-supported grasp.

The default ``anchored-close`` experiment places the 20 mm soft sphere in
front of the palm at ``(0.045, -0.005, 0.085) m``. A palm-facing cap is fixed
in world coordinates before constructing the solver; all remaining nodes
can deform. The cap selects reference nodes with local ``x <= -0.7 R``:
5, 13 and 29 fixed nodes for the coarse, medium and fine meshes respectively.
These artificial constraints test contact stability, not free grasping.

The base URDF retains its 22 finger degrees of freedom and fixed root.
This experiment has no carriage or support plane. Gravity remains enabled,
and hand self-collision remains disabled. Fixed nodes use the solver's
existing Dirichlet treatment; the example never overwrites returned states.

Prepare the verified hand collision assets as described in
:doc:`monolithic_sharpa_assets`. Then run from the repository root::

   uv run --extra examples -m newton.examples monolithic_sharpa_soft_ball \
     --asset-dir /path/to/left_sharpa_wave \
     --contact-dir /path/to/prepared-contact-assets \
     --trajectory '/path/to/Ball_catch - Left_hand_motion_rad.csv' \
     --calibration scripts/monolithic_reference/fixtures/sharpa_contact_calibration_v1.json

Use ``--viewer null`` for headless simulation, or ``--headless`` for offscreen
OpenGL rendering. The default is 1 ms per physical step and ten physical steps
per displayed frame. The default schedule is 4.5 simulated seconds:
prepare 0.5 s, close 2 s, then hold 2 s (450 displayed frames).

The CSV's original two-second closure and two-second hold are preserved. Targets are sampled at physical-step end and frozen inside each
solver step. A rollback stops the example without advancing physical time.

The default ball is ``ball_r2.npz``. Select ``ball_r1.npz`` or ``ball_r3.npz``
with ``--ball`` to inspect other mesh resolutions. All use Smith material and
consistent mass. These are simulation parameters, not identified hardware or
material properties. ``--friction-off`` creates an independent run with zero
contact friction; joint friction remains enabled.

The output directory contains the manifest, actual state and target traces,
per-finger and support forces, history diagnostics, convergence results, and
per-call PCG iterations. Forces are world-space forces on the rigid body;
negate them to obtain forces on the sphere. Palm and finger forces are
recorded separately. Stage snapshots record actual simulated positions.
The trace checks fixed positions and zero fixed velocities at every step.
Step, collision, PCG and remaining step time are reported separately;
rendering and trace serialization are outside solver step timings.

Plot the output with::

   uv run --extra examples python scripts/monolithic_reference/plot_sharpa_grasp.py \
     --run output/monolithic-sharpa-soft-ball \
     --calibration scripts/monolithic_reference/fixtures/sharpa_contact_calibration_v1.json

The frozen pre-integration calibration uses a double-plane clamp and three
sphere meshes. The lowest passing sampled stiffness is ``2e6 N/m³``;
the example retains ``1e7 N/m³``. The PCG iteration p95 budget is 64 per
actual solve, evaluated separately by stage, including retries. The existing
200-iteration hard limit and residual criteria remain unchanged.

Collision uses two conservative AABB gates followed by the existing centroid
and P1Q3 SDF queries. It has no BVH and retains the full static pair table,
stable history keys, and allocation capacities. ``--disable-aabb`` selects
the internal full-table oracle. AABB reduces query work; it does not reduce
the static global assembly capacity or guarantee a frame-rate improvement.

Full hand SDF simulation requires CUDA. CPU verification covers analytic
collision, controls and articulation components; CPU mesh SDF is not required.
Final placement/control calibration, friction-off negative acceptance, G6/G7
and any triggered patch-Schur work belong to PR-7B.

With ``--test``, anchored closure requires at least 99% normal convergence,
the unchanged residual criteria, penetration at most 2 mm, min(detF) at least
0.1, unchanged anchors, and finger force at least 1 mN on 90% of hold steps.
A completed schedule with no sustained finger loading does not pass.
PCG iteration p95 above 64 is reported separately as an efficiency trigger.

The earlier exploratory nine-second grasp experiment remains available with
``--experiment grasp --num-frames 900``. Its original all-dynamic ball,
carriage, support and unsuccessful initial placement remain available for
reproduction. Its outcomes are separate from anchored closure.

For a sequential resolution comparison, use the same asset, contact,
trajectory and calibration arguments with::

   uv run --extra examples scripts/monolithic_reference/run_sharpa_anchored.py \
     --asset-dir /path/to/left_sharpa_wave \
     --contact-dir /path/to/prepared-contact-assets \
     --trajectory '/path/to/Ball_catch - Left_hand_motion_rad.csv' \
     --calibration scripts/monolithic_reference/fixtures/sharpa_contact_calibration_v1.json \
     --refinements 1 2 3 --output output/anchored-resolution

This runner executes the complete closure for each mesh, then profiles a
separate 20-step continuous hold tail. The instrumented tail synchronizes
individual stages; overlapping stage timings must not be added. Formal
state snapshots remain at the end of the 4.5-second trajectory.

Measured scope of the initial fixture
-------------------------------------

The medium and fine meshes complete all 4500 steps with normal convergence,
unchanged anchors and sustained light middle-finger contact. Maximum
penetration is about 0.175 mm and 0.239 mm respectively. The other fingers
and palm have zero measured hold contact force; this is not a multiple-finger
high-load validation. The coarse mesh has detection records but zero force,
so its sustained-contact gate correctly fails.

Measured hold step p50 is approximately 77 ms (medium) and 156 ms (fine).
Using mean solver step times and ten substeps per displayed frame gives
about 1.63 and 0.76 FPS, excluding rendering and output. A separate
instrumented hold tail attributes about 44/114 ms per step to BSR construction,
compared with 3/5 ms to PCG. These are observed fixture results, not a final
G7 performance certificate. The resolution runner returns nonzero when the
coarse no-load case is included; it preserves all measured results.

Detailed r3 BSR profiling
-------------------------

To separate global scalar BSR construction, internal 3x3 BSR construction,
validation and host readbacks, run:

.. code-block:: bash

   uv run --extra examples scripts/monolithic_reference/profile_sharpa_bsr.py \
     --asset-dir /path/to/left_sharpa_wave \
     --contact-dir /path/to/pr6d-assets/hand \
     --trajectory '/path/to/Ball_catch - Left_hand_motion_rad.csv' \
     --calibration scripts/monolithic_reference/fixtures/sharpa_contact_calibration_v1.json \
     --output output/r3-bsr-profile

The default runs all 4500 trajectory steps with continuous friction history,
then takes 20 baseline and 20 detailed holding steps. Detailed timings insert
synchronization and report inclusive/exclusive time per call. A step may build
several matrices; per-build latency is different from per-step cost. The tool
also replays the final, unchanged triplets and checks bitwise matrix parity.
``--warmup-steps`` is available for smoke tests; a shortened run does not measure
the completed closure.

For native CUDA kernel attribution, prefix the command with
``nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
--capture-range=cudaProfilerApi --capture-range-end=stop -o output/r3-bsr``
and append ``--nsys-capture``. This requires Nsight Systems and
``libnvToolsExt.so.1``. Capture covers only a separate replay of the two BSR
builders, with NVTX labels ``global_scalar`` and ``internal_block3``. GPU kernels
must be attributed using their launch correlation IDs, since GPU execution can
outlive the host NVTX range. Replay does not advance State or history and is
not a full-step FPS measurement.

On an RTX 5070 Ti Laptop GPU with Warp 1.17.0, after the full 4500-step r3
closure, detailed finalize time averaged 72.54 ms per call: 71.00 ms in the
global scalar builder and 0.61 ms in the internal block builder. Baseline
measurement observed 1.5 finalize calls per step, totaling 112.44 ms per step.
In a separate unchanged-matrix replay, radix sort took 53.99 GPU ms per global
build, or 79.2% of its recorded GPU activity. The global input count was
198,565 triplets, but Warp sorted the full 97,318,186-entry allocated capacity.
The device count masks the unused tail without shortening the native sort.
These measurements identify an optimization target; this profiler does not
change production assembly. The portable result is recorded in
``scripts/monolithic_reference/fixtures/sharpa_bsr_profile_v1.json``.

PR-7C active-prefix optimization
--------------------------------

The production builder now receives continuous views limited to the validated
triplet count. The views share the preallocated device storage; static contact
keys, history, workspace capacity and solver tolerances are preserved. Empty
capacity bypasses slicing. Warp can still allocate temporary builder storage
and grow BSR output buffers as the active prefix increases; this is not an
allocation-free sparse backend.

Use ``scripts/monolithic_reference/validate_bsr_prefix.py`` with the same asset,
contact, trajectory and calibration arguments as the profiler above. Run
``--builder capacity --output output/r3-capacity`` and
``--builder production --output output/r3-prefix`` in separate processes. The
capacity option is an offline oracle; it is not a public solver mode. Each run
executes 4500 steps, saves committed history every 500 steps, then measures
continuous holding and five alternating replay groups after warm-up. Replay
uses identical frozen triplets and requires bitwise matrix parity. Append
``--nsys-capture`` under the Nsight command above for CUDA attribution, with
``capacity/global_scalar`` and ``prefix/global_scalar`` NVTX labels.

On the measured r3 fixture, finalize latency decreased from 72.86 to 0.90 ms
per call. Hold step p50/p95 decreased from 146.72/192.06 to 30.18/32.72 ms;
mean hold step time corresponds to 0.78 versus 4.46 solver-only FPS with ten
substeps, excluding rendering. Both complete trajectories converged normally
on all 4500 steps with unchanged acceptance gates. Independent CUDA trajectories
are not bitwise identical: maximum ball COM difference was 6.6 micrometers.
The unchanged full-capacity implementation itself showed 12.8 micrometers
between independent runs; this observation is not a new tolerance. The fixture
still exercises light middle-finger contact and does not certify free grasping.

The portable acceptance record is
``scripts/monolithic_reference/fixtures/pr7c_bsr_prefix_acceptance_v1.json``.

P1 exit and P2 continuation
---------------------------

P1 is accepted as ``PASS_LIMITED_SCOPE``: joint, material, contact/history,
transaction and asset/AABB gates, plus the measured anchored-close stability
and BSR performance. The r3 fixture completed 4500 steps on the accepted
implementation. The r2 full-run evidence predates PR-7C, which adds a 100-step
r2 check. The coarse r1 fixture has no force-producing contact and remains
``NOT_EXERCISED`` for sustained loading. The supported hand-contact envelope
is the measured light middle-finger contact, with the original tolerances.

Original G6 and the remaining G7 requirements move to P2. P2 must calibrate
the free-ball grasp, validate Close-Hold-Lift-Release and the friction-off
negative control, then complete multiple-finger loading, resolution, capacity,
memory and full-step performance measurements. The original PCG p95 budget of
64 and physical contact-stiffness floor still determine whether normal-only
patch Schur becomes mandatory. CPU mesh SDF remains ``NOT_REQUIRED``.

P2 is planned, with development not started. P1 acceptance does not complete
the V0.2 Grasp MVP. The stage decision, evidence hashes and deferred items are
recorded in
``scripts/monolithic_reference/fixtures/p1_limited_acceptance_v1.json``.

Before choosing P2 performance targets, measure how the outer nonlinear
Newton iteration limit affects convergence, actual update counts and step
time, then profile the current implementation after PR-7C. The proposed
iteration-limit sweep is 1, 2, 3, 4, 6, 8, 10, 16 and 24, with the current
default of 10 as the baseline. Residual tolerances, physical parameters,
line-search policy and inner PCG settings stay fixed. Convergence can stop
early; the iteration cap is not the number of updates actually executed.

Use continuous r3 anchored-close trajectories and an r2 comparison, keeping
soft stops and failed runs visible. Repeat full-step timings before adding
detailed synchronization or Nsight instrumentation. Report current/trial
evaluation counts, assembly, PCG, retries, memory and host synchronization
separately, then use those results to freeze performance targets. The later
multiple-finger grasp fixture requires its own measurements before G6/G7.
This work is planned; measurements and numerical targets remain pending.
See ``scripts/monolithic_reference/fixtures/p2_performance_plan_v1.json``.


Larger sphere stress cases
---------------------------

Use the same hand arguments with a separately generated sphere and an explicit
expected radius. ``--ball-radius`` validates the asset; it does not rescale a
loaded model or overwrite particle positions. For a 25 mm radius sphere::

   --ball scripts/monolithic_reference/fixtures/soft_ball_25mm/ball_r3.npz --ball-radius 0.025

The 30 mm stress asset is selected with::

   --ball scripts/monolithic_reference/fixtures/soft_ball_30mm/ball_r3.npz --ball-radius 0.030

Both directories contain r1/r2/r3 meshes. Regenerate into a separate output
directory with ``scripts/monolithic_reference/prepare_soft_ball.py --radius 0.025
--output output/soft-ball-25mm``. The default 20 mm assets and their acceptance
records remain available. A radius/asset mismatch is rejected.

The center, control trajectory, density, material, contact stiffness and
relative anchor-cap rule are unchanged. Larger spheres therefore have more
mass, a physically larger fixed cap, and coarser surface sampling at the same
refinement. Mass scales with radius cubed: 25/30 mm have 1.953125/3.375 times
the corresponding 20 mm mass. These are changed loading cases, not isolated
solver performance comparisons. Radius changes require new calibration;
``radius_calibration`` marks them ``UNVALIDATED_RADIUS_STRESS_TEST``.

The 30 mm r3 trial stopped at 2.311 s with a tet determinant guard failure;
initial overlap already exceeded the inherited 2 mm penetration gate. Keep
this as a failed stress case, not an accepted timing baseline. Traces include
``force_producing_contacts`` as well as detection counts and per-finger forces,
so detection-only candidates cannot masquerade as additional physical loading.

The exploratory 25 mm r3 run completed all 4500 steps with 100% normal
convergence, maximum penetration about 0.694 mm and minimum detF about 0.431.
Mean middle-finger hold force was about 0.439 N. It is a stronger loading
candidate for the P2 studies, but still exercises one loaded finger; the new
radius has not inherited the 20 mm calibration or a G6/G7 certificate.

A fresh 20 mm r3 control also completed all 4500 steps. During hold, the
25 mm sphere had 21 force-producing contact samples versus 10 for 20 mm,
and mean middle-finger force increased from 0.0256 to 0.439 N (about 17 times).
This establishes increased loading, not additional loaded fingers or a
calibrated performance target. The iteration sweep and new hotspot profiling
remain separate P2 work.
