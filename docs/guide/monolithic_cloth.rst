.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Experimental monolithic cloth
=============================

.. experimental::

   ``SolverMonolithic.ClothInput`` supports independent triangular
   shells, including naturally folded reference geometry. Multiple disjoint
   cloth and cable objects can share the global solve with supported articulation,
   free rigid and tet actors. Cloth contacts supported articulation, free rigid
   and static shapes. Self-collision, soft-soft contact, attachments and moving
   cloth boundary controls remain outside this interface.

Use native Model triangles and specify their row IDs in ``cloth_inputs``. The
triangle ``tri_ke`` and ``tri_ka`` fields supply two-dimensional membrane mu and
lambda [N/m]; native damping, drag, lift, activation, hinges and springs must be
zero. Each ClothInput specifies bending ``(alpha, beta)`` [N m], thickness [m]
and surface density [kg/m²]. Thickness is an authored envelope quantity; it does
not automatically rescale the supplied membrane or bending coefficients.
For an isotropic plane-stress material, a convenient small-strain mapping is
``mu=E*h/(2*(1+nu))``, ``lambda=E*h*nu/(1-nu**2)``,
``alpha=lambda*h**2/12`` and ``beta=2*mu*h**2/12``.

Reference positions define both the natural metric and natural curvature.
Dynamic Model masses must equal one third of each adjacent reference triangle's
area times surface density. Fixed particles use zero inverse mass; they keep
their input position during a step. The physical mass ledger still includes
fixed vertices. External nodal forces use ``state.particle_f``; gravity and
forces are frozen at step entry. No mechanical tree or placeholder tet is needed.
Create state/control through the solver. Reset and checkpoint restore preserve
reference shape; failure leaves published coordinates, velocities and force
history unchanged.

Material contact
----------------

Enable particle collision on the target shapes. Each cloth triangle contributes
P1Q3 samples on both sides of its middle surface, with reference area divided
by three per sample and a half-thickness gap offset. Back-facing terms use the
existing contact law's normal selection. The middle-surface xyz coordinates
carry the forces; this model adds no through-thickness rotational coordinates.
Both endpoints contribute to the global residual and full cross blocks.

Targets support analytic planes, spheres and boxes, and CUDA mesh targets with
an existing bound C2 SDF. Planes must be infinite and static. Material samples
respect Model world membership; solver actors still share one world.
Reference areas stay fixed during deformation.
C-EE, C-IE and C-II share the existing smoothed penalty/friction law. Use
``globalization="ResNorm"``; energy-based acceptance with material face selection
requires ``allow_nonconservative_energy=True`` and is reported as heuristic.

Contact uses the bounded deterministic pool. Insufficient material contact
capacity rolls back the step; it does not publish a truncated force sum.
``get_contact_forces()`` returns particle forces and body/joint reactions from
the same finally assembled candidate. Material checkpoint schema 6 also carries
cable contact torque. Earlier material schemas are rejected because they lack
this history; the existing A/R/T schema compatibility remains unchanged.

Numerical contract
------------------

The membrane uses stable Neo-Hookean energy with native-precision energy and
residual. Its complete 6-by-6 material Hessian is spectrally projected before
mapping to nodal coordinates. Six-node curvature bending uses an analytic
strain Jacobian and its full 18-by-18 material tangent. Missing boundary
neighbors are eliminated by affine extrapolation before differentiation.
All cross-node terms remain in the FP32 global matrix. Three-coordinate vertex
Cholesky blocks are only a preconditioner. Cloth and cable coordinates use one
Newton direction, line search and transaction.

Each cloth has an independent xyz residual RMS goal of
``1e-7 N + 1e-6 * initial_RMS`` by default. The existing FP32 absolute-tolerance
experiment override also applies to cloth xyz groups. Native FP64 is the
acceptance path; B and PCG remain FP32 with native direction qualification.
The full element matrix is checked against independent NumPy/complex-step
references, including negative lambda, compression, missing neighbors and
natural folds. Scalar inertia never substitutes for material PSD checks.

Current and adjacent triangles must retain relative area and corner sine
above 1e-8, with squared adjacent-normal sum above 1e-8. The whole trial segment
must retain positive area projection onto each accepted triangle normal. This
conservative chart guard can shorten a valid large rotation, and rejects
collapse hidden between valid endpoints. An invalid inertial predictor falls
back to the input positions. These checks do not provide self-collision or CCD.

The material nonlinear loop uses host decisions. CUDA graph mode captures the
linear solve and supported segments, not an entire cloth step. This small-scene
qualification does not establish large-mesh or accelerator performance.

Loading and recovery example
----------------------------

.. code-block:: console

   python -m newton.examples monolithic_cloth --kind patch --viewer null --test
   python -m newton.examples monolithic_cloth --kind prefold --viewer null --test
   python -m scripts.monolithic_reference.run_cloth_benchmark prefold --output /tmp/cloth-prefold

The frozen recipes in ``scripts/monolithic_reference/cloth_recipes.json`` use
32 triangles, fixed borders and a central force ramp, hold, unload and recovery.
The pre-fold is a 60-degree natural crease. The viewer overlays the reference
mesh in orange; rendered positions have no displacement magnification.
Recovery uses backward Euler's numerical dissipation without added physical
damping. The example validates deformation, independent elastic energy, area,
residuals, final fold angle and fresh-solver checkpoint continuation. It is a
no-contact functional benchmark; absolute constitutive accuracy is qualified
separately by element and assembly tests. A partial run or stopped solve cannot
pass. The runner retains source hashes, per-step measurements and failures.
