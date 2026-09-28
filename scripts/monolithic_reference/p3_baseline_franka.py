# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parameterize only the two baseline scene constructors, leaving stepping/recording inherited.

Constructor bodies are copied from 39f80c48; physical helpers and all runtime
methods come from the explicitly selected baseline checkout. This adapter is
private T0 tooling and will be consolidated with the scene entry points in T3.
"""

import gzip
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton
from newton.examples.softbody.example_monolithic_franka_grasp import Example
from scripts.monolithic_reference.franka_grasp import (
    DT,
    REPO,
    TIMES,
    FrankaGrasp,
    build_model,
    plan_poses,
    sha,
    smooth_fields,
)
from scripts.monolithic_reference.verify_p3_coupled import save, state_arrays


class BaselineFranka(FrankaGrasp):
    """Select the existing law and globalization without changing the physical scene."""

    def __init__(self, cache, device="cuda:0", *, law, globalization):
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        self.model, self.metadata = build_model(cache, device=device)
        self.high, self.low, plan = plan_poses(self.model)
        smooth, checks = smooth_fields(self.model)
        self.metadata.update(
            plan=plan,
            sdf_checks=checks,
            sdf_identity=smooth.identity,
            sdf_geometric_certification="not yet certified for new Franka assets",
            publish_policy="current_iterate",
            dt=DT,
            duration=float(TIMES[-1]),
        )
        widths = (0.01,) * 7 + (0.001,) * 2
        self.solver = SolverMonolithic._create_coupled_variational(
            self.model,
            contact_model=law,
            globalization=globalization,
            alignment_policy="mask",
            publish_policy="current_iterate",
            allow_nonconservative_energy=True,
            friction_velocity_scale=0.01,
            contact_stiffness=1e7,
            friction_coefficient=0.5,
            normal_smoothing=1e-4,
            material_model="smith_log_stabilized",
            mass_mode="consistent",
            tet_rest_density=wp.full(self.model.tet_count, 1000.0, dtype=float, device=self.model.device),
            joint_terms=SolverMonolithic.JointTerms(
                implicit_pd=True, limits=True, friction=True, limit_width=widths, friction_velocity_scale=widths
            ),
            pcg_mode="production",
            use_cuda_graph=True,
            smooth_sdf=smooth,
        )
        source = self.model.state()
        source.joint_q.assign(self.high.astype(np.float32))
        source.joint_qd.zero_()
        newton.eval_fk(self.model, source.joint_q, source.joint_qd, source)
        self.state = self.solver._create_coupled_state(dt=DT, source=source)
        self.control = self.model.control()
        self.steps = 0
        self.failed = False
        self.phase = "settle"
        self.metadata["actual_config"] = self.solver._config
        self.metadata["config_digest"] = self.solver._kinematics.digest.hex()
        self.metadata["baseline_options"] = {
            "law": law,
            "globalization": globalization,
            "mode": "production",
            "graph": True,
        }


class BaselineFrankaExample(Example):
    """Keep the original frame/step writer and physical acceptance checks."""

    def __init__(self, viewer, args):
        if wp.get_device(args.device).is_cpu:
            raise ValueError("This example requires CUDA mesh SDFs")
        self.output = Path(args.output) if args.output else None
        if self.output:
            self.output.mkdir(parents=True, exist_ok=False)
        self.case = BaselineFranka(args.asset_cache, device=args.device, law=args.law, globalization=args.globalization)
        self.viewer = viewer
        self.viewer.set_model(self.case.model)
        self.viewer.set_camera(pos=wp.vec3(1.35, -1.2, 0.8), pitch=-22.0, yaw=135.0)
        self.display = self.case.model.state()
        self.sim_time = 0.0
        self.rows = []
        self.frames = []
        self.chunks = []
        self.closed = False
        self.weights = self.case.model.particle_mass.numpy().astype(float)
        self.initial_com = np.average(self.case.state.particle_q.numpy(), axis=0, weights=self.weights)
        c = self.case.solver._contact
        self.plane_keys = self.case.model.shape_type.numpy()[
            c.shapes.numpy()[np.arange(c.count) % c.shapes.size]
        ] == int(newton.GeoType.PLANE)
        if self.output:
            self.log = gzip.open(self.output / "steps.jsonl.gz", "wt")
            save(self.output / "configuration.json", self.case.metadata)
            save(self.output / "command.json", {"argv": sys.argv, "cwd": str(Path.cwd())})
            save(
                self.output / "source-hashes.json",
                {
                    str(p.relative_to(REPO)): sha(p)
                    for folder in ("newton", "scripts/monolithic_reference")
                    for p in (REPO / folder).rglob("*.py")
                },
            )
            np.savez_compressed(self.output / "initial.npz", **state_arrays(self.case.state))
            np.savez_compressed(
                self.output / "model.npz",
                **{name: value.numpy() for name, value in vars(self.case.model).items() if isinstance(value, wp.array)},
            )
            self.record("INITIAL")
        self.begin = time.perf_counter()
