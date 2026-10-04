# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capture native failure prefixes for offline precision audit, without feedback."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_recording import save, serial, state_arrays
from scripts.monolithic_reference import run_precision as runner
from scripts.monolithic_reference.retained_assets import ORIGINAL_FROZEN


@wp.kernel
def composed_norms(poses: wp.array[Any], transforms: wp.array[Any], bodies: wp.array[int], norms: wp.array[Any]):
    shape = wp.tid()
    value = transforms[shape]
    if bodies[shape] >= 0:
        value = poses[bodies[shape]] * value
    norms[shape] = wp.length(wp.transform_get_rotation(value))


def arrays(owner):
    """Copy device arrays after native execution; never write solver storage."""
    return {name: value.numpy().copy() for name, value in vars(owner).items() if isinstance(value, wp.array)}


class Observer:
    """Record the native evaluator, trial, scale and physical posterior boundaries."""

    def __init__(self, run, output):
        self.run, self.output = run, output
        self.rows = []
        self.alpha = None
        self.scale = None
        self.base = None
        self.direction = None
        self.linear = []
        self.snapshot("model", arrays(run.model))
        self.snapshot("tet", arrays(run._tet))
        self.snapshot("physics", arrays(run._tet.physics))
        self.snapshot("contact", arrays(run._contact))
        self.snapshot("objective", arrays(run._objective))
        self.snapshot("kinematics", arrays(run._kinematics))
        smooth = run._contact.smooth_sdf
        sdf = {}
        for i, field in enumerate(smooth.fields if smooth is not None else ()):
            for name in ("lower", "upper", "inv_dx", "dims"):
                sdf[f"{i}.{name}"] = np.asarray(getattr(field.data, name))
            sdf[f"{i}.coefficients"] = field.data.coefficients.numpy()
        self.snapshot("sdf", sdf)
        save(
            output / "physics.json",
            {
                "contact_parameters": run._contact.parameters,
                "quaternion_norm_tolerance": run._contact.quaternion_norm_tolerance,
                "law": run._contact.contact_model,
                "alignment": run._contact.alignment_policy,
                "material": run._tet.physics.material_model,
                "mass": run._tet.physics.mass_mode,
                "guard": run._guard,
                "config": serial(run._config),
                "sdf_scale_baked": run._contact._descriptors["scale_baked"].tolist()
                if run._contact._descriptors is not None
                else None,
            },
        )
        for name in ("_evaluate", "_take_trial", "_assemble", "_physical_linear_error"):
            original = getattr(run, name)
            setattr(self, name + "_native", original)
            setattr(run, name, getattr(self, name))

    def snapshot(self, name, data):
        """Write an immutable numeric snapshot and return its identity."""
        path = self.output / (name + ".npz")
        np.savez_compressed(path, **data)
        return {"file": path.name, "sha256": runner.sha(path)}

    def _assemble(self, iteration, *, device_result=False):
        result = self._assemble_native(iteration, device_result=device_result)
        self.scale = result[1].numpy().astype(self.run._numpy_dtype) if device_result else result[1].copy()
        self.rows[-1]["scale"] = self.snapshot(
            f"scale-{len(self.rows):04d}",
            {
                "scale": self.scale,
                "dynamic": self.run._diagonal.numpy(),
                "diagonal": self.run._linear.k_global_scalar_bsr.values.numpy(),
                "offsets": self.run._linear.k_global_scalar_bsr.offsets.numpy(),
                "columns": self.run._linear.k_global_scalar_bsr.columns.numpy(),
            },
        )
        return result

    def _physical_linear_error(self, matrix, direction, residual, scale, *, device_result=False):
        result = self._physical_linear_error_native(matrix, direction, residual, scale, device_result=device_result)
        self.direction = direction.numpy().astype(self.run._numpy_dtype) if device_result else direction.copy()
        # Capture actual B32 storage; offline audit applies the matrix independently.
        data = {"direction": self.direction, "residual": residual, "scale": scale, "native_error": result}
        data = {
            name: value.numpy().astype(self.run._numpy_dtype) if isinstance(value, wp.array) else value
            for name, value in data.items()
        }
        for name in ("offsets", "columns", "values"):
            data[name] = getattr(self.run._linear.k_global_scalar_bsr, name).numpy()
        item = self.snapshot(f"linear-{len(self.linear):04d}", data)
        self.linear.append(dict(step=self.run._step_generation, **item))
        return result

    def _take_trial(self, alpha, dt):
        self.alpha = float(alpha)
        self.base = state_arrays(self.run._accepted)
        return self._take_trial_native(alpha, dt)

    def _evaluate(self, state, dt, *, build_tangent=True, device_result=False):
        run = self.run
        role = "original" if state is run._original else "trial" if state is run._trial else "accepted"
        error, result = None, None
        try:
            result = self._evaluate_native(state, dt, build_tangent=build_tangent, device_result=device_result)
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            data = {"state." + k: v for k, v in state_arrays(state).items()}
            for prefix, owner in (("objective", run._objective), ("kinematics", run._kinematics)):
                data.update({prefix + "." + k: v for k, v in arrays(owner).items()})
            for prefix, owner in (("original", run._original),):
                data.update({prefix + "." + k: v for k, v in state_arrays(owner).items()})
            if role == "trial":
                data.update({"base." + k: v for k, v in self.base.items()})
            data["native.contact_flags"] = run._contact.flags.numpy()
            data["native.poses"] = run._kinematics.poses.numpy()
            norms = wp.empty(run.model.shape_count, dtype=run._dtype, device=run.model.device)
            wp.launch(
                composed_norms,
                run.model.shape_count,
                [run._kinematics.poses, run._contact.transforms, run._contact.bodies, norms],
                device=run.model.device,
            )
            data["native.composed_norms"] = norms.numpy()
            data["native.particle_force"] = run._frozen_force.numpy()
            data["native.gravity"] = run._frozen_gravity.numpy()
            if result is not None:
                data["native.energy"], data["native.residual"], data["native.absolute"] = result
                if device_result:
                    data["native.residual"] = result[1].numpy().copy()
            if self.scale is not None:
                data["native.scale"] = self.scale
            if self.direction is not None:
                data["native.direction"] = self.direction
            item = self.snapshot(f"evaluation-{len(self.rows):04d}", data)
            self.rows.append(
                dict(
                    step=run._step_generation,
                    role=role,
                    alpha=self.alpha if role == "trial" else None,
                    dt=float(dt),
                    error=error,
                    **item,
                )
            )
            save(self.output / "evaluations.json", self.rows)
            save(self.output / "linear.json", self.linear)


def main():
    """Reproduce a failure prefix, preserving every difference from the prior run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original_build = runner.build
    prior = json.loads((args.reference / "receipt.json").read_text())
    case = prior["case"]
    observed = []

    def build(case, *, input_manifest=runner.FROZEN):
        result = original_build(case, input_manifest=input_manifest)
        observed.append(Observer(result[1], args.output))
        return result

    runner.build = build
    result = runner.worker(
        case,
        args.controls.resolve(),
        args.output.resolve(),
        prior["first_failure_step"],
        input_manifest=Path(prior.get("input_manifest", ORIGINAL_FROZEN)),
    )
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt.get("exception") is None, receipt.get("exception")
    assert receipt["first_failure_step"] == prior["first_failure_step"]
    assert receipt["counts"] == prior["counts"] and receipt["rollback_verified"]
    differences = []
    pairs = [("initial.npz", "initial.npz"), ("final.npz", "final.npz")]
    assert len(receipt["chunks"]) == len(prior["chunks"])
    pairs.extend((a["file"], b["file"]) for a, b in zip(prior["chunks"], receipt["chunks"], strict=True))
    for left_name, right_name in pairs:
        with np.load(args.reference / left_name) as left, np.load(args.output / right_name) as right:
            assert left.files == right.files
            for key in left.files:
                if left_name == "initial.npz" or left[key].dtype.kind not in "fc":
                    np.testing.assert_array_equal(left[key], right[key], err_msg=left_name + ":" + key)
                if not np.array_equal(left[key], right[key]):
                    delta = left[key].astype(float) - right[key].astype(float)
                    differences.append(
                        {
                            "file": left_name,
                            "field": key,
                            "max": float(abs(delta).max()),
                            "rms": float(np.sqrt(np.mean(delta**2))),
                        }
                    )
    save(
        args.output / "observation.json",
        {
            "reference": str(args.reference.resolve()),
            "published_bytes_equal": not differences,
            "initial_bytes_equal": True,
            "classification_equal": True,
            "differences": differences,
            "failure_step": prior["first_failure_step"],
            "evaluations": len(observed[0].rows),
            "native_exit_code": result,
            "boundary": "This instrumented native run is audited on its own states. Differences from the uninstrumented run are retained without a widened equivalence tolerance.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
