# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay one recorded hand failure and inspect contact changes at a chosen Newton iterate."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

import newton
from newton.examples.softbody.monolithic_recording import serial

from .run_hand_tasks import DT, HandTask, source_identity


def replay_parameters(identity, manifest):
    """Resolve recorded inputs; only the identified PR09 recipe has historical defaults."""
    legacy = manifest.get("source_sha256", {}).get("scripts/monolithic_reference/run_hand_tasks.py") == (
        "f6636403e8fc1e058860f3fd4e10a7a5574d3fc62cb57f896c83fb78db619820"
    )
    defaults = {"height": 0.0, "control_mode": "open-loop", "contact_stiffness_N_m3": 1e7} if legacy else {}

    def recorded(name, scene_name=None):
        scene_name = scene_name or name
        if name in manifest and scene_name in identity and manifest[name] != identity[scene_name]:
            raise ValueError(f"Scene and manifest disagree on {name}")
        if scene_name in identity:
            return identity[scene_name]
        if name in manifest:
            return manifest[name]
        if name in defaults:
            return defaults[name]
        raise ValueError(f"Missing recorded {name}; cannot assume current defaults")

    task = identity["task"]
    if task not in ("grasp-soft", "grasp-rigid", "pull-slider") or manifest["task"] not in (task, "all"):
        raise ValueError("Scene and manifest task disagree or are unsupported")
    if manifest["dt"] != DT or manifest["precision"] != "fp64":
        raise ValueError("Unsupported recorded timestep or precision")
    parameters = {
        "task": task,
        "surface_source": recorded("surface_source"),
        "friction": recorded("friction"),
        "offset": recorded("offset"),
        "height": recorded("height", "height_m"),
        "control_mode": recorded("control_mode"),
        "contact_stiffness": recorded("contact_stiffness_N_m3"),
    }
    if parameters["control_mode"] not in ("open-loop", "contact") or parameters["surface_source"] not in (
        "cad",
        "qualified",
    ):
        raise ValueError("Unsupported recorded control_mode or surface_source")
    if not np.isfinite([parameters[k] for k in ("friction", "offset", "height", "contact_stiffness")]).all():
        raise ValueError("Nonfinite recorded scene parameter")
    expected = [0.0, parameters["offset"], 0.0202 + parameters["height"]]
    if not np.allclose(identity["object_position_m"], expected, rtol=0, atol=1e-12):
        raise ValueError("Recorded object position disagrees with offset/height")
    return parameters


def restore_control(scene, payload, next_step):
    """Restore the last published feedback and controller before its failed advance."""
    # The historical PR09 source snapshot has no feedback module.
    from .hand_task_control import ContactHandControl, HandObservation  # noqa: PLC0415

    if not np.isclose(payload["time"], next_step * DT, rtol=0, atol=1e-12):
        raise ValueError("Controller snapshot time disagrees with failed step")
    controller = ContactHandControl.from_snapshot(payload["controller"])
    if not np.isclose(controller.last_time, (next_step - 1) * DT, rtol=0, atol=1e-12):
        raise ValueError("Controller last_time disagrees with checkpoint")
    for name in ("task", "mass", "friction", "required_load", "max_rate"):
        if not np.array_equal(getattr(controller, name), getattr(scene.controller, name)):
            raise ValueError(f"Controller snapshot configuration differs: {name}")
    feedback = None if payload["feedback"] is None else HandObservation.from_snapshot(payload["feedback"])
    if feedback is None:
        if next_step != 1:
            raise ValueError("Missing published feedback for checkpoint generation")
    elif not feedback.valid or feedback.generation != next_step - 1:
        raise ValueError("Feedback generation disagrees with checkpoint")
    scene.controller, scene.feedback = controller, feedback


def contact_chunk(batch, mode):
    """Copy only the valid native candidate prefix after its law evaluation."""
    count = int(batch.pool.candidates.count.numpy()[0])
    geometry = batch.current
    normal = geometry.soft_normal.numpy()[:count]
    gradient = geometry.gradient.numpy()[:count]
    return {
        "identity": batch.pool.candidates.identities.numpy()[:count],
        "mode": np.full(count, mode),
        "source": batch.samples.sources.numpy()[:count],
        "target": batch.samples.targets.numpy()[:count],
        "mask": batch.mask.numpy()[:count],
        "alignment": np.einsum("ij,ij->i", normal, gradient),
        "point": geometry.y.numpy()[:count],
        "gradient": gradient,
        "source_normal": normal,
        "distance": geometry.distance.numpy()[:count],
        "energy": batch.pool.candidates.energy.numpy()[:count].sum(axis=1),
        "omitted_energy": batch.omitted.numpy()[:count, :2].sum(axis=1),
        "local_residual_norm": np.linalg.norm(batch.local_r.numpy()[:count], axis=1),
    }


def compare_contacts(before, after):
    """Compare complete snapshots by sample identity, independent of storage order."""
    snapshots = []
    for values in (before, after):
        ids = np.asarray(values["identity"])
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate contact identities in snapshot")
        order = np.argsort(ids)
        snapshots.append({name: np.asarray(data)[order] for name, data in values.items()})
    before, after = snapshots
    if not np.array_equal(before["identity"], after["identity"]):
        return {"same_candidates": False}
    delta = after["energy"] - before["energy"]
    changed = np.flatnonzero(after["mask"] != before["mask"])
    ranked = changed[np.argsort(-np.abs(delta[changed]))][:12]
    return {
        "same_candidates": True,
        "mask_changes": len(changed),
        "mask_change_energy": float(delta[changed].sum()),
        "all_contact_energy_change": float(delta.sum()),
        "changed_contacts": [
            {
                "identity": int(after["identity"][i]),
                "source": int(after["source"][i]),
                "target": int(after["target"][i]),
                "before": {name: serial(values[i]) for name, values in before.items()},
                "after": {name: serial(values[i]) for name, values in after.items()},
            }
            for i in ranked
        ],
    }


class ContactObserver:
    """Observe one Newton iteration with batch graphs explicitly disabled."""

    def __init__(self, solver, iteration, *, freeze_contact=None):
        self.solver, self.iteration = solver, iteration
        self.index, self.alpha, self.capture = -1, None, False
        self.rows, self.chunks, self.base = [], [], None
        self.freeze_contact = freeze_contact
        self.refresh, self.evaluate = solver._refresh_candidates, solver._evaluate
        self.take_trial = solver._take_trial
        batch = solver._contact._batch
        self.stage = batch._stage
        self.query = batch._query
        batch.graphs.enabled = False
        solver._refresh_candidates = self._refresh
        solver._take_trial = self._take_trial
        solver._evaluate = self._evaluate
        batch._stage = self._stage
        batch._query = self._query

    def _refresh(self):
        self.index += 1
        return self.refresh()

    def _take_trial(self, alpha, dt):
        self.alpha = float(alpha)
        return self.take_trial(alpha, dt)

    def _stage(self, state, mode, dt, **kwargs):
        result = self.stage(state, mode, dt, **kwargs)
        if self.capture and not kwargs["begin"]:
            self.chunks.append(contact_chunk(self.solver._contact._batch, mode))
        return result

    def _query(self, state, begin):
        result = self.query(state, begin)
        if self.capture and not begin and self.freeze_contact is not None and self.base is not None:
            batch = self.solver._contact._batch
            count = int(batch.pool.candidates.count.numpy()[0])
            ids = batch.pool.candidates.identities.numpy()[:count]
            selected = np.flatnonzero(ids == self.freeze_contact)
            base = self.base[0]
            original = np.flatnonzero(base["identity"] == self.freeze_contact)
            if len(selected) and len(original):
                i, j = int(selected[0]), int(original[0])
                normal = batch.current.soft_normal.numpy()
                gradient = batch.current.gradient.numpy()[i]
                dot = float(normal[i] @ gradient)
                sign = 1 if base["alignment"][j] > 0 else -1
                # Diagnostic only: reflect the normal component across the mask boundary.
                # No distance, collider gradient, slip, or force projection is changed.
                norm2 = gradient @ gradient
                if norm2 == 0:
                    raise ValueError("Cannot freeze alignment with a zero collider gradient")
                normal[i] += (sign * max(abs(dot), 1e-30) - dot) * gradient / norm2
                batch.current.soft_normal.assign(normal)
        return result

    def _evaluate(self, state, dt, **kwargs):
        self.capture = self.index == self.iteration
        self.chunks = []
        result = self.evaluate(state, dt, **kwargs)
        if not self.capture:
            return result
        run = self.solver
        contact = {name: np.concatenate([chunk[name] for chunk in self.chunks]) for name in self.chunks[0]}
        order = np.argsort(contact["identity"])
        contact = {name: values[order] for name, values in contact.items()}
        row = {
            "iteration": self.index,
            "role": "accepted" if state is run._accepted else "trial",
            "alpha": None if state is run._accepted else self.alpha,
            "energy": float(result[0]),
            "contact_energy": run._contact.energy.numpy().tolist(),
            "candidate_count": len(order),
            "mask_counts": {str(key): int(np.sum(contact["mask"] == key)) for key in np.unique(contact["mask"])},
        }
        residual = run._complete_r.numpy().copy()
        if state is run._accepted:
            self.base = (contact, row, residual)
            if self.freeze_contact is not None and self.freeze_contact not in contact["identity"]:
                raise ValueError("Frozen contact is absent from the observed candidate set")
        elif self.base is not None:
            base, base_row, base_r = self.base
            row["energy_change"] = row["energy"] - base_row["energy"]
            row["residual_change_norm"] = float(np.linalg.norm(residual - base_r))
            # Generalized coordinates mix translation and angle units; this is not a distance in metres.
            row["effective_displacement_max"] = float(np.max(np.abs(run._effective_displacement())))
            row.update(compare_contacts(base, contact))
        self.rows.append(row)
        return result


def main():
    """Load the unchanged checkpoint configuration and execute its next physical step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observe-iteration", type=int)
    parser.add_argument(
        "--freeze-contact", type=int, help="Diagnostic-only identity whose mask is held at the observed base"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.freeze_contact is not None and args.observe_iteration is None:
        parser.error("--freeze-contact requires --observe-iteration")
    if args.observe_iteration is not None and args.observe_iteration < 0:
        parser.error("--observe-iteration must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    source_before = source_identity()
    identity = json.loads((args.case / "scene.json").read_text())
    manifest_path = args.case.parent / "manifest.json"
    parameters = replay_parameters(identity, json.loads(manifest_path.read_text()))
    summary = json.loads((args.case / "summary.json").read_text())
    constructor = dict(parameters)
    for name, historical in (("control_mode", "open-loop"), ("contact_stiffness", 1e7)):
        if name not in inspect.signature(HandTask).parameters:
            if constructor.pop(name) != historical:
                raise ValueError(f"Imported historical runner cannot reconstruct {name}")
    scene = HandTask(assets=args.assets, device=args.device, **constructor)
    for name in ("urdf_sha256", "mesh_sha256", "contact_manifest_sha256", "cube_sha256"):
        if name in identity and scene.identity.get(name) != identity[name]:
            raise ValueError(f"Reconstructed asset identity differs: {name}")
    checkpoint = args.case / "last-published-state.npz"
    scene.solver.load_checkpoint(scene.state, checkpoint)
    if "configuration_digest" in identity and scene.solver._kinematics.digest.hex() != identity["configuration_digest"]:
        raise ValueError("Scene configuration digest disagrees with checkpoint")
    next_step = int(scene.state._monolithic_coupled.force_generation.numpy()[0]) + 1
    if next_step != summary["steps"]:
        raise ValueError("Checkpoint and failed-step summary disagree")
    control_path = args.case / "control-before-failed-step.json"
    if parameters["control_mode"] == "contact":
        restore_control(scene, json.loads(control_path.read_text()), next_step)
    elif control_path.exists() or getattr(scene, "controller", None) is not None:
        raise ValueError("Open-loop replay has unexpected feedback control state")
    input_paths = [manifest_path, args.case / "scene.json", args.case / "summary.json"]
    input_paths += [
        args.assets / name
        for name in ("recorded-control.json", "Ball_catch - Left_hand_motion_rad.csv", "close-18/manifest.json")
    ]
    if control_path.exists():
        input_paths.append(control_path)
    input_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    observer = (
        ContactObserver(scene.solver, args.observe_iteration, freeze_contact=args.freeze_contact)
        if args.observe_iteration is not None
        else None
    )
    scene.step(next_step * DT)
    source_after = source_identity()
    payload = {
        "newton_import": newton.__file__,
        "case": str(args.case.resolve()),
        "attempt_step": next_step,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "configuration_digest": scene.solver._kinematics.digest.hex(),
        "replay_parameters": parameters,
        "input_sha256": input_hashes,
        "contact_batch_graphs_disabled": observer is not None,
        "observed_iteration": args.observe_iteration,
        "diagnostic_frozen_contact": args.freeze_contact,
        "stats": serial(scene.solver.last_stats),
        "observations": [] if observer is None else observer.rows,
        "source_sha256": source_before,
        "source_unchanged": source_before == source_after,
        "changed_source_paths": sorted(
            name
            for name in source_before.keys() | source_after.keys()
            if source_before.get(name) != source_after.get(name)
        ),
    }
    (args.output / "audit.json").write_text(json.dumps(serial(payload), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "step": next_step,
                "status": str(scene.solver.last_stats.status),
                "observations": len(payload["observations"]),
            }
        )
    )


if __name__ == "__main__":
    main()
