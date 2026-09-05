# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate offline comparison inputs without importing either physics runtime."""

import hashlib
import json
import math
import re
import struct
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class ComparisonManifest:
    """Hold versioned JSON inputs; a DRAFT never certifies measured evidence."""

    data: dict


@dataclass(frozen=True, slots=True)
class WorktreeIdentity:
    """Record an exact clean repository identity."""

    path: Path
    sha: str


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Hold a JSON step record using the separate physical-force/residual schema."""

    data: dict


@dataclass(frozen=True, slots=True)
class MappingEntry:
    """Describe an explicit physical convention mapping and blocked conclusions."""

    mapping_id: str
    manifest_path: str
    newton_binding: str
    superdex_binding: str
    units: str
    status: Literal["MATCHED", "INTENTIONALLY_DIFFERENT", "UNMAPPED"]
    rationale: str
    blocks: tuple[str, ...]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_manifest(path: Path) -> ComparisonManifest:
    """Load and validate a manifest while rejecting duplicate JSON keys."""
    manifest = ComparisonManifest(json.loads(path.read_text(), object_pairs_hook=_unique_object))
    validate_manifest(manifest, require_frozen=False)
    return manifest


def canonical_manifest_bytes(manifest: ComparisonManifest) -> bytes:
    """Encode every field as sorted, compact UTF-8 JSON, rejecting nonfinite floats."""
    return json.dumps(
        manifest.data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def compute_manifest_sha256(manifest: ComparisonManifest) -> str:
    """Hash canonical inputs including draft status and repository identities."""
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def _validate(value, schema, path, *, frozen):
    # Only the JSON Schema keywords used by the checked-in schema are supported.
    if value is None and not frozen:
        return
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: invalid enum value {value!r}")
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int, "number": (int, float)}
    if kind and (not isinstance(value, types[kind]) or (kind in ("integer", "number") and isinstance(value, bool))):
        raise ValueError(f"{path}: expected {kind}")
    if kind == "object":
        properties = schema["properties"]
        unknown = value.keys() - properties.keys()
        missing = set(schema["required"]) - value.keys()
        if unknown or (frozen and missing):
            raise ValueError(f"{path}: unknown {sorted(unknown)}; missing required {sorted(missing)}")
        for key, child in value.items():
            _validate(child, properties[key], f"{path}.{key}", frozen=frozen)
    elif kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise ValueError(f"{path}: invalid array length")
        for index, child in enumerate(value):
            _validate(child, schema["items"], f"{path}[{index}]", frozen=frozen)
    elif kind in ("number", "integer"):
        if frozen and kind == "number" and path.startswith("manifest.physics"):
            value = _stored_float32(value)
        if (
            not math.isfinite(value)
            or value < schema.get("minimum", -math.inf)
            or value <= schema.get("exclusiveMinimum", -math.inf)
        ):
            raise ValueError(f"{path}: invalid numeric value")
    elif kind == "string":
        if len(value) < schema.get("minLength", 0) or (
            "pattern" in schema and not re.fullmatch(schema["pattern"], value)
        ):
            raise ValueError(f"{path}: invalid string")


def validate_manifest(manifest: ComparisonManifest, *, require_frozen: bool) -> None:
    """Validate populated draft fields or the complete frozen V0.1 input contract."""
    canonical_manifest_bytes(manifest)
    data = manifest.data
    required_identity = {"schema_version", "fixture_id", "status", "repositories", "units", "mappings"}
    if not isinstance(data, dict) or not required_identity <= data.keys():
        raise ValueError("manifest: missing required identity, units or mappings")
    if data["status"] not in ("DRAFT", "FROZEN"):
        raise ValueError("manifest status must be DRAFT or FROZEN")
    frozen = data["status"] == "FROZEN"
    if require_frozen and not frozen:
        raise ValueError("a FROZEN manifest is required; draft inputs are not measured evidence")
    schema = json.loads(Path(__file__).with_name("manifest.schema.json").read_text())
    _validate(data, schema, "manifest", frozen=frozen)
    for key in required_identity:
        _validate(data[key], schema["properties"][key], key, frozen=True)
    if len({entry["mapping_id"] for entry in data["mappings"]}) != len(data["mappings"]):
        raise ValueError("duplicate mapping_id")
    mapping_ids = {entry["mapping_id"] for entry in data["mappings"]}
    required_mappings = {
        "frame.world",
        "mesh.boundary",
        "material.mass",
        "material.neo_hookean",
        "contact.gap",
        "contact.gradient_norm",
        "control.command",
        "contact.normal_law",
    }
    if frozen and not required_mappings <= mapping_ids:
        raise ValueError("FROZEN manifest is missing required physics mappings")
    if frozen and any(entry["status"] == "UNMAPPED" and entry["blocks"] for entry in data["mappings"]):
        raise ValueError("FROZEN manifest has UNMAPPED entries blocking physics conclusions")
    physics = data.get("physics") or {}
    rest = physics.get("rest_positions_m")
    if rest is not None:
        count = len(rest)
        for key, width in (("tet_indices", 4), ("boundary_faces", 3)):
            for indices in physics.get(key) or []:
                if len(set(indices)) != width or any(index >= count for index in indices):
                    raise ValueError(f"physics.{key}: degenerate or out-of-range topology")
        for key in ("fixed_nodes", "node_masses_kg"):
            if physics.get(key) is not None and len(physics[key]) != count:
                raise ValueError(f"physics.{key}: expected one value per node")
        for key in ("x_m", "v_m_s"):
            initial = physics.get("initial") or {}
            if initial.get(key) is not None and len(initial[key]) != count:
                raise ValueError(f"physics.initial.{key}: expected one vector per node")
    tets, materials = physics.get("tet_indices"), physics.get("tet_materials_pa_pa_pas")
    if tets is not None and materials is not None and len(tets) != len(materials):
        raise ValueError("physics: expected one material per tet")
    if frozen:
        _validate_frozen_physics(physics, data["benchmark"])


def _stored_float32(value):
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("physics value exceeds float32 storage") from error


def _determinant(matrix):
    a, b, c = matrix
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def _require_unit(vector, path):
    # Permit float32 representation roundoff, not an unnormalized input axis/quaternion.
    if not math.isclose(sum(value * value for value in vector), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{path}: expected a unit vector or quaternion")


def _validate_frozen_physics(physics, benchmark):
    joints, links, initial = physics["joints"], physics["links"], physics["initial"]
    dof_count = sum(joint["type"] != "FIXED" for joint in joints)
    for key in ("q", "qd"):
        if len(initial[key]) != dof_count:
            raise ValueError(f"physics.initial.{key}: expected one value per dynamic joint DoF")
    parents = {}
    for joint in joints:
        parent, child = joint["parent"], joint["child"]
        if parent >= len(links) or child >= len(links) or parent == child or child in parents:
            raise ValueError("physics.joints: invalid parent/child indices or duplicate child")
        parents[child] = parent
        if joint["type"] != "FIXED":
            _require_unit(joint["axis"], "physics.joints.axis")
        for key in ("parent_xform", "child_xform"):
            _require_unit(joint[key][3:], f"physics.joints.{key}")
    if set(parents) != set(range(len(links))):
        raise ValueError("physics.joints: every link must belong to a world-anchored tree")
    for start in parents:
        current = start
        visited = set()
        while current != -1:
            if current in visited:
                raise ValueError("physics.joints: cycle in world-anchored tree")
            visited.add(current)
            current = parents[current]
    for link in links:
        _require_unit(link["xform"][3:], "physics.links.xform")
        inertia = link["inertia_kg_m2"]
        if (
            any(inertia[i][j] != inertia[j][i] for i in range(3) for j in range(3))
            or inertia[0][0] <= 0
            or inertia[0][0] * inertia[1][1] - inertia[0][1] ** 2 <= 0
            or _determinant(inertia) <= 0
        ):
            raise ValueError("physics.links.inertia_kg_m2: expected symmetric positive-definite inertia")

    faces = Counter()
    rest = physics["rest_positions_m"]
    if len({tuple(sorted(tet)) for tet in physics["tet_indices"]}) != len(physics["tet_indices"]):
        raise ValueError("physics.tet_indices: duplicate tet")
    for tet in physics["tet_indices"]:
        edges = [[rest[index][axis] - rest[tet[0]][axis] for axis in range(3)] for index in tet[1:]]
        if _determinant(edges) <= 0:
            raise ValueError("physics.tet_indices: rest tet must have positive volume")
        faces.update(tuple(sorted(tet[:opposite] + tet[opposite + 1 :])) for opposite in range(4))
    actual_faces = [tuple(sorted(face)) for face in physics["boundary_faces"]]
    boundary = {face for face, count in faces.items() if count == 1}
    if (
        any(count > 2 for count in faces.values())
        or len(set(actual_faces)) != len(actual_faces)
        or set(actual_faces) != boundary
    ):
        raise ValueError("physics.boundary_faces: expected exact manifold tet boundary without duplicates")
    for node, fixed in enumerate(physics["fixed_nodes"]):
        if not fixed and _stored_float32(physics["node_masses_kg"][node]) <= 0:
            raise ValueError("physics.node_masses_kg: dynamic nodes need positive mass")
        if fixed and any(initial["v_m_s"][node]):
            raise ValueError("physics.initial.v_m_s: fixed nodes must have zero velocity")
    if any(_stored_float32(material[0]) <= 0 for material in physics["tet_materials_pa_pa_pas"]):
        raise ValueError("physics.tet_materials_pa_pa_pas: shear modulus must be positive")

    for shape in physics["shapes"]:
        if shape["link"] >= len(links):
            raise ValueError("physics.shapes.link: out-of-range link")
        _require_unit(shape["xform"][3:], "physics.shapes.xform")
        sdf = shape["sdf"]
        scale = [_stored_float32(value) for value in sdf["runtime_scale"]]
        if any(value <= 0 for value in scale) or any(value <= 0 for value in sdf["resolution"]):
            raise ValueError("physics.shapes.sdf: scale and resolution must be positive")
        if shape["type"] == "volume_sdf" and not sdf["scale_baked"] and len(set(scale)) != 1:
            raise ValueError("physics.shapes.sdf: unbaked volume scale must be exact-uniform float32")
        dimensions = shape["dimensions_m"]
        used_dimensions = {
            "sphere": 1,
            "box": 3,
            "capsule": 2,
            "cylinder": 2,
            "cone": 2,
            "infinite_plane": 0,
            "volume_sdf": 3,
        }
        if any(value <= 0 for value in dimensions[: used_dimensions[shape["type"]]]):
            raise ValueError("physics.shapes.dimensions_m: active primitive dimensions must be positive")

    for slot, sample in enumerate(physics["contact"]["barycentric"]):
        expected = [2 / 3 if axis == slot else 1 / 6 for axis in range(3)]
        if [_stored_float32(value) for value in sample] != [_stored_float32(value) for value in expected]:
            raise ValueError("physics.contact.barycentric: expected the three fixed P1Q3 slots")
    drive = physics["drive"]
    for key in ("stiffness", "damping"):
        if len(drive[key]) != dof_count:
            raise ValueError(f"physics.drive.{key}: expected one value per dynamic joint DoF")
    previous_time = -math.inf
    for sample in drive["trajectory"]:
        if sample["time_s"] <= previous_time or any(len(sample[key]) != dof_count for key in ("target_q", "target_qd")):
            raise ValueError("physics.drive.trajectory: require increasing times and joint-sized targets")
        previous_time = sample["time_s"]

    _require_unit(benchmark["approach_axis_world"], "benchmark.approach_axis_world")
    if any(node >= len(rest) for node in benchmark["soft_probe"]):
        raise ValueError("benchmark.soft_probe: out-of-range node")
    stages = benchmark["stages"]
    if not stages["free_space_end_step"] <= stages["contact_loading_end_step"] <= stages["settle_end_step"]:
        raise ValueError("benchmark.stages: stage boundaries must be ordered")
    for low, high in [benchmark["actual_closure_interval_m"], *benchmark["reference_envelope"].values()]:
        if low > high:
            raise ValueError("benchmark: interval lower bound exceeds upper bound")


def require_reference_worktree(path: Path, *, expected_sha: str) -> WorktreeIdentity:
    """Reject another SHA, repository subdirectories and any visible Git dirt."""
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise ValueError("expected_sha must be a full 40-character commit SHA")
    path = path.resolve()

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.PIPE).strip()
        except subprocess.CalledProcessError as error:
            raise ValueError(f"not a readable Git worktree: {path}") from error

    if Path(git("rev-parse", "--show-toplevel")).resolve() != path:
        raise ValueError("reference path must be the worktree root")
    sha = git("rev-parse", "HEAD")
    if sha != expected_sha:
        raise ValueError(f"reference SHA mismatch: expected {expected_sha}, got {sha}")
    if git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("reference worktree is dirty (including untracked files or submodules)")
    return WorktreeIdentity(path, sha)


def validate_step_record(record: dict) -> None:
    """Validate output structure and keep residual gradients distinct from forces."""
    if "force" in record:
        raise ValueError("ambiguous force field: separate residual_contribution and physical forces")
    schema = json.loads(Path(__file__).with_name("step_record.schema.json").read_text())
    _validate(record, schema, "step_record", frozen=True)
    json.dumps(record, allow_nan=False)
    parameters = json.loads(record["actual_parameters_json"], object_pairs_hook=_unique_object)
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("actual_parameters_json must encode effective parameters as a nonempty object")
    json.dumps(parameters, allow_nan=False)
    if record["converged"] and (record["rolled_back"] or record["committed_unconverged"]):
        raise ValueError("converged records cannot roll back or commit an unconverged candidate")
    if record["rolled_back"] and record["committed_unconverged"]:
        raise ValueError("a rolled-back record cannot commit an unconverged candidate")


def run_adapter(
    implementation: Literal["newton", "superdex"],
    manifest_path: Path,
    output_dir: Path,
    *,
    worktree: Path,
    device: str,
    build_type: str,
    timeout_s: int,
) -> int:
    """Reserve the offline adapter interface without producing simulation evidence."""
    raise NotImplementedError("PR-0 provides manifest and worktree checks; physics adapters are not implemented")


def compare_runs(
    newton_output: Path, superdex_output: Path, mapping_report: list[MappingEntry], output_path: Path
) -> int:
    """Reserve mapped comparison until actual adapters and evidence are available."""
    raise NotImplementedError("PR-0 cannot compare physics outputs before reference adapters exist")
