# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-capacity BVH contact candidates; geometry and law envelope construction remain separate."""

import warnings
from typing import Any

import numpy as np
import warp as wp

from .contact_pool import _CONTACT_WORKERS, _merge_contact_keys


@wp.func_native("""
    float value = static_cast<float>(x);
    if (wp::float64(value) > x) {
        wp::uint32 bits;
        memcpy(&bits, &value, sizeof(bits));
        if (value == 0.0f) bits = 0x80000001U;
        else if (value > 0.0f) --bits;
        else ++bits;
        memcpy(&value, &bits, sizeof(value));
    }
    return value;
""")
def _lower_float(x: wp.float64) -> float: ...


@wp.kernel(enable_backward=False)
def _copy_bounds(
    lower: wp.array[Any],
    upper: wp.array[Any],
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
    status: wp.array[int],
):
    i = wp.tid()
    lo, hi = wp.vec3(), wp.vec3()
    valid = True
    for axis in range(3):
        a, b = wp.float64(lower[i][axis]), wp.float64(upper[i][axis])
        lo[axis] = _lower_float(a)
        hi[axis] = -_lower_float(-b)
        valid = valid and wp.isfinite(a) and wp.isfinite(b) and a <= b
    if not valid or not wp.isfinite(lo) or not wp.isfinite(hi):
        wp.atomic_or(status, 0, 1)
        lo, hi = wp.vec3(), wp.vec3()
    out_lower[i] = lo
    out_upper[i] = hi


@wp.func
def _allowed(source: int, target: int, offsets: wp.array[int], sources: wp.array[int]):
    lo, hi = offsets[target] + 0, offsets[target + 1] + 0
    end = hi
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if sources[mid] < source:
            lo = mid + 1
        else:
            hi = mid
    return lo < end and sources[lo] == source


@wp.func
def _emit_candidates(
    face: int,
    target: int,
    targets: int,
    samples_per_patch: int,
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
    count: wp.array[int],
    dropped: wp.array[wp.int64],
):
    slot = wp.atomic_add(count, 0, 0)
    reserved = int(0)
    while slot < keys.shape[0]:
        reserved = wp.min(samples_per_patch, keys.shape[0] - slot)
        previous = wp.atomic_cas(count, 0, slot, slot + reserved)
        if previous == slot:
            break
        slot = previous
        reserved = 0
    for sample in range(reserved):
        # Stable geometric ordinal, independent of the reservation order.
        keys[slot + sample] = (wp.uint64(face) * wp.uint64(targets) + wp.uint64(target)) * wp.uint64(
            samples_per_patch
        ) + wp.uint64(sample)
        indices[slot + sample] = slot + sample
    if reserved < samples_per_patch:
        wp.atomic_add(dropped, 0, wp.int64(samples_per_patch - reserved))


@wp.kernel(enable_backward=False)
def _finite_bounds(
    targets: wp.array[int],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    finite_lower: wp.array[wp.vec3],
    finite_upper: wp.array[wp.vec3],
):
    i = wp.tid()
    finite_lower[i] = lower[targets[i]]
    finite_upper[i] = upper[targets[i]]


@wp.kernel(enable_backward=False)
def _query_faces(
    bvh: wp.uint64,
    face_sources: wp.array[int],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    finite_targets: wp.array[int],
    targets: int,
    samples_per_patch: int,
    offsets: wp.array[int],
    sources: wp.array[int],
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
    count: wp.array[int],
    dropped: wp.array[wp.int64],
    status: wp.array[int],
):
    face = wp.tid()
    if status[0] != 0:
        return
    query = wp.bvh_query_aabb(bvh, lower[face], upper[face])
    item = int(0)
    while wp.bvh_query_next(query, item):
        target = finite_targets[item]
        if _allowed(face_sources[face], target, offsets, sources):
            _emit_candidates(face, target, targets, samples_per_patch, keys, indices, count, dropped)


@wp.kernel(enable_backward=False)
def _query_planes(
    face_sources: wp.array[int],
    planes: wp.array[int],
    targets: int,
    samples_per_patch: int,
    offsets: wp.array[int],
    sources: wp.array[int],
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
    count: wp.array[int],
    dropped: wp.array[wp.int64],
    status: wp.array[int],
):
    face, plane = wp.tid()
    target = planes[plane]
    # Keep allowed plane patches, parallel across faces instead of one long loop.
    if status[0] == 0 and _allowed(face_sources[face], target, offsets, sources):
        _emit_candidates(face, target, targets, samples_per_patch, keys, indices, count, dropped)


class _ContactQuery:
    """Query supplied patch/target envelopes without asserting their physical coverage.

    Faces are unique (source shape or -1 for tet, face) pairs in canonical order;
    bounds passed to update follow ``faces`` and ``targets``. Directed permissions
    come from the sampling owner, including particle flags and shape pair filters.
    Each patch retains all three P1Q3 slots. Candidate capacity drops individual
    whole identities; downstream contact resource admission is still required.

    Call check before consuming a generation outside capture, or propagate status
    in its owning device pipeline. No query is triggered by an out-of-range trial:
    a frozen list can omit contacts. This component supplies no automatic repair.
    """

    def __init__(
        self, faces, targets, allowed, *, planes=(), capacity, device, initial_bounds=None, _samples_per_patch=3
    ):
        def integers(values, columns):
            a = np.asarray(values)
            if a.size == 0:
                return np.empty((0, columns), dtype=np.int32) if columns > 1 else np.empty(0, dtype=np.int32)
            if a.ndim != (2 if columns > 1 else 1):
                raise ValueError("Invalid contact query metadata")
            shape = (len(a), columns) if columns > 1 else (len(a),)
            if a.shape != shape or a.dtype.kind not in "iu" or np.any(a > np.iinfo(np.int32).max) or np.any(a < -1):
                raise ValueError("Invalid contact query metadata")
            return a.astype(np.int32)

        if type(_samples_per_patch) is not int or _samples_per_patch not in (1, 3):
            raise ValueError("Contact patches require one surface sample or three P1Q3 slots")
        self.samples_per_patch = _samples_per_patch
        self.faces = integers(faces, 2)
        self.targets = np.unique(integers(targets, 1))
        self.allowed = np.unique(integers(allowed, 2), axis=0)
        self.planes = tuple(np.unique(integers(planes, 1)))
        if (
            np.any(self.faces[:, 1] < 0)
            or np.any(self.targets < 0)
            or len(np.unique(self.faces, axis=0)) != len(self.faces)
            or not set(self.planes).issubset(self.targets)
            or not set(self.allowed[:, 0]).issubset(self.faces[:, 0])
            or not set(self.allowed[:, 1]).issubset(self.targets)
        ):
            raise ValueError("Invalid contact query identities or permissions")
        self.faces = self.faces[np.lexsort((self.faces[:, 1], self.faces[:, 0]))]
        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or not 0 <= capacity <= np.iinfo(np.int32).max
            or len(self.faces) * len(self.targets) * self.samples_per_patch > np.iinfo(np.int64).max
        ):
            raise ValueError("Invalid contact query capacity")
        self.device = wp.get_device(device)
        self.capacity = capacity
        offsets, sources = [0], []
        for target in self.targets:
            sources.extend(self.allowed[self.allowed[:, 1] == target, 0])
            offsets.append(len(sources))
        self.face_sources = wp.array(self.faces[:, 0], dtype=int, device=self.device)
        self.offsets = wp.array(offsets, dtype=int, device=self.device)
        self.sources = wp.array(sources, dtype=int, device=self.device)
        self.plane_targets = wp.array(
            [i for i, t in enumerate(self.targets) if t in self.planes], dtype=int, device=self.device
        )
        self.finite_targets = wp.array(
            [i for i, t in enumerate(self.targets) if t not in self.planes], dtype=int, device=self.device
        )
        self.face_lo = wp.zeros(len(self.faces), dtype=wp.vec3, device=self.device)
        self.face_hi = wp.zeros_like(self.face_lo)
        self.target_lo = wp.zeros(len(self.targets), dtype=wp.vec3, device=self.device)
        self.target_hi = wp.zeros_like(self.target_lo)
        self.finite_lo = wp.zeros(len(self.finite_targets), dtype=wp.vec3, device=self.device)
        self.finite_hi = wp.zeros_like(self.finite_lo)
        self._keys = wp.empty(capacity, dtype=wp.uint64, device=self.device)
        self._scratch_keys = wp.empty_like(self._keys)
        self._indices = wp.empty(capacity, dtype=int, device=self.device)
        self._scratch_indices = wp.empty_like(self._indices)
        self.count = wp.zeros(1, dtype=int, device=self.device)
        self.dropped = wp.zeros(1, dtype=wp.int64, device=self.device)
        self.status = wp.zeros(1, dtype=int, device=self.device)
        if initial_bounds is not None:
            for lo, hi, out_lo, out_hi in (
                (*initial_bounds[:2], self.face_lo, self.face_hi),
                (*initial_bounds[2:], self.target_lo, self.target_hi),
            ):
                if len(lo) != len(out_lo) or len(hi) != len(out_hi):
                    raise ValueError("Invalid initial contact query bounds")
                wp.launch(_copy_bounds, len(lo), [lo, hi, out_lo, out_hi, self.status], device=self.device)
            if self.status.numpy()[0]:
                raise ValueError("Invalid initial contact query bounds")
        # Query faces in parallel against a fixed finite-target BVH. Plane
        # patches use their separate path; original global target ordinals stay stable.
        self.bvh = None
        if len(self.faces) and len(self.finite_targets):
            self._update_finite_bounds()
            self.bvh = wp.Bvh(self.finite_lo, self.finite_hi)
        self._bvh = self.bvh
        self._owned = {name: value for name, value in vars(self).items() if isinstance(value, wp.array)}
        self._result_keys = self._scratch_keys if (max(1, capacity) - 1).bit_length() % 2 else self._keys
        for a in (self.faces, self.targets, self.allowed):
            a.setflags(write=False)

    def buffers(self):
        return tuple(self._owned.values())

    @property
    def keys(self):
        return self._result_keys

    def validate_storage(self):
        """Check owner bindings even when a captured graph bypasses update."""
        if self.bvh is not self._bvh or any(getattr(self, name) is not a for name, a in self._owned.items()):
            raise ValueError("Contact query storage changed")

    def _update_finite_bounds(self):
        wp.launch(
            _finite_bounds,
            len(self.finite_targets),
            [self.finite_targets, self.target_lo, self.target_hi, self.finite_lo, self.finite_hi],
            device=self.device,
        )

    def update(self, face_lo, face_hi, target_lo, target_hi):
        """Refit, enumerate and sort on device, without host reads or capacity changes."""
        self.validate_storage()
        for a, size in (
            (face_lo, len(self.faces)),
            (face_hi, len(self.faces)),
            (target_lo, len(self.targets)),
            (target_hi, len(self.targets)),
        ):
            if (
                not isinstance(a, wp.array)
                or a.dtype not in (wp.vec3, wp.vec3d)
                or a.shape != (size,)
                or a.device != self.device
                or not a.is_contiguous
            ):
                raise ValueError("Invalid contact query bounds")
            if a.size and any(
                b.size and a.ptr < b.ptr + b.capacity and b.ptr < a.ptr + a.capacity for b in self.buffers()
            ):
                raise ValueError("Contact query bounds alias owned storage")
        self.count.zero_()
        self.dropped.zero_()
        self.status.zero_()
        wp.launch(
            _copy_bounds,
            len(self.faces),
            [face_lo, face_hi, self.face_lo, self.face_hi, self.status],
            device=self.device,
        )
        wp.launch(
            _copy_bounds,
            len(self.targets),
            [target_lo, target_hi, self.target_lo, self.target_hi, self.status],
            device=self.device,
        )
        if self.bvh is not None:
            self._update_finite_bounds()
            self.bvh.refit()
            wp.launch(
                _query_faces,
                len(self.faces),
                [
                    self.bvh.id,
                    self.face_sources,
                    self.face_lo,
                    self.face_hi,
                    self.finite_targets,
                    len(self.targets),
                    self.samples_per_patch,
                    self.offsets,
                    self.sources,
                    self._keys,
                    self._indices,
                    self.count,
                    self.dropped,
                    self.status,
                ],
                device=self.device,
            )
        if len(self.planes) and len(self.faces):
            wp.launch(
                _query_planes,
                (len(self.faces), len(self.planes)),
                [
                    self.face_sources,
                    self.plane_targets,
                    len(self.targets),
                    self.samples_per_patch,
                    self.offsets,
                    self.sources,
                    self._keys,
                    self._indices,
                    self.count,
                    self.dropped,
                    self.status,
                ],
                device=self.device,
            )
        keys, scratch = self._keys, self._scratch_keys
        indices, scratch_indices = self._indices, self._scratch_indices
        run = 1
        workers = min(self.capacity, _CONTACT_WORKERS)
        while run < self.capacity:
            wp.launch(
                _merge_contact_keys,
                workers,
                [keys, indices, scratch, scratch_indices, self.count, run, workers],
                device=self.device,
            )
            keys, scratch = scratch, keys
            indices, scratch_indices = scratch_indices, indices
            run *= 2

    def check(self, *, warn=True):
        """Report hard errors separately from the permitted capacity warning."""
        if self.status.numpy()[0] != 0:
            raise ValueError("Invalid contact query bounds")
        dropped = int(self.dropped.numpy()[0])
        if dropped and warn:
            warnings.warn(f"Contact query capacity dropped {dropped} candidates", RuntimeWarning, stacklevel=2)
