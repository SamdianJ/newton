# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ancestor motion bounds and a separate native FK roundoff allowance."""

from typing import Any

import numpy as np
import warp as wp

from ...sim import JointType
from .articulation import _forward


@wp.func_native("""
    // All callers supply nonnegative binary64 values. Preserve infinity/NaN.
    wp::uint64 bits;
    memcpy(&bits, &x, sizeof(bits));
    if (x == 0.0) bits = 1;
    else if (bits < 0x7ff0000000000000ULL) ++bits;
    wp::float64 result;
    memcpy(&result, &bits, sizeof(result));
    return result;
""")
def _next_upper(x: wp.float64) -> wp.float64: ...


@wp.func
def _sum_upper(a: wp.float64, b: wp.float64):
    if a == wp.float64(0.0):
        return b
    if b == wp.float64(0.0):
        return a
    return _next_upper(a + b)


@wp.func
def _product_upper(a: wp.float64, b: wp.float64):
    if a == wp.float64(0.0) or b == wp.float64(0.0):
        return wp.float64(0.0)
    return _next_upper(a * b)


@wp.kernel(enable_backward=False)
def _ancestor_displacement(
    incoming: wp.array[int],
    parents: wp.array[int],
    types: wp.array[int],
    starts: wp.array[int],
    parent_length: wp.array[wp.float64],
    child_length: wp.array[wp.float64],
    axis_length: wp.array[wp.float64],
    radii: wp.array[wp.float64],
    q0: wp.array[Any],
    q1: wp.array[Any],
    joint_floating: wp.array[int],
    com_length: wp.array[wp.float64],
    root_com0: wp.array[Any],
    root_rotation0: wp.array[Any],
    root_com1: wp.array[Any],
    root_rotation1: wp.array[Any],
    displacement: wp.array[wp.float64],
    numerical: wp.array[wp.float64],
    unit_roundoff: wp.float64,
    tiny: wp.float64,
    status: wp.array[int],
):
    body = wp.tid()
    cursor = body
    tail = radii[body]
    distance = wp.float64(0.0)
    operations, angle = wp.float64(0.0), wp.float64(0.0)
    valid = wp.isfinite(tail) and tail >= wp.float64(0.0)
    # The bounded loop also rejects a corrupted parent cycle instead of hanging.
    for _depth in range(types.shape[0]):
        if cursor == -1 or not valid:
            break
        if cursor < 0 or cursor >= incoming.shape[0]:
            valid = False
            break
        joint = incoming[cursor]
        if joint < 0 or joint >= types.shape[0]:
            valid = False
            break
        a, b, axis = parent_length[joint], child_length[joint], axis_length[joint]
        if not wp.isfinite(a + b + axis) or a < wp.float64(0.0) or b < wp.float64(0.0) or axis < wp.float64(0.0):
            valid = False
            break
        radius = _sum_upper(tail, b)
        # <=512 primitive rounding/trig allowances per FK joint, with a
        # factor-two propagation allowance through quaternion normalization.
        operations = _sum_upper(_product_upper(wp.float64(2.0), operations), wp.float64(512.0))
        kind = types[joint]
        if kind == JointType.FREE:
            # Floating root: points move by the COM step plus a rotation about the COM, whose
            # chord is <= min(theta, 2) * rho with rho bounding the points about the root COM.
            f = joint_floating[joint]
            if f < 0 or f >= root_com0.shape[0]:
                valid = False
                break
            rho = _sum_upper(tail, com_length[joint])
            c0, c1 = root_com0[f], root_com1[f]
            r0, r1 = root_rotation0[f], root_rotation1[f]
            step2, reach2 = wp.float64(0.0), wp.float64(0.0)
            for k in range(3):
                d = wp.float64(c1[k]) - wp.float64(c0[k])
                step2 = _sum_upper(step2, _product_upper(d, d))
                reach2 = _sum_upper(
                    reach2,
                    wp.max(
                        _product_upper(wp.float64(c0[k]), wp.float64(c0[k])),
                        _product_upper(wp.float64(c1[k]), wp.float64(c1[k])),
                    ),
                )
            # Relative rotation r1 r0^-1 of unit quaternions, angle 2 atan2(|v|, |w|).
            w0 = wp.float64(r0[3])
            w1 = wp.float64(r1[3])
            v0 = wp.vec3d(wp.float64(r0[0]), wp.float64(r0[1]), wp.float64(r0[2]))
            v1 = wp.vec3d(wp.float64(r1[0]), wp.float64(r1[1]), wp.float64(r1[2]))
            rw = w1 * w0 + wp.dot(v1, v0)
            rv = w0 * v1 - w1 * v0 - wp.cross(v1, v0)
            # A few ulps of the FP64 composition and atan2, rounded outward.
            theta = _sum_upper(
                _product_upper(wp.float64(2.0), wp.atan2(wp.length(rv), wp.abs(rw))), wp.float64(64.0) * unit_roundoff
            )
            if not wp.isfinite(step2 + reach2 + theta) or not wp.isfinite(rho):
                valid = False
                break
            distance = _sum_upper(
                distance, _sum_upper(_next_upper(wp.sqrt(step2)), _product_upper(wp.min(theta, wp.float64(2.0)), rho))
            )
            # Native positions carry the absolute COM magnitude; any root rotation is <= pi.
            tail = _sum_upper(_next_upper(wp.sqrt(reach2)), rho)
            angle = _sum_upper(angle, wp.float64(3.141592653589794))
            cursor = parents[joint]
            continue
        tail = _sum_upper(a, radius)
        if kind == JointType.REVOLUTE or kind == JointType.PRISMATIC:
            index = starts[joint]
            if index < 0 or index >= q0.shape[0]:
                valid = False
                break
            start, end = wp.float64(q0[index]), wp.float64(q1[index])
            if not wp.isfinite(start) or not wp.isfinite(end):
                valid = False
                break
            delta = wp.float64(0.0)
            if start != end:
                delta = _next_upper(wp.abs(end - start))
            if kind == JointType.REVOLUTE:
                angle = _sum_upper(angle, wp.max(wp.abs(start), wp.abs(end)))
                # ||R(alpha*dq)-I|| <= min(|dq|,2); never wrap whole revolutions.
                distance = _sum_upper(distance, _product_upper(wp.min(delta, wp.float64(2.0)), radius))
            else:
                distance = _sum_upper(distance, _product_upper(delta, axis))
                tail = _sum_upper(tail, _product_upper(wp.max(wp.abs(start), wp.abs(end)), axis))
        elif kind != JointType.FIXED:
            valid = False
            break
        cursor = parents[joint]
    nu = _product_upper(operations, unit_roundoff)
    error = wp.float64(0.0)
    if nu < wp.float64(0.125):
        gamma = _next_upper(nu / (wp.float64(1.0) - nu))
        # Both the stored start FK and an intermediate FK can differ from
        # the real path; angular coordinate rounding carries a length scale.
        error = _product_upper(
            wp.float64(2.0), _product_upper(gamma, _product_upper(tail, _sum_upper(wp.float64(1.0), angle)))
        )
        error = _sum_upper(error, _product_upper(operations, tiny))
    if (
        cursor != -1
        or not valid
        or not wp.isfinite(distance)
        or not wp.isfinite(tail)
        or not wp.isfinite(error)
        or not wp.isfinite(angle)
        or nu >= wp.float64(0.125)
    ):
        wp.atomic_or(status, 0, 1)
        distance = wp.float64(0.0)
        error = wp.float64(0.0)
    displacement[body] = distance
    numerical[body] = error


@wp.kernel(enable_backward=False)
def _invalidate_displacement(
    displacement: wp.array[wp.float64], numerical: wp.array[wp.float64], status: wp.array[int]
):
    if status[0] != 0:
        displacement[wp.tid()] = wp.float64(0.0)
        numerical[wp.tid()] = wp.float64(0.0)


def _length_upper(vectors):
    """Round every nonnegative norm operation outward, including underflow."""
    vectors = np.asarray(vectors, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore"):
        squares = np.where(vectors == 0, 0.0, np.nextafter(vectors * vectors, np.inf))
        total = np.zeros(len(vectors))
        for axis in range(3):
            total = np.where(squares[:, axis] == 0, total, np.nextafter(total + squares[:, axis], np.inf))
        lengths = np.where(total == 0, 0.0, np.nextafter(np.sqrt(total), np.inf))
    if not np.isfinite(lengths).all():
        raise ValueError("Articulation sweep length exceeds finite float64 range")
    return lengths


class _ArticulationSweep:
    """Bound a body's local radius along q0→q1, with separate native FK allowance.

    Radii are about body-frame origins, including all local offsets. This scalar
    component does not select contacts, assert law support, perform CCD, or cover
    an arbitrary trial outside the defined linear joint path.
    """

    def __init__(self, kinematics, radii):
        kinematics.validate_identity()
        if wp.get_module_options(_forward.module).get("fast_math", False):
            raise ValueError("Native articulation sweep requires fast_math disabled")
        value = np.asarray(radii)
        if (
            value.dtype.kind not in "fiu"
            or value.shape != (kinematics.bodies,)
            or not np.isfinite(value).all()
            or np.any(value < 0)
        ):
            raise ValueError("Articulation sweep requires finite nonnegative body radii")
        self.kinematics, self.device = kinematics, kinematics.device
        incoming = np.empty(kinematics.bodies, dtype=np.int32)
        incoming[kinematics.children.numpy()] = np.arange(kinematics.joints)
        axes = kinematics.axes.numpy().astype(float)
        types, starts = kinematics.types.numpy(), kinematics.starts.numpy()
        lengths = np.zeros(kinematics.joints)
        moving = np.flatnonzero(types != int(JointType.FIXED))
        lengths[moving] = _length_upper(axes[starts[moving]])
        self.incoming = wp.array(incoming, dtype=int, device=self.device)
        self.parent_length = wp.array(_length_upper(kinematics.xp.numpy()[:, :3]), dtype=wp.float64, device=self.device)
        self.child_length = wp.array(_length_upper(kinematics.xc.numpy()[:, :3]), dtype=wp.float64, device=self.device)
        self.axis_length = wp.array(lengths, dtype=wp.float64, device=self.device)
        joint_floating = np.full(kinematics.joints, -1, dtype=np.int32)
        root_joints = kinematics.root_joints.numpy()
        joint_floating[root_joints] = np.arange(len(root_joints))
        com_length = np.zeros(kinematics.joints)
        children = kinematics.children.numpy()
        com_length[root_joints] = _length_upper(kinematics.local_com.numpy()[children[root_joints]])
        self.joint_floating = wp.array(joint_floating, dtype=int, device=self.device)
        self.com_length = wp.array(com_length, dtype=wp.float64, device=self.device)
        self.radii = wp.array(value.astype(np.float64), dtype=wp.float64, device=self.device)
        self.displacement = wp.zeros(kinematics.bodies, dtype=wp.float64, device=self.device)
        self.numerical = wp.zeros_like(self.displacement)
        self.status = wp.zeros(1, dtype=int, device=self.device)
        self._owned = {
            name: getattr(self, name)
            for name in (
                "incoming",
                "parent_length",
                "child_length",
                "axis_length",
                "joint_floating",
                "com_length",
                "radii",
                "displacement",
                "numerical",
                "status",
            )
        }
        self._borrowed = {name: getattr(kinematics, name) for name in ("parents", "types", "starts")}
        self.payload_bytes = 28 * kinematics.bodies + 36 * kinematics.joints + 4

    def buffers(self):
        return tuple(self._owned.values())

    def validate_storage(self):
        """Check native policy and bindings before captured query replay."""
        k = self.kinematics
        if wp.get_module_options(_forward.module).get("fast_math", False):
            raise ValueError("Native articulation sweep requires fast_math disabled")
        for source, fields in ((self, self._owned), (k, self._borrowed)):
            if any(getattr(source, name) is not original for name, original in fields.items()):
                raise ValueError("Articulation sweep storage changed")

    def update(self, q0, q1, roots0=None, roots1=None):
        """Overwrite fixed scalar scratch; callers must reject nonzero status.

        ``roots0``/``roots1`` are the native floating-root COM and rotation arrays at both ends.
        """
        self.validate_storage()
        k = self.kinematics
        if (roots0 is None or roots1 is None) != (not k.floating_count):
            raise ValueError("Articulation sweep requires both native floating-root poses")
        if roots0 is None:
            roots0 = roots1 = k._empty_roots
        if any(
            (q.shape, q.dtype, q.device) != ((k.count,), k.dtype, self.device) or not q.is_contiguous for q in (q0, q1)
        ):
            raise ValueError("Articulation sweep requires native coordinates on its device")
        if any(
            q.size and a.size and q.ptr < a.ptr + a.capacity and a.ptr < q.ptr + q.capacity
            for q in (q0, q1)
            for a in self.buffers()
        ):
            raise ValueError("Articulation sweep input aliases owned output")
        self.status.zero_()
        wp.launch(
            _ancestor_displacement,
            k.bodies,
            [
                self.incoming,
                k.parents,
                k.types,
                k.starts,
                self.parent_length,
                self.child_length,
                self.axis_length,
                self.radii,
                q0,
                q1,
                self.joint_floating,
                self.com_length,
                *roots0,
                *roots1,
                self.displacement,
                self.numerical,
                wp.float64(np.finfo(k.numpy_dtype).eps / 2),
                wp.float64(np.finfo(k.numpy_dtype).tiny),
                self.status,
            ],
            device=self.device,
        )
        wp.launch(
            _invalidate_displacement, k.bodies, [self.displacement, self.numerical, self.status], device=self.device
        )
