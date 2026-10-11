"""Reaction-wheel attitude control with a quaternion-error PID controller.

Rigid-body dynamics with three orthogonal reaction wheels:

    J w_dot = -w x (J w + h_w) + tau_c + tau_dist      (body)
    h_w_dot = -tau_c                                   (wheels absorb momentum)
    q_dot   = 0.5 * q (x) [0, w]                       (kinematics)

The controller drives the attitude-error quaternion to zero:

    tau_c = -Kp e - Ki int(e) dt - Kd w,  e = vector part of q_err (sign-fixed)

and respects wheel torque and momentum limits, with anti-windup on the
integral term (conditional integration + clamping). Gains come from a target natural frequency and damping
ratio: Kp = J wn^2, Kd = 2 zeta wn J.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


# --- Quaternion helpers (scalar-first, Hamilton convention) -------------------
def quat_mul(p, q):
    p0, pv = p[0], p[1:]
    q0, qv = q[0], q[1:]
    return np.concatenate(([p0 * q0 - pv @ qv], p0 * qv + q0 * pv + np.cross(pv, qv)))


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate(([math.cos(angle_rad / 2)], math.sin(angle_rad / 2) * axis))


def quat_from_euler_deg(roll, pitch, yaw):
    """3-2-1 (yaw-pitch-roll) Euler angles in degrees -> quaternion."""
    qz = quat_from_axis_angle([0, 0, 1], math.radians(yaw))
    qy = quat_from_axis_angle([0, 1, 0], math.radians(pitch))
    qx = quat_from_axis_angle([1, 0, 0], math.radians(roll))
    return quat_mul(quat_mul(qz, qy), qx)


def quat_to_euler_deg(q):
    q0, q1, q2, q3 = q
    roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q0 * q2 - q3 * q1))))
    yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
    return np.degrees([roll, pitch, yaw])


def quat_angle_deg(q):
    """Rotation angle represented by a quaternion [deg], in [0, 180]."""
    return math.degrees(2 * math.acos(min(1.0, abs(q[0]))))


# --- Configuration ------------------------------------------------------------
@dataclass
class Spacecraft:
    inertia: np.ndarray = field(default_factory=lambda: np.diag([2.0, 2.5, 1.8]))  # [kg m^2]
    wheel_max_torque: float = 0.2      # per wheel [N m]
    wheel_max_momentum: float = 1.0    # per wheel [N m s]


@dataclass
class PIDGains:
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    integral_limit: float = 0.05       # anti-windup clamp on int(e) [rad s]
    integral_zone: float = 0.035       # only integrate when |e| < ~2 deg (conditional integration)

    @classmethod
    def from_bandwidth(cls, inertia, wn=1.0, zeta=0.9, ki_ratio=0.1):
        """Second-order design: Kp = J wn^2, Kd = 2 zeta wn J, Ki = ki_ratio wn Kp."""
        J = np.diag(inertia)
        kp = J * wn ** 2
        kd = 2 * zeta * wn * J
        ki = ki_ratio * wn * kp
        return cls(kp, ki, kd)


@dataclass
class AttitudeResult:
    t: np.ndarray
    q: np.ndarray            # attitude quaternion (N x 4)
    w: np.ndarray            # body rates [rad/s] (N x 3)
    torque: np.ndarray       # commanded torque [N m] (N x 3)
    h_wheel: np.ndarray      # wheel momentum [N m s] (N x 3)
    error_deg: np.ndarray    # total pointing error [deg]
    q_target: np.ndarray

    def euler_error_deg(self):
        return np.array([quat_to_euler_deg(quat_mul(quat_conj(self.q_target), q)) for q in self.q])

    def settling_time(self, band_deg: float | None = None, band_fraction: float = 0.02):
        """First time after which the error stays inside the band.

        Default band: 2% of the initial error (the usual control definition).
        """
        band = band_deg if band_deg is not None else band_fraction * self.error_deg[0]
        outside = np.nonzero(self.error_deg > band)[0]
        if len(outside) == 0:
            return 0.0
        k = outside[-1] + 1
        return float(self.t[k]) if k < len(self.t) else math.inf

    def overshoot_deg(self):
        """Largest error after the first zero-crossing of any Euler-angle error."""
        e = self.euler_error_deg()
        worst = 0.0
        for ax in range(3):
            sgn = np.sign(e[0, ax])
            if sgn == 0:
                continue
            crossed = np.nonzero(np.sign(e[:, ax]) == -sgn)[0]
            if len(crossed):
                worst = max(worst, float(np.max(np.abs(e[crossed[0]:, ax]))))
        return worst

    def steady_state_error_deg(self, last_seconds: float = 5.0):
        mask = self.t >= self.t[-1] - last_seconds
        return float(np.mean(self.error_deg[mask]))


def simulate_attitude(q0, q_target, *, spacecraft: Spacecraft | None = None,
                      gains: PIDGains | None = None, w0=(0.0, 0.0, 0.0),
                      disturbance=(0.0, 0.0, 0.0), duration: float = 30.0,
                      dt: float = 0.01) -> AttitudeResult:
    """Simulate a reaction-wheel slew from q0 to q_target with RK4 integration."""
    sc = spacecraft or Spacecraft()
    J = np.asarray(sc.inertia, float)
    J_inv = np.linalg.inv(J)
    g = gains or PIDGains.from_bandwidth(J)
    tau_d = np.asarray(disturbance, float)
    qt = np.asarray(q_target, float) / np.linalg.norm(q_target)
    qt_conj = quat_conj(qt)

    def error_vec(q):
        qe = quat_mul(qt_conj, q)
        # shortest rotation: force a positive scalar part
        return (qe[1:] if qe[0] >= 0 else -qe[1:]) * 2.0   # ~ small-angle rotation vector

    def dynamics(q, w, h, tau_c):
        q_dot = 0.5 * quat_mul(q, np.concatenate(([0.0], w)))
        w_dot = J_inv @ (-np.cross(w, J @ w + h) + tau_c + tau_d)
        h_dot = -tau_c
        return q_dot, w_dot, h_dot

    n = int(round(duration / dt))
    t = np.arange(n + 1) * dt
    Q = np.zeros((n + 1, 4)); W = np.zeros((n + 1, 3)); H = np.zeros((n + 1, 3))
    TAU = np.zeros((n + 1, 3)); ERR = np.zeros(n + 1)
    q = np.asarray(q0, float) / np.linalg.norm(q0)
    w = np.asarray(w0, float)
    h = np.zeros(3)
    integ = np.zeros(3)

    for k in range(n + 1):
        e = error_vec(q)
        # PID with anti-windup: integrate only near the target, and clamp the state.
        if np.max(np.abs(e)) < g.integral_zone:
            integ = np.clip(integ + e * dt, -g.integral_limit, g.integral_limit)
        tau_c = -(g.kp * e + g.ki * integ + g.kd * w)
        tau_c = np.clip(tau_c, -sc.wheel_max_torque, sc.wheel_max_torque)
        # Wheel momentum saturation: a saturated wheel cannot absorb more momentum.
        sat = (np.abs(h) >= sc.wheel_max_momentum) & (np.sign(-tau_c) == np.sign(h))
        tau_c[sat] = 0.0

        Q[k], W[k], H[k], TAU[k] = q, w, h, tau_c
        ERR[k] = quat_angle_deg(quat_mul(qt_conj, q))
        if k == n:
            break
        # RK4 step with the control torque held constant over dt (zero-order hold).
        k1 = dynamics(q, w, h, tau_c)
        k2 = dynamics(q + dt / 2 * k1[0], w + dt / 2 * k1[1], h + dt / 2 * k1[2], tau_c)
        k3 = dynamics(q + dt / 2 * k2[0], w + dt / 2 * k2[1], h + dt / 2 * k2[2], tau_c)
        k4 = dynamics(q + dt * k3[0], w + dt * k3[1], h + dt * k3[2], tau_c)
        q = q + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w = w + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        h = h + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        q /= np.linalg.norm(q)

    return AttitudeResult(t, Q, W, TAU, H, ERR, qt)
