"""Full quadrotor model for simulation (same model and limits as quad_time_optimal.py / rpg_time_optimal quad.yaml):
rigid body, 4 rotor thrusts clipped to [T_min, T_max] (inputs), RK4 step; optional constant external acceleration
ext_acc (wind). Body-rate limits 15 rad/s (xy) / 0.3 rad/s (z) are left to the controllers.
"""
from pathlib import Path

import numpy as np
import yaml

G = 9.81
EZ = np.array([0.0, 0.0, 1.0])
QUAD = Path(__file__).resolve().parents[1] / "third_party/rpg_time_optimal/quads/quad.yaml"
STARLING_2_MAX = Path(__file__).resolve().parents[1] / "models/starling_2_max.yaml"


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_from_z(z):
    z = z / np.linalg.norm(z)
    ax = np.cross(EZ, z)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0])
    ang = np.arctan2(s, z[2])
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * ax / s])


class Quad:
    def __init__(self, path=None):
        """path: model yaml (default: rpg quad.yaml). With a 'rotors' list ([x, y, yaw torque / thrust] per rotor,
        body FLU) the mixer comes from it; otherwise the symmetric X with arm_length and torque_coeff."""
        q = yaml.safe_load((QUAD if path is None else Path(path)).read_text())
        self.m, l, c = q["mass"], q["arm_length"], q["torque_coeff"]
        self.J = np.array(q["inertia"], float)
        self.Jinv = np.linalg.inv(self.J)
        self.Tmax = q["TWR_max"] * self.m * G / 4
        self.Tmin = q["thrust_min"]
        self.wxy, self.wz = q["omega_max_xy"], q["omega_max_z"]
        if "rotors" in q:
            r = np.array(q["rotors"], float)
            self.Mx = np.vstack([np.ones(len(r)), r[:, 1], -r[:, 0], r[:, 2]])
        else:
            self.Mx = np.array([[1, 1, 1, 1], [l, -l, -l, l], [-l, -l, l, l], [c, -c, c, -c]], float)
        self.Mxinv = np.linalg.inv(self.Mx)
        self.A = 4 * self.Tmax / self.m                  # max thrust acceleration

    def deriv(self, x, u):
        v, q, w = x[3:6], x[6:10], x[10:13]
        F, tx, ty, tz = self.Mx @ u
        dv = rot(q) @ np.array([0, 0, F / self.m]) - G * EZ + getattr(self, "ext_acc", 0.0)   # ext_acc: wind etc.
        dq = 0.5 * quat_mul(q, np.concatenate([[0], w]))
        dw = self.Jinv @ (np.array([tx, ty, tz]) - np.cross(w, self.J @ w))
        return np.concatenate([v, dv, dq, dw])

    def step(self, x, u, dt):
        k1 = self.deriv(x, u)
        k2 = self.deriv(x + dt / 2 * k1, u)
        k3 = self.deriv(x + dt / 2 * k2, u)
        k4 = self.deriv(x + dt * k3, u)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[6:10] /= np.linalg.norm(x[6:10])
        return x
