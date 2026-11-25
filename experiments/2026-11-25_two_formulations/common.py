"""Shared by w_version.py and q_version.py (user, 2026-11-25): path, quad, limits, physical check.

Path: CPC lap of track k, periodic quintic B-spline p(u), u in [0, 1) ~ s / L (fitted to arc length).
Heading left free: body yaw rate 0, limits hold for every heading.
f = a + g e3 (thrust per mass; direction = body z, |f| = c = collective thrust / mass), j = df/dt, s = dj/dt.
    |w| = |f x j| / (f.f)                       (|f x j| = c |j_perp|)
    w'  = N / (f.f)^2,  N = (f x s)(f.f) - 2 (f x j)(f.j),  torque tau = J_xy w'  (heading free, J_x = J_y)
    rotor i = m c / 4 +- (tau_x +- tau_y) / (4 l),  worst heading: m c / 4 +- sqrt(2) |tau| / (4 l)
Limits (all >= 0), in physical units (user, 2026-11-25: "直接看物理量"; divides by f.f, which is 0 only at zero thrust):
    1. 15 - |w|                                        [rad/s]
    2. 4 T_max / m - c                                 [m/s^2]
    3. T_max - m c / 4 - sqrt(2) J |w'| / (4 l)       [N]   highest rotor, worst heading
    4. m c / 4 - sqrt(2) J |w'| / (4 l)               [N]   lowest rotor (T_min = 0), worst heading
Each version passes |f x j| and N built from its own variables.
"""
import os

import casadi as ca
import numpy as np
import yaml
from pathlib import Path
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[2]
Q = yaml.safe_load((ROOT / "third_party/rpg_time_optimal/quads/quad.yaml").read_text())
MASS, ARM = Q["mass"], Q["arm_length"]
JXY = float(np.array(Q["inertia"], float)[0, 0])
G = 9.81
TMAX = Q["TWR_max"] * MASS * G / 4
assert Q["thrust_min"] == 0.0
WMAX = Q["omega_max_xy"]
DEG = 4                       # speed spline degree (s continuous needs the 3rd derivative continuous)
HK = 1.0                      # speed knots every HK m
SPS = 4                       # constraint midpoints per path knot interval (plus the knots)
NAMES = ["|w| <= 15", "c <= 4 Tmax/m", "rotors <= Tmax", "rotors >= 0"]


def prows(u, n, deg, d):
    """Periodic uniform B-spline, n control points: d-th derivative rows at u."""
    card = BSpline.basis_element(np.arange(deg + 2), extrapolate=False)
    fn = card.derivative(d) if d else card
    x = np.asarray(u) * n
    out = np.zeros((len(x), n))
    i0 = np.floor(x).astype(int)
    for k in range(deg + 1):
        jj = i0 - k
        out[np.arange(len(x)), jj % n] += np.nan_to_num(fn(x - jj)) * n ** d
    return out


def cpc_lap(k):
    d = np.load(ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")
    x, T, N = d["x"], d["T"], int(d["N"])
    dts = np.repeat(T / N, N)                 # T: time of each gate-to-gate segment, N nodes per segment
    P = []
    for i in range(len(x)):
        h = dts[i]
        p0, v0, p1, v1 = x[i, :3], x[i, 3:6], x[(i + 1) % len(x), :3], x[(i + 1) % len(x), 3:6]
        for s in np.arange(20) / 20:
            P.append((2 * s ** 3 - 3 * s ** 2 + 1) * p0 + (s ** 3 - 2 * s ** 2 + s) * h * v0
                     + (-2 * s ** 3 + 3 * s ** 2) * p1 + (s ** 3 - s ** 2) * h * v1)
    return np.array(P), float(T.sum())


class Path_:
    def __init__(self, k, hp=None):
        hp = float(os.environ.get("HP", 0.5)) if hp is None else hp
        P, self.lap_cpc = cpc_lap(k)
        self.cut = os.environ.get("CUT", "0") == "1"
        self.i0 = 0
        if self.cut:                               # start the lap at CPC's slowest node (the tip of the V turn)
            x = np.load(ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")["x"]
            self.i0 = int(np.argmin(np.linalg.norm(x[:, 3:6], axis=1)))
            P = np.roll(P, -20 * self.i0, axis=0)
        self.P = P
        sl = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        s_ = np.r_[0, np.cumsum(sl)[:-1]]
        self.L = sl.sum()
        self.nref = int(round(self.L / hp))        # path knots every hp m
        if self.cut:
            self._fit_cut(s_ / self.L, P, int(os.environ.get("CUTC", 3)))
        else:
            self.Cp = np.linalg.lstsq(prows(s_ / self.L, self.nref, 5, 0), P, rcond=None)[0]
        self.n = int(round(self.L / HK))          # speed spline control points
        self.M0 = SPS * max(self.nref, self.n)
        self.us = self.grid(1)                    # constraint samples
        self.M = len(self.us)

    def grid(self, fac):
        """fac * M0 midpoints plus every path knot and speed knot (the snap has a corner at each knot: the path's 4th
        derivative and the speed spline's 3rd derivative are piecewise linear, so the torque peaks there)."""
        mid = (np.arange(fac * self.M0) + 0.5) / (fac * self.M0)
        return np.unique(np.r_[mid, np.arange(self.nref) / self.nref, np.arange(self.n) / self.n])

    def _fit_cut(self, u, P, cont):
        """CUT=1 (user, 2026-11-25: "直接在尖點用兩條不同的 spline"): the lap is one open (clamped) quintic B-spline that
        starts and ends at the tip of the V turn, so the tip is a junction instead of the middle of a uniform span.
        At the junction p and its u-derivatives 1..cont are equal on both sides (cont = 3: p, velocity, acceleration
        and jerk continuous; the 4th derivative (torque) may jump). No assumption on the speed there.
        Least squares with these equalities (KKT system)."""
        self.tk = np.r_[np.zeros(5), np.linspace(0, 1, self.nref + 1), np.ones(5)]
        nc = len(self.tk) - 6
        A = self._rows(u, 0)
        E = np.vstack([self._rows(np.array([0.0]), d) - self._rows(np.array([1.0]), d) for d in range(cont + 1)])
        K = np.block([[A.T @ A, E.T], [E, np.zeros((len(E), len(E)))]])
        self.Cp = np.linalg.solve(K, np.vstack([A.T @ P, np.zeros((len(E), 3))]))[:nc]

    def _rows(self, u, d):
        nc = len(self.tk) - 6
        b = BSpline(self.tk, np.eye(nc), 5)
        b = b.derivative(d) if d else b
        return b(np.clip(np.asarray(u, float), 0, 1))

    def prow(self, u, d):
        return self._rows(u, d) if self.cut else prows(u, self.nref, 5, d)

    def at(self, u):
        """p_u .. p_uuuu (12 x len(u)) and speed-spline rows (0..3 derivatives)."""
        Pd = np.vstack([(self.prow(u, d) @ self.Cp).T for d in range(1, 5)])
        B = [prows(u, self.n, DEG, d) for d in range(4)]
        return Pd, B


def rows(f, Anorm, N):
    ff = ca.dot(f, f)
    c = ca.sqrt(ff)
    w = Anorm / ff
    tq = np.sqrt(2) * JXY * ca.norm_2(N) / ff ** 2 / (4 * ARM)       # worst-heading rotor share of the torque [N]
    return ca.vertcat(WMAX - w, 4 * TMAX / MASS - c, TMAX - MASS * c / 4 - tq, MASS * c / 4 - tq)


def physical(f, j, s):
    """Check only (divides): |w| [rad/s], worst-heading torque needed / allowed, c [m/s^2]. Arrays 3 x M."""
    ff = (f * f).sum(0)
    c = np.sqrt(ff)
    A = np.cross(f.T, j.T).T
    N = np.cross(f.T, s.T).T * ff - 2 * A * (f * j).sum(0)
    w = np.linalg.norm(A, axis=0) / ff
    tn = np.sqrt(2) * JXY * np.linalg.norm(N, axis=0) / ff ** 2
    ta = 4 * ARM * np.minimum(TMAX - MASS * c / 4, MASS * c / 4)
    return w, tn / ta, c


def report(lab, w, tr, c):
    return (f"{lab}: |w| max {w.max():.2f} rad/s (15) | torque needed/allowed max {tr.max():.3f} (1) | "
            f"collective [{c.min():.2f}, {c.max():.2f}] (max {4 * TMAX / MASS:.2f}) m/s^2")


def ipopt(x, fobj, g, x0, lbx):
    S = ca.nlpsol("s", "ipopt", {"x": x, "f": fobj, "g": g},
                  {"print_time": False, "ipopt": {"print_level": int(__import__("os").environ.get("PL", 0)), "max_iter": int(__import__("os").environ.get("MAXIT", 3000)), "print_frequency_iter": 25}})
    import time
    t0 = time.perf_counter()
    sol = S(x0=x0, lbx=lbx, lbg=0, ubg=np.inf)
    st = S.stats()
    return np.array(sol["x"]).ravel(), st["return_status"], st["iter_count"], time.perf_counter() - t0
