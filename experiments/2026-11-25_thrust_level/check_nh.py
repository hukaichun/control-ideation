"""Checks of limits.LIM_NH (physical units, no squaring, no scaling) before optimising (user, 2026-11-25).
1. CPC laps: |w| and the torque from f, j, s against CPC's own body rates and rotor torques.
2. Random points: each row has the sign of the physical margin, and row / (f.f)^k equals that margin exactly.
3. Gradient: finite, matches central differences; what happens where f x j = 0 or N = 0 (hover, straight line).
    uv run --with casadi --with pyyaml --with numpy python check_nh.py
"""
import casadi as ca
import numpy as np

import limits as LM
from check_limits import quat_R


def cpc(k):
    d = np.load(LM.ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")
    x, u = d["x"], d["u"]
    J, e3 = LM.JD, np.array([0, 0, 1.0])
    ew, et = [], []
    for i in range(len(x)):
        R = quat_R(x[i, 6:10]); w = x[i, 10:13]
        tau = LM.MIX[1:] @ u[i]
        wd = np.linalg.solve(J, tau - np.cross(w, J @ w))
        c = u[i].sum() / LM.MASS
        f = c * R[:, 2]; j = c * R @ np.cross(w, e3)
        s = c * R @ (np.cross(wd, e3) + np.cross(w, np.cross(w, e3)))
        wn, tn, _, _ = (float(v) for v in LM.CHK_NH(np.r_[f, j, s]))
        ew.append(abs(wn - np.linalg.norm(w[:2])))
        et.append(abs(tn - np.sqrt(2) * J[0, 0] * np.linalg.norm(wd[:2])))
    print(f"[{LM.FORM}] 1. CPC track {k}: |w_xy| error max {max(ew):.1e} rad/s, sqrt2 J |w'_xy| error max {max(et):.1e} N m")


def signs():
    rng = np.random.default_rng(0)
    bad, rel = 0, 0.0
    for _ in range(20000):
        c = 10 ** rng.uniform(-2, 1.6)
        f = c * (lambda v: v / np.linalg.norm(v))(rng.normal(size=3))
        inp = np.r_[f, rng.normal(0, 50, 3), rng.normal(0, 3000, 3)]
        g = np.array(LM.LIM_NH(inp)).ravel()
        wn, tn, ta, _ = (float(v) for v in LM.CHK_NH(inp))
        ff = c * c
        up = 4 * LM.ARM * (LM.TMAX - LM.MASS * c / 4)
        lo = 4 * LM.ARM * (LM.MASS * c / 4 - LM.TMIN)
        phys = np.array([(LM.WXY - wn) * ff, (up - tn) * ff ** 2, (lo - tn) * ff ** 2])
        if LM.FORM == "norm":
            bad += int(np.any(np.sign(g) != np.sign(phys)))
            rel = max(rel, np.max(np.abs(g - phys) / np.maximum(np.abs(phys), 1e-300)))
        else:   # squared rows: (rate), (up >= 0 and upper), (lo >= 0 and lower) must agree with the physical signs
            ok = [g[0] >= 0, g[1] >= 0 and g[2] >= 0, g[3] >= 0 and g[4] >= 0]
            bad += int(any(o != (p >= 0) for o, p in zip(ok, phys)))
            phs = np.array([ff ** 2 * (LM.WXY ** 2 - wn ** 2), ff ** 4 * (up ** 2 - tn ** 2), ff ** 4 * (lo ** 2 - tn ** 2)])
            rel = max(rel, np.max(np.abs(g[[0, 2, 4]] - phs) / np.maximum(np.abs(phs), 1e-300)))
    print(f"2. 20000 random points (c in [0.01, 40]): sign mismatches {bad}, max relative difference to the physical "
          f"margin (or squared margins) x (f.f)^k {rel:.1e}")


def grads():
    x = ca.SX.sym("x", 9)
    Jf = ca.Function("J", [x], [ca.jacobian(LM.LIM_NH(x), x)])
    rng = np.random.default_rng(1)
    worst = 0
    for _ in range(20):
        x0 = np.r_[rng.normal(0, 5, 3) + [0, 0, 10], rng.normal(0, 30, 3), rng.normal(0, 300, 3)]
        Ja = np.array(Jf(x0)); Jn = np.zeros_like(Ja)
        for q in range(9):
            h = 1e-6 * max(1, abs(x0[q])); xp, xm = x0.copy(), x0.copy(); xp[q] += h; xm[q] -= h
            Jn[:, q] = (np.array(LM.LIM_NH(xp)).ravel() - np.array(LM.LIM_NH(xm)).ravel()) / (2 * h)
        worst = max(worst, np.abs(Ja - Jn).max() / np.abs(Ja).max())
    print(f"3. gradient vs central differences, 20 random points: max error {worst:.1e}")
    for nm, x0 in {"hover (f x j = 0, N = 0)": np.r_[0, 0, 9.81, 0, 0, 0, 0, 0, 0],
                   "straight climb, jerk along f": np.r_[0, 0, 9.81, 0, 0, 5, 0, 0, 3]}.items():
        print(f"   {nm}: limits {np.round(np.array(LM.LIM_NH(x0)).ravel(), 3)}, gradient finite "
              f"{bool(np.isfinite(np.array(Jf(x0))).all())}")


if __name__ == "__main__":
    for k in (0, 1, 5):
        cpc(k)
    signs()
    grads()
