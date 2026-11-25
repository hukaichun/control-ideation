"""Checks of limits.py before any optimisation (user, 2026-11-25).

1. CPC trajectory (logs/cpc_traj_tol0.24_k.npz: states and rotor thrusts at every node): from the state and the
   rotor thrusts build f, j, s exactly (rotors constant on a node interval: c' = 0 there) and the heading psi with its
   derivatives (h horizontal, perpendicular to body y); the division-free formulas must give back CPC's rotor thrusts
   and body rates.
2. Zero collective thrust: values finite (no NaN / inf), limits satisfied when the trajectory is physically fine
   (vertical, free fall), violated when it asks to turn the thrust without thrust.
3. Gradient of the limits vs central differences at random points.

    cd experiments/2026-11-25_thrust_level
    uv run --with casadi --with pyyaml --with numpy python check_limits.py
"""
import casadi as ca
import numpy as np

import limits as LM

ROOT = LM.ROOT


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# heading angle and its time derivatives from body y and its derivatives (h = +-(y x e3)/|y x e3|, x.h > 0)
_y, _yd, _ydd = (ca.SX.sym(nm, 3) for nm in ("y", "yd", "ydd"))
_sg = ca.SX.sym("sg")
_k = _sg * ca.cross(_y, ca.DM([0, 0, 1]))
_psi = ca.atan2(_k[1], _k[0])
_psid = ca.jtimes(_psi, _y, _yd)
_psidd = ca.jtimes(_psid, ca.vertcat(_y, _yd), ca.vertcat(_yd, _ydd))
PSI = ca.Function("psi", [_y, _yd, _ydd, _sg], [ca.vertcat(_psi, _psid, _psidd)])


def check_cpc(k):
    d = np.load(ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")
    x, u = d["x"], d["u"]
    J = LM.JD
    e2, e3 = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    rows = []
    for i in range(len(x)):
        R = quat_R(x[i, 6:10])
        w = x[i, 10:13]
        tau = LM.MIX[1:] @ u[i]
        wd = np.linalg.solve(J, tau - np.cross(w, J @ w))
        c = u[i].sum() / LM.MASS
        f = c * R[:, 2]
        j = c * R @ np.cross(w, e3)
        s = c * R @ (np.cross(wd, e3) + np.cross(w, np.cross(w, e3)))
        y = R[:, 1]
        yd = R @ np.cross(w, e2)
        ydd = R @ (np.cross(wd, e2) + np.cross(w, np.cross(w, e2)))
        sg = 1.0 if R[:, 0] @ np.cross(y, e3) > 0 else -1.0
        ps = np.array(PSI(y, yd, ydd, sg)).ravel()
        inp = np.r_[f, j, s, ps]
        uw = np.array(LM.UW(inp)).ravel()
        g = np.array(LM.LIM(inp)).ravel()
        rows.append((np.abs(uw[:4] - u[i]).max(), np.abs(uw[4:] - w).max(), c, g.min(), np.isfinite(g).all()))
    r = np.array(rows)
    print(f"1. CPC track {k} ({len(x)} nodes): rotor thrust error max {r[:, 0].max():.1e} N, body rate error max "
          f"{r[:, 1].max():.1e} rad/s | collective thrust range [{r[:, 2].min():.2f}, {r[:, 2].max():.2f}] m/s^2 | "
          f"all finite {bool(r[:, 4].all())} | min scaled limit value {r[:, 3].min():.1e} (CPC sits on its limits: ~0)")


def check_zero():
    print("2. zero / tiny collective thrust:")
    cases = {
        "free fall, vertical, nothing turning (fine)": np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0],
        "c = 1e-6, vertical, no jerk (fine)": np.r_[0, 0, 1e-6, 0, 0, 0, 0, 0, 0, 0.3, 0, 0],
        "c = 1e-3, sideways jerk 0.1 (needs w = 100 rad/s: violated)": np.r_[0, 0, 1e-3, 0.1, 0, 0, 0, 0, 0, 0.3, 0, 0],
        "c = 1, sideways jerk 0.1 (w = 0.1 rad/s: fine)": np.r_[0, 0, 1, 0.1, 0, 0, 0, 0, 0, 0.3, 0, 0],
        "hover c = g (fine)": np.r_[0, 0, 9.81, 0, 0, 0, 0, 0, 0, 0.3, 0, 0],
    }
    for nm, inp in cases.items():
        g = np.array(LM.LIM(inp)).ravel()
        uw = np.array(LM.UW(inp)).ravel()
        bad = [LM.NAMES[i] for i in np.flatnonzero(g < -1e-15)]
        print(f"   {nm}: finite {np.isfinite(g).all()}, min limit value {g.min():+.2e}, violated: {bad or 'none'} | "
              f"(check only, divides) rotors {np.round(uw[:4], 3)}, w {np.round(uw[4:], 3)}")


def check_grad():
    rng = np.random.default_rng(1)
    inp = ca.SX.sym("i", 12)
    Jf = ca.Function("J", [inp], [ca.jacobian(LM.LIM(inp), inp)])
    worst = 0.0
    for _ in range(20):
        x0 = np.r_[rng.normal(0, 5, 3) + [0, 0, 10], rng.normal(0, 30, 3), rng.normal(0, 300, 3),
                   rng.uniform(-3, 3), rng.normal(0, 0.3), rng.normal(0, 1)]
        Ja = np.array(Jf(x0))
        Jn = np.zeros_like(Ja)
        for q in range(12):
            h = 1e-6 * max(1.0, abs(x0[q]))
            xp, xm = x0.copy(), x0.copy()
            xp[q] += h
            xm[q] -= h
            Jn[:, q] = (np.array(LM.LIM(xp)).ravel() - np.array(LM.LIM(xm)).ravel()) / (2 * h)
        worst = max(worst, np.abs(Ja - Jn).max() / max(np.abs(Ja).max(), 1e-30))
    print(f"3. gradient vs central differences at 20 random points: max error relative to the largest entry {worst:.1e}")


if __name__ == "__main__":
    for k in (0, 1, 5):
        check_cpc(k)
    check_zero()
    check_grad()
