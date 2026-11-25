"""Home-made CPC-style time-optimal lap with the full quadrotor model (rigid body, 4 rotor thrusts,
attitude quaternion, body-rate limits), parameters from uzh-rpg/rpg_time_optimal quads/quad.yaml.

Difference from CPC: instead of complementary progress constraints, each gate-to-gate segment has its own
free duration (gate order is fixed, so both search the same set of trajectories). Periodic lap, gate
tolerance 0.3 m, RK4 multiple shooting, IPOPT. Warm-started from the point-mass optimum.

    uv run --with casadi --with pyyaml --with scipy --with numba examples/quad_time_optimal.py
"""
import argparse
import time
from pathlib import Path

import casadi as ca
import numpy as np
import yaml

import race_compare as rc

QUAD = Path(__file__).resolve().parents[1] / "third_party/rpg_time_optimal/quads/quad.yaml"


def quad_model():
    q = yaml.safe_load(QUAD.read_text())
    m, l, c = q["mass"], q["arm_length"], q["torque_coeff"]
    J = np.array(q["inertia"], float)
    Tmax = q["TWR_max"] * m * rc.G / 4
    Tmin = q["thrust_min"]
    wxy, wz = q["omega_max_xy"], q["omega_max_z"]

    x = ca.MX.sym("x", 13)  # p(3) v(3) q(4, w x y z) w(3)
    u = ca.MX.sym("u", 4)   # rotor thrusts
    p, v, qt, w = x[0:3], x[3:6], x[6:10], x[10:13]
    qw, qx, qy, qz = qt[0], qt[1], qt[2], qt[3]
    R = ca.vertcat(
        ca.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
        ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))
    F = ca.sum1(u)
    dv = R @ ca.vertcat(0, 0, F / m) - ca.vertcat(0, 0, rc.G)
    dq = 0.5 * ca.vertcat(-qx * w[0] - qy * w[1] - qz * w[2],
                          qw * w[0] + qy * w[2] - qz * w[1],
                          qw * w[1] + qz * w[0] - qx * w[2],
                          qw * w[2] + qx * w[1] - qy * w[0])
    d = l  # moment arm as in rpg_time_optimal quad.py (X configuration)
    tau = ca.vertcat(d * (u[0] - u[1] - u[2] + u[3]),
                     d * (-u[0] - u[1] + u[2] + u[3]),
                     c * (u[0] - u[1] + u[2] - u[3]))
    dw = ca.solve(ca.DM(J), tau - ca.cross(w, ca.DM(J) @ w))
    f = ca.Function("f", [x, u], [ca.vertcat(v, dv, dq, dw)])
    return f, dict(m=m, Tmax=Tmax, Tmin=Tmin, wxy=wxy, wz=wz)


def rk4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def quat_from_thrust_dir(z):
    """Quaternion rotating body z onto z (yaw-free, shortest arc)."""
    z = z / np.linalg.norm(z)
    ax = np.cross([0, 0, 1], z)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0])
    ang = np.arctan2(s, z[2])
    ax /= s
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * ax])


def solve(gates, tol, N, T_pm, p_pm, v_pm, cold=False, vel_guess=3.0, thrust_scale=1.0, wxy=None, init=None):
    """thrust_scale, wxy: a restricted vehicle (max rotor thrust x thrust_scale, |w_x|, |w_y| <= wxy).
    init: (T, x, u) of an earlier solution to start from (instead of the point-mass / cold guess)."""
    f, P = quad_model()
    P = dict(P, Tmax=P["Tmax"] * thrust_scale, wxy=P["wxy"] if wxy is None else wxy)
    M = len(gates)
    opti = ca.Opti()
    T = opti.variable(M)
    X = [opti.variable(13, N + 1) for _ in range(M)]
    U = [opti.variable(4, N) for _ in range(M)]

    # warm start from the point-mass optimum (N nodes per segment there too)
    acc = (np.roll(v_pm, -1, 0) - v_pm)                # per node, scaled below
    for i in range(M):
        dt = T_pm[i] / N
        ps, vs = p_pm[i * N:(i + 1) * N], v_pm[i * N:(i + 1) * N]
        a = acc[i * N:(i + 1) * N] / dt
        qs = np.array([quat_from_thrust_dir(ai + rc.G * rc.EZ) for ai in a])
        x0 = np.hstack([ps, vs, qs, np.zeros((N, 3))])
        x0 = np.vstack([x0, x0[-1:]])
        opti.set_initial(X[i], x0.T)
        opti.set_initial(T[i], T_pm[i])
        opti.set_initial(U[i], np.tile(P["m"] * np.linalg.norm(a + rc.G * rc.EZ, axis=1) / 4, (4, 1)))
        if cold:  # CPC-style guess: straight line between gates at vel_guess, hover attitude and thrust
            g0, g1 = gates[i], gates[(i + 1) % M]
            Tg = np.linalg.norm(g1 - g0) / vel_guess
            x0 = np.zeros((N + 1, 13))
            x0[:, 0:3] = np.linspace(g0, g1, N + 1)
            x0[:, 3:6] = (g1 - g0) / Tg
            x0[:, 6] = 1.0
            opti.set_initial(X[i], x0.T)
            opti.set_initial(T[i], Tg)
            opti.set_initial(U[i], np.full((4, N), P["m"] * rc.G / 4))

    if init is not None:
        T0, x0_, u0_ = init
        for i in range(M):
            xs = x0_[i * N:(i + 1) * N]
            opti.set_initial(X[i], np.vstack([xs, x0_[((i + 1) * N) % len(x0_)][None]]).T)
            opti.set_initial(U[i], np.clip(u0_[i * N:(i + 1) * N].T, P["Tmin"], P["Tmax"]))
            opti.set_initial(T[i], T0[i])

    for i in range(M):
        dt = T[i] / N
        for k in range(N):
            opti.subject_to(X[i][:, k + 1] == rk4(f, X[i][:, k], U[i][:, k], dt))
        j = (i + 1) % M
        opti.subject_to(X[j][:, 0] == X[i][:, N])
        opti.subject_to(ca.sumsqr(X[i][0:3, 0] - gates[i]) <= tol ** 2)
        opti.subject_to(opti.bounded(P["Tmin"], ca.vec(U[i]), P["Tmax"]))
        opti.subject_to(opti.bounded(-P["wxy"], ca.vec(X[i][10:12, :]), P["wxy"]))
        opti.subject_to(opti.bounded(-P["wz"], ca.vec(X[i][12, :]), P["wz"]))
        opti.subject_to(T[i] >= 0.05)
    opti.subject_to(ca.sumsqr(X[0][6:10, 0]) == 1)   # quaternion norm (propagated by the dynamics)
    opti.minimize(ca.sum1(T))
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 5000})
    sol = opti.solve()
    x = np.hstack([sol.value(X[i])[:, :-1] for i in range(M)]).T
    u = np.hstack([sol.value(U[i]) for i in range(M)]).T
    return sol.value(T), x, u, P


def main(N, out, cold):
    gates = rc.load_track()
    A, tol = 3.3 * rc.G, 0.3
    T_pm_total, p_pm, v_pm, T_pm = rc.free_path_optimum(gates, A, tol, N=N, return_segments=True)
    print(f"point-mass optimum: lap {T_pm_total:.3f} s", flush=True)
    t0 = time.time()
    T, x, u, P = solve(gates, tol, N, T_pm, p_pm, v_pm, cold)
    lap = float(np.sum(T))
    w = x[:, 10:13]
    print(f"full-quadrotor optimum ({'cold' if cold else 'warm'} start): lap {lap:.3f} s   [{time.time() - t0:.0f} s solve]")
    print(f"  thrust per rotor in [{u.min():.2f}, {u.max():.2f}] N (limit {P['Tmax']:.2f}), "
          f"max |w_xy| {np.abs(w[:, :2]).max():.1f} rad/s, max |w_z| {np.abs(w[:, 2]).max():.2f} rad/s")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.hstack([x, u]), delimiter=",",
               header="px,py,pz,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,T1,T2,T3,T4", comments="")
    print(f"trajectory -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30, help="nodes per gate-to-gate segment")
    ap.add_argument("--cold", action="store_true", help="straight-line guess instead of the point-mass optimum")
    ap.add_argument("-o", "--out", type=Path, default=Path("logs/quad_time_optimal.csv"))
    a = ap.parse_args()
    main(a.N, a.out, a.cold)
