"""CPC-style speed planning on a fixed path: the full quadrotor (p, v, q, w; 4 rotor thrusts; RK4 multiple shooting,
same model as quad_time_optimal.py / rpg quad.yaml), minimum lap time, but position is not free:
each node carries its progress s_k along the given path and must stay in a thin tube |p_k - c(s_k)| <= delta.
No gate / progress complementarity is needed (the path already passes the gates).

Optional margins: total thrust <= cap(|v|) * max and |w_xy| <= w_margin * 15 rad/s.

Output: sigma(s) (speed along the path) as a periodic cubic B-spline.
"""
import time

import casadi as ca
import numpy as np

import quad_time_optimal as qto
import speed_opt as so

G = 9.81


def path_table(path, ds=0.05):
    ss = np.arange(0.0, path.length, ds)
    P, T = [], []
    for s in ss:
        for pc in path.pieces:
            if pc[1] <= s < pc[1] + pc[2] or pc is path.pieces[-1]:
                sl = min(s - pc[1], pc[2])
                if pc[0] == "line":
                    P.append(pc[3] + sl * pc[4]); T.append(pc[4])
                else:
                    p, t, _, _ = pc[3].point(sl)
                    P.append(p); T.append(t)
                break
    return ss, np.array(P), np.array(T)


def warm_start(path, sigma_fn, N, ss, P, Tn, m):
    sig = sigma_fn(ss)
    tcum = np.concatenate([[0], np.cumsum(np.diff(np.append(ss, path.length)) / sig)])
    lap = tcum[-1]
    tk = np.arange(N) * lap / N
    sk = np.interp(tk, tcum, np.append(ss, path.length))
    pk = np.array([np.interp(sk, ss, P[:, i], period=path.length) for i in range(3)]).T
    tan = np.array([np.interp(sk, ss, Tn[:, i], period=path.length) for i in range(3)]).T
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    vk = sigma_fn(sk)[:, None] * tan
    dt = lap / N
    ak = (np.roll(vk, -1, 0) - np.roll(vk, 1, 0)) / (2 * dt)
    f = ak + G * np.array([0, 0, 1.0])
    z = f / np.linalg.norm(f, axis=1)[:, None]
    qk = np.array([qto.quat_from_thrust_dir(zz) for zz in z])
    zd = (np.roll(z, -1, 0) - np.roll(z, 1, 0)) / (2 * dt)
    wk = []
    for i in range(N):
        w_world = np.cross(z[i], zd[i])
        R = qto_rot(qk[i])
        wk.append(R.T @ w_world)
    X = np.hstack([pk, vk, qk, np.array(wk)])
    U = np.tile(m * np.linalg.norm(f, axis=1)[:, None] / 4, (1, 4))
    return X, U, sk, lap


def qto_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def solve(path, sigma_fn, ds=0.3, delta=0.1, cap=None, w_margin=1.0, max_iter=1500, verbose=False):
    """Nodes at fixed spacing ds along the path (like TOPP); the time of each step is a variable."""
    t0 = time.perf_counter()
    f, Q = qto.quad_model()
    m, Tmax = Q["m"], Q["Tmax"]
    A = 4 * Tmax / m
    L = path.length
    ss, P, Tn = path_table(path, ds=0.01)
    N = int(round(L / ds))
    sk = np.arange(N) * L / N
    ck = np.array([np.interp(sk, ss, P[:, i], period=L) for i in range(3)]).T
    tk = np.array([np.interp(sk, ss, Tn[:, i], period=L) for i in range(3)]).T
    tk /= np.linalg.norm(tk, axis=1)[:, None]
    sig = sigma_fn(sk)
    dt0 = (L / N) / (0.5 * (sig + np.roll(sig, -1)))
    lap0 = dt0.sum()
    # warm start on the path at the given speed: thrust direction from the finite-difference acceleration
    vk = sig[:, None] * tk
    ak = (np.roll(vk, -1, 0) - np.roll(vk, 1, 0)) / (dt0 + np.roll(dt0, 1))[:, None]
    fz = ak + G * np.array([0, 0, 1.0])
    z = fz / np.linalg.norm(fz, axis=1)[:, None]
    qk = np.array([qto.quat_from_thrust_dir(zz) for zz in z])
    zd = (np.roll(z, -1, 0) - np.roll(z, 1, 0)) / (dt0 + np.roll(dt0, 1))[:, None]
    wk = np.array([qto_rot(qk[i]).T @ np.cross(z[i], zd[i]) for i in range(N)])
    X0 = np.hstack([ck, vk, qk, wk])
    U0 = np.tile(m * np.linalg.norm(fz, axis=1)[:, None] / 4, (1, 4))

    opti = ca.Opti()
    X = opti.variable(13, N)
    U = opti.variable(4, N)
    DT = opti.variable(N)
    for k in range(N):
        opti.subject_to(X[:, (k + 1) % N] == qto.rk4(f, X[:, k], U[:, k], DT[k]))   # periodic lap
        opti.subject_to(ca.sumsqr(X[0:3, k] - ck[k]) <= delta ** 2)                 # tube around the path
        if cap is not None:
            opti.subject_to(ca.sum1(U[:, k]) <= m * A * cap(ca.norm_2(X[3:6, k])))
        opti.subject_to(ca.sumsqr(X[10:12, k]) <= (w_margin * Q["wxy"]) ** 2)
    opti.subject_to(ca.sumsqr(X[6:10, 0]) == 1)       # unit quaternion (propagated by the dynamics, as in CPC)
    opti.subject_to(opti.bounded(-Q["wz"], X[12, :], Q["wz"]))
    opti.subject_to(opti.bounded(Q["Tmin"], ca.vec(U), Tmax))
    opti.subject_to(opti.bounded(1e-3, DT, 0.3))
    opti.minimize(ca.sum1(DT))
    opti.set_initial(X, X0.T)
    opti.set_initial(U, U0.T)
    opti.set_initial(DT, dt0)
    opti.solver("ipopt", {"print_time": False},
                {"max_iter": max_iter, "print_level": 5 if verbose else 0, "mu_strategy": "adaptive", "tol": 1e-6})
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    try:
        sol = opti.solve()
        status = "ok"
    except RuntimeError:
        sol = opti.debug
        status = opti.stats()["return_status"]
    t_solve = time.perf_counter() - t0
    Xs, Us, DTs = sol.value(X), sol.value(U), sol.value(DT)
    sig_k = np.sum(Xs[3:6].T * tk, axis=1)
    return dict(lap=float(DTs.sum()), lap0=lap0, X=Xs, U=Us, DT=DTs, sk=sk, sig_k=sig_k, spline=fit_sigma(sk, sig_k, L),
                t_build=t_build, t_solve=t_solve, status=status, iters=opti.stats()["iter_count"],
                dev=np.linalg.norm(Xs[0:3].T - ck, axis=1), N=N)


def fit_sigma(sk, sig_k, L, M=150, smooth=1e-3):
    """Least-squares periodic cubic B-spline through the node speeds (light second-difference smoothing)."""
    rows = np.array([so._basis_at(M, L, s)[0] for s in sk])
    D = np.roll(np.eye(M), 1, 1) - 2 * np.eye(M) + np.roll(np.eye(M), -1, 1)
    Amat = np.vstack([rows, np.sqrt(smooth) * D])
    b = np.concatenate([sig_k, np.zeros(M)])
    coef = np.linalg.lstsq(Amat, b, rcond=None)[0]
    return so.SigmaSpline(coef, L)
