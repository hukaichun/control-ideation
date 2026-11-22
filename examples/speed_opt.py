"""Speed along a fixed path, optimised with the drone's limits:  minimise  T + lam * integral |j|^2 dt
over sigma(s) at N nodes (periodic), subject to
    |a + g| <= A            (total thrust)
    |a + g| >= f_min        (thrust direction stays steerable)
    |j_perp| <= w_max * c   (body rate: j = c' z + c w x z, so |w| = |j_perp| / c)
with  a = 1/2 u' T + u K,   j = sigma (1/2 u'' T + 3/2 u' K + u K'),   u = sigma^2.
The result is a sigma table (speed along the path).
"""
import time

import casadi as ca
import numpy as np

G = 9.81


def path_grid(path, N):
    """Path geometry on N equally spaced nodes: tangent T, curvature vector K, dK/ds."""
    L = path.length
    ss = np.arange(N) * L / N
    P, T, K = [], [], []
    for s in ss:
        for pc in path.pieces:
            if pc[1] <= s < pc[1] + pc[2] or pc is path.pieces[-1]:
                sl = min(s - pc[1], pc[2])
                if pc[0] == "line":
                    P.append(pc[3] + sl * pc[4]); T.append(pc[4]); K.append(np.zeros(3))
                else:
                    p, t, kv, _ = pc[3].point(sl)
                    P.append(p); T.append(t); K.append(kv)
                break
    P, T, K = map(np.array, (P, T, K))
    ds = L / N
    Kd = (np.roll(K, -1, 0) - np.roll(K, 1, 0)) / (2 * ds)
    return ss, ds, P, T, K, Kd


def bspline_basis(M, L, n_per):
    """Uniform periodic cubic B-spline with M control points on [0, L): value / d/ds / d2/ds2 matrices at
    n_per points per knot interval."""
    h = L / M
    tau = (np.arange(n_per) + 0.5) / n_per
    Bv = np.stack([(1 - tau) ** 3, 3 * tau ** 3 - 6 * tau ** 2 + 4, -3 * tau ** 3 + 3 * tau ** 2 + 3 * tau + 1, tau ** 3], 1) / 6
    B1 = np.stack([-3 * (1 - tau) ** 2, 9 * tau ** 2 - 12 * tau, -9 * tau ** 2 + 6 * tau + 3, 3 * tau ** 2], 1) / (6 * h)
    B2 = np.stack([6 * (1 - tau), 18 * tau - 12, -18 * tau + 6, 6 * tau], 1) / (6 * h * h)
    B3 = np.array([-1.0, 3.0, -3.0, 1.0]) / h ** 3            # third derivative: constant on each interval
    n = M * n_per
    V, D1, D2, D3 = (np.zeros((n, M)) for _ in range(4))
    for i in range(M):
        for m in range(4):
            col = (i + m - 1) % M
            V[i * n_per:(i + 1) * n_per, col] += Bv[:, m]
            D1[i * n_per:(i + 1) * n_per, col] += B1[:, m]
            D2[i * n_per:(i + 1) * n_per, col] += B2[:, m]
            D3[i * n_per:(i + 1) * n_per, col] += B3[m]
    ss = (np.repeat(np.arange(M), n_per) + np.tile(tau, M)) * h
    return ss, V, D1, D2, D3


def _basis_at(M, L, s):
    h = L / M
    x = (s % L) / h
    i = int(np.floor(x)) % M
    tau = x - np.floor(x)
    Bv = np.array([(1 - tau) ** 3, 3 * tau ** 3 - 6 * tau ** 2 + 4, -3 * tau ** 3 + 3 * tau ** 2 + 3 * tau + 1, tau ** 3]) / 6
    B1 = np.array([-3 * (1 - tau) ** 2, 9 * tau ** 2 - 12 * tau, -9 * tau ** 2 + 6 * tau + 3, 3 * tau ** 2]) / (6 * h)
    B2 = np.array([6 * (1 - tau), 18 * tau - 12, -18 * tau + 6, 6 * tau]) / (6 * h * h)
    V, D1, D2 = np.zeros(M), np.zeros(M), np.zeros(M)
    for m in range(4):
        V[(i + m - 1) % M] += Bv[m]; D1[(i + m - 1) % M] += B1[m]; D2[(i + m - 1) % M] += B2[m]
    return V, D1, D2


class SigmaSpline:
    def __init__(self, coef, L):
        self.c, self.L, self.M = np.asarray(coef), L, len(coef)

    def __call__(self, x):
        x = np.mod(np.asarray(x, float), self.L) / (self.L / self.M)
        i = np.floor(x).astype(int) % self.M
        tau = x - np.floor(x)
        B = [(1 - tau) ** 3, 3 * tau ** 3 - 6 * tau ** 2 + 4, -3 * tau ** 3 + 3 * tau ** 2 + 3 * tau + 1, tau ** 3]
        return sum(self.c[(i + m - 1) % self.M] * B[m] for m in range(4)) / 6


def grid_at(path, ss):
    T, K = [], []
    for s in ss:
        for pc in path.pieces:
            if pc[1] <= s < pc[1] + pc[2] or pc is path.pieces[-1]:
                sl = min(s - pc[1], pc[2])
                if pc[0] == "line":
                    T.append(pc[4]); K.append(np.zeros(3))
                else:
                    _, t, kv, _ = pc[3].point(sl)
                    T.append(t); K.append(kv)
                break
    T, K = np.array(T), np.array(K)
    ds = np.diff(np.append(ss, ss[0] + path.length))
    Kd = (np.roll(K, -1, 0) - np.roll(K, 1, 0)) / (np.roll(ds, 1) + ds)[:, None]
    return T, K, Kd, ds


def optimise(path, A, sig0_fn, M=150, n_per=4, lam=1e-5, w_max=15.0, f_min=2.0, eta=0.9, max_iter=300, verbose=False,
             use_kd=True, joint_deg=None, cap=None, motor=None):
    """sigma(s) = periodic cubic B-spline with M control points; limits checked at n_per points per interval
    with exact derivatives (the flight uses this same spline).

    motor (dict m, Ix, l, Tmax, Tmin, margin): torque first. From the snap along the path (flatness) the needed
    angular acceleration alpha_perp = z x z'' and torque |tau| = Ix |alpha_perp| are reserved first, the
    collective thrust T gets what is left, for every motor and any yaw (worst direction, factor sqrt 2):
        T + sqrt2 |tau| / l <= 4 margin Tmax,    T - sqrt2 |tau| / l >= 4 Tmin.
    Replaces the plain thrust cap |a + g| <= A."""
    t0 = time.perf_counter()
    L = path.length
    ss, V, D1, D2, D3 = bspline_basis(M, L, n_per)
    T, K, Kd, ds = grid_at(path, ss)
    if not use_kd:                     # line + arc: curvature jumps at joints are left to the control law
        Kd = np.zeros_like(Kd)
    c = ca.SX.sym("c", M)
    sig, sp, spp = ca.mtimes(ca.DM(V), c), ca.mtimes(ca.DM(D1), c), ca.mtimes(ca.DM(D2), c)
    u, up, upp = sig ** 2, 2 * sig * sp, 2 * sp ** 2 + 2 * sig * spp
    n = len(ss)
    Tm, Km, Kdm = ca.DM(T), ca.DM(K), ca.DM(Kd)
    a = 0.5 * ca.repmat(up, 1, 3) * Tm + ca.repmat(u, 1, 3) * Km
    f = a + ca.repmat(ca.DM([[0, 0, G]]), n, 1)
    j = ca.repmat(sig, 1, 3) * (0.5 * ca.repmat(upp, 1, 3) * Tm + 1.5 * ca.repmat(up, 1, 3) * Km
                                + ca.repmat(u, 1, 3) * Kdm)
    c2 = ca.sum2(f * f)
    jj = ca.sum2(j * j)
    jperp2 = jj - ca.sum2(j * f) ** 2 / c2
    J_time = ca.sum1(ca.DM(ds) / sig)
    J_jerk = ca.sum1(jj * ca.DM(ds) / sig)
    if cap is not None:                # optional thrust cap that depends on speed: A * cap(sigma)
        frac = cap(sig)
        g_thrust = c2 / (A * frac) ** 2
    else:
        g_thrust = c2 / A ** 2
    g = ca.vertcat(g_thrust, jperp2 / ((eta * w_max) ** 2 * c2), sig)
    lb = np.concatenate([np.full(n, (f_min / A) ** 2), np.full(n, -np.inf), np.full(n, 0.5)])
    ub = np.concatenate([np.ones(n) if motor is None else np.full(n, np.inf), np.ones(n), np.full(n, 60.0)])
    if motor is not None:
        R3 = lambda x: ca.repmat(x, 1, 3)
        sppp = ca.mtimes(ca.DM(D3), c)
        uppp = 6 * sp * spp + 2 * sig * sppp
        Kdd = np.zeros_like(Kd) if not use_kd else (np.roll(Kd, -1, 0) - np.roll(Kd, 1, 0)) / (np.roll(ds, 1) + ds)[:, None]
        Jv = 0.5 * R3(upp) * Tm + 1.5 * R3(up) * Km + R3(u) * Kdm                  # j = sigma J
        Jd = 0.5 * R3(uppp) * Tm + 2 * R3(upp) * Km + 2.5 * R3(up) * Kdm + R3(u) * ca.DM(Kdd)
        snap = R3(sig * sp) * Jv + R3(u) * Jd
        cn = ca.sqrt(c2)
        z = f / R3(cn)
        cd = ca.sum2(z * j)
        zd = (j - R3(cd) * z) / R3(cn)
        cdd = ca.sum2(z * snap) + cn * ca.sum2(zd * zd)
        zdd = (snap - R3(cdd) * z - 2 * R3(cd) * zd) / R3(cn)
        cr = lambda A_, B_: ca.horzcat(A_[:, 1] * B_[:, 2] - A_[:, 2] * B_[:, 1], A_[:, 2] * B_[:, 0] - A_[:, 0] * B_[:, 2],
                                       A_[:, 0] * B_[:, 1] - A_[:, 1] * B_[:, 0])
        alpha = cr(z, zdd)
        tau2 = motor["Ix"] ** 2 * ca.sum2(alpha * alpha)
        Tc = motor["m"] * cn
        k_ = motor["l"] / np.sqrt(2)
        hi = 4 * motor["margin"] * motor["Tmax"] - Tc
        lo = Tc - 4 * motor["Tmin"]
        sc = (4 * motor["Tmax"] * k_) ** 2
        g = ca.vertcat(g, hi, lo, ((k_ * hi) ** 2 - tau2) / sc, ((k_ * lo) ** 2 - tau2) / sc)
        lb = np.concatenate([lb, np.zeros(4 * n)])
        ub = np.concatenate([ub, np.full(4 * n, np.inf)])
    if joint_deg is not None:          # curvature jumps at line/arc joints: limit the jump of the thrust direction
        gj = []
        for pc in path.pieces:
            if pc[0] != "corner":
                continue
            cr = pc[3]
            for sj, tj, kj in ((pc[1], *cr.point(0.0)[1:3]), (pc[1] + pc[2], *cr.point(cr.L)[1:3])):
                Vj, D1j, _ = (np.array(x) for x in _basis_at(M, L, sj))
                sgj = ca.mtimes(ca.DM(Vj[None, :]), c)
                spj = ca.mtimes(ca.DM(D1j[None, :]), c)
                f1 = sgj * spj * tj + np.array([0, 0, G])
                f2 = f1 + sgj ** 2 * kj
                cr2 = ca.sum1(ca.cross(f1, f2) ** 2)
                gj.append(cr2 - np.sin(np.radians(joint_deg)) ** 2 * ca.sum1(f1 ** 2) * ca.sum1(f2 ** 2))
        g = ca.vertcat(g, *gj)
        lb = np.concatenate([lb, np.full(len(gj), -np.inf)])
        ub = np.concatenate([ub, np.zeros(len(gj))])
    nlp = {"x": c, "f": J_time + lam * J_jerk, "g": g}
    opts = {"ipopt.max_iter": max_iter, "ipopt.print_level": 5 if verbose else 0, "print_time": False,
            "ipopt.mu_strategy": "adaptive", "ipopt.tol": 1e-6}
    solver = ca.nlpsol("speed", "ipopt", nlp, opts)
    t_build = time.perf_counter() - t0
    x0 = np.linalg.lstsq(V, sig0_fn(ss), rcond=None)[0]
    t0 = time.perf_counter()
    sol = solver(x0=x0, lbg=lb, ubg=ub)
    t_solve = time.perf_counter() - t0
    coef = np.array(sol["x"]).ravel()
    st = solver.stats()
    sg = V @ coef
    out = dict(coef=coef, spline=SigmaSpline(coef, L), lap=float(np.sum(ds / sg)), t_solve=t_solve,
               t_build=t_build, status=st["return_status"], iters=st["iter_count"])
    if motor is not None:                       # planned worst-case motor thrusts (for checking)
        ev = ca.Function("ev", [c], [Tc, ca.sqrt(tau2)])(coef)
        Tn, tn = (np.array(x).ravel() for x in ev)
        out["motor_max"] = (Tn / 4 + tn / (4 * k_)) / motor["Tmax"]
        out["motor_min"] = (Tn / 4 - tn / (4 * k_)) / motor["Tmax"]
        out["ss"] = ss
    return out
