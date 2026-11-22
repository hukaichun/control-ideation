"""Speed planning directly in the space of V (no fitting): on each piece V(p) = g_i(l_i(p)) with
  l_i(p) = e_i.(p - o_i)  a linear function of p (line: e = direction, o = start; arc: e = chord direction start->end,
                          o = centre; monotone along an arc shorter than half a turn)
  g_i(x) = sum_k c_ik T_k(x),  x = 2 (l - l0)/(l1 - l0) - 1  (Chebyshev, well conditioned)
Speed on the path: sigma^2 = u = 2 (H* - V), H* = 0.
Optimisation over the coefficients c_ik: minimum time (+ lam * jerk^2), with the drone limits along the path from
flatness (thrust floor, body rate, torque-first motor limits), and V, V', V'' continuous at the joints.
Along the path: l' = e.t, l'' = e.K, l''' = -(e.t) / R^2 on arcs (0 on lines); jerk and snap include the turning of the
curvature vector on arcs (dK/ds = -T/R^2, d2K/ds2 = -K/R^2); curvature jumps at joints are not included (left to the
control law).
"""
import time

import casadi as ca
import numpy as np
from numpy.polynomial import chebyshev as C

import path_pieces as pp

G = 9.81


def pieces_V(path):
    out = []
    for P in pp.pieces_of(path):
        if P["kind"] == "line":
            e, o = P["u"], P["a"]
            l0, l1 = 0.0, P["len"]
        else:
            ps = P["c"] - P["R"] * P["e2"]
            pe = P["c"] + P["R"] * (np.sin(P["th"]) * P["e1"] - np.cos(P["th"]) * P["e2"])
            e = (pe - ps) / np.linalg.norm(pe - ps)
            o = P["c"]
            l0, l1 = e @ (ps - o), e @ (pe - o)
        out.append(dict(P=P, e=e, o=o, l0=l0, l1=l1))
    return out


def _cheb_basis(x, K, xd, xdd, xddd):
    """Rows: T_k(x) and its first three derivatives along the path (chain rule through x(s))."""
    I = np.eye(K + 1)
    B = [C.chebval(x, I[k]) for k in range(K + 1)]
    D1 = [C.chebval(x, C.chebder(I[k], 1)) for k in range(K + 1)]
    D2 = [C.chebval(x, C.chebder(I[k], 2)) for k in range(K + 1)]
    D3 = [C.chebval(x, C.chebder(I[k], 3)) for k in range(K + 1)]
    B, D1, D2, D3 = (np.array(z).T for z in (B, D1, D2, D3))
    S1 = D1 * xd[:, None]
    S2 = D2 * (xd ** 2)[:, None] + D1 * xdd[:, None]
    S3 = D3 * (xd ** 3)[:, None] + 3 * D2 * (xd * xdd)[:, None] + D1 * xddd[:, None]
    return B, S1, S2, S3


_SAMPLE = None
COMPILE = True    # per-sample function as compiled C (cached in build/); False: CasADi virtual machine


def _sample_fun():
    """Per sample: x = V and its first three s-derivatives, geo = (tangent, curvature vector, ds, 1/R^2), par = motor and limits ->
    (7 constraint values, cost ds/sigma (1 + lam |j|^2)). Built once, mapped over all samples of any track."""
    global _SAMPLE
    if _SAMPLE is not None:
        return _SAMPLE
    x, geo, par = ca.SX.sym("x", 4), ca.SX.sym("geo", 8), ca.SX.sym("par", 9)
    m, Ix, l, Tmax, Tmin, margin, w_max, eta, lam = (par[i] for i in range(9))
    Tm, Km, dsm, k2 = geo[0:3], geo[3:6], geo[6], geo[7]
    Kd, Kdd = -k2 * Tm, -k2 * Km                     # arc: dK/ds = -T/R^2, d2K/ds2 = -K/R^2 (line: k2 = 0)
    u, up, upp, uppp = -2 * x[0], -2 * x[1], -2 * x[2], -2 * x[3]          # H* = 0
    sig = ca.sqrt(ca.fmax(u, 1e-6))
    sp = up / (2 * sig)
    a = 0.5 * up * Tm + u * Km
    f = a + ca.DM([0, 0, G])
    Jv = 0.5 * upp * Tm + 1.5 * up * Km + u * Kd
    j = sig * Jv
    Jd = 0.5 * uppp * Tm + 2 * upp * Km + 2.5 * up * Kd + u * Kdd
    snap = sig * sp * Jv + u * Jd
    c2 = ca.dot(f, f)
    jj = ca.dot(j, j)
    jperp2 = jj - ca.dot(j, f) ** 2 / c2
    cn = ca.sqrt(c2)
    z = f / cn
    cd = ca.dot(z, j)
    zd = (j - cd * z) / cn
    cdd = ca.dot(z, snap) + cn * ca.dot(zd, zd)
    zdd = (snap - cdd * z - 2 * cd * zd) / cn
    alpha = ca.cross(z, zdd)
    tau2 = Ix ** 2 * ca.dot(alpha, alpha)
    Tc = m * cn
    k_ = l / np.sqrt(2)
    hi = 4 * margin * Tmax - Tc
    lo = Tc - 4 * Tmin
    scl = (4 * Tmax * k_) ** 2
    g = ca.vertcat(u, c2, jperp2 / ((eta * w_max) ** 2 * c2), hi, lo, ((k_ * hi) ** 2 - tau2) / scl,
                   ((k_ * lo) ** 2 - tau2) / scl)
    F = ca.Function("vplan_sample", [x, geo, par], [g, dsm / sig * (1 + lam * jj)])
    _SAMPLE = _compiled(F) if COMPILE else F
    return _SAMPLE


def _compiled(F):
    """F with its first and second derivatives as compiled C, cached on disk (built once, about 10 s)."""
    import hashlib
    import os
    import subprocess
    key = hashlib.sha1(F.serialize().encode()).hexdigest()[:12]
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "vplan_" + key)
    so_ = os.path.join(d, F.name() + ".so")
    if not os.path.exists(so_):
        os.makedirs(d, exist_ok=True)
        J = F.jacobian()
        cg = ca.CodeGenerator(F.name() + ".c")
        for f in (F, J, J.jacobian()):
            cg.add(f)
        cg.generate(d + os.sep)
        subprocess.run(["gcc", "-O2", "-fPIC", "-shared", os.path.join(d, F.name() + ".c"), "-o", so_ + ".tmp"], check=True)
        os.replace(so_ + ".tmp", so_)
    return ca.external(F.name(), so_)


def _samples(pv, K, ds):
    """Sample points of every piece: rows (per piece), A (c -> V and 3 s-derivatives at each sample), geo."""
    rows = []                    # per sample: piece, T, K, ds, B, S1, S2, S3
    for i, q in enumerate(pv):
        P = q["P"]
        n = max(int(np.ceil(P["len"] / ds)), 2 * (K + 1))     # fewer samples than coefficients: free wiggles
        sl = (np.arange(n) + 0.5) / n * P["len"]
        if P["kind"] == "line":
            T = np.tile(P["u"], (n, 1)); Kv = np.zeros((n, 3)); l = sl
            ld, ldd, lddd = np.ones(n), np.zeros(n), np.zeros(n)
        else:
            al = sl / P["R"]
            T = np.outer(np.cos(al), P["e1"]) + np.outer(np.sin(al), P["e2"])
            rad = np.outer(np.sin(al), P["e1"]) - np.outer(np.cos(al), P["e2"])
            Kv = -rad / P["R"]
            l = P["R"] * rad @ q["e"]
            ld, ldd = T @ q["e"], Kv @ q["e"]
            lddd = -ld / P["R"] ** 2
        sc = 2 / (q["l1"] - q["l0"])
        x = sc * (l - q["l0"]) - 1
        B, S1, S2, S3 = _cheb_basis(x, K, sc * ld, sc * ldd, sc * lddd)
        k2 = np.zeros(n) if P["kind"] == "line" else np.full(n, 1 / P["R"] ** 2)
        rows.append(dict(i=i, T=T, K=Kv, k2=k2, ds=np.full(n, P["len"] / n), B=B, S1=S1, S2=S2, S3=S3, s=P["s0"] + sl))
    nc = K + 1
    nP = len(pv)
    N = sum(r["B"].shape[0] for r in rows)
    # one linear map c -> (V and its first three s-derivatives) at every sample, then one small per-sample function
    # mapped over all samples
    A = np.zeros((4 * N, nP * nc))
    geo = np.zeros((8, N))
    k0 = 0
    for r in rows:
        n = r["B"].shape[0]
        for d, key in enumerate(("B", "S1", "S2", "S3")):
            A[4 * np.arange(k0, k0 + n) + d, r["i"] * nc:(r["i"] + 1) * nc] = r[key]
        geo[:, k0:k0 + n] = np.vstack([r["T"].T, r["K"].T, r["ds"], r["k2"]])
        k0 += n
    return rows, A, geo, N


def plan_V(path, motor, sigma0_fn, K=8, ds=0.25, lam=1e-6, w_max=15.0, eta=0.9, f_min=2.0, max_iter=500, tol=1e-6, opts=None):
    t_start = time.perf_counter()
    pv = pieces_V(path)
    nP = len(pv)
    rows, A, geo, N = _samples(pv, K, ds)
    nc = K + 1
    par = np.array([motor["m"], motor["Ix"], motor["l"], motor["Tmax"], motor["Tmin"], motor["margin"], w_max, eta, lam])
    # decision variables: coefficients c and the sample values X (= A c, linear equality), so the nonlinear part is
    # block diagonal (cheap derivatives)
    c = ca.MX.sym("c", nP * nc)
    Xv = ca.MX.sym("X", 4 * N)
    gS, jS = _sample_fun().map(N)(ca.reshape(Xv, 4, N), ca.DM(geo), ca.repmat(ca.DM(par), 1, N))
    gs = [ca.vec(gS), ca.mtimes(ca.sparsify(ca.DM(A)), c) - Xv]
    lbs = [np.tile([0.25, f_min ** 2, -np.inf, 0, 0, 0, 0], N), np.zeros(4 * N)]
    ubs = [np.tile([3600.0, np.inf, 1, np.inf, np.inf, np.inf, np.inf], N), np.zeros(4 * N)]
    J_all = ca.sum2(jS)
    # joints: V, dV/ds, d2V/ds2 continuous (end of piece i = start of piece i+1); linear in c
    Jrows = []
    for i, q in enumerate(pv):
        k2 = (i + 1) % nP
        ends = []
        for idx, xe in ((i, 1.0), (k2, -1.0)):
            qq = pv[idx]
            P = qq["P"]
            if P["kind"] == "line":
                ld, ldd = 1.0, 0.0
            else:
                al = P["th"] if xe > 0 else 0.0
                t = np.cos(al) * P["e1"] + np.sin(al) * P["e2"]
                rad = np.sin(al) * P["e1"] - np.cos(al) * P["e2"]
                ld, ldd = t @ qq["e"], (-rad / P["R"]) @ qq["e"]
            sc = 2 / (qq["l1"] - qq["l0"])
            B, S1, S2, _ = _cheb_basis(np.array([xe]), K, np.array([sc * ld]), np.array([sc * ldd]), np.zeros(1))
            row = np.zeros((3, nP * nc))
            row[:, idx * nc:(idx + 1) * nc] = np.vstack([B, S1, S2])
            ends.append(row)
        Jrows.append(ends[0] - ends[1])
    gs.append(ca.mtimes(ca.DM(np.vstack(Jrows)), c)); lbs.append(np.zeros(3 * nP)); ubs.append(np.zeros(3 * nP))
    g = ca.vertcat(*gs)
    nlp = {"x": ca.vertcat(c, Xv), "f": J_all, "g": g}
    solver = ca.nlpsol("vplan", "ipopt", nlp, {"ipopt.max_iter": max_iter, "ipopt.print_level": 0, "print_time": False,
                                               "ipopt.mu_strategy": "adaptive", "ipopt.tol": tol, **(opts or {})})
    t_build = time.perf_counter() - t_start
    # warm start: least squares of each g_i to V = -1/2 sigma0^2
    x0 = []
    for r in rows:
        x0.append(np.linalg.lstsq(r["B"], -0.5 * sigma0_fn(r["s"]) ** 2, rcond=None)[0])
    t0 = time.perf_counter()
    x0 = np.concatenate(x0)
    sol = solver(x0=np.concatenate([x0, A @ x0]), lbg=np.concatenate(lbs), ubg=np.concatenate(ubs))
    t_solve = time.perf_counter() - t0
    coef = np.array(sol["x"]).ravel()[:nP * nc]
    lap = 0.0
    for r in rows:
        V = r["B"] @ coef[r["i"] * nc:(r["i"] + 1) * nc]
        lap += np.sum(r["ds"] / np.sqrt(np.maximum(-2 * V, 1e-6)))
    spec = [dict(e=q["e"], o=q["o"], l0=q["l0"], l1=q["l1"], coef=coef[i * nc:(i + 1) * nc]) for i, q in enumerate(pv)]
    return dict(spec=spec, lap=lap, status=solver.stats()["return_status"], t_build=t_build, t_solve=t_solve,
                iters=solver.stats()["iter_count"])


def check_V(path, spec, motor, ds=0.02, K=8, w_max=15.0, eta=0.9, f_min=2.0):
    """Constraints of a plan on a fine grid: worst value of each (normalised so that <= 0 is satisfied)."""
    pv = pieces_V(path)
    rows, A, geo, N = _samples(pv, K, ds)
    coef = np.concatenate([q["coef"] for q in spec])
    par = np.array([motor["m"], motor["Ix"], motor["l"], motor["Tmax"], motor["Tmin"], motor["margin"], w_max, eta, 0.0])
    gS = np.array(_sample_fun().map(N)((A @ coef).reshape(N, 4).T, geo, np.tile(par[:, None], (1, N)))[0])
    return dict(speed2=(0.25 - gS[0]).max(), thrust_floor=(f_min ** 2 - gS[1]).max(), body_rate=(gS[2] - 1).max(),
                thrust_hi=(-gS[3]).max(), thrust_lo=(-gS[4]).max(), motor_hi=(-gS[5]).max(), motor_lo=(-gS[6]).max())


def warm_sigma(path, quad, margin):
    """sigma(s) for the warm start: the simple speed limit along the path (path_g2.plan_speed, thrust only).
    Returns (sigma_fn, None, motor, S0); the optimum of plan_V does not depend on it (checked on 8 tracks)."""
    import path_g2 as pg
    lap0, sig0, s0, S0 = pg.plan_speed(path, 0.6 * quad.A, window_m=2.0, f_min=2.0)
    motor = dict(m=quad.m, Ix=quad.J[0, 0], l=0.15, Tmax=quad.Tmax, Tmin=quad.Tmin, margin=margin)
    return (lambda x: np.interp(x, s0, sig0, period=path.length)), None, motor, S0
