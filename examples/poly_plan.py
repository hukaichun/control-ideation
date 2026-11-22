"""Path and speed optimised together (CPC-style), with the path restricted to piecewise polynomials that stay implicit.

Piece i (fixed frame O, e1, e2, e3 and chord length l from the starting path):
  x = e1.(p - O) in [0, l],  t = 2 x / l - 1
  path   : e2.(p - O) = f(x),  e3.(p - O) = g(x)        -> phi = (e2.(p - O) - f(x), e3.(p - O) - g(x))
  speed  : V(p) = h(x),  sigma^2 = -2 V  (H* = 0)
  f, g, h Chebyshev polynomials in t (degrees DEG_P, DEG_V).
Variables: coefficients of f, g, h of every piece, and per joint the reparametrisation derivatives (b1 > 0, b2, b3).
Constraints:
  every sample: drone limits from flatness (thrust floor and ceiling, body rate, torque-first motor limits), as v_plan;
  every joint : position, and G3 (first three derivatives equal up to the change of parameter) -> body rate continuous;
                V, V', V'' continuous;
  every gate  : |p(x_g) - gate| <= tol at the gate's x (fixed from the starting path).
Cost: lap time (+ lam jerk^2).
Start: the line + arc path (path_g2, q = 0), cut at the middles of the lines (and at the middle of turns > MAX_TURN),
fitted by least squares; speed from path_g2.plan_speed.
"""
import time

import casadi as ca
import numpy as np
from numpy.polynomial import chebyshev as C

import path_g2 as pg
import v_plan as vp

G = 9.81
DEG_P, DEG_V = 7, 8
MAX_TURN = np.deg2rad(120.0)


def cheb_rows(t, deg, nder, scale):
    """Rows k = 0..nder: d^k/dx^k of T_0..T_deg at t (x = l (t + 1) / 2, dt/dx = scale)."""
    I = np.eye(deg + 1)
    t = np.atleast_1d(t)
    out = []
    for k in range(nder + 1):
        out.append(np.array([C.chebval(t, C.chebder(I[m], k)) if k else C.chebval(t, I[m]) for m in range(deg + 1)]).T
                   * scale ** k)
    return out                                   # list of (n, deg+1)


# ---------------------------------------------------------------- starting pieces from the line + arc path
def init_pieces(path, gates, ds=0.01):
    P, T, K, D, sl = path.sample(ds)
    s = np.concatenate([[0.0], np.cumsum(sl)[:-1]])
    cuts = []                                    # cut at the middle of every line; long turns also at their middle
    for pc in path.pieces:
        if pc[0] == "line":
            cuts.append(pc[1] + 0.5 * pc[2])
        elif pc[3].th > MAX_TURN:
            cuts.append(pc[1] + 0.5 * pc[2])
    cuts = np.sort(np.array(cuts))
    n = len(P)
    idx = [int(np.argmin(np.abs(s - c))) for c in cuts]
    pieces = []
    for i in range(len(idx)):
        i0, i1 = idx[i], idx[(i + 1) % len(idx)]
        ii = np.arange(i0, i1 + 1) if i1 > i0 else np.concatenate([np.arange(i0, n), np.arange(0, i1 + 1)])
        Q = P[ii]
        O = Q[0]
        ch = Q[-1] - O
        l = np.linalg.norm(ch)
        e1 = ch / l
        dev = Q - O - np.outer((Q - O) @ e1, e1)
        k = np.argmax(np.linalg.norm(dev, axis=1))
        if np.linalg.norm(dev[k]) > 1e-6:
            e2 = dev[k] / np.linalg.norm(dev[k])
        else:
            t_ = np.array([1.0, 0, 0]) if abs(e1[0]) < 0.9 else np.array([0, 1.0, 0])
            e2 = np.cross(e1, t_); e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        x = (Q - O) @ e1
        assert np.all(np.diff(x) > 0), "piece not monotone along its chord"
        t = 2 * x / l - 1
        fc = C.chebfit(t, (Q - O) @ e2, DEG_P)
        gc = C.chebfit(t, (Q - O) @ e3, DEG_P)
        pieces.append(dict(O=O, e1=e1, e2=e2, e3=e3, l=l, f=fc, g=gc, s=s[ii] if i1 > i0 else None, ii=ii))
    # gates: piece and x of the nearest starting-path point
    gate_at = []
    for gt in gates:
        k = int(np.argmin(np.linalg.norm(P - gt, axis=1)))
        for i, pc in enumerate(pieces):
            if k in pc["ii"][1:-1] or (k in pc["ii"]):
                gate_at.append((i, float((P[k] - pc["O"]) @ pc["e1"])))
                break
    return pieces, gate_at, (P, s)


# ---------------------------------------------------------------- per-sample function (built once, compiled)
_SAMPLE = None


def _kin():
    """From the x-derivatives of f, g (0..4), of V (0..3) and the frame: v, a, j, snap and dt/dx. Motion along the
    path with x' = w(x) = sigma / |p_x|, sigma^2 = -2 V; d/dt = w d/dx."""
    F, Gg, U, fr = ca.SX.sym("F", 5), ca.SX.sym("Gg", 5), ca.SX.sym("U", 4), ca.SX.sym("fr", 9)
    e1, e2, e3 = fr[0:3], fr[3:6], fr[6:9]
    xi = ca.SX.sym("xi")
    fact = [1, 1, 2, 6, 24]
    p = e1 * xi
    for k in range(5):
        p = p + (F[k] * e2 + Gg[k] * e3) * xi ** k / fact[k]
    V = U[0] + U[1] * xi + U[2] * xi ** 2 / 2 + U[3] * xi ** 3 / 6
    u = -2 * V
    px = ca.jacobian(p, xi)
    w = ca.sqrt(ca.fmax(u, 1e-6)) / ca.norm_2(px)
    Dt = lambda e: w * ca.jacobian(e, xi)
    v = Dt(p); a = Dt(v); j = Dt(a); sn = Dt(j)
    out = [ca.substitute(e, xi, 0) for e in (v, a, j, sn, 1 / w, u)]
    return F, Gg, U, fr, out


def _sample_fun():
    global _SAMPLE
    if _SAMPLE is not None:
        return _SAMPLE
    F, Gg, U, fr, (v, a, j, sn, dtdx, u) = _kin()
    dx, par = ca.SX.sym("dx"), ca.SX.sym("par", 9)
    m, Ix, l, Tmax, Tmin, margin, w_max, eta, lam = (par[i] for i in range(9))
    f = a + ca.DM([0, 0, G])
    c2 = ca.dot(f, f)
    jj = ca.dot(j, j)
    jperp2 = jj - ca.dot(j, f) ** 2 / c2
    cn = ca.sqrt(c2)
    z = f / cn
    cd = ca.dot(z, j)
    zd = (j - cd * z) / cn
    cdd = ca.dot(z, sn) + cn * ca.dot(zd, zd)
    zdd = (sn - cdd * z - 2 * cd * zd) / cn
    alpha = ca.cross(z, zdd)
    tau2 = Ix ** 2 * ca.dot(alpha, alpha)
    Tc = m * cn
    k_ = l / np.sqrt(2)
    hi = 4 * margin * Tmax - Tc
    lo = Tc - 4 * Tmin
    scl = (4 * Tmax * k_) ** 2
    g = ca.vertcat(u, c2, jperp2 / ((eta * w_max) ** 2 * c2), hi, lo, ((k_ * hi) ** 2 - tau2) / scl,
                   ((k_ * lo) ** 2 - tau2) / scl)
    x = ca.vertcat(F, Gg, U)
    Fn = ca.Function("pplan_sample", [x, fr, dx, par], [g, dx * dtdx * (1 + lam * jj)])
    _SAMPLE = vp._compiled(Fn) if vp.COMPILE else Fn
    return _SAMPLE


_KIN = None


def kin_fun():
    global _KIN
    if _KIN is None:
        F, Gg, U, fr, out = _kin()
        _KIN = ca.Function("pkin", [ca.vertcat(F, Gg, U), fr], out[:4])
    return _KIN


# ---------------------------------------------------------------- the optimisation
def plan(path, gates, motor, sigma0_fn, ds=0.25, tol=0.24, lam=1e-6, w_max=15.0, eta=0.9, f_min=2.0, max_iter=1000,
         ipopt_tol=1e-6, stages=True, rounds=5, trust=0.3):
    """stages: 1 smooth the path alone, 2 speed on that path, 3 both together (each stage starts feasible)."""
    t_start = time.perf_counter()
    pcs, gate_at, (P0, s0) = init_pieces(path, gates)
    zp, bp = smooth_path(pcs, gate_at, gates, tol)
    t_smooth = time.perf_counter() - t_start
    r2 = _plan(pcs, gate_at, gates, path, P0, s0, motor, sigma0_fn, ds, tol, lam, w_max, eta, f_min, max_iter, ipopt_tol,
               zp, bp, fix_path=True)
    r3 = r2
    for _ in range(rounds if stages else 0):     # path moves at most `trust` metres per round (trust region)
        rn = _plan(pcs, gate_at, gates, path, P0, s0, motor, sigma0_fn, ds, tol, lam, w_max, eta, f_min, max_iter,
                   ipopt_tol, r3["zall"], r3["beta"], fix_path=False, trust=trust)
        print(f"  round: {rn['status']} it {rn['iters']} lap {rn['lap']:.3f} ({rn['t_solve']:.1f}s)", flush=True)
        if not rn["status"].startswith("Solve_Succeeded") or rn["lap"] > r3["lap"] - 1e-3:
            break
        r3 = rn
    r3.update(t_smooth=t_smooth, lap_fixed=r2["lap"], status_fixed=r2["status"], t_fixed=r2["t_solve"],
              t_total=time.perf_counter() - t_start)
    return r3


def smooth_path(pcs, gate_at, gates, tol, w_fit=1.0, n=40, w_d3=0.0):
    """Path alone: G3 joints and gates, close to the start, small third derivative. Returns (f, g coefficients per
    piece, joint betas)."""
    nf = DEG_P + 1
    nP = len(pcs)
    z = ca.MX.sym("z", nP * 2 * nf)
    beta = ca.MX.sym("b", 3 * nP)
    J, gs, lb, ub = 0, [], [], []
    z0 = np.concatenate([np.concatenate([pc["f"], pc["g"]]) for pc in pcs])
    for i, pc in enumerate(pcs):
        t = np.linspace(-1, 1, n)
        R0, _, _, R3 = cheb_rows(t, DEG_P, 3, 2 / pc["l"])
        for q in (0, 1):
            zz = z[i * 2 * nf + q * nf:i * 2 * nf + (q + 1) * nf]
            d = ca.mtimes(ca.DM(R0), zz - z0[i * 2 * nf + q * nf:i * 2 * nf + (q + 1) * nf])
            J += w_fit * ca.sumsqr(d) / n + w_d3 * ca.sumsqr(ca.mtimes(ca.DM(R3), zz)) * pc["l"] / n
    def ends(i, tt):
        pc = pcs[i]
        Rp = cheb_rows(np.array([tt]), DEG_P, 3, 2 / pc["l"])
        fz, gz = z[i * 2 * nf:i * 2 * nf + nf], z[i * 2 * nf + nf:(i + 1) * 2 * nf]
        x_ = pc["l"] * (tt + 1) / 2
        out = []
        for d in range(4):
            base = ca.DM(pc["O"] + x_ * pc["e1"]) if d == 0 else (ca.DM(pc["e1"]) if d == 1 else ca.DM.zeros(3))
            out.append(base + ca.mtimes(ca.DM(Rp[d]), fz) * ca.DM(pc["e2"]) + ca.mtimes(ca.DM(Rp[d]), gz) * ca.DM(pc["e3"]))
        return out
    for i in range(nP):
        k = (i + 1) % nP
        Pa, Pb = ends(i, 1.0), ends(k, -1.0)
        b1, b2, b3 = beta[3 * i], beta[3 * i + 1], beta[3 * i + 2]
        for e in (Pa[0] - Pb[0], Pa[1] - b1 * Pb[1], Pa[2] - b1 ** 2 * Pb[2] - b2 * Pb[1],
                  Pa[3] - b1 ** 3 * Pb[3] - 3 * b1 * b2 * Pb[2] - b3 * Pb[1]):
            gs.append(e); lb.append(np.zeros(3)); ub.append(np.zeros(3))
    for gi, (i, xg) in enumerate(gate_at):
        pc = pcs[i]
        Rp = cheb_rows(np.array([2 * xg / pc["l"] - 1]), DEG_P, 0, 1.0)[0]
        pg_ = (ca.DM(pc["O"] + xg * pc["e1"]) + ca.mtimes(ca.DM(Rp), z[i * 2 * nf:i * 2 * nf + nf]) * ca.DM(pc["e2"])
               + ca.mtimes(ca.DM(Rp), z[i * 2 * nf + nf:(i + 1) * 2 * nf]) * ca.DM(pc["e3"]))
        gs.append(ca.sumsqr(pg_ - ca.DM(gates[gi]))); lb.append(np.zeros(1)); ub.append(np.full(1, tol ** 2))
    lbx = np.full(nP * 2 * nf + 3 * nP, -np.inf); ubx = -lbx.copy()
    lbx[nP * 2 * nf::3] = 0.2
    b0 = np.tile([1.0, 0, 0], nP)
    S = ca.nlpsol("smooth", "ipopt", {"x": ca.vertcat(z, beta), "f": J, "g": ca.vertcat(*gs)},
                  {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 500})
    sol = S(x0=np.concatenate([z0, b0]), lbx=lbx, ubx=ubx, lbg=np.concatenate(lb), ubg=np.concatenate(ub))
    xs = np.array(sol["x"]).ravel()
    return xs[:nP * 2 * nf], xs[nP * 2 * nf:]


def _plan(pcs, gate_at, gates, path, P0, s0, motor, sigma0_fn, ds, tol, lam, w_max, eta, f_min, max_iter, ipopt_tol,
          z_start, b_start, fix_path, trust=None):
    t_start = time.perf_counter()
    nP = len(pcs)
    nf, nv = DEG_P + 1, DEG_V + 1
    nz = nP * (2 * nf + nv)
    off = lambda i: i * (2 * nf + nv)
    # sample rows
    blocks, frames, dxs, N = [], [], [], 0
    for i, pc in enumerate(pcs):
        n = max(int(np.ceil(pc["l"] / ds)), 2 * nv)
        t = (np.arange(n) + 0.5) / n * 2 - 1
        sc = 2 / pc["l"]
        Rp = cheb_rows(t, DEG_P, 4, sc)
        Rv = cheb_rows(t, DEG_V, 3, sc)
        blocks.append((i, n, Rp, Rv))
        frames.append(np.tile(np.concatenate([pc["e1"], pc["e2"], pc["e3"]])[:, None], (1, n)))
        dxs.append(np.full(n, pc["l"] / n))
        N += n
    A = np.zeros((14 * N, nz))
    k0 = 0
    for i, n, Rp, Rv in blocks:
        r = 14 * np.arange(k0, k0 + n)
        for d in range(5):
            A[r + d, off(i):off(i) + nf] = Rp[d]
            A[r + 5 + d, off(i) + nf:off(i) + 2 * nf] = Rp[d]
        for d in range(4):
            A[r + 10 + d, off(i) + 2 * nf:off(i) + 2 * nf + nv] = Rv[d]
        k0 += n
    frames, dxs = np.hstack(frames), np.concatenate(dxs)
    par = np.array([motor["m"], motor["Ix"], motor["l"], motor["Tmax"], motor["Tmin"], motor["margin"], w_max, eta, lam])
    z = ca.MX.sym("z", nz)
    X = ca.MX.sym("X", 14 * N)
    beta = ca.MX.sym("b", 3 * nP)
    gS, jS = _sample_fun().map(N)(ca.reshape(X, 14, N), ca.DM(frames), ca.DM(dxs).T, ca.repmat(ca.DM(par), 1, N))
    gs = [ca.vec(gS), ca.mtimes(ca.sparsify(ca.DM(A)), z) - X]
    lbs = [np.tile([0.25, f_min ** 2, -np.inf, 0, 0, 0, 0], N), np.zeros(14 * N)]
    ubs = [np.tile([3600.0, np.inf, 1, np.inf, np.inf, np.inf, np.inf], N), np.zeros(14 * N)]

    def ends(i, tt):
        """World-frame position and x-derivatives 1..3 of the path, and V and its x-derivatives 0..2, at t = tt."""
        pc = pcs[i]
        sc = 2 / pc["l"]
        Rp = cheb_rows(np.array([tt]), DEG_P, 3, sc)
        Rv = cheb_rows(np.array([tt]), DEG_V, 2, sc)
        fz = z[off(i):off(i) + nf]; gz = z[off(i) + nf:off(i) + 2 * nf]; hz = z[off(i) + 2 * nf:off(i) + 2 * nf + nv]
        x_ = pc["l"] * (tt + 1) / 2
        Pd = []
        for d in range(4):
            fv, gv = ca.mtimes(ca.DM(Rp[d]), fz), ca.mtimes(ca.DM(Rp[d]), gz)
            base = ca.DM(pc["O"] + x_ * pc["e1"]) if d == 0 else (ca.DM(pc["e1"]) if d == 1 else ca.DM.zeros(3))
            Pd.append(base + fv * ca.DM(pc["e2"]) + gv * ca.DM(pc["e3"]))
        Vd = [ca.mtimes(ca.DM(Rv[d]), hz) for d in range(3)]
        return Pd, Vd

    for i in range(nP):
        k = (i + 1) % nP
        (Pa, Va), (Pb, Vb) = ends(i, 1.0), ends(k, -1.0)
        b1, b2, b3 = beta[3 * i], beta[3 * i + 1], beta[3 * i + 2]
        jc = [Va[0] - Vb[0], Va[1] - b1 * Vb[1], Va[2] - b1 ** 2 * Vb[2] - b2 * Vb[1]]
        if not fix_path:
            jc += [Pa[0] - Pb[0], Pa[1] - b1 * Pb[1], Pa[2] - b1 ** 2 * Pb[2] - b2 * Pb[1],
                   Pa[3] - b1 ** 3 * Pb[3] - 3 * b1 * b2 * Pb[2] - b3 * Pb[1]]
        for e in jc:
            gs.append(e); lbs.append(np.zeros(e.shape[0])); ubs.append(np.zeros(e.shape[0]))
    for gi, (i, xg) in enumerate(gate_at if not fix_path else []):
        pc = pcs[i]
        Rp = cheb_rows(np.array([2 * xg / pc["l"] - 1]), DEG_P, 0, 1.0)[0]
        fv = ca.mtimes(ca.DM(Rp), z[off(i):off(i) + nf]); gv = ca.mtimes(ca.DM(Rp), z[off(i) + nf:off(i) + 2 * nf])
        pg_ = ca.DM(pc["O"] + xg * pc["e1"]) + fv * ca.DM(pc["e2"]) + gv * ca.DM(pc["e3"])
        dvec = pg_ - ca.DM(gates[gi])
        gs.append(ca.dot(dvec, dvec)); lbs.append(np.zeros(1)); ubs.append(np.full(1, tol ** 2))
    g = ca.vertcat(*gs)
    # starting point
    z0 = np.zeros(nz)
    Lp = path.length
    full = len(z_start) == nz
    for i, pc in enumerate(pcs):
        if full:
            z0[off(i):off(i + 1)] = z_start[off(i):off(i + 1)]
            continue
        z0[off(i):off(i) + 2 * nf] = z_start[i * 2 * nf:(i + 1) * 2 * nf]
        Q = P0[pc["ii"]]
        t = 2 * ((Q - pc["O"]) @ pc["e1"]) / pc["l"] - 1
        sq = np.unwrap(s0[pc["ii"]], period=Lp)
        z0[off(i) + 2 * nf:off(i) + 2 * nf + nv] = C.chebfit(t, -0.5 * sigma0_fn(sq % Lp) ** 2, DEG_V)
    b0 = np.asarray(b_start, float)
    lbx = np.full(nz + 14 * N + 3 * nP, -np.inf); ubx = -lbx.copy()
    lbx[nz + 14 * N::3] = 0.2
    if fix_path:                                   # speed only: path and joint betas fixed
        for i in range(nP):
            lbx[off(i):off(i) + 2 * nf] = ubx[off(i):off(i) + 2 * nf] = z0[off(i):off(i) + 2 * nf]
        lbx[nz + 14 * N:] = ubx[nz + 14 * N:] = b0
    elif trust is not None:                        # Chebyshev |T_k| <= 1: box of trust/nf per coefficient
        for i in range(nP):
            lbx[off(i):off(i) + 2 * nf] = z0[off(i):off(i) + 2 * nf] - trust / nf
            ubx[off(i):off(i) + 2 * nf] = z0[off(i):off(i) + 2 * nf] + trust / nf
    xall = ca.vertcat(z, X, beta)
    nlp = {"x": xall, "f": ca.sum2(jS), "g": g}
    solver = ca.nlpsol("pplan", "ipopt", nlp, {"ipopt.max_iter": max_iter, "ipopt.print_level": 0, "print_time": False,
                                               "ipopt.mu_strategy": "adaptive", "ipopt.tol": ipopt_tol})
    t_build = time.perf_counter() - t_start
    t0 = time.perf_counter()
    sol = solver(x0=np.concatenate([z0, A @ z0, b0]), lbx=lbx, ubx=ubx, lbg=np.concatenate(lbs), ubg=np.concatenate(ubs))
    t_solve = time.perf_counter() - t0
    xs_all = np.array(sol["x"]).ravel()
    zs = xs_all[:nz]
    spec = []
    for i, pc in enumerate(pcs):
        spec.append(dict(O=pc["O"], e1=pc["e1"], e2=pc["e2"], e3=pc["e3"], l=pc["l"], f=zs[off(i):off(i) + nf],
                         g=zs[off(i) + nf:off(i) + 2 * nf], h=zs[off(i) + 2 * nf:off(i) + 2 * nf + nv]))
    Xs = A @ zs
    gv, jv = (np.array(y) for y in _sample_fun().map(N)(Xs.reshape(N, 14).T, frames, dxs[None, :], np.tile(par[:, None], (1, N))))
    st = solver.stats()
    return dict(spec=spec, lap=float(jv.sum()), status=st["return_status"], iters=st["iter_count"], t_build=t_build,
                t_solve=t_solve, gate_at=gate_at, zall=zs, beta=xs_all[nz + 14 * N:])


# ---------------------------------------------------------------- helpers for flying / checking
def point(sp, x):
    t = 2 * x / sp["l"] - 1
    return sp["O"] + x * sp["e1"] + C.chebval(t, sp["f"]) * sp["e2"] + C.chebval(t, sp["g"]) * sp["e3"]


def sample(spec, n=200):
    return np.vstack([np.array([point(sp, x) for x in np.linspace(0, sp["l"], n, endpoint=False)]) for sp in spec])


def state_at(sp, x):
    """p, v, a, j of the planned motion at x on piece sp."""
    t = 2 * x / sp["l"] - 1
    sc = 2 / sp["l"]
    Rp = cheb_rows(np.array([t]), DEG_P, 4, sc)
    Rv = cheb_rows(np.array([t]), DEG_V, 3, sc)
    X = np.concatenate([[R[0] @ sp["f"] for R in Rp], [R[0] @ sp["g"] for R in Rp], [R[0] @ sp["h"] for R in Rv]])
    v, a, j, _ = (np.array(y).ravel() for y in kin_fun()(X, np.concatenate([sp["e1"], sp["e2"], sp["e3"]])))
    return point(sp, x), v, a, j


# ---------------------------------------------------------------- from a given trajectory (e.g. CPC)
def dense_cpc(d, per=20):
    """Dense positions and velocities of a saved CPC trajectory (cubic Hermite between nodes); lap time."""
    x, T, N = d["x"], d["T"], int(d["N"])
    dts = np.repeat(T / N, N)
    P, V = [], []
    M = len(x)
    for k in range(M):
        p0, v0, p1, v1, h = x[k, :3], x[k, 3:6], x[(k + 1) % M, :3], x[(k + 1) % M, 3:6], dts[k]
        for s in np.arange(per) / per:
            h00, h10, h01, h11 = 2*s**3-3*s**2+1, s**3-2*s**2+s, -2*s**3+3*s**2, s**3-s**2
            P.append(h00*p0 + h10*h*v0 + h01*p1 + h11*h*v1)
            dh00, dh10, dh01, dh11 = 6*s**2-6*s, 3*s**2-4*s+1, -6*s**2+6*s, 3*s**2-2*s
            V.append((dh00*p0 + dh10*h*v0 + dh01*p1 + dh11*h*v1) / h)
    return np.array(P), np.array(V), float(T.sum())

def pieces_from(P, gates, maxang=np.deg2rad(60.0), maxlen=6.0):
    """Cut a dense closed path into graph pieces (turn <= maxang from the piece start, length <= maxlen) and fit."""
    n = len(P)
    tan = np.roll(P, -1, 0) - P; tan /= np.linalg.norm(tan, axis=1)[:, None]
    sl_ = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    cuts, start, run = [0], 0, 0.0
    for k in range(1, n):
        run += sl_[k - 1]
        if np.arccos(np.clip(tan[k] @ tan[start], -1, 1)) > maxang or run > maxlen:
            cuts.append(k); start = k; run = 0.0
    pcs = []
    for i in range(len(cuts)):
        i0, i1 = cuts[i], (cuts[(i + 1) % len(cuts)] if i + 1 < len(cuts) else n)
        ii = np.arange(i0, i1 + 1) % n
        Q = P[ii]; O = Q[0]; ch = Q[-1] - O; l = np.linalg.norm(ch); e1 = ch / l
        dev = Q - O - np.outer((Q - O) @ e1, e1); k = np.argmax(np.linalg.norm(dev, axis=1))
        e2 = dev[k] / np.linalg.norm(dev[k]); e3 = np.cross(e1, e2)
        x = (Q - O) @ e1
        if not np.all(np.diff(x) > 0): return None
        t = 2 * x / l - 1
        pcs.append(dict(O=O, e1=e1, e2=e2, e3=e3, l=l, f=C.chebfit(t, (Q - O) @ e2, DEG_P), g=C.chebfit(t, (Q - O) @ e3, DEG_P), ii=ii))
    gate_at = []
    for gt in gates:
        k = int(np.argmin(np.linalg.norm(P - gt, axis=1)))
        for i, pc in enumerate(pcs):
            if k in pc["ii"]:
                gate_at.append((i, float((P[k] - pc["O"]) @ pc["e1"]))); break
    return pcs, gate_at

def spec_of(pcs, zp):
    nf = DEG_P + 1
    return [dict(O=pc["O"], e1=pc["e1"], e2=pc["e2"], e3=pc["e3"], l=pc["l"], f=zp[i*2*nf:i*2*nf+nf], g=zp[i*2*nf+nf:(i+1)*2*nf]) for i, pc in enumerate(pcs)]


def plan_from_points(P, V, gates, motor, maxang=np.deg2rad(60.0), maxlen=6.0, tol=0.24, **kw):
    """Fit a given closed trajectory (dense P, V) with G3 piecewise polynomials, then plan only the speed on it."""
    import types
    sl = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0], np.cumsum(sl)[:-1]])
    L = sl.sum()
    spd = np.linalg.norm(V, axis=1)
    res = pieces_from(P, gates, maxang, maxlen)
    if res is None:
        return None
    pcs, gate_at = res
    t0 = time.perf_counter()
    zp, bp = smooth_path(pcs, gate_at, gates, tol)
    t_smooth = time.perf_counter() - t0
    a = dict(ds=0.25, tol=tol, lam=1e-6, w_max=15.0, eta=0.9, f_min=2.0, max_iter=1000, ipopt_tol=1e-6)
    a.update(kw)
    r = _plan(pcs, gate_at, gates, types.SimpleNamespace(length=L), P, s, motor, lambda q: np.interp(q, s, spd, period=L),
              a["ds"], a["tol"], a["lam"], a["w_max"], a["eta"], a["f_min"], a["max_iter"], a["ipopt_tol"], zp, bp,
              fix_path=True)
    r["t_smooth"] = t_smooth
    r["path_err"] = np.linalg.norm(sample(r["spec"], 300)[:, None] - P[None], axis=2).min(1).max()
    return r
