"""Speed as a function of position, with the switching points (knots) placed along the path (user, 2026-11-24):
"directly decide where to fire".

Path fixed: CPC's own path (examples/quad_time_optimal.py, full model), fitted as a periodic quintic B-spline r(u),
u ~ arc length / L (spacing 0.5 m). Only the speed is optimised:
  q(u) = u'^2 = (v / L)^2 (so V = h = -1/2 |r'|^2 q, kinetic energy = -h): periodic uniform B-spline of degree DEG,
  knots every HQ m of path. DEG 4 = C3 (h''' continuous, snap continuous: what the law needs once a planned snap
  step is allowed, user 2026-11-24); DEG 5 for comparison with the earlier experiment.
  lap = int du / sqrt(q)
Limits per sample (the same as ../2026-11-24_time_optimal_in_our_space, flatness from path + speed):
  collective thrust in [F_MIN, margin x 4 Tmax], roll/pitch torque inside the disc that fits in the X-rotor torque
  diamond (holds for any yaw: conservative vs CPC's per-rotor limits), |body rate xy| <= eta x 15 rad/s.
  q >= 0 on the control points (convex hull: q > 0 everywhere).
Checked afterwards on a 4x finer grid. Start: CPC's own speed, fitted into the spline.

    cd experiments/2026-11-24_speed_knots_in_position
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [tracks...]
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.interpolate import BSpline

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
TRACKS = [int(a) for a in sys.argv[1:]] or list(range(10))


def _load(name, path):
    argv, sys.argv = sys.argv, sys.argv[:1]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


R = _load("prev", EXP / "2026-11-24_time_optimal_in_our_space" / "run.py")   # rows (quintic), SMP, fit_ref, load_cpc
qs = R.qs
HQS = tuple(float(h) for h in os.environ.get("HQS", "2.0,1.0,0.5,0.25").split(","))
DEGS = (4, 5) if os.environ.get("LIMITS", "exact") == "approx" else (4,)   # exact: C3 only (deg 4 ~ deg 5 in the approx run)
SPS = 4
EXTRA = {"ipopt.hessian_approximation": "limited-memory"} if os.environ.get("LBFGS") else {}
K_IN = 1 / np.sqrt(2)
SETS = {"100%": (1.0, 1.0), "90%": (0.9, 0.9)}       # (thrust margin, body-rate factor)
LIMITS = os.environ.get("LIMITS", "exact")           # "exact": CPC's own limits; "approx": the first version
NY = 1.0                                             # m, knot spacing of CPC's heading h(u) (fixed, from CPC)


def exact_fun():
    """CPC's limits, exactly, from path + speed + yaw (differential flatness, Mellinger's yaw convention):
    each rotor thrust in [Tmin, margin Tmax], |w_x|, |w_y| <= eta 15, |w_z| <= 0.3 rad/s (quad_time_optimal.py:
    X rotor layout with moment arm l, torque coefficient c, inertia J). Collective floor F_MIN kept as before."""
    qd = R.qs
    import yaml
    Q = yaml.safe_load((R.ROOT / "third_party/rpg_time_optimal/quads/quad.yaml").read_text())
    m, l, cq = Q["mass"], Q["arm_length"], Q["torque_coeff"]
    J = ca.DM(np.array(Q["inertia"], float))
    Tmax, Tmin = Q["TWR_max"] * m * R.G / 4, Q["thrust_min"]
    wxy, wz = Q["omega_max_xy"], Q["omega_max_z"]
    Mix = np.array([[1, 1, 1, 1], [l, -l, -l, l], [-l, -l, l, l], [cq, -cq, cq, -cq]], float)
    Minv = ca.DM(np.linalg.inv(Mix))
    r = [ca.SX.sym(f"r{k}", 3) for k in range(1, 5)]
    qv = [ca.SX.sym(f"q{k}") for k in range(4)]
    ps = [ca.SX.sym(f"h{k}", 3) for k in range(3)]               # heading reference h, dh/du, d2h/du2
    par = ca.SX.sym("par", 2)                                     # margin, eta
    r1, r2, r3, r4 = r
    q, q1, q2, q3 = qv
    sq = ca.sqrt(ca.fmax(q, 1e-12))
    v = r1 * sq
    a = r2 * q + r1 * q1 / 2
    Bj = r3 * q + 1.5 * r2 * q1 + 0.5 * r1 * q2
    j = sq * Bj
    sn = q * (r4 * q + 2.5 * r3 * q1 + 2 * r2 * q2 + 0.5 * r1 * q3) + 0.5 * q1 * Bj
    psi, psd, psdd = ps[0], ps[1] * sq, ps[2] * q + ps[1] * q1 / 2          # heading ref and its time derivatives
    # rotation as a function of (a, h): body x = h projected off body z (no singularity at 90 deg tilt, unlike a yaw
    # angle); body rates by the chain rule (symbolic Jacobians)
    A_ = ca.SX.sym("A", 3); P_ = ca.SX.sym("P", 3); Jk = ca.SX.sym("Jk", 3); Pd = ca.SX.sym("Pd", 3)
    f = A_ + ca.DM([0, 0, R.G])
    zb = f / ca.norm_2(f)
    xb = P_ - ca.dot(P_, zb) * zb; xb = xb / ca.norm_2(xb)
    yb = ca.cross(zb, xb)
    Rm = ca.horzcat(xb, yb, zb)
    Rd = ca.reshape(ca.jtimes(ca.vec(Rm), ca.vertcat(A_, P_), ca.vertcat(Jk, Pd)), 3, 3)
    W = ca.mtimes(Rm.T, Rd)
    w = ca.vertcat(W[2, 1], W[0, 2], W[1, 0])
    wf = ca.Function("w", [A_, P_, Jk, Pd], [w, ca.norm_2(f)])
    w_, cn = wf(a, psi, j, psd)
    S_ = ca.SX.sym("S", 3); Pdd = ca.SX.sym("Pdd", 3)
    wd_expr = ca.jtimes(wf(A_, P_, Jk, Pd)[0], ca.vertcat(A_, P_, Jk, Pd), ca.vertcat(Jk, Pd, S_, Pdd))
    wdf = ca.Function("wd", [A_, P_, Jk, Pd, S_, Pdd], [wd_expr])
    wd = wdf(a, psi, j, psd, sn, psdd)
    tau = ca.mtimes(J, wd) + ca.cross(w_, ca.mtimes(J, w_))
    u = ca.mtimes(Minv, ca.vertcat(m * cn, tau))
    margin, eta = par[0], par[1]
    g = ca.vertcat(cn ** 2, ca.dot(v, v), (margin * Tmax - u) / Tmax, (u - Tmin) / Tmax,
                   1 - ca.fabs(w_[0]) / (eta * wxy), 1 - ca.fabs(w_[1]) / (eta * wxy), 1 - ca.fabs(w_[2]) / wz, q)
    X = ca.vertcat(*r, *qv, *ps)
    return ca.Function("smp_exact", [X, par], [g, 1 / sq])


EXACT = exact_fun() if LIMITS == "exact" else None
EX_LB = [R.F_MIN ** 2, 0.0] + [0.0] * 4 + [0.0] * 4 + [0.0] * 3 + [1e-6]
EX_UB = [np.inf, 3600.0] + [np.inf] * 11 + [np.inf]
EX_NAMES = ["c floor", "speed"] + [f"rotor {i + 1} hi" for i in range(4)] + [f"rotor {i + 1} lo" for i in range(4)] \
    + ["w_x", "w_y", "w_z", "q"]


def cpc_yaw(k, u_dense, L):
    """CPC's heading: its body x axis at the nodes, as a periodic quintic B-spline in u (3 components). Returns
    (coefficients, fit error in deg)."""
    d = np.load(R.ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")
    hx = []
    for w_, x_, y_, z_ in d["x"][:, 6:10]:
        hx.append([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ + w_ * z_), 2 * (x_ * z_ - w_ * y_)])
    hx = np.array(hx)
    un = u_dense[::20][:len(hx)]                      # dense_cpc: 20 samples per node
    ny = max(8, int(round(L / NY)))
    C = np.linalg.lstsq(R.rows(un, ny, 0), hx, rcond=None)[0]
    fit = R.rows(un, ny, 0) @ C
    fit /= np.linalg.norm(fit, axis=1, keepdims=True)
    err = np.degrees(np.arccos(np.clip(np.sum(fit * hx, 1), -1, 1))).max()
    return C, float(err)


def yaw_rows(C, u):
    n = len(C)
    return np.vstack([(R.rows(u, n, d) @ C).T for d in range(3)])


def rows_deg(u, n, d, deg):
    """Periodic uniform B-spline of degree deg (n control points): d-th u-derivative rows at u."""
    card = BSpline.basis_element(np.arange(deg + 2), extrapolate=False)
    f = card.derivative(d) if d else card
    x = np.asarray(u) * n
    out = np.zeros((len(x), n))
    i0 = np.floor(x).astype(int)
    for k in range(deg + 1):
        j = i0 - k
        out[np.arange(len(x)), j % n] += np.nan_to_num(f(x - j)) * n ** d
    return out


def path_derivs(Cref, u):
    n = len(Cref)
    return [R.rows(u, n, d) @ Cref for d in range(1, 5)]


def evaluate(rd, Cq, nq, deg, u, L, par, yaw=None):
    Q = [rows_deg(u, nq, d, deg) @ Cq / L ** 2 for d in range(4)]
    X = np.vstack([e.T for e in rd] + [q[None] for q in Q] + ([yaw_rows(yaw[0], u)] if EXACT else []))
    fn, lb, ub = (EXACT, EX_LB, EX_UB) if EXACT else (R.SMP, R.G_LB, R.G_UB)
    g, dt = (np.array(y) for y in fn.map(len(u))(X, np.tile(par[:, None], (1, len(u)))))
    viol = np.maximum(np.array(lb)[:, None] - g, g - np.array(ub)[:, None])
    return g, dt.ravel(), viol


def solve(Cref, L, nq, deg, Cq0, par, max_iter=3000, yaw=None):
    nref = len(Cref)
    M = SPS * max(nref, nq)
    u = (np.arange(M) + 0.5) / M
    rd = path_derivs(Cref, u)
    ny = len(yaw[0]) if EXACT else 0
    z = ca.SX.sym("z", nq + 3 * ny)
    Q = [ca.mtimes(ca.DM(rows_deg(u, nq, d, deg) / L ** 2), z[:nq]) for d in range(4)]
    pen = 0
    if EXACT:   # heading: a variable too (CPC optimises its yaw; |w_z| <= 0.3 is active on CPC)
        Ch = ca.reshape(z[nq:], ny, 3)
        Hr = [ca.mtimes(ca.DM(R.rows(u, ny, d)), Ch).T for d in range(3)]
        pen = 1e-4 * ca.sumsqr(ca.sum2(Ch ** 2) - 1)        # fixes the free scale of the heading reference
    X = ca.vertcat(*[ca.DM(e.T) for e in rd], *[q.T for q in Q], *(Hr if EXACT else []))
    fn, lb, ub = (EXACT, EX_LB, EX_UB) if EXACT else (R.SMP, R.G_LB, R.G_UB)
    g, dt = fn.map(M)(X, ca.repmat(ca.DM(par), 1, M))
    S = ca.nlpsol("lap", "ipopt", {"x": z, "f": ca.sum2(dt) / M + pen, "g": ca.vec(g)},
                  {"ipopt.print_level": int(os.environ.get("PL", 0)), "print_time": False, "ipopt.max_iter": max_iter, **EXTRA,
                   "ipopt.mu_strategy": "adaptive", "ipopt.tol": 1e-8})
    t0 = time.perf_counter()
    lbx = np.r_[np.full(nq, 1e-6 * L ** 2), np.full(3 * ny, -np.inf)]
    sol = S(x0=np.r_[Cq0, yaw[0].T.ravel() if EXACT else []], lbx=lbx, ubx=np.inf,
            lbg=np.tile(lb, M), ubg=np.tile(ub, M))
    t = time.perf_counter() - t0
    st = S.stats()
    zs = np.array(sol["x"]).ravel()
    Cq = zs[:nq]
    yaw0 = yaw
    if EXACT:
        yaw = (zs[nq:].reshape(3, ny).T,)
    Mf = 4 * M
    uf = (np.arange(Mf) + 0.5) / Mf
    _, dtf, viol = evaluate(path_derivs(Cref, uf), Cq, nq, deg, uf, L, par, yaw)
    _, _, viol0 = evaluate(path_derivs(Cref, uf), Cq0, nq, deg, uf, L, par, yaw0)
    names = EX_NAMES if EXACT else ["c floor", "speed", "body rate", "thrust hi", "thrust lo", "torque hi", "torque lo", "q"]
    w = int(np.unravel_index(np.argmax(viol), viol.shape)[0])
    return dict(Cq=Cq, Ch=yaw[0] if EXACT else None, status=st["return_status"], iters=st["iter_count"], t=t, lap=float(sol["f"]),
                lap_fine=float(dtf.mean()), viol=float(viol.max()), worst=names[w], start_viol=float(viol0.max()))


def main():
    quad = qs.Quad()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"speed knots along the path: CPC's path fixed, q = (v/L)^2 periodic B-spline, degrees {DEGS}, "
        f"spacings {HQS} m, limits {LIMITS}, tracks {TRACKS}")
    summary = []
    for k in TRACKS:
        P, V, lap_cpc, gates = R.load_cpc(k)
        L = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum()
        nref = int(round(L / 0.5))
        Cref, u, qL2, L, err = R.fit_ref(P, V, nref)
        yaw = cpc_yaw(k, u, L) if EXACT else None
        log(f"track {k}: CPC {lap_cpc:.3f} s, length {L:.1f} m, path fit {err * 1000:.2f} mm"
            + (f", CPC heading fit {yaw[1]:.2f} deg" if EXACT else ""))
        save = dict(Cref=Cref, L=L, lap_cpc=lap_cpc, gates=gates)
        for sname, (margin, eta) in SETS.items():
            par = np.array([margin, eta]) if EXACT else \
                np.array([quad.m, quad.J[0, 0], 0.15, quad.Tmax, quad.Tmin, margin, 15.0, eta, K_IN])
            for deg in DEGS:
                for hq in HQS:
                    nq = max(8, int(round(L / hq)))
                    Cq0 = np.linalg.lstsq(rows_deg(u, nq, 0, deg), qL2, rcond=None)[0]
                    r = solve(Cref, L, nq, deg, Cq0, par, yaw=yaw)
                    ok = r["status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
                    gap = 100 * (r["lap_fine"] / lap_cpc - 1)
                    log(f"  {sname:4s} deg {deg} knots {L / nq:4.2f} m ({nq:3d} vars): {r['status'][:22]:22s} "
                        f"{r['iters']:4d} it {r['t']:6.1f} s | lap {r['lap']:.3f}, 4x finer {r['lap_fine']:.3f} "
                        f"({gap:+5.1f}%) | worst limit {r['viol']:+.4f} ({r['worst']}) | start worst {r['start_viol']:+.3f}")
                    summary.append((k, sname, deg, hq, gap, ok, r["viol"]))
                    save[f"Cq_{sname[:-1]}_{deg}_{hq}"] = r["Cq"]
                    if EXACT:
                        save[f"Ch_{sname[:-1]}_{deg}_{hq}"] = r["Ch"]
        (HERE / "results" / LIMITS).mkdir(exist_ok=True)
        np.savez(HERE / "results" / LIMITS / f"track{k}.npz", **save)
    log("\nsummary: track | set | degree | knot spacing m | lap vs CPC | solver ok | worst limit (4x finer)")
    for k, s, d, h, gap, ok, v in summary:
        log(f"{k} | {s} | {d} | {h} | {gap:+.1f}% | {'ok' if ok else 'FAIL'} | {v:+.4f}")
    (HERE / "results" / f"run_{LIMITS}_{'_'.join(map(str, TRACKS))}.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
