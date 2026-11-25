"""Time-optimal lap inside our space, written the way CPC is written (user, 2026-11-25): the quantities between the
path and the rotors are variables, tied by equalities that only multiply (no division by the collective thrust).

Planning space (fixed parameter count):
  path   p(u) = r_ref(u) + alpha(u) n1(u) + beta(u) n2(u), u in [0, 1) periodic (sideways offsets only; r_ref = a
         reference lap, n1, n2 its rotation-minimising frame), alpha, beta periodic quintic B-splines, spacing HA m
  speed  q(u) = u'^2 periodic quintic B-spline, spacing HQ m (speed is a function of position)
  heading reference h(u) (3D) periodic quintic B-spline, spacing HH m (body x lies in span(h, body z))
Per sample i (lifted, as CPC's states): body axes x_b, y_b (z_b = x_b x y_b), collective thrust c and its first two
time derivatives, body rates w and their derivative, the 4 rotor thrusts, s = u' (= sqrt(q)).
Equalities per sample (polynomial):
  x.x = 1, y.y = 1, x.y = 0
  c z = a + g e3
  c' z + c R(w x e3) = j
  c'' z + 2 c' R(w x e3) + c R(w' x e3 + w x (w x e3)) = snap
  y.h = 0 and its first two time derivatives (heading)
  m c = sum(u),  J w' + w x J w = tau(u)        (quad_time_optimal / rpg_time_optimal X layout)
  s^2 = q
  with a, j, snap from the path and speed spline coefficients (linear in them per sample, after lifting r1..r4, q0..q3).
Bounds (the same as CPC): T_min <= u_i <= T_max, |w_x|, |w_y| <= 15, |w_z| <= 0.3, gates within TOL.
Objective: lap = sum_i du / s_i.
Checked afterwards on a 4x finer grid with the exact flatness map (../2026-11-24_speed_knots_in_position).

    cd experiments/2026-11-25_lifted_our_space
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [tracks...]
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
import yaml

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


R = _load("prev", EXP / "2026-11-24_time_optimal_in_our_space" / "run.py")    # rows, fit_ref, rm_frame, Problem
os.environ["LIMITS"] = "exact"
K = _load("knots", EXP / "2026-11-24_speed_knots_in_position" / "run.py")     # exact flatness check, CPC heading
rows = R.rows
G = 9.81
TOL = 0.24
HQ = float(os.environ.get("HQ", 1.0))
HA = float(os.environ.get("HA", 1.0))
HH = float(os.environ.get("HH", 1.0))
SPS = int(os.environ.get("SPS", 2))
TRUST = 0.5
ROUNDS = int(os.environ.get("ROUNDS", 4))
SETS = {"100%": (1.0, 1.0), "90%": (0.9, 0.9)}
SET_NAMES = os.environ.get("SETS", "100%").split(",")

Q = yaml.safe_load((R.ROOT / "third_party/rpg_time_optimal/quads/quad.yaml").read_text())
MASS, ARM, CQ = Q["mass"], Q["arm_length"], Q["torque_coeff"]
JD = np.array(Q["inertia"], float)
TMAX, TMIN = Q["TWR_max"] * MASS * G / 4, Q["thrust_min"]
WXY, WZ = Q["omega_max_xy"], Q["omega_max_z"]
NY = 20   # lifted variables per sample


def tau_of(u):
    return ca.vertcat(ARM * (u[0] - u[1] - u[2] + u[3]), ARM * (-u[0] - u[1] + u[2] + u[3]),
                      CQ * (u[0] - u[1] + u[2] - u[3]))


def eq_fun():
    X = ca.SX.sym("X", 16)            # r1..r4 (3 each), q0..q3 (u-derivatives of u'^2)
    H = ca.SX.sym("H", 9)             # h, h_u, h_uu
    Y = ca.SX.sym("Y", NY)
    r1, r2, r3, r4 = X[0:3], X[3:6], X[6:9], X[9:12]
    q, q1, q2, q3 = X[12], X[13], X[14], X[15]
    h, hu, huu = H[0:3], H[3:6], H[6:9]
    xb, yb = Y[0:3], Y[3:6]
    c, cd, cdd = Y[6], Y[7], Y[8]
    w, wd, u, s = Y[9:12], Y[12:15], Y[15:19], Y[19]
    zb = ca.cross(xb, yb)
    a = r2 * q + r1 * q1 / 2
    Bj = r3 * q + 1.5 * r2 * q1 + 0.5 * r1 * q2
    j = s * Bj
    sn = q * (r4 * q + 2.5 * r3 * q1 + 2 * r2 * q2 + 0.5 * r1 * q3) + 0.5 * q1 * Bj
    hd = hu * s
    hdd = huu * q + hu * q1 / 2

    def world(b):
        return b[0] * xb + b[1] * yb + b[2] * zb
    wx, wy, wz = w[0], w[1], w[2]
    W3 = world(ca.vertcat(wy, -wx, 0))                                   # R (w x e3)
    A3 = world(ca.vertcat(wd[1] + wx * wz, -wd[0] + wy * wz, -(wx ** 2 + wy ** 2)))   # R (w' x e3 + w x (w x e3))
    W2 = world(ca.vertcat(-wz, 0, wx))                                   # R (w x e2)
    A2 = world(ca.vertcat(-wd[2] + wx * wy, -(wx ** 2 + wz ** 2), wd[0] + wy * wz))   # R (w' x e2 + w x (w x e2))
    Jm = ca.DM(JD)
    e = ca.vertcat(
        ca.dot(xb, xb) - 1, ca.dot(yb, yb) - 1, ca.dot(xb, yb),
        c * zb - a - ca.DM([0, 0, G]),
        cd * zb + c * W3 - j,
        cdd * zb + 2 * cd * W3 + c * A3 - sn,
        ca.dot(yb, h), ca.dot(W2, h) + ca.dot(yb, hd), ca.dot(A2, h) + 2 * ca.dot(W2, hd) + ca.dot(yb, hdd),
        MASS * c - ca.sum1(u),
        ca.mtimes(Jm, wd) + ca.cross(w, ca.mtimes(Jm, w)) - tau_of(u),
        s ** 2 - q)
    return ca.Function("eq", [X, H, Y], [e])


EQ = eq_fun()


def init_fun():
    """Flatness (with division) only to build the starting guess of the lifted variables."""
    X = ca.SX.sym("X", 16)
    H = ca.SX.sym("H", 9)
    r1, r2, r3, r4 = X[0:3], X[3:6], X[6:9], X[9:12]
    q, q1, q2, q3 = X[12], X[13], X[14], X[15]
    s = ca.sqrt(ca.fmax(q, 1e-12))
    a = r2 * q + r1 * q1 / 2
    Bj = r3 * q + 1.5 * r2 * q1 + 0.5 * r1 * q2
    j = s * Bj
    sn = q * (r4 * q + 2.5 * r3 * q1 + 2 * r2 * q2 + 0.5 * r1 * q3) + 0.5 * q1 * Bj
    hd, hdd = H[3:6] * s, H[6:9] * q + H[3:6] * q1 / 2
    A_, P_, Jk, Pd, S_, Pdd = (ca.SX.sym(n, 3) for n in ("A", "P", "J", "Pd", "S", "Pdd"))
    f = A_ + ca.DM([0, 0, G])
    cn = ca.norm_2(f)
    zb = f / cn
    xb = P_ - ca.dot(P_, zb) * zb
    xb = xb / ca.norm_2(xb)
    yb = ca.cross(zb, xb)
    Rm = ca.horzcat(xb, yb, zb)
    Rd = ca.reshape(ca.jtimes(ca.vec(Rm), ca.vertcat(A_, P_), ca.vertcat(Jk, Pd)), 3, 3)
    W = ca.mtimes(Rm.T, Rd)
    w = ca.vertcat(W[2, 1], W[0, 2], W[1, 0])
    cd = ca.jtimes(cn, A_, Jk)
    wf = ca.Function("w", [A_, P_, Jk, Pd], [w, cd])
    wv, cdv = wf(a, H[0:3], j, hd)
    full = ca.vertcat(A_, P_, Jk, Pd)
    dfull = ca.vertcat(Jk, Pd, S_, Pdd)
    wd_e = ca.jtimes(wf(A_, P_, Jk, Pd)[0], full, dfull)
    cdd_e = ca.jtimes(wf(A_, P_, Jk, Pd)[1], full, dfull)
    dfun = ca.Function("d", [A_, P_, Jk, Pd, S_, Pdd], [wd_e, cdd_e])
    wdv, cddv = dfun(a, H[0:3], j, hd, sn, hdd)
    xv, yv, cv = ca.Function("R", [A_, P_], [xb, yb, cn])(a, H[0:3])
    Jm = ca.DM(JD)
    tau = ca.mtimes(Jm, wdv) + ca.cross(wv, ca.mtimes(Jm, wv))
    Mix = np.array([[1, 1, 1, 1], [ARM, -ARM, -ARM, ARM], [-ARM, -ARM, ARM, ARM], [CQ, -CQ, CQ, -CQ]], float)
    u = ca.mtimes(ca.DM(np.linalg.inv(Mix)), ca.vertcat(MASS * cv, tau))
    return ca.Function("init", [X, H], [ca.vertcat(xv, yv, cv, cdv, cddv, wv, wdv, u, s)])


INIT = init_fun()


def solve(ref, gates, ug, Ch0, Cq0, na, nq, margin, eta):
    Cref, nref, CN1, CN2, L = ref
    nh = len(Ch0)
    pb = R.Problem(ref, na, nq, gates, ug, R.qs.Quad(), margin, eta, 1.0)
    M = SPS * max(nref, na, nq)
    us = (np.arange(M) + 0.5) / M
    pb.us, pb.M, pb.du = us, M, 1.0 / M
    pb.Rref = [rows(us, nref, d) @ Cref for d in range(5)]
    pb.N1 = [rows(us, nref, d) @ CN1 for d in range(5)]
    pb.N2 = [rows(us, nref, d) @ CN2 for d in range(5)]
    A, b0 = pb.lift_map()                                  # X = b0 + A [alpha; beta; q coefficients]
    nz = 2 * na + nq
    Hrows = [rows(us, nh, d) for d in range(3)]
    z = ca.MX.sym("z", nz)
    ch = ca.MX.sym("ch", 3 * nh)                           # heading coefficients (nh x 3, column-major)
    Ys = ca.MX.sym("Y", NY * M)
    Xm = ca.reshape(ca.DM(b0) + ca.mtimes(ca.sparsify(ca.DM(A)), z), 16, M)
    Chm = ca.reshape(ch, nh, 3)
    Hm = ca.vertcat(*[ca.mtimes(ca.DM(Hd), Chm).T for Hd in Hrows])        # 9 x M
    Ym = ca.reshape(Ys, NY, M)
    E = EQ.map(M)(Xm, Hm, Ym)
    J = ca.sum2(1.0 / Ym[19, :]) / M
    Rg = ca.DM(rows(ug, na, 0))
    ag, bg = ca.mtimes(Rg, z[:na]), ca.mtimes(Rg, z[na:2 * na])
    B0 = rows(ug, nref, 0)
    pg = ca.DM(B0 @ Cref) + ca.repmat(ag, 1, 3) * ca.DM(B0 @ CN1) + ca.repmat(bg, 1, 3) * ca.DM(B0 @ CN2)
    gg = ca.sum2((pg - ca.DM(gates)) ** 2)
    xall = ca.vertcat(z, ch, Ys)
    S = ca.nlpsol("lap", "ipopt", {"x": xall, "f": J, "g": ca.vertcat(ca.vec(E), gg)},
                  {"expand": True, "print_time": False,
                   "ipopt": {"print_level": int(os.environ.get("PL", 0)), "max_iter": 3000, "tol": 1e-7,
                             "mu_strategy": "adaptive"}})
    # bounds on the lifted variables (CPC's)
    ylb = np.full(NY, -np.inf); yub = np.full(NY, np.inf)
    ylb[9:11], yub[9:11] = -eta * WXY, eta * WXY
    ylb[11], yub[11] = -WZ, WZ
    ylb[15:19], yub[15:19] = TMIN, margin * TMAX
    ylb[19] = 1e-3
    # start
    X0 = (b0 + A @ np.r_[np.zeros(2 * na), Cq0]).reshape(M, 16).T
    H0 = np.vstack([(Hd @ Ch0).T for Hd in Hrows])
    Y0 = np.array(INIT.map(M)(X0, H0))
    Y0[9:12] = np.clip(Y0[9:12], ylb[9:12, None], yub[9:12, None])
    Y0[15:19] = np.clip(Y0[15:19], ylb[15:19, None], yub[15:19, None])
    zc, chc, yc = np.r_[np.zeros(2 * na), Cq0], Ch0.T.ravel(), Y0.T.ravel()
    lbg = np.r_[np.zeros(NY * M), np.zeros(len(gates))]
    ubg = np.r_[np.zeros(NY * M), np.full(len(gates), TOL ** 2)]
    hist, best, t = [], None, 0.0
    for rd in range(ROUNDS):
        lbx = np.r_[zc[:2 * na] - TRUST, np.full(nq, 1e-6 * L ** 2), np.full(3 * nh, -np.inf), np.tile(ylb, M)]
        ubx = np.r_[zc[:2 * na] + TRUST, np.full(nq, np.inf), np.full(3 * nh, np.inf), np.tile(yub, M)]
        t0 = time.perf_counter()
        sol = S(x0=np.r_[zc, chc, yc], lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        t += time.perf_counter() - t0
        st = S.stats()
        xs = np.array(sol["x"]).ravel()
        lap = float(sol["f"])
        ok = st["return_status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        hist.append((st["return_status"], st["iter_count"], lap))
        if not ok:
            break
        improved = best is None or lap < best[1] - 1e-4
        if best is None or lap < best[1]:
            best = (xs, lap)
        zc, chc, yc = xs[:nz], xs[nz:nz + 3 * nh], xs[nz + 3 * nh:]
        if not improved:
            break
    if best is None:
        best = (xs, float(sol["f"]))
    xs, lap = best
    Ca = np.column_stack([xs[:na], xs[na:2 * na]])
    Cq = xs[2 * na:nz]
    Ch = xs[nz:nz + 3 * nh].reshape(3, nh).T
    # exact check on a 4x finer grid (flatness map of ../2026-11-24_speed_knots_in_position, without its c floor)
    Mf = 4 * M
    uf = (np.arange(Mf) + 0.5) / Mf
    from math import comb
    Al = [rows(uf, na, d) @ Ca[:, 0] for d in range(5)]
    Be = [rows(uf, na, d) @ Ca[:, 1] for d in range(5)]
    rd_ = []
    for d in range(1, 5):
        e_ = rows(uf, nref, d) @ Cref
        for k in range(d + 1):
            e_ = e_ + comb(d, k) * (Al[k][:, None] * (rows(uf, nref, d - k) @ CN1) + Be[k][:, None] * (rows(uf, nref, d - k) @ CN2))
        rd_.append(e_)
    Xf = np.vstack([e_.T for e_ in rd_] + [rows(uf, nq, d) @ Cq / L ** 2 for d in range(4)]
                   + [(rows(uf, nh, d) @ Ch).T for d in range(3)])
    gf, dtf = (np.array(y) for y in K.EXACT.map(Mf)(Xf, np.tile(np.array([[margin], [eta]]), (1, Mf))))
    viol = np.maximum(np.array(K.EX_LB)[:, None] - gf, gf - np.array(K.EX_UB)[:, None])[1:]   # drop the c floor row
    names = K.EX_NAMES[1:]
    wi = int(np.unravel_index(np.argmax(viol), viol.shape)[0])
    gate = float(max(np.linalg.norm(pb.points(Ca, np.linspace(0, 1, 20000, endpoint=False)) - g_, axis=1).min()
                     for g_ in gates))
    good = [h_ for h_ in hist if h_[0] in ("Solve_Succeeded", "Solved_To_Acceptable_Level")]
    return dict(Ca=Ca, Cq=Cq, Ch=Ch, lap=lap, lap_fine=float(dtf.mean()), viol=float(viol.max()), worst=names[wi],
                gate=gate, t=t, status=(good[-1] if good else hist[-1])[0], iters=sum(h_[1] for h_ in hist),
                rounds=" / ".join(f"{h_[0][:5]} {h_[1]} it {h_[2]:.3f}" for h_ in hist), M=M,
                nvar=nz + 3 * nh, nlift=NY * M, ok=bool(good))


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"lifted (CPC-style) optimisation in our space: offsets {HA} m, speed {HQ} m, heading {HH} m, "
        f"{SPS} samples per span, trust {TRUST} m x {ROUNDS} rounds, gate tol {TOL} m, tracks {TRACKS}")
    for k in TRACKS:
        P, V, lap_ref, gates = R.load_cpc(k)                 # starting point: the saved CPC lap (only a start)
        L = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum()
        nref = int(round(L / 0.5))
        Cref, u, qL2, L, err = R.fit_ref(P, V, nref)
        CN1, CN2, _ = R.rm_frame(Cref, nref)
        ref = (Cref, nref, CN1, CN2, L)
        ug = np.array([u[np.argmin(np.linalg.norm(P - g_, axis=1))] for g_ in gates])
        Ch0, herr = K.cpc_yaw(k, u, L) if HH == K.NY else _heading(k, u, L)
        na, nq = int(round(L / HA)), int(round(L / HQ))
        Cq0 = np.linalg.lstsq(rows(u, nq, 0), qL2, rcond=None)[0]
        log(f"track {k}: start lap {lap_ref:.3f} s (saved CPC lap, start only), length {L:.1f} m, path fit {err * 1000:.2f} mm, "
            f"heading fit {herr:.2f} deg")
        res = {}
        for sname in SET_NAMES:
            margin, eta = SETS[sname]
            r = solve(ref, gates, ug, Ch0, Cq0, na, nq, margin, eta)
            res[sname] = r
            log(f"  {sname:4s} ({r['nvar']} parameters + {r['nlift']} lifted, {r['M']} samples): {r['status'][:22]:22s} "
                f"{r['iters']:4d} it {r['t']:6.1f} s | lap {r['lap']:.3f}, 4x finer exact {r['lap_fine']:.3f} | "
                f"worst limit (4x finer, exact) {r['viol']:+.4f} ({r['worst']}) | gates <= {r['gate']:.3f} m | "
                f"rounds: {r['rounds']}")
        (HERE / "results").mkdir(exist_ok=True)
        np.savez(HERE / "results" / f"track{k}.npz", Cref=Cref, CN1=CN1, CN2=CN2, L=L, gates=gates,
                 **{f"{c_}_{s_[:-1]}": res[s_][c_] for s_ in res for c_ in ("Ca", "Cq", "Ch")})
    with open(HERE / "results" / "run.log", "a") as f:
        f.write("\n".join(lines) + "\n")


def _heading(k, u, L):
    K.NY = HH
    return K.cpc_yaw(k, u, L)


if __name__ == "__main__":
    main()
