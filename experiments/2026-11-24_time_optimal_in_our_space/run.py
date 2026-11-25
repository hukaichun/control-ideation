"""How close to time-optimal (CPC) can our representation get, when the lap is optimised inside our space instead of
fitting CPC's path into it (user, 2026-11-24)?

Our space (what the blended E-L law needs, experiments/2026-11-24_blended_bspline_pieces):
  path  r(u) = r_ref(u) + alpha(u) n1(u) + beta(u) n2(u), u in [0, 1) periodic: r_ref = CPC's path as a uniform
        quintic B-spline (spacing 0.5 m), n1, n2 its rotation-minimising frame (fitted as quintic B-splines),
        alpha, beta quintic B-splines (spacing max(HQ, 1 m)): only sideways changes of the path (a first version with
        free 3D control points could slide along the path without changing it: degenerate, did not converge)
  speed u' = sqrt(q(u)): q periodic uniform quintic B-spline, spacing ~ HQ m; speed is a function of position
        (V = -1/2 |r'|^2 q), as the law requires
  d/dt = sqrt(q) d/du:  v = r1 sqrt(q),  a = r2 q + r1 q1 / 2,  j = sqrt(q) (r3 q + 1.5 r2 q1 + 0.5 r1 q2),
  snap = q (r4 q + 2.5 r3 q1 + 2 r2 q2 + 0.5 r1 q3) + 0.5 q1 (r3 q + 1.5 r2 q1 + 0.5 r1 q2)
  lap  = int du / sqrt(q)
Limits per sample (same formulas as poly_plan's speed plan): collective thrust <= margin * 4 Tmax, >= 4 Tmin,
roll/pitch torque within what the remaining thrust allows, |body rate xy| <= eta * w_max (CPC: 15 rad/s), speed <= 60.
Gates: |r(u_g) - gate| <= 0.24 m (CPC's tolerance for these trajectories), u_g fixed at the start.
Steps (gap split, record section 17):
  A   path fixed (CPC's path), only q: V-space loss vs knot spacing HQ
  B   path offsets and q together: path-space loss (best path in our space, not CPC's)
  B*  as B with the torque limit as the disc inside the X-rotor torque diamond (holds for any yaw; poly_plan's
      disc is up to sqrt(2) looser than per-rotor limits)
  C*  as B* with 90% thrust and body rates <= 0.9 x 15 (reserve)
CPC: examples/quad_time_optimal.py, full model, per-rotor thrust limits, body rates <= 15 (xy) / 0.3 (z) rad/s.

    cd experiments/2026-11-24_time_optimal_in_our_space
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [tracks...]
"""
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.interpolate import BSpline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import poly_plan as pp      # noqa: E402
import quad_sim as qs       # noqa: E402

TRACKS = [int(a) for a in sys.argv[1:]] or list(range(10))
TOL = 0.24
HQS = (2.0, 1.0, 0.5)          # 0.25 m: one solve ran > 1 h without finishing (dropped)
HR_A = 0.5                   # path spacing for step A (fixed CPC path)
DEG = 5
EXTRA = {}
TRUST = 0.5                  # m, sideways offsets move at most this much per round
ROUNDS = 6
SPS = 4                      # constraint samples per span of the finer spline
G = 9.81
# previous results (record section 11: CPC path fitted into G3 Chebyshev pieces, speed only, loose torque model):
# (no body-rate limit, 100%) and (body rate <= 0.9 x 15, 100%)
PREV = {0: (2.6, 2.6), 1: (3.6, 4.2), 2: (10.0, 11.2), 3: (8.2, 8.3), 4: (6.5, 7.1), 5: None, 6: None, 7: (5.4, 6.8),
        8: (9.1, 11.4), 9: (8.0, 10.2)}
K_IN = 1 / np.sqrt(2)
# step: (thrust margin, body-rate factor eta, free path, torque factor, warm start from step)
# start: CPC of a restricted vehicle (cpc_restricted.py: 90% thrust, |w_x|, |w_y| <= 0.9 x 15 / sqrt(2)), which keeps a
# margin to our 100% limits (user, 2026-11-24). Starting from the full CPC was infeasible in our model on 19-37% of
# the samples (smoothed bang-bang thrust over the limit, per-axis vs magnitude body-rate limit) and diverged.
import os
SET = os.environ.get("SET", "100")
# step: (thrust margin, body-rate factor, free path, torque factor, start = restricted CPC file tag)
STEPS = {"100": {"ours 100%": (1.0, 1.0, True, 1.0, ""), "ours 100%, safe torque": (1.0, 1.0, True, K_IN, "")},
         "90": {"ours 90%, safe torque": (0.9, 0.9, True, K_IN, "_0.8")}}[SET]   # 90%: started from the 80% CPC

_card = BSpline.basis_element(np.arange(DEG + 2), extrapolate=False)
_cardd = [_card] + [_card.derivative(d) for d in range(1, 5)]


def rows(u, n, d):
    """Periodic uniform B-spline (n control points) d-th u-derivative rows at u (len(u) x n)."""
    x = np.asarray(u) * n
    R = np.zeros((len(x), n))
    i0 = np.floor(x).astype(int)
    for k in range(DEG + 1):
        j = i0 - k
        val = _cardd[d](x - j)
        R[np.arange(len(x)), j % n] += np.nan_to_num(val) * n ** d
    return R


def sample_fun():
    r = [ca.SX.sym(f"r{k}", 3) for k in range(1, 5)]
    qv = [ca.SX.sym(f"q{k}") for k in range(4)]
    par = ca.SX.sym("par", 9)
    m, Ix, l, Tmax, Tmin, margin, w_max, eta, kfac = (par[i] for i in range(9))
    r1, r2, r3, r4 = r
    q, q1, q2, q3 = qv
    sq = ca.sqrt(ca.fmax(q, 1e-9))
    v = r1 * sq
    a = r2 * q + r1 * q1 / 2
    Bj = r3 * q + 1.5 * r2 * q1 + 0.5 * r1 * q2
    j = sq * Bj
    sn = q * (r4 * q + 2.5 * r3 * q1 + 2 * r2 * q2 + 0.5 * r1 * q3) + 0.5 * q1 * Bj
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
    k_ = l / np.sqrt(2) * kfac             # kfac 1: |tau| <= k (4 Tmax - Tc) as in poly_plan (loose, any yaw best case);
    hi = 4 * margin * Tmax - Tc             # kfac 1/sqrt(2): disc inside the X-rotor torque diamond (any yaw)
    lo = Tc - 4 * Tmin
    scl = (4 * Tmax * k_) ** 2
    g = ca.vertcat(c2, ca.dot(v, v), jperp2 / ((eta * w_max) ** 2 * c2), hi / Tmax, lo / Tmax,
                   ((k_ * hi) ** 2 - tau2) / scl, ((k_ * lo) ** 2 - tau2) / scl, q)
    X = ca.vertcat(*r, *qv)
    return ca.Function("smp", [X, par], [g, 1 / sq])


SMP = sample_fun()
F_MIN = 2.0                  # m/s^2, collective thrust floor as in poly_plan (body rate = |j_perp| / c blows up near c = 0;
                             # without it solutions reached c = 0.02 and broke the limits between the samples)
G_LB = [F_MIN ** 2, 0.0, -np.inf, 0, 0, 0, 0, 1e-6]
G_UB = [np.inf, 3600.0, 1.0, np.inf, np.inf, np.inf, np.inf, np.inf]


def load_cpc(k, restricted=False, tag=""):
    d = np.load(HERE / "results" / f"cpc_restricted{tag}_{k}.npz" if restricted else ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")
    P, V, lap = pp.dense_cpc(d)
    return P, V, lap, d["gates"]


def fit_ref(P, V, nref):
    """CPC path as a periodic quintic B-spline r_ref(u) (u ~ arc length / L); the speed as q * L^2 samples."""
    sl = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.r_[0, np.cumsum(sl)[:-1]]
    L = sl.sum()
    u = s / L
    Cref = np.linalg.lstsq(rows(u, nref, 0), P, rcond=None)[0]
    err = np.linalg.norm(rows(u, nref, 0) @ Cref - P, axis=1).max()
    qL2 = np.sum(V ** 2, 1) / np.sum((rows(u, nref, 1) @ Cref) ** 2, 1) * L ** 2
    return Cref, u, qL2, L, err


def rm_frame(Cref, nref, n=4000):
    """Rotation-minimising frame (double reflection) of the closed reference, the closure twist spread evenly,
    fitted as periodic quintic B-splines (smooth, with derivatives): offset directions n1, n2."""
    u = np.arange(n) / n
    p = rows(u, nref, 0) @ Cref
    t = rows(u, nref, 1) @ Cref
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    r0 = np.cross(t[0], [0, 0, 1.0])
    if np.linalg.norm(r0) < 0.3:
        r0 = np.cross(t[0], [1.0, 0, 0])
    R = [r0 / np.linalg.norm(r0)]
    for i in range(n):                                  # one more step closes the loop
        a, b = i, (i + 1) % n
        v1 = p[b] - p[a]
        c1 = v1 @ v1
        rL = R[-1] - 2 / c1 * (v1 @ R[-1]) * v1
        tL = t[a] - 2 / c1 * (v1 @ t[a]) * v1
        v2 = t[b] - tL
        c2 = v2 @ v2
        R.append(rL - 2 / c2 * (v2 @ rL) * v2 if c2 > 1e-18 else rL)
    R = np.array(R)
    end, start = R[-1], R[0]
    ang = np.arctan2(np.cross(start, end) @ t[0], start @ end)      # closure twist about the tangent
    R = R[:-1]
    B_ = np.cross(t, R)
    th = -ang * u
    n1 = np.cos(th)[:, None] * R + np.sin(th)[:, None] * B_
    n2 = np.cross(t, n1)
    A0 = rows(u, nref, 0)
    return np.linalg.lstsq(A0, n1, rcond=None)[0], np.linalg.lstsq(A0, n2, rcond=None)[0], float(np.degrees(ang))


class Problem:
    """Path r = r_ref + alpha n1 + beta n2 (alpha, beta: periodic quintic B-splines, spacing ~ L / na): only sideways
    changes, so the parametrisation cannot slide along the path (that freedom made the first version degenerate)."""

    def __init__(self, ref, na, nq, gates, ug, quad, margin, eta, kfac, w_max=15.0):
        self.Cref, self.nref, self.CN1, self.CN2, self.L = ref
        self.na, self.nq = na, nq
        M = SPS * max(self.nref, na, nq)
        us = (np.arange(M) + 0.5) / M
        self.us, self.M, self.du = us, M, 1.0 / M
        self.Rref = [rows(us, self.nref, d) @ self.Cref for d in range(5)]          # M x 3, d = 0..4
        self.N1 = [rows(us, self.nref, d) @ self.CN1 for d in range(5)]
        self.N2 = [rows(us, self.nref, d) @ self.CN2 for d in range(5)]
        self.Ra = [ca.sparsify(ca.DM(rows(us, na, d))) for d in range(5)]
        self.Rq = [ca.sparsify(ca.DM(rows(us, nq, d))) for d in range(4)]
        self.ug = ug
        self.par = np.array([quad.m, quad.J[0, 0], 0.15, quad.Tmax, quad.Tmin, margin, w_max, eta, kfac])
        self.gates = gates

    def path_derivs(self, ca_al, ca_be, lib=ca):
        """r^(d), d = 1..4 (M x 3) for alpha, beta coefficient vectors (Leibniz rule)."""
        from math import comb
        out = []
        A = [ca.mtimes(R, ca_al) for R in self.Ra]
        Bt = [ca.mtimes(R, ca_be) for R in self.Ra]
        for d in range(1, 5):
            e = ca.DM(self.Rref[d])
            for k in range(d + 1):
                e = e + comb(d, k) * (ca.repmat(A[k], 1, 3) * ca.DM(self.N1[d - k]) + ca.repmat(Bt[k], 1, 3) * ca.DM(self.N2[d - k]))
            out.append(e)
        return out

    def points(self, Ca, u):
        """Path positions at u (numpy)."""
        al, be = rows(u, self.na, 0) @ Ca[:, 0], rows(u, self.na, 0) @ Ca[:, 1]
        return rows(u, self.nref, 0) @ self.Cref + al[:, None] * (rows(u, self.nref, 0) @ self.CN1) + be[:, None] * (rows(u, self.nref, 0) @ self.CN2)

    def lift_map(self):
        """X (16 x M, column-major vec) = b0 + A [alpha; beta; q coefficients]: per sample r1..r4 (3 each), q0..q3."""
        from math import comb
        na, nq, M, L = self.na, self.nq, self.M, self.L
        nx = 2 * na + nq
        A = np.zeros((16 * M, nx))
        b0 = np.zeros(16 * M)
        Ra = [rows(self.us, na, d) for d in range(5)]
        Rq = [rows(self.us, nq, d) for d in range(4)]
        idx = 16 * np.arange(M)
        for d in range(1, 5):
            for c in range(3):
                row = idx + 3 * (d - 1) + c
                b0[row] = self.Rref[d][:, c]
                for k in range(d + 1):
                    A[row, :na] += comb(d, k) * self.N1[d - k][:, c][:, None] * Ra[k]
                    A[row, na:2 * na] += comb(d, k) * self.N2[d - k][:, c][:, None] * Ra[k]
        for d in range(4):
            A[idx + 12 + d, 2 * na:] = Rq[d] / L ** 2
        return A, b0

    def solve(self, Ca0, Cq0, free_path, max_iter=1000, trust=TRUST, rounds=ROUNDS):
        """Lifted (as CPC / poly_plan): every sample's path and speed derivatives are variables X, tied to the
        B-spline coefficients by linear equalities; the limits act on X only. The sideways offsets move at most
        `trust` m per round (box around the previous round), up to `rounds` rounds while the lap improves."""
        na, nq, M, L = self.na, self.nq, self.M, self.L
        A, b0 = self.lift_map()
        nx = 2 * na + nq
        z = ca.MX.sym("z", nx)
        X = ca.MX.sym("X", 16 * M)
        Xm = ca.reshape(X, 16, M)
        gS, dtS = SMP.map(M)(Xm, ca.repmat(ca.DM(self.par), 1, M))
        J = ca.sum2(dtS) * self.du
        g = [ca.vec(gS), X - ca.mtimes(ca.sparsify(ca.DM(A)), z)]
        lb = [np.tile(G_LB, M), b0]
        ub = [np.tile(G_UB, M), b0]
        if free_path:
            Rg = ca.DM(rows(self.ug, na, 0))
            ag, bg = ca.mtimes(Rg, z[:na]), ca.mtimes(Rg, z[na:2 * na])
            pg = ca.DM(rows(self.ug, self.nref, 0) @ self.Cref) + ca.repmat(ag, 1, 3) * ca.DM(rows(self.ug, self.nref, 0) @ self.CN1) \
                + ca.repmat(bg, 1, 3) * ca.DM(rows(self.ug, self.nref, 0) @ self.CN2)
            g.append(ca.sum2((pg - ca.DM(self.gates)) ** 2)); lb.append(np.zeros(len(self.gates))); ub.append(np.full(len(self.gates), TOL ** 2))
        S = ca.nlpsol("lap", "ipopt", {"x": ca.vertcat(z, X), "f": J, "g": ca.vertcat(*g)},
                      {"expand": True, "ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": max_iter,
                       "ipopt.mu_strategy": "adaptive", "ipopt.tol": 1e-7, **EXTRA})
        zc = np.r_[Ca0[:, 0], Ca0[:, 1], Cq0]
        best, hist, t = None, [], 0.0
        for rd in range(rounds if free_path else 1):
            lbx, ubx = np.full(nx + 16 * M, -np.inf), np.full(nx + 16 * M, np.inf)
            if free_path:
                lbx[:2 * na], ubx[:2 * na] = zc[:2 * na] - trust, zc[:2 * na] + trust
            else:
                lbx[:2 * na] = ubx[:2 * na] = zc[:2 * na]
            t0 = time.perf_counter()
            sol = S(x0=np.r_[zc, b0 + A @ zc], lbx=lbx, ubx=ubx, lbg=np.concatenate(lb), ubg=np.concatenate(ub))
            t += time.perf_counter() - t0
            st = S.stats()
            ok = st["return_status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
            lap = float(sol["f"])
            hist.append((st["return_status"], st["iter_count"], lap))
            if not ok:
                break
            if best is not None and lap > best[1] - 1e-4:
                best = (np.array(sol["x"]).ravel()[:nx], lap, st) if lap < best[1] else best
                break
            best = (np.array(sol["x"]).ravel()[:nx], lap, st)
            zc = best[0]
        if best is None:                                   # first round failed: report its end point
            best = (np.array(sol["x"]).ravel()[:nx], float(sol["f"]), S.stats())
        xs, lap_best, stb = best
        Ca, Cq = np.column_stack([xs[:na], xs[na:2 * na]]), xs[2 * na:]
        good = [h[0] for h in hist if h[0] in ("Solve_Succeeded", "Solved_To_Acceptable_Level")]
        status = good[-1] if good else hist[-1][0]          # the kept solution comes from the last good round
        iters = sum(h[1] for h in hist)
        rounds_txt = " / ".join(f"{h[0][:5]} {h[1]} it {h[2]:.3f}" for h in hist)
        # check the limits on a 4x finer grid than the one optimised on
        Mf = 4 * M
        uf = (np.arange(Mf) + 0.5) / Mf
        from math import comb
        A = [rows(uf, na, d) @ Ca[:, 0] for d in range(5)]
        Bt = [rows(uf, na, d) @ Ca[:, 1] for d in range(5)]
        Rd = []
        for d in range(1, 5):
            e = rows(uf, self.nref, d) @ self.Cref
            for k in range(d + 1):
                e = e + comb(d, k) * (A[k][:, None] * (rows(uf, self.nref, d - k) @ self.CN1) + Bt[k][:, None] * (rows(uf, self.nref, d - k) @ self.CN2))
            Rd.append(e)
        Xf = np.vstack([e.T for e in Rd] + [rows(uf, nq, d) @ Cq / L ** 2 for d in range(4)])
        gf, dtf = (np.array(y) for y in SMP.map(Mf)(Xf, np.tile(self.par[:, None], (1, Mf))))
        viol = np.maximum(np.array(G_LB)[:, None] - gf, gf - np.array(G_UB)[:, None]).max()
        ug = np.linspace(0, 1, 20000, endpoint=False)
        Pg = self.points(Ca, ug)
        gate = float(max(np.linalg.norm(Pg - gg, axis=1).min() for gg in self.gates))
        off = float(np.abs(Ca).max())
        return dict(Ca=Ca, Cq=Cq, lap=lap_best, lap_fine=float(dtf.sum() / Mf), status=status, iters=iters, t=t,
                    viol=float(viol), gate=gate, off=off, rounds=rounds_txt,
                    ok=status in ("Solve_Succeeded", "Solved_To_Acceptable_Level"))


def refit(u_from_na, C_from, n_to):
    uu = np.linspace(0, 1, 4000, endpoint=False)
    return np.linalg.lstsq(rows(uu, n_to, 0), rows(uu, u_from_na, 0) @ C_from, rcond=None)[0]


def main():
    quad = qs.Quad()
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    log(f"time-optimal inside our space: quintic periodic B-splines, gate tol {TOL} m, V spacings {HQS} m, tracks {TRACKS}")
    rows_out = []
    for k in TRACKS:
        _, _, lap_cpc, gates = load_cpc(k)
        tag = next(iter(STEPS.values()))[4]                 # all steps of a set share the start
        P, V, lap_r, _ = load_cpc(k, restricted=True, tag=tag)
        L = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum()
        nref = int(round(L / HR_A))
        Cref, u, qL2, L, err = fit_ref(P, V, nref)
        CN1, CN2, twist = rm_frame(Cref, nref)
        ref = (Cref, nref, CN1, CN2, L)
        ug = np.array([u[np.argmin(np.linalg.norm(P - g, axis=1))] for g in gates])
        pv = PREV.get(k)
        log(f"track {k}: CPC {lap_cpc:.3f} s, restricted CPC (start) {lap_r:.3f} s ({100 * (lap_r / lap_cpc - 1):+.1f}%), "
            f"length {L:.1f} m, reference fit {err * 1000:.2f} mm | "
            "previous (fitted G3 pieces, speed only) "
            + (f"+{pv[0]:.1f}% (no rate limit) / +{pv[1]:.1f}% (rate 0.9 x 15)" if pv else "failed"))
        res = {}
        for step, (margin, eta, free, kfac, src) in STEPS.items():
            prev = None
            for hq in HQS:
                nq = max(8, int(round(L / hq)))
                na = int(round(L / max(hq, 1.0)))              # sideways offsets: spacing max(hq, 1 m)
                # every solve starts from CPC itself: its path (offsets 0) and its speed (user, 2026-11-24; warm starts
                # from the previous solution at a coarser spacing diverged)
                Ca0 = np.zeros((na, 2))
                Cq0 = np.linalg.lstsq(rows(u, nq, 0), qL2, rcond=None)[0]
                pb = Problem(ref, na, nq, gates, ug, quad, margin, eta, kfac)
                st = pb.solve(Ca0, Cq0, False, max_iter=0)              # evaluate the start only (lap, limits)
                r = pb.solve(Ca0, Cq0, free)
                r["start_viol"], r["start_lap"] = st["viol"], st["lap_fine"]
                r.update(na=na, nq=nq)
                res[(step, hq)] = r
                if r["ok"]:
                    prev = r
                log(f"  {step:22s} V {hq:4.2f} m offsets {L / na:4.2f} m ({nq} + {2 * na if free else 0} vars): {r['status'][:22]:22s} "
                    f"{r['iters']:4d} it {r['t']:6.1f} s | lap {r['lap']:.3f} ({100 * (r['lap'] / lap_cpc - 1):+5.1f}%), "
                    f"4x finer check {r['lap_fine']:.3f}, worst limit {r['viol']:+.3f} | gates <= {r['gate']:.3f} m, "
                    f"max offset coef {r['off']:.2f} m | start: lap {r['start_lap']:.3f}, worst limit {r['start_viol']:+.3f} | rounds: {r['rounds']}")
                rows_out.append((k, step, hq, lap_cpc, r, r["ok"]))
        np.savez(HERE / "results" / f"track{k}{'' if SET == '100' else '_' + SET}.npz", Cref=Cref, CN1=CN1, CN2=CN2, L=L, lap_cpc=lap_cpc, gates=gates,
                 **{f"{s_.replace(' ', '_').replace(',', '').replace('%', 'pct')}_{h}_{c}": res[(s_, h)][c] for (s_, h) in res for c in ("Ca", "Cq")})
    log("\nsummary: track | step | V spacing m | lap vs CPC | status")
    for k, step, hq, lc, r, ok in rows_out:
        log(f"{k} | {step} | {hq} | {100 * (r['lap'] / lc - 1):+.1f}% | {'ok' if ok else r['status']}")
    (HERE / "results" / f"run{'' if SET == '100' else '_' + SET}_{'_'.join(map(str, TRACKS))}.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
