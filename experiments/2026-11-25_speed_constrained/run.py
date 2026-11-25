"""Speed assignment on a fixed path with the limits as constraints (user, 2026-11-25: no penalties; constraints stay
constraints; first-order method).

Variables: knots (periodic degree-4 B-splines, every HK m): Q (= v^2 when |r'| = L), heading reference h (3D);
           per sample: the 20 lifted quantities of ../2026-11-25_lifted_own_optimizer (body axes, c, c', c'', w, w',
           4 rotors, S), tied to the knots by equalities with products only (no division).
Constraints: E(knots, Y) = 0 (per sample); T_min <= rotor <= T_max, |w_x|, |w_y| <= 15, |w_z| <= 0.3 (bounds on Y);
             Q >= eps (convex hull: q > 0 everywhere).
Objective: T = int du / sqrt(q(u)), Gauss-Legendre (8 points per knot interval) on the Q spline.
Method (gradient projection for nonlinear constraints, first order):
  1. step: d = argmin grad(T).d + 1/(2 alpha) |d|_W^2  s.t.  E + J_E d = 0, bounds   (QP, OSQP)
  2. projection back onto the feasible set over all variables: min |D|_W^2 s.t. E + J_E D = 0, bounds, repeated
     until E = 0 and the bounds hold (a first version only moved the sample variables with the knots fixed: it
     could not undo a small |w_z| overshoot, every step was rejected and halved, 0.003 s per step)
  3. accept if T decreases (Armijo), else alpha / 2; on success alpha x 2
Path: the saved CPC lap (test path only). Start: 2 m/s (3 m/s broke |w_z| <= 0.3 at one sample), heading (1, 0, 0).

    cd experiments/2026-11-25_speed_constrained
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib --with osqp python run.py [tracks...]
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
TRACKS = [int(a) for a in sys.argv[1:]] or list(range(10))
V_START = float(os.environ.get("V_START", 2.0))
MAX_IT = int(os.environ.get("MAX_IT", 300))
argv, sys.argv = sys.argv, sys.argv[:1]
os.environ["MODE"] = "speed"
spec = importlib.util.spec_from_file_location("lift", EXP / "2026-11-25_lifted_own_optimizer" / "run.py")
LO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LO)
sys.argv = argv
R, K, NYV, rows = LO.R, LO.K, LO.NYV, LO.rows
# typical sizes of the variables (metric of the step)
YSCALE = np.r_[np.ones(6), 30, 300, 3000, 15, 15, 15, 300, 300, 300, np.full(4, 7.0), 15]


def quad(n):
    xg, wg = np.polynomial.legendre.leggauss(8)
    e = np.arange(n + 1) / n
    uq = ((e[:-1, None] + e[1:, None]) / 2 + (e[1:, None] - e[:-1, None]) / 2 * xg[None, :]).ravel()
    wq = ((e[1:, None] - e[:-1, None]) / 2 * wg[None, :]).ravel()
    return sp.csr_matrix(rows(uq, n, 0)), wq


class Problem:
    def __init__(self, k):
        Pth, V, lap_ref, gates = R.load_cpc(k)
        Ln = np.linalg.norm(np.roll(Pth, -1, 0) - Pth, axis=1).sum()
        n = int(round(Ln / LO.HK))
        sl = np.linalg.norm(np.roll(Pth, -1, 0) - Pth, axis=1)
        s_ = np.r_[0, np.cumsum(sl)[:-1]]
        C = np.linalg.lstsq(rows(s_ / Ln, n, 0), Pth, rcond=None)[0]
        self.L = LO.Lifted(n, Ln, "speed", Cpath=C)
        self.B0, self.wq = quad(n)
        P = self.L
        self.qidx = np.arange(P.Qi, P.nk, P.kd)
        lb = np.full(P.nv, -np.inf)
        ub = np.full(P.nv, np.inf)
        lb[self.qidx] = 1e-3
        for j in range(NYV):
            lb[P.nk + j::NYV] = P.blb[j]
            ub[P.nk + j::NYV] = P.bub[j]
        lb[P.nk + 19::NYV] = 1e-3
        self.lb, self.ub = lb, ub
        bnd = np.where(np.isfinite(ub), np.abs(ub), np.where(np.isfinite(lb), np.abs(lb), 1.0))
        self.btol = 1e-6 * np.maximum(bnd, 1e-3)              # numerical tolerance, relative to each limit
        wk = np.full(P.nk, float(os.environ.get("HSCALE", 10.0)))       # heading reference: free to turn
        wk[self.qidx] = 100.0
        self.scale = np.r_[wk, np.tile(YSCALE, P.M)]
        self.lap_ref = lap_ref

    def T(self, v):
        q = self.B0 @ v[self.qidx] * self.L.qs
        return float(self.wq @ q ** -0.5)

    def gradT(self, v):
        q = self.B0 @ v[self.qidx] * self.L.qs
        g = np.zeros(self.L.nv)
        g[self.qidx] = self.B0.T @ (-0.5 * self.wq * q ** -1.5) * self.L.qs
        return g

    def correct(self, v, iters=15, tol=1e-10):
        """Knots fixed: Newton per sample on its 20 equations in its 20 unknowns."""
        P = self.L
        for _ in range(iters):
            e, J = P.eq_jac(v)
            if np.abs(e).max() < tol:
                return v, True
            Jy = J[:, P.nk:]                                     # block diagonal (20 x 20 per sample)
            try:
                dy = spl.spsolve(Jy.tocsc(), -e)
            except Exception:
                return v, False
            if not np.all(np.isfinite(dy)):
                return v, False
            v = v.copy()
            v[P.nk:] += dy
        return v, np.abs(P.eqs(v)).max() < 1e-8


def restore(Pb, v, W, iters=20):
    """Projection back onto the feasible set over all variables (knots included): repeat
    min |D|_W^2  s.t.  E(v) + J_E D = 0,  lb <= v + D <= ub   until E = 0 and the bounds hold."""
    P = Pb.L
    nv = P.nv
    for _ in range(iters):
        e, J = P.eq_jac(v)
        inb = np.all(v >= Pb.lb - Pb.btol) and np.all(v <= Pb.ub + Pb.btol)
        if np.abs(e).max() < 1e-10 and inb:
            return v, True
        m = osqp.OSQP()
        m.setup(P=W.tocsc(), q=np.zeros(nv), A=sp.vstack([J, sp.identity(nv)]).tocsc(),
                l=np.r_[-e, Pb.lb - v], u=np.r_[-e, Pb.ub - v], verbose=False, eps_abs=1e-10, eps_rel=1e-10,
                max_iter=20000, polish=True)
        res = m.solve()
        if res.x is None or res.info.status_val not in (1, 2):
            return v, False
        v = v + res.x
    e = P.eqs(v)
    return v, bool(np.abs(e).max() < 1e-8 and np.all(v >= Pb.lb - Pb.btol) and np.all(v <= Pb.ub + Pb.btol))


def solve(Pb, v, log=None):
    P = Pb.L
    nv = P.nv
    alpha = 1e-2
    W = sp.diags(1.0 / Pb.scale ** 2)
    T = Pb.T(v)
    st = dict(it=0, rej=0, t=0.0)
    t0 = time.perf_counter()
    for it in range(MAX_IT):
        g = Pb.gradT(v)
        e, J = P.eq_jac(v)
        lo, hi = Pb.lb - v, Pb.ub - v
        A = sp.vstack([J, sp.identity(nv)]).tocsc()
        l = np.r_[-e, lo]
        u = np.r_[-e, hi]
        accepted = False
        for _ in range(30):
            m = osqp.OSQP()
            m.setup(P=(W / alpha).tocsc(), q=g, A=A, l=l, u=u, verbose=False, eps_abs=1e-9, eps_rel=1e-9,
                    max_iter=20000, polish=True)
            res = m.solve()
            d = res.x
            if d is None or res.info.status_val not in (1, 2):
                alpha *= 0.5
                st["rej"] += 1
                continue
            vn, ok = restore(Pb, v + d, W)
            if not ok:
                st["why_corr"] = st.get("why_corr", 0) + 1
            elif not (np.all(vn >= Pb.lb - Pb.btol) and np.all(vn <= Pb.ub + Pb.btol)):
                st["why_bound"] = st.get("why_bound", 0) + 1
                bad = np.flatnonzero((vn < Pb.lb - Pb.btol) | (vn > Pb.ub + Pb.btol))
                st["bad_kind"] = sorted(set(int((b - P.nk) % NYV) if b >= P.nk else -1 for b in bad))
                st["bad_amt"] = float(max(np.max(Pb.lb - vn), np.max(vn - Pb.ub)))
            else:
                Tn = Pb.T(vn)
                if Tn <= T + 1e-4 * g @ (vn - v):
                    accepted = True
                    break
                st["why_armijo"] = st.get("why_armijo", 0) + 1
            alpha *= 0.5
            st["rej"] += 1
        if not accepted:
            break
        dv = np.abs((vn - v) / Pb.scale).max()
        v, T = vn, Tn
        st["it"] = it + 1
        alpha *= 2
        if log and (it < 5 or it % 20 == 0):
            log(f"      it {it:3d}: T {T:.4f} s, alpha {alpha:.1e}, scaled step {dv:.1e}, rejections {st['rej']} "
                f"(correction {st.get('why_corr', 0)}, bounds {st.get('why_bound', 0)} {st.get('bad_kind')} by {st.get('bad_amt', 0):.1e}, "
                f"Armijo {st.get('why_armijo', 0)})")
        if dv < 1e-7:
            break
    st["t"] = time.perf_counter() - t0
    return v, T, st


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"speed assignment, constraints as constraints (QP step + per-sample correction), knots {LO.HK} m, "
        f"{LO.SPS} samples per span, tracks {TRACKS}")
    (HERE / "results").mkdir(exist_ok=True)
    for k in TRACKS:
        Pb = Problem(k)
        P = Pb.L
        Kn = np.column_stack([np.full(P.n, V_START ** 2), np.tile([1.0, 0, 0], (P.n, 1))])
        v0 = LO.start_values(P, Kn)
        v0, ok = Pb.correct(v0)
        inb = np.all(v0 >= Pb.lb - 1e-9) and np.all(v0 <= Pb.ub + 1e-9)
        log(f"track {k}: {P.nv} variables, start T {Pb.T(v0):.3f} s, start feasible {ok and inb}")
        v, T, st = solve(Pb, v0, log)
        lap_f, viol_f, worst = LO.check(P, v)
        log(f"  track {k}: T {T:.4f} s (saved CPC lap {Pb.lap_ref:.3f}) | {st['it']} steps, {st['rej']} rejected, "
            f"{st['t']:.1f} s | |eq| {np.abs(P.eqs(v)).max():.1e} | between samples (check only): {viol_f:.1e} ({worst})")
        np.savez(HERE / "results" / f"track{k}.npz", v=v, n=P.n, L=P.L, Cpath=P.Cpath)
    with open(HERE / "results" / "run.log", "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
