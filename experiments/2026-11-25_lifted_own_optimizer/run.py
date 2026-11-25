"""Speed (and path) optimisation without any division (user, 2026-11-25): the quantities between the trajectory and the
rotors are variables at every sample, tied by equalities that only multiply; heading is a variable (not chosen by
us); our own optimiser.

Knot variables (periodic degree-4 B-splines on the same knots, u in [0, 1)):
  MODE=speed: Q (speed), h (3D heading reference); the path is fixed (the saved CPC lap, test path only)
  MODE=joint: x, y, z (path), Q, h
  q = u'^2 = Q / L^2 (Q ~ v^2)
Sample variables (20 each): body x, body y, c, c', c'', w (body rates), w', 4 rotor thrusts, S = L u' (~ speed)
Equalities per sample (../2026-11-25_lifted_our_space, checked to vanish on exact flatness values):
  x.x = 1, y.y = 1, x.y = 0;  c z = a + g;  c' z + c R(w x e3) = j;  c'' z + 2 c' R(w x e3) + c R(w' x e3 + w x (w x e3)) = snap
  y.h = 0 and its first two time derivatives;  m c = sum u;  J w' + w x J w = tau(u);  s^2 = q
  (a, j, snap, h and its derivatives are linear in the knot variables per sample, apart from the products with q, s)
Bounds (CPC's): T_min <= u_i <= T_max, |w_x|, |w_y| <= 15, |w_z| <= 0.3; gates (joint): nearest point within TOL.
Objective: T = mean over samples of 1/s.
Optimiser (own): augmented Lagrangian for equalities and bounds; inner Gauss-Newton / Newton with Marquardt damping
(diagonal scaling) and Armijo backtracking; sparse matrices (each sample only touches its own 20 variables and the 5
nearby knots).

    cd experiments/2026-11-25_lifted_own_optimizer
    MODE=speed uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [tracks...]
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[1]
TRACKS = [int(a) for a in sys.argv[1:]] or list(range(10))
MODE = os.environ.get("MODE", "speed")
HK = float(os.environ.get("HK", 1.0))      # knot spacing (m)
DEG = 4
SPS = int(os.environ.get("SPS", 2))
TOL = 0.24
V0 = 3.0
JIT = os.environ.get("JIT", "1") == "1"
NYV = 20


def _load(name, path):
    argv, sys.argv = sys.argv, sys.argv[:1]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


LF = _load("lifted", EXP / "2026-11-25_lifted_our_space" / "run.py")      # EQ, INIT, quad constants
K = LF.K
R = LF.R
JS = _load("joint", EXP / "2026-11-25_path_speed_joint" / "run.py")        # gates_of, start_path, nearest point
G = 9.81


def rows(u, n, d):
    return K.rows_deg(u, n, d, DEG)


# equation scaling (rows of EQ): unit 3, accel 3, jerk 3, snap 3, heading 3, thrust 1, torque 3, s^2 = q 1
def eq_scale(L):
    return np.r_[np.ones(3), np.full(3, 1 / G), np.full(3, 1e-2), np.full(3, 1e-3), 1.0, 1e-1, 1e-2,
                 1 / (4 * LF.TMAX), np.full(3, 1.0), L ** 2]


class Lifted:
    def __init__(self, n, L, mode, Cpath=None, gates=None, margin=1.0, eta=1.0):
        self.n, self.L, self.mode = n, L, mode
        self.M = SPS * n
        self.u = (np.arange(self.M) + 0.5) / self.M
        B = [rows(self.u, n, d) for d in range(5)]
        nz = np.zeros_like(B[0], dtype=bool)
        for b in B:
            nz |= b != 0
        self.W = int(nz.sum(1).max())
        cols = np.zeros((self.M, self.W), int)
        for i in range(self.M):
            c = np.flatnonzero(nz[i])
            cols[i, :len(c)] = c
            cols[i, len(c):] = c[0]
        valid = np.arange(self.W)[None, :] < nz.sum(1)[:, None]
        self.cols = cols
        self.Bv = np.stack([np.where(valid, np.take_along_axis(b, cols, 1), 0.0) for b in B])      # 5 x M x W
        self.Bs = [sp.csr_matrix(b) for b in B]
        # knot components
        self.comp = (["x", "y", "z"] if mode == "joint" else []) + ["Q", "hx", "hy", "hz"]
        self.kd = len(self.comp)
        self.nk = self.kd * n
        self.nv = self.nk + NYV * self.M
        self.Cpath = Cpath
        self.gates = gates
        self.qs = 1 / L ** 2
        self.margin, self.eta = margin, eta
        # per-sample function: inputs X (16), H (9), Y (20) with S = L s in place of s
        X = ca.SX.sym("X", 16)
        Hh = ca.SX.sym("H", 9)
        Y = ca.SX.sym("Y", NYV)
        Yeq = ca.vertcat(Y[:19], Y[19] / L)
        e = LF.EQ(X, Hh, Yeq) * ca.DM(eq_scale(L))
        Z = ca.vertcat(X, Hh, Y)
        opts = {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O2"]}} if JIT else {}
        self.Fe = ca.Function("e", [Z], [e], opts).map(self.M)
        self.Fj = ca.Function("ej", [Z], [e, ca.vec(ca.jacobian(e, Z))], opts).map(self.M)
        # bounds on sample variables
        lb = np.full(NYV, -np.inf)
        ub = np.full(NYV, np.inf)
        lb[9:11], ub[9:11] = -eta * LF.WXY, eta * LF.WXY
        lb[11], ub[11] = -LF.WZ, LF.WZ
        lb[15:19], ub[15:19] = LF.TMIN, margin * LF.TMAX
        self.blb, self.bub = lb, ub
        self.bidx = np.flatnonzero(np.isfinite(lb) | np.isfinite(ub))      # bounded components
        self.Qi = self.comp.index("Q")

    # ---- layout helpers
    def split(self, v):
        Kn = v[:self.nk].reshape(self.n, self.kd)
        Y = v[self.nk:].reshape(self.M, NYV).T                    # 20 x M
        return Kn, Y

    def inputs(self, v):
        Kn, Y = self.split(v)
        c = {nm: Kn[:, i] for i, nm in enumerate(self.comp)}
        if self.mode == "joint":
            C = Kn[:, :3]
        else:
            C = self.Cpath
        r = [(self.Bs[d] @ C).T for d in range(1, 5)]
        q = [self.Bs[d] @ c["Q"] * self.qs for d in range(4)]
        Hm = np.vstack([(self.Bs[d] @ Kn[:, self.kd - 3:]).T for d in range(3)])
        return np.vstack(r + [np.vstack(q), Hm, Y])                   # 45 x M

    def eqs(self, v):
        return np.array(self.Fe(self.inputs(v))).ravel(order="F")    # sample-major, 20 per sample

    def eq_jac(self, v):
        e, Je = self.Fj(self.inputs(v))
        e = np.array(e).ravel(order="F")
        Je = np.array(Je).reshape(NYV, 45, self.M, order="F")         # rows x inputs x samples
        M, W, kd = self.M, self.W, self.kd
        # input -> (knot component, basis derivative, scale)
        imap = []
        if self.mode == "joint":
            for d in range(1, 5):
                for c in range(3):
                    imap.append((3 * (d - 1) + c, c, d, 1.0))
        for d in range(4):
            imap.append((12 + d, self.Qi, d, self.qs))
        for d in range(3):
            for c in range(3):
                imap.append((16 + 3 * d + c, kd - 3 + c, d, 1.0))
        Jk = np.zeros((M, NYV, W, kd))
        for inp, comp, d, s in imap:
            Jk[:, :, :, comp] += np.einsum("ri,iw->irw", Je[:, inp, :], self.Bv[d]) * s
        kcols = (kd * self.cols[:, :, None] + np.arange(kd)[None, None, :]).reshape(M, W * kd)
        ycols = self.nk + NYV * np.arange(M)[:, None] + np.arange(NYV)[None, :]
        cols = np.concatenate([kcols, ycols], 1)                       # M x (W kd + 20)
        data = np.concatenate([Jk.reshape(M, NYV, W * kd), np.transpose(Je[:, 25:45, :], (2, 0, 1))], 2)
        nc = cols.shape[1]
        J = sp.csr_matrix((data.ravel(), np.repeat(cols[:, None, :], NYV, 1).ravel(),
                           np.arange(0, NYV * M * nc + 1, nc)), shape=(NYV * M, self.nv))
        return e, J

    # ---- inequalities g(v) >= 0: sample bounds, gates
    def ineqs(self, v, us=None):
        _, Y = self.split(v)
        rows_, gl = [], []
        for k in self.bidx:
            if np.isfinite(self.blb[k]):
                gl.append(Y[k] - self.blb[k]); rows_.append((k, 1.0))
            if np.isfinite(self.bub[k]):
                gl.append(self.bub[k] - Y[k]); rows_.append((k, -1.0))
        g = np.concatenate(gl)
        data, ri, ci = [], [], []
        for j, (k, s) in enumerate(rows_):
            ri.append(j * self.M + np.arange(self.M))
            ci.append(self.nk + NYV * np.arange(self.M) + k)
            data.append(np.full(self.M, s))
        J = sp.csr_matrix((np.concatenate(data), (np.concatenate(ri), np.concatenate(ci))), shape=(len(g), self.nv))
        if self.mode == "joint":
            gg, Jg = self.gate_terms(v, us)
            g = np.r_[g, gg]
            J = sp.vstack([J, Jg]).tocsr()
        # normalise bound rows (thrust by T_max, rates by their limit)
        return g, J

    def gate_terms(self, v, us):
        Kn, _ = self.split(v)
        C = Kn[:, :3]
        R0 = rows(us, self.n, 0)
        e = R0 @ C - self.gates
        g = TOL ** 2 - np.sum(e ** 2, 1)
        J = np.zeros((len(us), self.nv))
        for c in range(3):
            J[:, c:self.nk:self.kd] = -2 * e[:, c:c + 1] * R0
        return g, sp.csr_matrix(J)

    def nearest(self, v, us_prev=None):
        Kn, _ = self.split(v)
        C = Kn[:, :3]
        nG = len(self.gates)
        uu = np.linspace(0, 1, 20 * self.n, endpoint=False)
        P = rows(uu, self.n, 0) @ C
        out = np.zeros(nG)
        for k, g_ in enumerate(self.gates):
            d = np.linalg.norm(P - g_, axis=1)
            if us_prev is not None:
                lo, hi = us_prev[k - 1], us_prev[(k + 1) % nG]
                d = np.where(((uu - lo) % 1.0) < ((hi - lo) % 1.0), d, np.inf)
            u0 = uu[int(np.argmin(d))]
            for _ in range(8):
                r0, r1, r2 = (rows(np.array([u0]), self.n, dd) @ C for dd in range(3))
                f1 = 2 * float((r0 - g_) @ r1.T)
                f2 = 2 * float(r1 @ r1.T + (r0 - g_) @ r2.T)
                if f2 <= 0:
                    break
                u0 = (u0 - f1 / f2) % 1.0
            out[k] = u0
        return out

    # ---- objective: T = mean(L / S)
    def obj(self, v):
        _, Y = self.split(v)
        S = Y[19]
        T = float(np.mean(self.L / S))
        g = np.zeros(self.nv)
        g[self.nk + 19::NYV] = -self.L / S ** 2 / self.M
        Hd = np.zeros(self.nv)
        Hd[self.nk + 19::NYV] = 2 * self.L / S ** 3 / self.M
        return T, g, Hd


def solve(P, v0, log=None, tol_eq=1e-6, tol_in=1e-4, max_outer=40, max_inner=60):
    v = np.array(v0, float)
    nv = P.nv
    pos = np.r_[np.arange(P.Qi, P.nk, P.kd), P.nk + 19 + NYV * np.arange(P.M)]    # Q, S stay positive
    us = P.nearest(v) if P.mode == "joint" else None
    lam_e = np.zeros(NYV * P.M)
    lam_i = None
    mu = 10.0
    lm = 1e-4
    st = dict(inner=0, t=0.0)

    def merit(v, us):
        T = P.obj(v)[0]
        e = P.eqs(v)
        g, _ = P.ineqs(v, us)
        s = np.maximum(0, lam_i - mu * g)
        return T + lam_e @ e + 0.5 * mu * e @ e + (s @ s - lam_i @ lam_i) / (2 * mu)

    prev = np.inf
    for outer in range(max_outer):
        for it in range(max_inner):
            t0 = time.perf_counter()
            if P.mode == "joint":
                us = P.nearest(v, us)
            T, gT, HdT = P.obj(v)
            e, Je = P.eq_jac(v)
            g, Jg = P.ineqs(v, us)
            if lam_i is None:
                lam_i = np.zeros(len(g))
            s = np.maximum(0, lam_i - mu * g)
            act = s > 0
            grad = gT + Je.T @ (lam_e + mu * e) - Jg.T @ s
            Ja = Jg[act]
            Hm = (sp.diags(HdT) + mu * (Je.T @ Je) + mu * (Ja.T @ Ja)).tocsc()
            st["inner"] += 1
            if np.linalg.norm(grad, np.inf) < 1e-9:
                st["t"] += time.perf_counter() - t0
                break
            D = np.maximum(Hm.diagonal(), 1e-12)
            m0 = merit(v, us)
            ok = False
            for _ in range(15):
                dv = spl.spsolve((Hm + sp.diags(lm * D)).tocsc(), -grad)
                neg = dv[pos] < 0
                step = min(1.0, 0.95 * np.min(v[pos][neg] / -dv[pos][neg])) if neg.any() else 1.0
                slope = grad @ dv
                while step > 1e-4:
                    if merit(v + step * dv, us) <= m0 + 1e-4 * step * slope:
                        ok = True
                        break
                    step *= 0.5
                if ok:
                    break
                lm *= 10
            st["t"] += time.perf_counter() - t0
            if not ok:
                break
            v = v + step * dv
            lm = max(lm / 3, 1e-10) if step == 1.0 else lm
            if np.linalg.norm(step * dv, np.inf) < 1e-12 * (1 + np.linalg.norm(v, np.inf)):
                break
        if P.mode == "joint":
            us = P.nearest(v, us)
        e = P.eqs(v)
        g, _ = P.ineqs(v, us)
        ve, vi = float(np.abs(e).max()), float(max(0.0, -g.min()))
        if log:
            log(f"      outer {outer:2d}: T {P.obj(v)[0]:.4f}, |eq| {ve:.1e}, ineq viol {vi:.1e}, mu {mu:.0e}, "
                f"Newton {st['inner']}, {st['t']:.1f} s")
        lam_e = lam_e + mu * e
        lam_i = np.maximum(0, lam_i - mu * g)
        viol = max(ve / tol_eq, vi / tol_in)
        if viol < 1 and outer > 0:
            break
        if viol > 0.25 * prev:
            mu *= 10
        prev = viol
    return v, us, dict(outer=outer + 1, eq=ve, ineq=vi, **st)


def start_values(P, Kn):
    """Lifted start: sample values from the knot values (flatness only to build the start)."""
    v = np.r_[Kn.ravel(), np.zeros(NYV * P.M)]
    X = P.inputs(v)[:25]
    Y0 = np.array(LF.INIT.map(P.M)(X[:16], X[16:25]))
    Y0[19] *= P.L
    v[P.nk:] = Y0.T.ravel()
    return v


def check(P, v):
    """Between the samples: the exact flatness map on a 4x finer grid (only as a check)."""
    Kn, _ = P.split(v)
    Mf = 4 * P.M
    uf = (np.arange(Mf) + 0.5) / Mf
    C = Kn[:, :3] if P.mode == "joint" else P.Cpath
    X = np.vstack([(rows(uf, P.n, d) @ C).T for d in range(1, 5)]
                  + [rows(uf, P.n, d) @ Kn[:, P.Qi] * P.qs for d in range(4)]
                  + [(rows(uf, P.n, d) @ Kn[:, P.kd - 3:]).T for d in range(3)])
    g, dt = (np.array(x) for x in K.EXACT.map(Mf)(X, np.tile(np.array([[P.margin], [P.eta]]), (1, Mf))))
    viol = np.maximum(np.array(K.EX_LB)[:, None] - g, g - np.array(K.EX_UB)[:, None])[2:13]
    w = int(np.unravel_index(viol.argmax(), viol.shape)[0])
    return float(dt.mean()), float(viol.max()), K.EX_NAMES[2 + w]


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"lifted, no division, own optimiser: MODE {MODE}, degree {DEG}, knots {HK} m, {SPS} samples per span, "
        f"heading a variable, tracks {TRACKS}")
    (HERE / "results").mkdir(exist_ok=True)
    for k in TRACKS:
        if MODE == "speed":
            Pth, V, lap_ref, gates = R.load_cpc(k)
            Ln = np.linalg.norm(np.roll(Pth, -1, 0) - Pth, axis=1).sum()
            n = int(round(Ln / HK))
            sl = np.linalg.norm(np.roll(Pth, -1, 0) - Pth, axis=1)
            s_ = np.r_[0, np.cumsum(sl)[:-1]]
            C = np.linalg.lstsq(rows(s_ / Ln, n, 0), Pth, rcond=None)[0]
            P = Lifted(n, Ln, "speed", Cpath=C)
            Kn = np.column_stack([np.full(n, V0 ** 2), np.tile([1.0, 0, 0], (n, 1))])
        else:
            gates = JS.gates_of(k)
            G_ = np.vstack([gates, gates[:1]])
            n0 = int(round(np.linalg.norm(np.diff(G_, axis=0), axis=1).sum() * 1.15 / HK))
            C0, Ln = JS.start_path(gates, n0)
            n = int(round(Ln / HK))
            C0, Ln = JS.start_path(gates, n)
            P = Lifted(n, Ln, "joint", gates=gates)
            Kn = np.column_stack([C0, np.full(n, V0 ** 2), np.tile([1.0, 0, 0], (n, 1))])
        v0 = start_values(P, Kn)
        e0 = np.abs(P.eqs(v0)).max()
        t0 = time.perf_counter()
        v, us, st = solve(P, v0, log=log)
        lap_f, viol_f, worst = check(P, v)
        log(f"track {k}: {P.nk} knot + {NYV * P.M} sample variables, start |eq| {e0:.1e} | lap {P.obj(v)[0]:.4f} s "
            f"(4x finer {lap_f:.4f}) | {st['outer']} outer / {st['inner']} Newton, {st['t']:.1f} s | |eq| {st['eq']:.1e}, "
            f"bounds/gates viol {st['ineq']:.1e} | between samples (check): {viol_f:.1e} ({worst})")
        np.savez(HERE / "results" / f"{MODE}_track{k}.npz", v=v, n=n, L=P.L, gates=gates,
                 Cpath=P.Cpath if P.Cpath is not None else 0)
    with open(HERE / "results" / f"run_{MODE}.log", "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
