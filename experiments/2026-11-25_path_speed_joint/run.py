"""Path and speed optimised together, with our own optimiser (user, 2026-11-25).

Still one dimensional: path r(u) and speed q(u) = u'^2 are functions of the same parameter u in [0, 1) (periodic),
both uniform periodic B-splines of degree 4 (C3) on the same knots (every H m of the start path): 4 numbers per knot
(x, y, z, q). v = r'(u) sqrt(q): the tangent comes from the path, the rate from q; a, j, snap are products of path
derivatives and q derivatives, so every sample depends only on the 5 nearby knots -> banded problem.
  min    T = int du / sqrt(q)
  s.t.   per sample: T_min <= rotor thrust_i <= T_max, |w_x|, |w_y| <= 15 rad/s (exact flatness map,
         ../2026-11-24_speed_knots_in_position); heading = a fixed horizontal direction (body yaw rate 0)
         gates: |r(u*_k) - g_k| <= TOL, u*_k = the point of the current path nearest to gate k, searched between the
         neighbouring gates' u*, recomputed every Newton step (d|r(u*) - g|/du = 0 there, so u* is held fixed when
         differentiating)
         q control points >= eps
  no penalty on the parametrisation (user: try without)
Start: a closed spline through the gate centres (chord-length periodic cubic, refitted), constant 3 m/s.
Optimiser: the one of ../2026-11-25_speed_assignment (augmented Lagrangian, damped Newton with Gauss-Newton
constraint terms, banded per-sample derivatives), with Levenberg damping because the path block has no curvature of
its own in T.

    cd experiments/2026-11-25_path_speed_joint
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [tracks...]
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
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[1]
TRACKS = [int(a) for a in sys.argv[1:]] or list(range(10))
H = float(os.environ.get("H", 1.0))        # knot spacing, m of the start path
DEG = 4
SPS = 4
TOL = 0.24
V0 = 3.0
JIT = os.environ.get("JIT", "1") == "1"


def _load(name, path):
    argv, sys.argv = sys.argv, sys.argv[:1]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


os.environ["LIMITS"] = "exact"
K = _load("knots", EXP / "2026-11-24_speed_knots_in_position" / "run.py")
sys.path.insert(0, str(ROOT / "examples"))
import race_compare as rc      # noqa: E402
import track_utils as tu       # noqa: E402

ROWS = slice(2, 12)            # rotor hi (4), rotor lo (4), w_x, w_y
NAMES = K.EX_NAMES[2:12]
NG = 10


def rows(u, n, d):
    return K.rows_deg(u, n, d, DEG)


def gates_of(k):
    return np.asarray(rc.load_track() if k == 0 else tu.random_track(k), float)


def start_path(gates, n):
    """Closed curve through the gate centres (periodic cubic in chord length), refitted as the degree-4 B-spline
    with n control points, u proportional to its arc length."""
    G_ = np.vstack([gates, gates[:1]])
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(G_, axis=0), axis=1))]
    cs = CubicSpline(t, G_, bc_type="periodic")
    tt = np.linspace(0, t[-1], 4000, endpoint=False)
    P = cs(tt)
    sl = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.r_[0, np.cumsum(sl)[:-1]]
    L = sl.sum()
    C = np.linalg.lstsq(rows(s / L, n, 0), P, rcond=None)[0]
    return C, L


def heading(a):
    f = a + np.array([0, 0, 9.81])
    z = f / np.linalg.norm(f, axis=1, keepdims=True)
    th = np.linspace(0, np.pi, 361)
    Hh = np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
    worst = np.abs(z @ Hh.T).max(0)
    i = int(np.argmin(worst))
    return Hh[i], float(np.degrees(np.arccos(worst[i])))


class Joint:
    """z = per knot (x, y, z, q) -> 4n variables. Per sample: inputs r1..r4 (12), q0..q3 (4)."""

    def __init__(self, n, gates, h, qs, margin=1.0, eta=1.0):
        self.n, self.M = n, SPS * n
        self.qs = qs                                              # variable Q = q / qs (~ v^2), qs = 1 / L^2
        self.u = (np.arange(self.M) + 0.5) / self.M
        self.B = [rows(self.u, n, d) for d in range(5)]           # M x n
        nz = self.B[0] != 0
        for B in self.B[1:]:
            nz |= B != 0
        self.W = int(nz.sum(1).max())
        cols = np.zeros((self.M, self.W), int)
        for i in range(self.M):
            c = np.flatnonzero(nz[i])
            cols[i, :len(c)] = c
            cols[i, len(c):] = c[0]
        valid = np.arange(self.W)[None, :] < nz.sum(1)[:, None]
        self.cols = cols
        self.Bv = np.stack([np.where(valid, np.take_along_axis(B, cols, 1), 0.0) for B in self.B])   # 5 x M x W
        self.gates, self.h = gates, h
        self.par = np.array([margin, eta])
        X = ca.SX.sym("X", 16)
        par = ca.SX.sym("par", 2)
        g, dt = K.EXACT(ca.vertcat(X, ca.DM(h), ca.DM.zeros(6)), par)
        g = g[ROWS]
        qv = X[12:16]
        opts = {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O2"]}} if JIT else {}
        self.Fd = ca.Function("d", [X, par], [g, ca.vec(ca.jacobian(g, X)), dt, ca.gradient(dt, qv),
                                              ca.vec(ca.hessian(dt, qv)[0])], opts).map(self.M)
        self.Fv = ca.Function("v", [X, par], [g, dt], opts).map(self.M)
        self.Pm = np.tile(self.par[:, None], (1, self.M))
        self.us = None

    def unpack(self, z):
        Z = z.reshape(self.n, 4)
        return Z[:, :3], Z[:, 3]

    def inputs(self, z):
        C, q = self.unpack(z)
        Bs = [sp.csr_matrix(B) for B in self.B] if not hasattr(self, "_Bs") else self._Bs
        self._Bs = Bs
        r = [(Bs[d] @ C).T for d in range(1, 5)]
        qd = [Bs[d] @ q * self.qs for d in range(4)]
        return np.vstack(r + [np.vstack(qd)])                     # 16 x M

    # gates: nearest point on the current path, between the neighbours' u*
    def nearest(self, z, us_prev=None):
        C, _ = self.unpack(z)
        nG = len(self.gates)
        uu = np.linspace(0, 1, 20 * self.n, endpoint=False)
        P = rows(uu, self.n, 0) @ C
        out = np.zeros(nG)
        for k, g_ in enumerate(self.gates):
            d = np.linalg.norm(P - g_, axis=1)
            if us_prev is not None:
                lo, hi = us_prev[k - 1], us_prev[(k + 1) % nG]
                span = (hi - lo) % 1.0
                ok = ((uu - lo) % 1.0) < span
                d = np.where(ok, d, np.inf)
            u0 = uu[int(np.argmin(d))]
            for _ in range(8):                                    # Newton on d/du |r(u) - g|^2 = 0
                r0, r1, r2 = (rows(np.array([u0]), self.n, dd) @ C for dd in range(3))
                f1 = 2 * (r0 - g_) @ r1.T
                f2 = 2 * (r1 @ r1.T + (r0 - g_) @ r2.T)
                if f2[0, 0] <= 0:
                    break
                u0 = (u0 - f1[0, 0] / f2[0, 0]) % 1.0
            out[k] = u0
        return out

    def gate_terms(self, z, us):
        C, _ = self.unpack(z)
        R0 = rows(us, self.n, 0)                                  # nG x n
        e = R0 @ C - self.gates                                   # nG x 3
        g = TOL ** 2 - np.sum(e ** 2, 1)                          # >= 0
        J = np.zeros((len(us), 4 * self.n))
        for c in range(3):
            J[:, c::4] = -2 * e[:, c:c + 1] * R0
        return g, sp.csr_matrix(J)

    def values(self, z, us):
        g, dt = self.Fv(self.inputs(z), self.Pm)
        gg, _ = self.gate_terms(z, us)
        return float(np.array(dt).mean()), np.r_[np.array(g).ravel(order="F"), gg]

    def derivs(self, z, us):
        g, Jg, dt, gd, Hd = (np.array(v) for v in self.Fd(self.inputs(z), self.Pm))
        M, W, n = self.M, self.W, self.n
        T = float(dt.mean())
        # input sensitivity: input (d, c) of sample i -> variable (cols[i, w], c) with weight Bv[d, i, w]
        # T depends only on q: q-input d=0..3 -> variable 4*col + 3 with Bv[d]
        gq = np.einsum("di,diw->iw", gd, self.Bv[:4]) * self.qs / M
        gT = np.bincount((4 * self.cols + 3).ravel(), gq.ravel(), minlength=4 * n)
        Hd = Hd.reshape(4, 4, M, order="F")
        Hb = np.einsum("dei,diw,eiv->iwv", Hd, self.Bv[:4], self.Bv[:4]) * self.qs ** 2 / M
        vi = 4 * self.cols + 3
        HT = sp.csc_matrix((Hb.ravel(), (np.repeat(vi, W, 1).ravel(), np.tile(vi, (1, W)).ravel())), shape=(4 * n, 4 * n))
        Jg = Jg.reshape(NG, 16, M, order="F")
        # columns of the per-sample Jacobian over its 4W variables
        Jl = np.zeros((M, NG, W, 4))
        for d in range(4):                                        # r_{d+1}: inputs 3d..3d+2, basis derivative d+1
            for c in range(3):
                Jl[:, :, :, c] += np.einsum("ki,iw->ikw", Jg[:, 3 * d + c, :], self.Bv[d + 1])
            Jl[:, :, :, 3] += np.einsum("ki,iw->ikw", Jg[:, 12 + d, :], self.Bv[d]) * self.qs
        vidx = (4 * self.cols[:, :, None] + np.arange(4)[None, None, :]).reshape(M, 4 * W)
        data = Jl.reshape(M, NG, 4 * W).ravel()
        idx = np.repeat(vidx[:, None, :], NG, 1).ravel()
        J = sp.csr_matrix((data, idx, np.arange(0, NG * M * 4 * W + 1, 4 * W)), shape=(NG * M, 4 * n))
        gg, Jgt = self.gate_terms(z, us)
        return T, gT, HT, np.r_[g.ravel(order="F"), gg], sp.vstack([J, Jgt]).tocsr()

    def accel(self, z):
        C, q = self.unpack(z)
        r1, r2 = self.B[1] @ C, self.B[2] @ C
        q0, q1 = self.B[0] @ q * self.qs, self.B[1] @ q * self.qs
        return r2 * q0[:, None] + r1 * q1[:, None] / 2


def solve(P, z0, eps=1e-6, tol=1e-4, mu=10.0, max_outer=40, max_inner=80, log=None):
    z = np.array(z0, float)
    nvar = len(z)
    qi = np.arange(3, nvar, 4)
    us = P.nearest(z)
    lam = None
    stats = dict(inner=0, t=0.0)
    lm = 1e-6
    I = sp.identity(nvar, format="csc")

    def lagr(z, lam, mu, us):
        T, g = P.values(z, us)
        s = np.maximum(0, lam + mu * (-g))
        return T + (s @ s - lam @ lam) / (2 * mu)

    viol_prev = np.inf
    for outer in range(max_outer):
        for it in range(max_inner):
            t0 = time.perf_counter()
            us = P.nearest(z, us)
            T, gT, HT, g, J = P.derivs(z, us)
            if lam is None:
                lam = np.zeros(len(g))
            c = -g
            s = np.maximum(0, lam + mu * c)
            act = s > 0
            grad = gT - J.T @ s
            Ja = J[act]
            Hm = (HT + mu * (Ja.T @ Ja)).tocsc()
            stats["inner"] += 1
            if np.linalg.norm(grad, np.inf) < 1e-8:
                stats["t"] += time.perf_counter() - t0
                break
            scale = max(Hm.diagonal().mean(), 1e-12)
            L0 = lagr(z, lam, mu, us)
            accepted = False
            for _ in range(12):                                   # Levenberg: grow the damping until descent
                dz = spl.spsolve((Hm + lm * scale * I).tocsc(), -grad)
                neg = dz[qi] < 0
                step = min(1.0, 0.99 * np.min((z[qi][neg] - eps) / -dz[qi][neg])) if neg.any() else 1.0
                slope = grad @ dz
                while step > 1e-6:
                    if lagr(z + step * dz, lam, mu, us) <= L0 + 1e-4 * step * slope:
                        accepted = True
                        break
                    step *= 0.5
                if accepted:
                    break
                lm *= 10
            stats["t"] += time.perf_counter() - t0
            if not accepted:
                break
            z = z + step * dz
            lm = max(lm / 3 if step == 1.0 else lm, 1e-9)
            if step * np.linalg.norm(dz, np.inf) < 1e-10 * (1 + np.linalg.norm(z, np.inf)):
                break
        us = P.nearest(z, us)
        T, g = P.values(z, us)
        c = -g
        viol = float(max(0.0, c.max()))
        if log:
            log(f"      outer {outer:2d}: T {T:.4f}, max violation {viol:.2e}, mu {mu:.0e}, Newton steps {stats['inner']}, "
                f"{stats['t']:.1f} s")
        lam = np.maximum(0, lam + mu * c)
        if viol < tol and outer > 0:
            break
        if viol > 0.25 * viol_prev:
            mu *= 10
        viol_prev = viol
    return z, us, dict(outer=outer + 1, viol=viol, **stats)


def check(P, z):
    """Exact limits on a 4x finer grid; gate distances on a dense curve."""
    n = P.n
    C, q = P.unpack(z)
    Mf = 4 * P.M
    uf = (np.arange(Mf) + 0.5) / Mf
    X = np.vstack([(rows(uf, n, d) @ C).T for d in range(1, 5)] + [rows(uf, n, d) @ q * P.qs for d in range(4)]
                  + [np.tile(P.h[:, None], (1, Mf)), np.zeros((6, Mf))])
    g, dt = (np.array(v) for v in K.EXACT.map(Mf)(X, np.tile(P.par[:, None], (1, Mf))))
    viol = np.maximum(0, -g[ROWS])
    w = int(np.unravel_index(viol.argmax(), viol.shape)[0])
    Pd = rows(np.linspace(0, 1, 40000, endpoint=False), n, 0) @ C
    gate = max(np.linalg.norm(Pd - g_, axis=1).min() for g_ in P.gates)
    L = np.linalg.norm(np.roll(Pd, -1, 0) - Pd, axis=1).sum()
    sp_ = np.linalg.norm(rows(uf, n, 1) @ C, axis=1)
    return float(dt.mean()), float(viol.max()), NAMES[w], float(gate), float(L), float(sp_.max() / sp_.min())


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"path + speed together, own optimiser: degree {DEG} on both, knots every {H} m of the start path, "
        f"{SPS} samples per span, gate tol {TOL} m, start: spline through the gates at {V0} m/s, tracks {TRACKS}")
    (HERE / "results").mkdir(exist_ok=True)
    for k in TRACKS:
        gates = gates_of(k)
        G_ = np.vstack([gates, gates[:1]])
        n = int(round(np.linalg.norm(np.diff(G_, axis=0), axis=1).sum() * 1.15 / H))
        C0, L0 = start_path(gates, n)
        n = int(round(L0 / H))
        C0, L0 = start_path(gates, n)
        q0 = np.full(n, V0 ** 2)                                  # Q = q L^2 = v^2 when |r'| = L
        z0 = np.column_stack([C0, q0]).ravel()
        # heading from a rough fast guess: the start path at 10 m/s
        zf = np.column_stack([C0, np.full(n, 100.0)]).ravel()
        tmp = Joint(n, gates, np.array([1.0, 0, 0]), 1 / L0 ** 2)
        h, hm = heading(tmp.accel(zf))
        t0 = time.perf_counter()
        P = Joint(n, gates, h, 1 / L0 ** 2)
        t_build = time.perf_counter() - t0
        z, us, st = solve(P, z0, log=log)
        lap, viol, worst, gate, L, ratio = check(P, z)
        _, hm_end = heading(P.accel(z))
        log(f"track {k}: start path {L0:.1f} m, {n} knots ({4 * n} variables), {P.M} samples | lap {lap:.3f} s, "
            f"path {L:.1f} m, worst gate {gate:.3f} m | {st['outer']} outer / {st['inner']} Newton steps, "
            f"{st['t']:.1f} s (+ build {t_build:.1f} s), violation {st['viol']:.1e}, 4x finer {viol:.1e} ({worst}) | "
            f"|dr/du| max/min {ratio:.2f} | heading margin start {hm:.1f} deg, end {np.degrees(np.arccos(np.clip(np.cos(np.radians(hm_end)), -1, 1))):.1f} deg")
        np.savez(HERE / "results" / f"track{k}.npz", z=z, n=n, h=h, gates=gates, us=us)
    with open(HERE / "results" / "run.log", "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
