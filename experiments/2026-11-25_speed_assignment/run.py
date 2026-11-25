"""Speed assignment on a fixed path with our own optimiser (user, 2026-11-25: split the problem; a given smooth path,
assign dp/dt on it; heading is an input; write the optimiser yourself, automatic differentiation allowed).

Problem (one dimensional along the path):
  path   r(u), u in [0, 1) periodic, fixed (here: the saved CPC lap, fitted as a quintic B-spline, 0.5 m spacing;
         only a test path, the source does not matter for the method)
  heading fixed: body x in span(h, body z) with h a constant horizontal direction, chosen as far as possible from
         every thrust direction of the start (no singularity); body yaw rate is then 0
  speed  q(u) = u'^2, periodic uniform B-spline of degree 4 (C3), knots every HQ m
  min    T(q) = int du / sqrt(q)
  s.t.   per sample: T_min <= rotor thrust_i <= T_max (i = 1..4), |w_x|, |w_y| <= 15 rad/s (exact flatness map,
         ../2026-11-24_speed_knots_in_position), q control points >= eps (convex hull: q > 0 everywhere)
Optimiser (own): augmented Lagrangian (PHR) for the inequalities; inner damped Newton: exact Hessian of T, Gauss-Newton
for the constraint terms, Armijo backtracking that keeps q's control points positive. Gradients and Jacobians by CasADi
(AD only). IPOPT on the same problem only as a cross-check.

    cd experiments/2026-11-25_speed_assignment
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [tracks...]
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
TRACKS = [int(a) for a in sys.argv[1:]] or list(range(10))
HQ = float(os.environ.get("HQ", 1.0))
DEG = 4
SPS = 4
CHECK_IPOPT = os.environ.get("IPOPT", "1") == "1"
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
R = K.R
ROWS = slice(2, 12)          # rotor hi (4), rotor lo (4), w_x, w_y  (g >= 0); the c floor, speed, w_z, q rows unused
NAMES = K.EX_NAMES[2:12]


def heading(a):
    """Constant horizontal h that stays farthest from all thrust directions z = (a + g)/|a + g|."""
    f = a + np.array([0, 0, 9.81])
    z = f / np.linalg.norm(f, axis=1, keepdims=True)
    th = np.linspace(0, np.pi, 361)
    H = np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
    worst = np.abs(z @ H.T).max(0)                       # max |z . h| over the lap, per candidate
    i = int(np.argmin(worst))
    return H[i], float(np.degrees(np.arccos(worst[i])))  # angle margin to the nearest thrust direction


class SpeedProblem:
    def __init__(self, Cref, L, nq, margin=1.0, eta=1.0, M=None):
        nref = len(Cref)
        self.nq, self.L = nq, L
        self.M = M or SPS * max(nref, nq)
        u = (np.arange(self.M) + 0.5) / self.M
        self.u = u
        rd = [R.rows(u, nref, d) @ Cref for d in range(1, 5)]
        self.rd = rd
        self.Bq = [K.rows_deg(u, nq, d, DEG) / L ** 2 for d in range(4)]
        self.par = np.array([margin, eta])
        self.h = None

    def set_heading(self, h):
        self.h = h
        z = ca.SX.sym("z", self.nq)
        Q = [ca.mtimes(ca.DM(B), z).T for B in self.Bq]
        Hm = ca.DM(np.vstack([np.tile(h[:, None], (1, self.M)), np.zeros((6, self.M))]))
        X = ca.vertcat(*[ca.DM(e.T) for e in self.rd], *Q, Hm)
        g, dt = K.EXACT.map(self.M)(X, ca.repmat(ca.DM(self.par), 1, self.M))
        g = g[ROWS, :]
        T = ca.sum2(dt) / self.M
        gv = ca.vec(g)                                   # column-major: sample-major blocks of 10
        self.fT = ca.Function("T", [z], [T, ca.gradient(T, z), ca.hessian(T, z)[0]])
        self.fg = ca.Function("g", [z], [gv, ca.jacobian(gv, z)])
        self.z_sym, self.T_sym, self.g_sym = z, T, gv

    def set_local(self, h, jit=True):
        """Per-sample function of the 4 numbers (q and its first three u-derivatives), with that sample's path data
        and the heading as parameters: the 10 constraint values, their 4-Jacobian, dt and its gradient/Hessian.
        Mapped over all samples in one call."""
        import scipy.sparse as sp
        self.h = h
        qv = ca.SX.sym("qv", 4)
        D = ca.SX.sym("D", 12)
        par = ca.SX.sym("par", 2)
        X = ca.vertcat(D, qv, ca.DM(h), ca.DM.zeros(6))
        g, dt = K.EXACT(X, par)
        g = g[ROWS]
        Jg = ca.jacobian(g, qv)
        gd, Hd = ca.gradient(dt, qv), ca.hessian(dt, qv)[0]
        opts = {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O2"]}} if jit else {}
        self.Fm = ca.Function("loc", [qv, D, par], [g, ca.vec(Jg), dt, gd, ca.vec(Hd)], opts).map(self.M)
        self.Fv = ca.Function("locv", [qv, D, par], [g, dt], opts).map(self.M)
        self.Dm = np.vstack([e.T for e in self.rd])                    # 12 x M
        self.Pm = np.tile(self.par[:, None], (1, self.M))
        self.Bs = [sp.csr_matrix(B) for B in self.Bq]                  # M x n, 5 nonzeros per row
        M = self.M
        self.perm = (np.arange(10)[None, :] * M + np.arange(M)[:, None]).ravel()   # k-major -> sample-major

    def qvals(self, z):
        return np.vstack([B @ z for B in self.Bs])                     # 4 x M

    def values(self, z):
        g, dt = self.Fv(self.qvals(z), self.Dm, self.Pm)
        return float(np.array(dt).mean()), np.array(g).ravel(order="F")   # sample-major

    def _pattern(self):
        """Per sample: the (at most W) control points it depends on, and the basis values for d = 0..3."""
        nz = np.zeros_like(self.Bq[0], dtype=bool)
        for B in self.Bq:
            nz |= B != 0
        W = int(nz.sum(1).max())
        cols = np.zeros((self.M, W), int)
        for i in range(self.M):
            c = np.flatnonzero(nz[i])
            cols[i, :len(c)] = c
            cols[i, len(c):] = c[0]                      # padding (zero weight)
        self.W, self.cols = W, cols
        self.Bv = np.stack([np.where(np.arange(W)[None, :] < nz.sum(1)[:, None],
                                     np.take_along_axis(B, cols, 1), 0.0) for B in self.Bq])     # 4 x M x W

    def derivs(self, z):
        import scipy.sparse as sp
        if not hasattr(self, "cols"):
            self._pattern()
        g, Jg, dt, gd, Hd = (np.array(v) for v in self.Fm(self.qvals(z), self.Dm, self.Pm))
        M, W, n = self.M, self.W, self.nq
        T = float(dt.mean())
        gT = np.bincount(self.cols.ravel(), (np.einsum("di,diw->iw", gd, self.Bv)).ravel(), minlength=n) / M
        Hd = Hd.reshape(4, 4, M, order="F")
        Hb = np.einsum("dei,diw,eiv->iwv", Hd, self.Bv, self.Bv) / M                 # M x W x W blocks
        r = np.repeat(self.cols, W, 1).ravel()
        c = np.tile(self.cols, (1, W)).ravel()
        HT = sp.csc_matrix((Hb.ravel(), (r, c)), shape=(n, n))
        Jg = Jg.reshape(10, 4, M, order="F")
        data = np.einsum("kdi,diw->ikw", Jg, self.Bv).ravel()                        # sample-major rows
        idx = np.repeat(self.cols[:, None, :], 10, 1).ravel()
        J = sp.csr_matrix((data, idx, np.arange(0, 10 * M * W + 1, W)), shape=(10 * M, n))
        return T, gT, HT, g.ravel(order="F"), J

    def accel(self, Cq):
        q, q1 = self.Bq[0] @ Cq, self.Bq[1] @ Cq
        return self.rd[1] * q[:, None] + self.rd[0] * q1[:, None] / 2


def auglag(P, z0, eps=1e-6, tol=1e-4, mu=10.0, max_outer=30, max_inner=60, log=None):
    """min T(z) s.t. g(z) >= 0, z >= eps. PHR augmented Lagrangian with c = -g <= 0:
    L = T + 1/(2 mu) sum(max(0, lam + mu c)^2 - lam^2). Inner: damped Newton (exact Hessian of T + Gauss-Newton
    mu J^T J on the active terms), Armijo backtracking, steps cut to keep z >= eps."""
    z = np.array(z0, float)
    nc = P.fg(z)[0].shape[0]
    lam = np.zeros(nc)
    stats = dict(inner=0, evals=0)

    def lagr(z, lam, mu):
        T = float(P.fT(z)[0])
        c = -np.array(P.fg(z)[0]).ravel()
        s = np.maximum(0, lam + mu * c)
        stats["evals"] += 1
        return T + (s @ s - lam @ lam) / (2 * mu), c

    viol_prev = np.inf
    for outer in range(max_outer):
        for it in range(max_inner):
            T, gT, HT = (np.array(v) for v in P.fT(z))
            gv, Jg = P.fg(z)
            c = -np.array(gv).ravel()
            Jc = -Jg.sparse()                            # scipy sparse
            s = np.maximum(0, lam + mu * c)
            act = s > 0
            grad = gT.ravel() + Jc.T @ s
            Ja = Jc[act]
            H = HT + mu * (Ja.T @ Ja).toarray()
            stats["inner"] += 1
            if np.linalg.norm(grad, np.inf) < 1e-7:
                break
            delta = 1e-8 * np.trace(H) / len(z)
            while True:
                try:
                    Lc = np.linalg.cholesky(H + delta * np.eye(len(z)))
                    break
                except np.linalg.LinAlgError:
                    delta = max(10 * delta, 1e-10)
            dz = -np.linalg.solve(Lc.T, np.linalg.solve(Lc, grad))
            neg = dz < 0                                 # keep z >= eps
            step = min(1.0, np.min((z[neg] - eps) / -dz[neg]) * 0.99) if neg.any() else 1.0
            L0, _ = lagr(z, lam, mu)
            slope = grad @ dz
            while step > 1e-12:
                L1, _ = lagr(z + step * dz, lam, mu)
                if L1 <= L0 + 1e-4 * step * slope:
                    break
                step *= 0.5
            z = z + step * dz
            if step * np.linalg.norm(dz, np.inf) < 1e-10 * (1 + np.linalg.norm(z, np.inf)):
                break
        c = -np.array(P.fg(z)[0]).ravel()
        viol = float(max(0.0, c.max()))
        if log:
            log(f"      outer {outer:2d}: T {float(P.fT(z)[0]):.4f}, max violation {viol:.2e}, mu {mu:.0e}, "
                f"inner total {stats['inner']}")
        lam = np.maximum(0, lam + mu * c)
        if viol < tol and outer > 0:
            break
        if viol > 0.25 * viol_prev:
            mu *= 10
        viol_prev = viol
    return z, dict(outer=outer + 1, **stats, viol=viol)


def auglag_banded(P, z0, eps=1e-6, tol=1e-4, mu=10.0, max_outer=30, max_inner=60, log=None):
    """Same method as auglag, using the banded structure: per-sample derivatives (4 numbers each) chained through
    the B-spline rows (5 nonzeros), sparse cyclic-banded Newton matrix, sparse LU solve."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    z = np.array(z0, float)
    n = len(z)
    lam = np.zeros(10 * P.M)
    stats = dict(inner=0, evals=0, t_steps=0.0)

    def lagr(z, lam, mu):
        T, g = P.values(z)
        c = -g
        s = np.maximum(0, lam + mu * c)
        stats["evals"] += 1
        return T + (s @ s - lam @ lam) / (2 * mu), c

    viol_prev = np.inf
    I = sp.identity(n, format="csc")
    for outer in range(max_outer):
        for it in range(max_inner):
            t0 = time.perf_counter()
            T, gT, HT, g, J = P.derivs(z)
            c = -g
            s = np.maximum(0, lam + mu * c)
            act = s > 0
            grad = gT - J.T @ s
            Ja = J[act]
            H = (HT + mu * (Ja.T @ Ja)).tocsc()
            stats["inner"] += 1
            if np.linalg.norm(grad, np.inf) < 1e-7:
                stats["t_steps"] += time.perf_counter() - t0
                break
            delta = 1e-8 * H.diagonal().mean()
            dz = spl.spsolve((H + delta * I).tocsc(), -grad)
            neg = dz < 0
            step = min(1.0, np.min((z[neg] - eps) / -dz[neg]) * 0.99) if neg.any() else 1.0
            L0, _ = lagr(z, lam, mu)
            slope = grad @ dz
            while step > 1e-12:
                L1, _ = lagr(z + step * dz, lam, mu)
                if L1 <= L0 + 1e-4 * step * slope:
                    break
                step *= 0.5
            z = z + step * dz
            stats["t_steps"] += time.perf_counter() - t0
            if step * np.linalg.norm(dz, np.inf) < 1e-10 * (1 + np.linalg.norm(z, np.inf)):
                break
        T, g = P.values(z)
        c = -g
        viol = float(max(0.0, c.max()))
        if log:
            log(f"      outer {outer:2d}: T {T:.4f}, max violation {viol:.2e}, mu {mu:.0e}, inner total {stats['inner']}")
        lam = np.maximum(0, lam + mu * c)
        if viol < tol and outer > 0:
            break
        if viol > 0.25 * viol_prev:
            mu *= 10
        viol_prev = viol
    return z, dict(outer=outer + 1, **stats, viol=viol)


def ipopt(P, z0, eps=1e-6):
    S = ca.nlpsol("ip", "ipopt", {"x": P.z_sym, "f": P.T_sym, "g": P.g_sym},
                  {"print_time": False, "ipopt": {"print_level": 0, "max_iter": 3000, "tol": 1e-8}})
    t0 = time.perf_counter()
    sol = S(x0=z0, lbx=eps, ubx=np.inf, lbg=0, ubg=np.inf)
    return np.array(sol["x"]).ravel(), S.stats()["return_status"], time.perf_counter() - t0


def check(P, Cq, h):
    """Exact limits on a 4x finer grid (same rows)."""
    Q = SpeedProblem.__new__(SpeedProblem)
    Q.__dict__.update(P.__dict__)
    Mf = 4 * P.M
    uf = (np.arange(Mf) + 0.5) / Mf
    nref = len(P.Cref)
    rd = [R.rows(uf, nref, d) @ P.Cref for d in range(1, 5)]
    X = np.vstack([e.T for e in rd] + [K.rows_deg(uf, P.nq, d, DEG) @ Cq / P.L ** 2 for d in range(4)]
                  + [np.tile(h[:, None], (1, Mf)), np.zeros((6, Mf))])
    g, dt = (np.array(v) for v in K.EXACT.map(Mf)(X, np.tile(P.par[:, None], (1, Mf))))
    g = g[ROWS]
    viol = np.maximum(0, -g)
    w = int(np.unravel_index(viol.argmax(), viol.shape)[0])
    return float(dt.mean()), float(viol.max()), NAMES[w]


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"speed assignment on a fixed path, own augmented-Lagrangian Newton, q degree {DEG}, knots {HQ} m, "
        f"{SPS} samples per span, tracks {TRACKS}")
    for k in TRACKS:
        Pth, V, lap_ref, gates = R.load_cpc(k)
        L = np.linalg.norm(np.roll(Pth, -1, 0) - Pth, axis=1).sum()
        nref = int(round(L / 0.5))
        Cref, u, qL2, L, err = R.fit_ref(Pth, V, nref)
        nq = max(8, int(round(L / HQ)))
        Cq0 = np.linalg.lstsq(K.rows_deg(u, nq, 0, DEG), qL2, rcond=None)[0]
        P = SpeedProblem(Cref, L, nq)
        P.Cref = Cref
        h, hmarg = heading(P.accel(Cq0))
        tb = time.perf_counter()
        P.set_local(h, jit=JIT)
        t_build = time.perf_counter() - tb
        # start: a slow, feasible lap (constant speed 3 m/s): Newton then only has to speed up
        Cs = np.full(nq, (3.0 / L) ** 2 * L ** 2)
        t0 = time.perf_counter()
        z, st = auglag_banded(P, Cs, log=log)
        t_own = time.perf_counter() - t0
        lap, viol, worst = check(P, z, h)
        log(f"track {k}: path {L:.1f} m (fit {err * 1000:.2f} mm), heading margin {hmarg:.1f} deg, {nq} variables, "
            f"{P.M} samples | own (banded): lap {P.values(z)[0]:.4f} s (4x finer {lap:.4f}), solve {t_own:.2f} s "
            f"(build {t_build:.1f} s), {st['outer']} outer / {st['inner']} Newton steps "
            f"({1000 * st['t_steps'] / st['inner']:.1f} ms per step), violation {st['viol']:.1e}, 4x finer {viol:.1e} ({worst})")
        if CHECK_IPOPT:
            P.set_heading(h)
            zi, sti, ti = ipopt(P, Cs)
            lapi, violi, _ = check(P, zi, h)
            log(f"          IPOPT (check only): {sti}, lap {float(P.fT(zi)[0]):.4f} s, {ti:.2f} s, 4x finer violation "
                f"{violi:.1e} | max |q_own - q_ipopt| / max q = {np.abs(z - zi).max() / zi.max():.1e}")
        (HERE / "results").mkdir(exist_ok=True)
        np.savez(HERE / "results" / f"track{k}.npz", Cref=Cref, L=L, Cq=z, h=h, gates=gates)
    with open(HERE / "results" / "run.log", "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
