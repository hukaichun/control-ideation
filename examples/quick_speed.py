"""Fast speed (V) for a given polynomial path and a requested average speed, without optimisation.

  1. speed limit from curvature: sigma <= sqrt(a_lat / kappa), and sigma <= cap
  2. forward / backward pass around the closed path: d(sigma^2)/ds <= 2 sqrt(a_max^2 - (sigma^2 kappa)^2);
     then rounded (moving average kept under the limit) so the attitude does not have to switch instantly
  3. V = -sigma^2 / 2 fitted per piece (Chebyshev in x), V, V', V'' continuous at joints (equality-constrained LS)
  4. check thrust / body rate / motor limits with the full model (flatness, poly_plan's compiled per-sample function);
     if broken, lower a_lat and repeat
  5. cap set by bisection so that the average speed is the requested one (or the highest feasible)

    PYTHONPATH=. uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python quick_speed.py
"""
import time

import numpy as np

import poly_plan as pp

G = 9.81


class QuickSpeed:
    def __init__(self, spec, motor, ds=0.25, w_max=15.0, eta=0.9, f_min=2.0):
        self.spec, self.motor = spec, motor
        nf, nv = pp.DEG_P + 1, pp.DEG_V + 1
        self.nv = nv
        self.blocks, Fs, Gs, frames, dxs, dss, kap = [], [], [], [], [], [], []
        for i, sp in enumerate(spec):
            n = max(int(np.ceil(sp["l"] / ds)), 2 * nv)
            t = (np.arange(n) + 0.5) / n * 2 - 1
            sc = 2 / sp["l"]
            Rp = pp.cheb_rows(t, pp.DEG_P, 4, sc)
            Rv = pp.cheb_rows(t, pp.DEG_V, 3, sc)
            F = np.array([R @ sp["f"] for R in Rp]); Gg = np.array([R @ sp["g"] for R in Rp])
            px = sp["e1"][None] + np.outer(F[1], sp["e2"]) + np.outer(Gg[1], sp["e3"])
            pxx = np.outer(F[2], sp["e2"]) + np.outer(Gg[2], sp["e3"])
            npx = np.linalg.norm(px, axis=1)
            kap.append(np.linalg.norm(np.cross(px, pxx), axis=1) / npx ** 3)
            dss.append(sp["l"] / n * npx)
            self.blocks.append((i, n, t, Rv))
            Fs.append(F); Gs.append(Gg)
            frames.append(np.tile(np.concatenate([sp["e1"], sp["e2"], sp["e3"]])[:, None], (1, n)))
            dxs.append(np.full(n, sp["l"] / n))
        self.F, self.G = np.hstack(Fs), np.hstack(Gs)
        self.frames, self.dx = np.hstack(frames), np.concatenate(dxs)
        self.ds, self.kap = np.concatenate(dss), np.concatenate(kap)
        self.N = len(self.ds)
        self.par = np.array([motor["m"], motor["Ix"], motor["l"], motor["Tmax"], motor["Tmin"], motor["margin"],
                             w_max, eta, 0.0])
        self.f_min = f_min
        self.fun = pp._sample_fun().map(self.N)
        # joint rows for V, V', V'' continuity (betas from the path)
        zf = np.concatenate([np.concatenate([sp["f"], sp["g"]]) for sp in spec])
        beta = pp._beta_est(spec, zf)
        nP = len(spec)
        rows = []
        for i in range(nP):
            k = (i + 1) % nP
            Ra = pp.cheb_rows(np.array([1.0]), pp.DEG_V, 2, 2 / spec[i]["l"])
            Rb = pp.cheb_rows(np.array([-1.0]), pp.DEG_V, 2, 2 / spec[k]["l"])
            b1, b2 = beta[3 * i], beta[3 * i + 1]
            for coef_a, coef_b in ((Ra[0], Rb[0]), (Ra[1], b1 * Rb[1]), (Ra[2], b1 ** 2 * Rb[2] + b2 * Rb[1])):
                r = np.zeros(nP * nv)
                r[i * nv:(i + 1) * nv] += coef_a[0]
                r[k * nv:(k + 1) * nv] -= coef_b[0]
                rows.append(r)
        self.E = np.array(rows)
        B = np.zeros((self.N, nP * nv))
        k0 = 0
        for i, n, t, Rv in self.blocks:
            B[k0:k0 + n, i * nv:(i + 1) * nv] = Rv[0]
            k0 += n
        self.B = B
        m = nP * nv                                   # KKT matrix of min |B c - y|^2 s.t. E c = 0 (factor once)
        K = np.block([[2 * B.T @ B + 1e-9 * np.eye(m), self.E.T], [self.E, np.zeros((len(rows), len(rows)))]])
        self.Kinv = np.linalg.pinv(K)
        self.Arows = []
        for i, n, t, Rv in self.blocks:
            self.Arows.append(Rv)

    def profile(self, cap, a_lat, smooth_m=None):
        u = self._passes(cap, a_lat)
        if smooth_m:                                  # rounded corners (body rate): moving average, kept under u
            w = max(int(round(smooth_m / np.mean(self.ds))), 1)
            ker = np.ones(2 * w + 1) / (2 * w + 1)
            for _ in range(4):
                us = np.convolve(np.concatenate([u[-w:], u, u[:w]]), ker, mode="valid")
                u = np.minimum(u, us)
        return u

    def _passes(self, cap, a_lat, u_lim=None):
        a_max = a_lat
        u = np.minimum(cap ** 2, a_lat / np.maximum(self.kap, 1e-6))
        if u_lim is not None:
            u = np.minimum(u, u_lim)
        for _ in range(2):                            # closed path: two rounds of forward / backward passes
            for k in range(self.N):
                k1 = (k + 1) % self.N
                at = np.sqrt(max(a_max ** 2 - (u[k] * self.kap[k]) ** 2, 0.0))
                u[k1] = min(u[k1], u[k] + 2 * at * self.ds[k])
            for k in range(self.N - 1, -1, -1):
                k0 = (k - 1) % self.N
                at = np.sqrt(max(a_max ** 2 - (u[k] * self.kap[k]) ** 2, 0.0))
                u[k0] = min(u[k0], u[k] + 2 * at * self.ds[k0])
        return u

    def fit_V(self, u):
        y = -0.5 * u
        m = self.B.shape[1]
        rhs = np.concatenate([2 * self.B.T @ y, np.zeros(self.E.shape[0])])
        return (self.Kinv @ rhs)[:m]

    def violations(self, c):
        """Per sample: True where any limit is broken."""
        nv = self.nv
        U = np.zeros((4, self.N))
        k0 = 0
        for (i, n, t, Rv) in self.blocks:
            for d in range(4):
                U[d, k0:k0 + n] = Rv[d] @ c[i * nv:(i + 1) * nv]
            k0 += n
        g, dt = (np.array(z) for z in self.fun(np.vstack([self.F, self.G, U]), self.frames, self.dx[None, :],
                                                np.tile(self.par[:, None], (1, self.N))))
        bad = ((g[0] < 0.25) | (g[1] < self.f_min ** 2) | (g[2] > 1) | (g[3] < 0) | (g[4] < 0) | (g[5] < 0) | (g[6] < 0))
        return bad.ravel(), float(dt.sum())

    def plan_local(self, v_avg, window_m=2.0, shrink=0.85, iters=40):
        """Start from the curvature / acceleration limits at full thrust; where the full-model check fails, lower the
        speed limit locally (within window_m) and repeat. Then cap for the requested average speed."""
        A = self.motor["margin"] * 4 * self.motor["Tmax"] / self.motor["m"]
        a_lat = np.sqrt(A ** 2 - G ** 2)
        s = np.concatenate([[0.0], np.cumsum(self.ds)[:-1]])
        L = self.ds.sum()
        u_lim = np.full(self.N, np.inf)

        def solve(cap):
            ul = u_lim.copy()
            for it in range(iters):
                c = self.fit_V(self._passes(cap, a_lat, ul))
                bad, T = self.violations(c)
                if not bad.any():
                    return c, T, it, ul
                u_now = -2 * (self.B @ c)
                for k in np.where(bad)[0]:
                    near = np.abs((s - s[k] + L / 2) % L - L / 2) < window_m
                    ul[near] = np.minimum(ul[near], shrink * np.maximum(u_now[near], 0.25))
            return None, None, iters, ul

        c, T, it, ul = solve(60.0)
        if c is None:
            return None
        u_lim[:] = ul
        if L / T <= v_avg:
            return dict(coef=c, lap=T, v_avg=L / T, reached=False, iters=it)
        lo, hi = 0.5, 60.0
        best = (c, T)
        for _ in range(25):
            cap = 0.5 * (lo + hi)
            c, T, _, _ = solve(cap)
            if c is None:
                hi = cap; continue
            best = (c, T)
            if abs(L / T - v_avg) < 0.05:
                break
            lo, hi = (cap, hi) if L / T < v_avg else (lo, cap)
        return dict(coef=best[0], lap=best[1], v_avg=L / best[1], reached=True, iters=it)

    def check(self, c):
        nv = self.nv
        U = np.zeros((4, self.N))
        k0 = 0
        for (i, n, t, Rv) in self.blocks:
            for d in range(4):
                U[d, k0:k0 + n] = Rv[d] @ c[i * nv:(i + 1) * nv]
            k0 += n
        X = np.vstack([self.F, self.G, U])
        g, dt = (np.array(z) for z in self.fun(X, self.frames, self.dx[None, :], np.tile(self.par[:, None], (1, self.N))))
        viol = max((0.25 - g[0]).max(), (self.f_min ** 2 - g[1]).max() / 10, (g[2] - 1).max(), (-g[3]).max() / 10,
                   (-g[4]).max() / 10, (-g[5]).max(), (-g[6]).max())
        return viol, float(dt.sum())

    def plan(self, v_avg, a_lat0=None, tol_v=0.05, smooth_m=3.0):
        A = self.motor["margin"] * 4 * self.motor["Tmax"] / self.motor["m"]
        a_lat = np.sqrt(A ** 2 - G ** 2) if a_lat0 is None else a_lat0
        L = self.ds.sum()
        for _ in range(12):                           # lower a_lat until the fastest profile is feasible
            c = self.fit_V(self.profile(60.0, a_lat, smooth_m))
            viol, T = self.check(c)
            if viol <= 1e-3:
                break
            a_lat *= 0.92
        best = (c, T)
        if L / T <= v_avg:                            # cannot reach the requested average: give the fastest feasible
            return dict(coef=c, lap=T, v_avg=L / T, a_lat=a_lat, reached=False)
        lo, hi = 0.5, 60.0                            # bisection on the speed cap
        for _ in range(30):
            cap = 0.5 * (lo + hi)
            c = self.fit_V(self.profile(cap, a_lat, smooth_m))
            viol, T = self.check(c)
            if viol > 1e-3:
                hi = cap; continue
            best = (c, T)
            if abs(L / T - v_avg) < tol_v:
                break
            lo, hi = (cap, hi) if L / T < v_avg else (lo, cap)
        return dict(coef=best[0], lap=best[1], v_avg=L / best[1], a_lat=a_lat, reached=True)


if __name__ == "__main__":
    import quad_sim as qs
    import flat_el_policy as fe
    quad = qs.Quad()
    D = np.load("../logs/astar_demo_1.npz", allow_pickle=True)
    spec = [dict(s) for s in D["spec"]]
    motor = dict(m=quad.m, Ix=quad.J[0, 0], l=0.15, Tmax=quad.Tmax, Tmin=quad.Tmin, margin=0.9)
    t0 = time.perf_counter()
    qsp = QuickSpeed(spec, motor)
    t_setup = time.perf_counter() - t0
    print(f"path: {len(spec)} pieces, {qsp.ds.sum():.1f} m, {qsp.N} samples | setup (once per path) {1e3 * t_setup:.0f} ms "
          f"| optimised plan (poly_plan) lap {float(D['lap']):.2f} s = {qsp.ds.sum() / float(D['lap']):.1f} m/s average")
    for v in (5.0, 8.0, 10.0, 12.0, 20.0):
        t0 = time.perf_counter()
        r = qsp.plan_local(v)
        dt = time.perf_counter() - t0
        print(f"  requested {v:4.1f} m/s -> {r['v_avg']:.2f} m/s (lap {r['lap']:.2f} s, reached {r['reached']}) in {1e3 * dt:.0f} ms",
              flush=True)
        if v == 10.0:
            sp2 = [dict(s, h=r["coef"][i * qsp.nv:(i + 1) * qsp.nv]) for i, s in enumerate(spec)]
            pol = fe.PolyELPolicy(sp2, wp=16, wH=12)
            p0, v0, a0, j0 = pp.state_at(sp2[0], 0.0)
            log = fe.fly(quad, pol, p0, v0, a0, j0, 2.2 * r["lap"], mode="B")
            L2 = log[(log[:, 0] >= r["lap"]) & (log[:, 0] < 2 * r["lap"])]
            print(f"    flown: ok {log[-1, 0] > 2.19 * r['lap']} | lap 2 |phi| max {L2[:, 1].max():.3f} m | sat "
                  f"{100 * log[:, 8].mean():.0f}% | max |w| {log[:, 7].max():.1f}", flush=True)
