"""G2 corners: line - clothoid - arc - clothoid - line (curvature continuous), fast plan through gates,
analytic path coordinates (closed form on lines, Fresnel + Newton on corners), and a speed profile that
respects thrust and an approximate body-rate limit (sigma^3 |dkappa/ds| <= eta * w_max * g on clothoids).
"""
import numpy as np
from scipy.special import fresnel

import race_compare as rc

G, EZ = rc.G, rc.EZ


class Corner:
    """Symmetric G2 corner in the plane (e1 = incoming direction, e2 = turn side), scaled to tangent length T."""

    def __init__(self, V, din, dout, T, q):
        self.th = np.arccos(np.clip(din @ dout, -1, 1))
        n = dout - (dout @ din) * din
        self.e1, self.e2 = din, n / np.linalg.norm(n)
        self.q = q
        if q == 0:                                     # plain circular arc (lines + arcs, curvature jumps at joints)
            self.r = T / np.tan(self.th / 2)
            self.Lc, self.La = 0.0, self.r * self.th
            self.L, self.a = self.La, 0.0
            self.P0 = V - T * din
            self.E2d = self.local(self.L)[0]
            return
        T1 = self._shape(1.0)[1]
        self.r = T / T1
        self.Lc = q * self.r * self.th
        self.La = self.r * self.th - self.Lc
        self.L = 2 * self.Lc + self.La
        self.a = np.sqrt(np.pi * self.r * self.Lc)
        self.P0 = V - T * din
        self.E2d = self._shape(self.r)[0]

    def _shape(self, r):
        """End point (2D) and tangent length for radius r."""
        Lc, La, th = self.q * r * self.th, r * self.th - self.q * r * self.th, self.th
        a = np.sqrt(np.pi * r * Lc)
        S, C = fresnel(Lc / a)
        Q1 = np.array([a * C, a * S])
        ph1 = Lc / (2 * r)
        c0 = Q1 + r * np.array([-np.sin(ph1), np.cos(ph1)])
        Q2 = c0 + r * np.array([np.sin(ph1 + La / r), -np.cos(ph1 + La / r)])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        E = Q2 + R @ np.array([a * C, -a * S])
        return E, E[1] / np.sin(th)

    def local(self, s):
        """2D point, heading, curvature, dcurvature/ds at arc length s in [0, L]."""
        r, Lc, La, a, th = self.r, self.Lc, self.La, self.a, self.th
        if self.q == 0:
            ph = s / r
            return np.array([r * np.sin(ph), r * (1 - np.cos(ph))]), ph, 1 / r, 0.0
        if s <= Lc:
            S, C = fresnel(s / a)
            return np.array([a * C, a * S]), s * s / (2 * r * Lc), s / (r * Lc), 1 / (r * Lc)
        S1, C1 = fresnel(Lc / a)
        ph1 = Lc / (2 * r)
        c0 = np.array([a * C1, a * S1]) + r * np.array([-np.sin(ph1), np.cos(ph1)])
        if s <= Lc + La:
            ph = ph1 + (s - Lc) / r
            return c0 + r * np.array([np.sin(ph), -np.cos(ph)]), ph, 1 / r, 0.0
        u = self.L - s
        S, C = fresnel(u / a)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return self.E2d - R @ np.array([a * C, -a * S]), th - u * u / (2 * r * Lc), u / (r * Lc), -1 / (r * Lc)

    def point(self, s):
        xy, ph, k, dk = self.local(s)
        pos = self.P0 + xy[0] * self.e1 + xy[1] * self.e2
        t = np.cos(ph) * self.e1 + np.sin(ph) * self.e2
        nrm = -np.sin(ph) * self.e1 + np.cos(ph) * self.e2
        return pos, t, k * nrm, dk


def corners_from(verts, frac, q):
    M = len(verts)
    e = np.roll(verts, -1, 0) - verts
    L = np.linalg.norm(e, axis=1)
    d = e / L[:, None]
    return [Corner(verts[i], d[i - 1], d[i], frac[i] * 0.5 * min(L[i - 1], L[i]), q) for i in range(M)]


def fast_plan_g2(gates, frac=0.9, q=0.5, iters=60):
    verts = gates.copy()
    fr = np.full(len(gates), frac)
    for _ in range(iters):
        cs = corners_from(verts, fr, q)
        mids = np.array([c.point(c.L / 2)[0] for c in cs])
        verts = verts + (gates - mids)
    return verts, fr


class PathG2:
    def __init__(self, verts, frac, q, sigma_fn=None):
        cs = corners_from(verts, frac, q)
        M = len(cs)
        self.pieces, s0 = [], 0.0
        for i in range(M):
            j = (i + 1) % M
            a = cs[i].P0 + cs[i].E2d[0] * cs[i].e1 + cs[i].E2d[1] * cs[i].e2
            b = cs[j].P0
            ln = np.linalg.norm(b - a)
            self.pieces.append(("line", s0, ln, a, (b - a) / ln))
            s0 += ln
            self.pieces.append(("corner", s0, cs[j].L, cs[j]))
            s0 += cs[j].L
        self.length = s0
        self.sigma_fn = sigma_fn

    def sample(self, ds=0.02):
        """Dense samples for speed planning: pos, tan, kv, dkappa, seglen."""
        P, T, K, D = [], [], [], []
        for pc in self.pieces:
            m = max(int(pc[2] / ds), 1)
            for k in range(m):
                s = (k + 0.5) / m * pc[2]
                if pc[0] == "line":
                    P.append(pc[3] + s * pc[4]); T.append(pc[4]); K.append(np.zeros(3)); D.append(0.0)
                else:
                    p, t, kv, dk = pc[3].point(s)
                    P.append(p); T.append(t); K.append(kv); D.append(dk)
        P, T, K, D = map(np.array, (P, T, K, D))
        sl = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        return P, T, K, D, sl

    def project(self, p):
        best = None
        for pc in self.pieces:
            if pc[0] == "line":
                _, s0, ln, a, u = pc
                t = np.clip((p - a) @ u, 0, ln)
                foot, tan, kv, s = a + t * u, u, np.zeros(3), s0 + t
            else:
                _, s0, ln, c = pc
                grid = np.linspace(0, ln, 9)
                sl = min(grid, key=lambda x: np.sum((p - c.point(x)[0]) ** 2))
                for _ in range(6):                        # Newton on |p - c(s)|^2
                    pos, tan, kv, _ = c.point(sl)
                    w = p - pos
                    h = 1 - w @ kv
                    sl = float(np.clip(sl + (w @ tan) / max(h, 0.2), 0, ln))
                foot, tan, kv, _ = c.point(sl)
                s = s0 + sl
            dist = np.sum((p - foot) ** 2)
            if best is None or dist < best[0]:
                best = (dist, s, foot, tan, kv)
        return best[1:]


def topp_capped(tan, kv, sl, A, umax_extra, laps=3):
    """rc.topp with an additional per-sample cap on u = sigma^2."""
    n = len(tan)
    tz = tan[:, 2]
    kap = np.linalg.norm(kv, axis=1)
    nz = np.where(kap > 0, kv[:, 2] / np.maximum(kap, 1e-12), 0.0)
    c0 = G ** 2 * tz ** 2 - G ** 2 + A ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        umax = np.where(kap > 1e-9, (-kap * G * nz + np.sqrt((kap * G * nz) ** 2 + kap ** 2 * c0)) / kap ** 2, 1e9)
    umax = np.minimum(umax, umax_extra)
    u = rc._passes(np.ascontiguousarray(tan), np.ascontiguousarray(kv), sl, umax, laps, A)
    u_lap = np.maximum(u[n:2 * n], 1e-4)
    lap = np.sum(sl / np.sqrt(0.5 * (u_lap + np.roll(u_lap, -1))))
    return lap, np.sqrt(u_lap)


def plan_speed(path, A, w_max=15.0, eta=0.8, ds=0.02, window_m=1.5, f_ref=None, f_min=4.0):
    """Body-rate cap on clothoids: thrust direction turns at about sigma^3 |dkappa/ds| / |f|; |f| taken as
    f_ref (default: the planning thrust A, since the time-optimal profile runs near full thrust)."""
    P, T, K, D, sl = path.sample(ds)
    f_ref = A if f_ref is None else f_ref
    with np.errstate(divide="ignore"):
        cap = np.where(np.abs(D) > 0, (eta * w_max * f_ref / np.abs(D)) ** (2 / 3), 1e9)
    cap = np.minimum(cap, min_thrust_cap(T, K, f_min))
    lap, sig = topp_capped(T, K, sl, A, cap)
    if window_m > 0:
        sig = smooth_sigma(sig, sl, window_m)
        lap = np.sum(sl / (0.5 * (sig + np.roll(sig, -1))))
    s = np.concatenate([[0], np.cumsum(sl)])[:-1] + 0.5 * sl[0]
    return lap, sig, s, (P, T, K, D, sl)


def min_thrust_cap(T, K, f_min):
    """Cap on u = sigma^2 keeping the normal-plane thrust |u kv + g e_perp| >= f_min (thrust direction stays
    defined over crests: at zero thrust the body cannot be steered). Below the lower root of the quadratic."""
    eperp = G * (EZ - T[:, 2:3] * T)
    A2 = np.sum(K * K, axis=1)
    B = np.sum(K * eperp, axis=1)
    C = np.sum(eperp * eperp, axis=1) - f_min ** 2
    disc = B * B - A2 * C
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = (-B - np.sqrt(np.maximum(disc, 0))) / A2
    cap = np.where((A2 > 1e-12) & (disc > 0) & (B < 0), np.where(C > 0, lo, 0.25), 1e9)
    return np.maximum(cap, 0.25)


def smooth_sigma(sig, sl, window_m, passes=3):
    """Min-filter then moving-average over +-window_m (periodic). Result never exceeds sig, and the
    bang-bang switches of the time-optimal profile become ramps (bounded jerk)."""
    w = max(int(window_m / np.mean(sl)), 1)
    n = len(sig)
    ext = np.concatenate([sig[-w:], sig, sig[:w]])
    from numpy.lib.stride_tricks import sliding_window_view
    mn = sliding_window_view(ext, 2 * w + 1).min(axis=1)          # length n
    sm = mn
    wb = max(w // passes, 1)                       # `passes` box filters: passes=3 makes sigma C^2 (bounded snap)
    for _ in range(passes):
        ext2 = np.concatenate([sm[-wb:], sm, sm[:wb]])
        sm = sliding_window_view(ext2, 2 * wb + 1).mean(axis=1)
    return np.minimum(sm, sig)


def _nearest_on_corner(c, x, n=33):
    ss = np.linspace(0, c.L, n)
    pts = np.array([c.point(s)[0] for s in ss])
    k = int(np.argmin(np.sum((pts - x) ** 2, axis=1)))
    sl = ss[k]
    for _ in range(4):
        pos, tan, kv, _ = c.point(sl)
        w = x - pos
        sl = float(np.clip(sl + (w @ tan) / max(1 - w @ kv, 0.2), 0, c.L))
    return c.point(sl)[0]


def fast_plan_g2_fit(gates, frac=0.9, q=0.5, offset=0.24, iters=40):
    """Corners whose size is set by frac and whose position slides so that the gate sits `offset` metres
    outside the curve (the path cuts the corner, staying inside the gate tolerance); any point of the
    corner may be the one nearest the gate, so the corner can turn/slide instead of centring on it."""
    verts = gates.copy()
    fr = np.full(len(gates), frac)
    for _ in range(iters):
        cs = corners_from(verts, fr, q)
        new = verts.copy()
        for i, c in enumerate(cs):
            mid = c.point(c.L / 2)[0]
            inward = mid - verts[i]
            inward /= np.linalg.norm(inward)
            target = gates[i] + offset * inward
            new[i] = verts[i] + (target - _nearest_on_corner(c, target))
        verts = new
    return verts, fr


def best_fit_plan(gates, A, q=0.5, offset=0.24, fracs=(0.6, 0.8, 0.95), **kw):
    best = None
    for fr in fracs:
        v, f = fast_plan_g2_fit(gates, fr, q, offset)
        path = PathG2(v, f, q)
        lap, sig, s, S = plan_speed(path, A, **kw)
        gd = rc.gate_dist(S[0], gates).max()
        if gd <= 0.3 and (best is None or lap < best[0]):
            best = (lap, path, sig, s, S, fr)
    return best


def ref_rates(sig, S):
    """What the reference asks of the drone: body rate |dz/dt| (z = thrust direction) and thrust-magnitude rate
    dc/dt (c = |a + g|), both along the plan."""
    P, T, K, D, sl = S
    u = sig ** 2
    dsig = (np.roll(sig, -1) - np.roll(sig, 1)) / (np.roll(sl, 1) + sl)
    a = (sig * dsig)[:, None] * T + u[:, None] * K
    f = a + G * EZ
    c = np.linalg.norm(f, axis=1)
    z = f / c[:, None]
    dt2 = (np.roll(sl, 1) + sl) / sig
    w = np.linalg.norm((np.roll(z, -1, 0) - np.roll(z, 1, 0)) / dt2[:, None], axis=1)
    cd = (np.roll(c, -1) - np.roll(c, 1)) / dt2
    return w, cd


def ref_body_rate(sig, S):
    return ref_rates(sig, S)[0]


def executable_plan(path, A, w_max=15.0, eta=0.8, cdot_max=None, windows=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0), **kw):
    """Smallest speed-smoothing window whose reference needs body rate <= eta * w_max and, if given,
    thrust-magnitude rate |dc/dt| <= cdot_max everywhere."""
    for wm in windows:
        lap, sig, s, S = plan_speed(path, A, w_max=w_max, eta=eta, window_m=wm, **kw)
        wr, cd = ref_rates(sig, S)
        if wr.max() <= eta * w_max and (cdot_max is None or np.abs(cd).max() <= cdot_max):
            break
    return lap, sig, s, S, wm, wr.max(), np.abs(cd).max()
