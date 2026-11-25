"""Speed planning directly in the control law's representation (user, 2026-11-25: "能不能直接對 f g h 做規劃",
step 1: path fixed).

Pieces (as the blended B-spline law, ../2026-11-24_blended_bspline_pieces, closed lap as in
../2026-11-24_time_optimal_in_our_space/fly_test.py): cut the lap every 45 deg of turning (cores >= 2.5 m), extend each
core by D = 1 m on both sides. Piece i: chord frame (e1 along the chord, e2 horizontal normal, e3 = e1 x e2), origin O,
chord coordinate x; path q(x) = O + x e1 + f(x) e2 + g(x) e3, f, g quintic B-splines (knots <= KP m), fitted to the
initial lap (here CPC's lap, test path only) and then FIXED.
Speed (my choice, to confirm with the user): r_i(x) = (dx/dt)^2, a clamped degree-4 B-spline per piece (knots <= KR m).
The law's potential follows exactly: h_i(x) = -1/2 |v|^2 = -1/2 r_i(x) (1 + f'^2 + g'^2).
"""
import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

TH_CORE = np.deg2rad(45.0)
D = 1.0
KP = 0.5
KR = 1.0
DEGR = 4


def cut(P):
    """BL.cut on an open sample: arc length s and cut positions (s) every 45 deg of turning."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    dth = np.r_[0, np.arccos(np.clip((T[1:] * T[:-1]).sum(1), -1, 1))]
    cuts, acc, last = [0.0], 0.0, 0.0
    for k in range(1, len(P)):
        acc += dth[k]
        if acc >= TH_CORE and s[k] - last >= 2 * D + 0.5 and s[-1] - s[k] >= 2 * D + 0.5:
            cuts.append(s[k]); acc, last = 0.0, s[k]
    cuts.append(s[-1])
    return s, np.array(cuts)


def clamped(a, b, h, k):
    n = max(1, int(np.ceil((b - a) / h)))
    return np.r_[[a] * (k + 1), np.linspace(a, b, n + 1)[1:-1], [b] * (k + 1)]


FITX = float(__import__("os").environ.get("FITX", 0.0))   # fit f, g on FITX m more at both ends (end effects)


class Piece:
    def __init__(self, P3, s3, a, b, core):
        m = (s3 >= a - 1e-9) & (s3 <= b + 1e-9)
        Q = P3[m]
        self.O = Q[0].copy()
        e1 = Q[-1] - Q[0]
        self.e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross([0, 0, 1.0], self.e1)
        self.e2 = e2 / np.linalg.norm(e2)
        self.e3 = np.cross(self.e1, self.e2)
        q = Q - self.O
        x, y, z = q @ self.e1, q @ self.e2, q @ self.e3
        if not np.all(np.diff(x) > 0):
            raise RuntimeError("piece not monotone along its chord (turns more than 90 deg)")
        self.xa, self.xb = x[0], x[-1]
        self.s_, self.x_ = s3[m].copy(), x.copy()        # arc length (unrolled lap) -> chord coordinate
        self.xc = tuple(float(np.interp(c, s3[m], x)) for c in core)       # core range in x
        if FITX > 0:      # fit on a longer stretch so that the used range [xa, xb] has no spline-end effects
            mw = (s3 >= a - FITX - 1e-9) & (s3 <= b + FITX + 1e-9)
            qw = P3[mw] - self.O
            x, y, z = qw @ self.e1, qw @ self.e2, qw @ self.e3
            if not np.all(np.diff(x) > 0):
                raise RuntimeError("widened fit range not monotone along the chord")
        kn = clamped(x[0], x[-1], KP, 5)
        self.f, self.g = (make_lsq_spline(x, v, kn, 5) for v in (y, z))
        self.fit_err = float(max(np.abs(self.f(x) - y).max(), np.abs(self.g(x) - z).max()))
        self.tk = clamped(x[0], x[-1], KR, DEGR)
        self.nr = len(self.tk) - DEGR - 1
        self.knots = np.unique(np.r_[kn, self.tk])

    def xof(self, p):
        return (np.atleast_2d(p) - self.O) @ self.e1

    def geom(self, x):
        """q_x .. q_xxxx (12 x len(x))."""
        out = []
        for d in range(1, 5):
            fd, gd = self.f.derivative(d)(x), self.g.derivative(d)(x)
            out.append((d == 1) * self.e1[:, None] + self.e2[:, None] * fd + self.e3[:, None] * gd)
        return np.vstack(out)

    def rrows(self, x, d):
        b = BSpline(self.tk, np.eye(self.nr), DEGR)
        return (b.derivative(d) if d else b)(np.clip(x, self.xa, self.xb))

    def metric(self, x):
        return 1 + self.f.derivative(1)(x) ** 2 + self.g.derivative(1)(x) ** 2


def build(P):
    """Closed lap P (one lap, dense). Lap unrolled three times, pieces cut on the middle copy (fly_test.build_closed)."""
    L1 = len(P)
    P3 = np.vstack([P, P, P])
    s3 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P3, axis=0), axis=1))]
    Llap = s3[L1]
    _, cuts = cut(P3[L1:2 * L1 + 1])
    cuts = cuts[:-1] + Llap
    nc = len(cuts)
    pieces = []
    for i in range(nc):
        c0, c1 = cuts[i], cuts[(i + 1) % nc] + (Llap if i == nc - 1 else 0)
        pieces.append(Piece(P3, s3, c0 - D, c1 + D, (c0, c1)))
    junctions = [P3[np.argmin(np.abs(s3 - (cuts[(i + 1) % nc] + (Llap if i == nc - 1 else 0))))] for i in range(nc)]
    return pieces, junctions, Llap
