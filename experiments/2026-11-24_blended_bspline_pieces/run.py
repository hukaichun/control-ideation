"""Local control laws on B-spline graph pieces cut by turning angle, blended smoothly (user's design, 2026-11-24).

Source: the planned path and V of experiments/2026-11-24_perfect_maze (minimum curvature, current Chebyshev G3 pieces
+ speed plan); same path, same speed, only the representation of the law changes.
  1. sample the source path densely (position, V); cut it into cores where the accumulated turning reaches TH_CORE;
     every core is extended by D on both sides (overlap with its neighbours)
  2. each extended piece: e1 = its chord, e2 = horizontal normal, e3 = e1 x e2; it must be monotone along e1
     f(x), g(x), h(x) = quintic B-splines (knots every <= KNOT m) fitted by linear least squares to the samples
     local law i: phi_i = (e2.q - f(x), e3.q - g(x)), V_i = h(x), x = e1.q, q = p - O_i (same E-L law as PolyELPolicy)
  3. blending: w = sum psi_i w_i / sum psi_i
     psi_i = rise_i(x) * fall_i(x) * lat_i(|phi_i|):  rise from 0 to 1 over the overlap with the previous piece
     (x of the source points at cut - D .. cut + D), fall likewise with the next piece, C3 smooth step (degree 7);
     lat_i = 1 for |phi_i| <= R_IN, smooth step to 0 at R_OUT (so a piece of a parallel leg of the maze, whose x-window
     also contains p, does not take part). No weight anywhere (far off the path): the piece with the smallest |phi_i|.
Compared with the current PolyELPolicy (hard switch between Chebyshev pieces) on the same path: flight on the path,
and starts displaced sideways (recovery).

    cd experiments/2026-11-24_blended_bspline_pieces
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [mazes...]
"""
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.interpolate import PPoly, make_lsq_spline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
SEEDS = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
import bench as B                           # noqa: E402

fe, pp = B.fe, B.pp
SRC = HERE.parent / "2026-11-24_perfect_maze" / "results" / "min_curvature"
TH_CORE = np.deg2rad(45.0)                  # turning of a core; extended pieces stay well below 90 deg
D = 1.0                                     # overlap half-length (m) on each side of a cut
KNOT = 2.0                                  # max knot spacing (m)
K = 5                                       # quintic
R_IN, R_OUT = 1.0, 2.0                      # lateral window of a piece (m)
WP, WH = 16.0, 12.0                         # same gains as the shared back end's PolyELPolicy
OFFSETS = (0.0, 0.5, 1.0)                   # sideways start offsets (m)
GUST_A, GUST_T = 10.0, 0.3                  # gust test: sideways acceleration (m/s^2) for GUST_T s, while crossing a cut


def sample_spec(spec, n=400):
    P, V = [], []
    for k, s in enumerate(spec):
        x = np.linspace(0, s["l"], n, endpoint=(k == len(spec) - 1))
        t = 2 * x / s["l"] - 1
        P.append(s["O"] + np.outer(x, s["e1"]) + np.outer(C.chebval(t, s["f"]), s["e2"]) + np.outer(C.chebval(t, s["g"]), s["e3"]))
        V.append(C.chebval(t, s["h"]))
    P, V = np.vstack(P), np.concatenate(V)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]
    return P[keep], V[keep]


def smooth_step(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 4 * (35 - 84 * u + 70 * u ** 2 - 20 * u ** 3)


def cut(P):
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


def spline_expr(sp, x):
    """casadi expression of a scipy B-spline (as its piecewise polynomial; end spans extrapolate)."""
    pc = PPoly.from_spline(sp)
    br, cf = pc.x, pc.c
    spans = [i for i in range(len(br) - 1) if br[i + 1] > br[i]]
    def poly(i):
        e, dx = 0, x - br[i]
        for c in cf[:, i]:
            e = e * dx + c
        return e
    val = poly(spans[-1])
    for i in reversed(spans[:-1]):
        val = ca.if_else(x < br[i + 1], poly(i), val)
    return val


class Piece:
    def __init__(self, P, V, s, a, b, rise, fall):
        m = (s >= a - 1e-9) & (s <= b + 1e-9)
        Q, Vm = P[m], V[m]
        self.O = Q[0].copy()
        e1 = Q[-1] - Q[0]
        e1[2] = 0 if abs(Q[-1, 2] - Q[0, 2]) < 1e-6 else e1[2]
        self.e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross([0, 0, 1.0], self.e1)
        self.e2 = e2 / np.linalg.norm(e2)
        self.e3 = np.cross(self.e1, self.e2)
        q = Q - self.O
        x, y, z = q @ self.e1, q @ self.e2, q @ self.e3
        if not np.all(np.diff(x) > 0):
            raise RuntimeError("piece not monotone along its chord")
        self.xa, self.xb = x[0], x[-1]
        nsp = max(1, int(np.ceil((x[-1] - x[0]) / KNOT)))
        kn = np.r_[[x[0]] * (K + 1), np.linspace(x[0], x[-1], nsp + 1)[1:-1], [x[-1]] * (K + 1)]
        self.f, self.g, self.h = (make_lsq_spline(x, val, kn, K) for val in (y, z, Vm))
        self.fit_err = float(max(np.abs(self.f(x) - y).max(), np.abs(self.g(x) - z).max()))
        self.V_err = float(np.abs(self.h(x) - Vm).max())
        self.ncoef = len(self.f.c)
        self.tilt = float(np.degrees(np.arccos(np.clip(np.abs(np.gradient(x) / np.linalg.norm(np.gradient(q, axis=0), axis=1)), 0, 1)).max()))
        xof = lambda sv: float((np.interp(sv, s, P[:, 0]) - self.O[0]) * self.e1[0] + (np.interp(sv, s, P[:, 1]) - self.O[1]) * self.e1[1]
                               + (np.interp(sv, s, P[:, 2]) - self.O[2]) * self.e1[2])
        self.rise = None if rise is None else (xof(rise[0]), xof(rise[1]))
        self.fall = None if fall is None else (xof(fall[0]), xof(fall[1]))
        p, v, a_, j = (ca.SX.sym(nm, 3) for nm in "pvaj")
        qq = p - ca.DM(self.O)
        xx = ca.dot(ca.DM(self.e1), qq)
        phi = ca.vertcat(ca.dot(ca.DM(self.e2), qq) - spline_expr(self.f, xx), ca.dot(ca.DM(self.e3), qq) - spline_expr(self.g, xx))
        w, info = fe._el_w(p, v, a_, j, phi, spline_expr(self.h, xx), 0.0, WP, WH)
        self.F = ca.Function("w", [p, v, a_, j], [w, info])

    def weight(self, p):
        q = p - self.O
        x = q @ self.e1
        if x < self.xa - 0.5 or x > self.xb + 0.5:
            return 0.0, np.inf
        d = np.hypot(q @ self.e2 - float(self.f(np.clip(x, self.xa, self.xb))), q @ self.e3 - float(self.g(np.clip(x, self.xa, self.xb))))
        wgt = 1.0
        if self.rise:
            wgt *= smooth_step((x - self.rise[0]) / (self.rise[1] - self.rise[0]))
        if self.fall:
            wgt *= 1 - smooth_step((x - self.fall[0]) / (self.fall[1] - self.fall[0]))
        wgt *= 1 - smooth_step((d - R_IN) / (R_OUT - R_IN))
        return wgt, d


class BlendPolicy:
    def __init__(self, pieces):
        self.pieces = pieces
        self.fun = [self._w]
        self.nmix = []

    def piece(self, p):
        return 0

    def _w(self, p, v, a, j):
        wd = [pc.weight(p) for pc in self.pieces]
        ws = np.array([x[0] for x in wd])
        if ws.sum() < 1e-9:
            i = int(np.argmin([x[1] for x in wd]))
            self.nmix.append(0)
            return self.pieces[i].F(p, v, a, j)
        act = np.nonzero(ws > 1e-12)[0]
        self.nmix.append(len(act))
        W, I = 0, 0
        for i in act:
            w_, inf_ = self.pieces[i].F(p, v, a, j)
            W = W + ws[i] * np.array(w_).ravel()
            I = I + ws[i] * np.array(inf_).ravel()
        return W / ws[act].sum(), I / ws[act].sum()


class Recorder:
    """Wraps a policy and records w at every control step (to see jumps)."""
    def __init__(self, pol):
        self.pol, self.W = pol, []
        self.fun = [self._mk(f) for f in pol.fun]

    def _mk(self, f):
        def g(p, v, a, j):
            w, info = f(p, v, a, j)
            self.W.append(np.array(w).ravel())
            return w, info
        return g

    def piece(self, p):
        return self.pol.piece(p)


def build(spec):
    P, V = sample_spec(spec)
    s, cuts = cut(P)
    n = len(cuts) - 1
    pieces = []
    for i in range(n):
        a = max(0.0, cuts[i] - (D if i > 0 else 0.0))
        b = min(s[-1], cuts[i + 1] + (D if i < n - 1 else 0.0))
        rise = (cuts[i] - D, cuts[i] + D) if i > 0 else None
        fall = (cuts[i + 1] - D, cuts[i + 1] + D) if i < n - 1 else None
        pieces.append(Piece(P, V, s, a, b, rise, fall))
    return pieces, P, s, cuts


class GustQuad(B.qs.Quad):
    """Quad with sideways gusts: list of (t0, acceleration vector), each lasting GUST_T s."""
    def __init__(self, gusts):
        super().__init__()
        self.gusts, self.t = gusts, 0.0

    def step(self, x, u, dt):
        self.ext_acc = sum((g for t0, g in self.gusts if t0 <= self.t < t0 + GUST_T), np.zeros(3))
        self.t += dt
        return super().step(x, u, dt)


def fly(quad, pol, spec, t_end, offset, gusts=None):
    p0, v0, a0, j0 = pp.state_at(spec[0], 0.0)
    p0 = p0 + offset * spec[0]["e2"]
    rec = Recorder(pol)
    plant = GustQuad(gusts) if gusts else None
    lg = fe.fly(quad, rec, p0, v0, a0, j0, t_end, mode="B", plant=plant)
    e_end = spec[-1]["e1"][:2]
    beyond = np.nonzero((lg[:, 4:6] - B.GOAL) @ e_end >= 0)[0]
    Wr = np.array(rec.W)
    if len(beyond):                        # only up to the goal line (the path ends there; w after it is meaningless)
        Wr = Wr[:beyond[0] // 2 + 1]       # one control step every 2 simulation steps
    dW = np.linalg.norm(np.diff(Wr, axis=0), axis=1)
    out = dict(ok=len(beyond) > 0, dw_max=float(dW.max()), dw_p99=float(np.percentile(dW, 99.9)),
               w_max=float(np.linalg.norm(Wr, axis=1).max()))
    F = lg[:beyond[0] + 1] if out["ok"] else lg
    dev = TREE.query(F[:, 4:7])[0]                     # true distance to the planned path (same for both laws)
    k_settle = np.nonzero(dev > 0.05)[0]
    out.update(t=float(F[-1, 0]), dev_after=float(dev[F[:, 0] > 1.0].max()) if (F[:, 0] > 1.0).any() else np.nan,
               dev_max=float(dev.max()), dev_rms=float(np.sqrt(np.mean(dev ** 2))), clear=B.clearance(F[:, 4:7], OBS),
               settle=float(F[k_settle[-1], 0]) if len(k_settle) else 0.0, sat=float(F[:, 8].mean()), F=F, dev=dev)
    return out


OBS = None
TREE = None


def main():
    global OBS, TREE
    from scipy.spatial import cKDTree
    quad = B.qs.Quad()
    lines = []

    def log(s_):
        print(s_, flush=True); lines.append(s_)

    log(f"blended B-spline pieces: core turning {np.degrees(TH_CORE):.0f} deg, overlap +-{D} m, knots <= {KNOT} m, degree {K}, "
        f"lateral window {R_IN}..{R_OUT} m, gains wp {WP} wH {WH}; source {SRC.relative_to(HERE.parent)}, mazes {SEEDS}")
    rows = []
    for seed in SEEDS:
        OBS = B.perfect_maze(seed)
        src = B.load(SRC, seed)
        spec = list(src["spec"])
        T_plan = float(src["time"])
        t0 = time.perf_counter()
        pieces, P, s, cuts = build(spec)
        tb = time.perf_counter() - t0
        TREE = cKDTree(P)
        n_old = len(spec) * (13 + len(spec[0]["f"]) + len(spec[0]["g"]) + len(spec[0]["h"]))
        n_new = sum(6 + 2 + 4 + 3 * pc.ncoef for pc in pieces)          # O, e1, x range, rise/fall, f g h
        n_new_2d = sum(6 + 2 + 4 + 2 * pc.ncoef for pc in pieces)       # g not stored (constant height)
        log(f"maze {seed}: path {s[-1]:.1f} m, plan {T_plan:.2f} s | old: {len(spec)} Chebyshev pieces, {n_old} numbers | "
            f"new: {len(pieces)} pieces ({tb:.1f} s), lengths {min(pc.xb - pc.xa for pc in pieces):.1f}..{max(pc.xb - pc.xa for pc in pieces):.1f} m, "
            f"max tilt from chord {max(pc.tilt for pc in pieces):.0f} deg, fit err {max(pc.fit_err for pc in pieces) * 1000:.2f} mm, "
            f"V err {max(pc.V_err for pc in pieces):.3f}, {n_new} numbers ({n_new_2d} without g)")
        for off in OFFSETS:
            r_old = fly(quad, fe.PolyELPolicy(spec, wp=WP, wH=WH), spec, 1.3 * T_plan, off)
            bp = BlendPolicy(pieces)
            r_new = fly(quad, bp, spec, 1.3 * T_plan, off)
            mix = np.array(bp.nmix[:int(r_new["t"] / 0.001) + 1])
            for name, r in (("old (switch)", r_old), ("new (blend)", r_new)):
                log(f"  offset {off:.1f} m {name:13s}: {'goal' if r['ok'] else 'NO GOAL'} at {r['t']:.2f} s | off path max {r['dev_max']:.3f} m, "
                    f"after 1 s {r['dev_after']:.3f} m, last > 5 cm at {r['settle']:.2f} s | clearance {r['clear']:.2f} m | "
                    f"sat {100 * r['sat']:.1f}% | max |w| {r['w_max']:.0f}, max step of w {r['dw_max']:.1f} (99.9% {r['dw_p99']:.1f})"
                    + (f" | steps blending 2+ laws {100 * np.mean(mix >= 2):.0f}%, fallback {100 * np.mean(mix == 0):.1f}%" if name.startswith("new") else ""))
                rows.append((seed, f"offset {off:.1f}", name, r))
            if off == OFFSETS[-1] or off == 0.0:
                plot(HERE / "results" / f"maze{seed}_off{off:g}.png", seed, pieces, P, cuts, s, r_old, r_new, f"start offset {off:g} m sideways")
        # gust test: push sideways while crossing every second cut (times from the undisturbed new-law flight)
        F0 = fly(quad, BlendPolicy(pieces), spec, 1.3 * T_plan, 0.0)["F"]
        gusts, sign = [], 1.0
        for c_ in cuts[1:-1:2]:
            q = np.array([np.interp(c_, s, P[:, k]) for k in range(3)])
            k = int(np.argmin(np.linalg.norm(F0[:, 4:7] - q, axis=1)))
            tang = np.array([np.interp(c_ + 0.1, s, P[:, i]) - np.interp(c_ - 0.1, s, P[:, i]) for i in range(3)])
            nrm = np.cross([0, 0, 1.0], tang / np.linalg.norm(tang))
            gusts.append((F0[k, 0] - 0.5 * GUST_T, sign * GUST_A * nrm)); sign = -sign
        r_old = fly(quad, fe.PolyELPolicy(spec, wp=WP, wH=WH), spec, 1.3 * T_plan, 0.0, gusts)
        r_new = fly(quad, BlendPolicy(pieces), spec, 1.3 * T_plan, 0.0, gusts)
        for name, r in (("old (switch)", r_old), ("new (blend)", r_new)):
            log(f"  {len(gusts)} gusts {name:13s}: {'goal' if r['ok'] else 'NO GOAL'} at {r['t']:.2f} s | off path max {r['dev_max']:.3f} m, "
                f"rms {r['dev_rms']:.3f} m | clearance {r['clear']:.2f} m | sat {100 * r['sat']:.1f}% | max |w| {r['w_max']:.0f}, "
                f"max step of w {r['dw_max']:.1f} (99.9% {r['dw_p99']:.1f})")
            rows.append((seed, f"{len(gusts)} gusts", name, r))
        plot(HERE / "results" / f"maze{seed}_gusts.png", seed, pieces, P, cuts, s, r_old, r_new,
             f"{len(gusts)} sideways gusts of {GUST_A:g} m/s^2 x {GUST_T:g} s while crossing cuts", gusts)
    log("\nsummary: maze | test | law | goal | time s | off path max m | off path rms m | max step of w")
    for seed, test, name, r in rows:
        log(f"{seed} | {test} | {name} | {'yes' if r['ok'] else 'NO'} | {r['t']:.2f} | {r['dev_max']:.3f} | {r['dev_rms']:.3f} | {r['dw_max']:.1f}")
    (HERE / "results" / "run.log").write_text("\n".join(lines) + "\n")


def plot(fn, seed, pieces, P, cuts, s, r_old, r_new, what, gusts=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(11, 10), gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax = axs[0]
    for x1, y1, x2, y2, rr in OBS:
        ax.plot([x1, x2], [y1, y2], color="0.45", lw=4, solid_capstyle="round")
    ax.plot(P[:, 0], P[:, 1], "-", color="0.7", lw=3, label="planned path")
    for c_ in cuts[1:-1]:
        q = np.array([np.interp(c_, s, P[:, k]) for k in range(2)])
        ax.plot(*q, "o", ms=5, mfc="none", color="tab:purple")
    ax.plot(r_old["F"][:, 4], r_old["F"][:, 5], "-", color="tab:red", lw=1, label="old: Chebyshev pieces, hard switch")
    ax.plot(r_new["F"][:, 4], r_new["F"][:, 5], "-", color="tab:blue", lw=1, label="new: B-spline pieces, blended")
    ax.set_aspect("equal"); ax.set_xlim(0, B.W); ax.set_ylim(0, B.H)
    for t0, g in gusts:
        k = int(np.searchsorted(r_new["F"][:, 0], t0))
        if k < len(r_new["F"]):
            ax.annotate("", xy=r_new["F"][k, 4:6] + 1.5 * g[:2] / GUST_A, xytext=r_new["F"][k, 4:6],
                        arrowprops=dict(arrowstyle="->", color="tab:orange"))
    ax.set_title(f"perfect maze {seed}: {what} ({len(pieces)} new pieces; circles = cuts)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax = axs[1]
    ax.semilogy(r_old["F"][:, 0], np.maximum(r_old["dev"], 1e-5), color="tab:red", lw=1, label="old")
    ax.semilogy(r_new["F"][:, 0], np.maximum(r_new["dev"], 1e-5), color="tab:blue", lw=1, label="new")
    for t0, g in gusts:
        ax.axvspan(t0, t0 + GUST_T, color="tab:orange", alpha=0.2)
    ax.set_xlabel("t s"); ax.set_ylabel("distance to planned path m"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fn, dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
