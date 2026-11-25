"""Minimum-curvature path (race-line style, Heilmeier et al. 2019) on the grid mazes; the result goes through the shared
back end (experiments/common/bench.py: G3 fit -> speed plan -> flight).

Reference line: A* (pillars inflated CLEAR + 0.75 cell) -> line-of-sight shortcut -> resampled every DS.
Each point may only move along its normal n_i:  p_i = r_i + alpha_i n_i,  ends fixed at start and goal.
Bounds (the "track width"): along the normal line, the free interval around alpha = 0 outside every pillar inflated by
CLEAR and inside the map (CLEAR from its edge); on the inner side of a bend also |alpha| <= 0.8 * (radius at which
neighbouring normals cross), so the points keep their order.
  min  sum_i k_i^2 h_i        (integral of curvature^2), k_i = circumcircle curvature of p_{i-1}, p_i, p_{i+1}
  s.t. lo_i <= alpha_i <= hi_i
Solved as an NLP with IPOPT (same optimum as the linearised QP iterations, without the linearisation).
Repeated N_ITER times, each time with the previous result as the reference line (new normals, new bounds), so the
first, polyline reference does not limit the result.
No weights, no design speed: the only number is CLEAR, which every method already has.

    cd experiments/2026-11-24_min_curvature
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [mazes...]
"""
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
SEEDS = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
import bench as B                           # noqa: E402

DS = 0.5                                    # reference point spacing (m)
SMOOTH = 1.0                                # Gaussian smoothing (m) of the tangent of the first, polyline reference
N_ITER = 3
A_MAX = 8.0                                 # search range along a normal (m)


def reference(pil):
    occ, cells = B.astar(pil, B.CLEAR + 0.75 * B.CELL)
    poly = B.shortcut(occ, cells)
    return B.resample(poly, DS), poly


def normals(R, smooth):
    t = np.gradient(R, axis=0)
    if smooth > 0:                          # Gaussian-smoothed tangent: a polyline has no normal at its corners
        from scipy.ndimage import gaussian_filter1d
        t = gaussian_filter1d(t, smooth / DS, axis=0, mode="nearest")
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.column_stack([-t[:, 1], t[:, 0]]), t


def bounds(R, N, T, pil):
    n = len(R)
    lo, hi = np.full(n, -A_MAX), np.full(n, A_MAX)
    for i in range(n):
        p, d = R[i], N[i]
        for cx, cy, r in pil:                       # |p + a d - c|^2 = (r + CLEAR)^2
            q = p - (cx, cy)
            b, c = q @ d, q @ q - (r + B.CLEAR) ** 2
            disc = b * b - c
            if disc <= 0:
                continue
            a1, a2 = -b - np.sqrt(disc), -b + np.sqrt(disc)
            if a2 <= 0:
                lo[i] = max(lo[i], a2)
            elif a1 >= 0:
                hi[i] = min(hi[i], a1)
            else:                                    # reference point inside (only by grid rounding): stay put
                lo[i] = hi[i] = 0.0
        for k, (mn, mx) in enumerate(((B.CLEAR, B.W - B.CLEAR), (B.CLEAR, B.H - B.CLEAR))):
            if abs(d[k]) > 1e-9:
                a, b2 = (mn - p[k]) / d[k], (mx - p[k]) / d[k]
                a, b2 = min(a, b2), max(a, b2)
                lo[i], hi[i] = max(lo[i], a), min(hi[i], b2)
    # neighbouring normals cross at radius ds / dtheta on the inner side of a bend: stay within 0.8 of it
    th = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
    dth = np.gradient(th)
    rad = DS / np.maximum(np.abs(dth), 1e-9)
    left = dth > 0                                   # bend to the left: inner side is +normal
    hi[left] = np.minimum(hi[left], 0.8 * rad[left])
    lo[~left] = np.maximum(lo[~left], -0.8 * rad[~left])
    lo[0] = hi[0] = lo[-1] = hi[-1] = 0.0
    return lo, hi


def min_curv(R, N, lo, hi):
    n = len(R)
    a = ca.MX.sym("a", n)
    p = ca.DM(R.T) + ca.repmat(a.T, 2, 1) * ca.DM(N.T)
    u, v = p[:, 1:-1] - p[:, :-2], p[:, 2:] - p[:, 1:-1]
    w = p[:, 2:] - p[:, :-2]
    lu, lv, lw = (ca.sqrt(ca.sum1(z ** 2) + 1e-12) for z in (u, v, w))
    k = 2 * (u[0, :] * v[1, :] - u[1, :] * v[0, :]) / (lu * lv * lw)
    J = ca.sum2(k ** 2 * (lu + lv) / 2)
    S = ca.nlpsol("mc", "ipopt", {"x": a, "f": J},
                  {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 3000, "ipopt.tol": 1e-8})
    t0 = time.perf_counter()
    sol = S(x0=np.clip(0, lo, hi), lbx=lo, ubx=hi)
    al = np.array(sol["x"]).ravel()
    return R + al[:, None] * N, dict(status=S.stats()["return_status"], iters=S.stats()["iter_count"],
                                     t=time.perf_counter() - t0, J=float(sol["f"]),
                                     at_bound=float(np.mean((al <= lo + 1e-3) | (al >= hi - 1e-3))))


def main():
    quad = B.qs.Quad()
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    log(f"minimum curvature: reference spacing {DS} m, first tangent smoothed {SMOOTH} m, {N_ITER} rounds, "
        f"clearance {B.CLEAR} m, mazes {SEEDS}")
    rows = []
    for seed in SEEDS:
        pil = B.maze(seed)
        R, poly = reference(pil)
        R0, msg = R, []
        for it in range(N_ITER):
            Nn, T = normals(R, SMOOTH if it == 0 else 0.0)
            lo, hi = bounds(R, Nn, T, pil)
            P, info = min_curv(R, Nn, lo, hi)
            st = B.shape_stats(P)
            msg.append(f"round {it + 1}: {info['status']} ({info['iters']} it, {info['t']:.1f} s) int k^2 {info['J']:.3f}, "
                       f"{st['length']:.1f} m, k max {st['kmax']:.2f}, {100 * info['at_bound']:.0f}% at a bound, "
                       f"clearance {B.clearance(P, pil):.2f} m")
            R = B.resample(P, DS)
        log(f"maze {seed}: reference {B.shape_stats(R0)['length']:.1f} m ({len(poly)}-vertex polyline)")
        for m in msg:
            log("  " + m)
        try:
            res = B.finish(P, pil, quad, log=log)
        except RuntimeError as e:
            log(f"  back end failed: {e}")
            continue
        rows.append((seed, "min curvature", res))
        B.save(HERE, seed, res)

        def extra(ax, R0=R0, P=P):
            ax.plot(R0[:, 0], R0[:, 1], ".", ms=2, color="0.6", label="reference (A* shortcut)")
            ax.plot(P[:, 0], P[:, 1], "-", lw=0.8, color="k", label="minimum curvature")
        B.plot(out_dir / f"maze{seed}.png", seed, pil, res, f"minimum curvature, maze {seed}", extra)
    B.summary(rows, log)
    (out_dir / "run.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
