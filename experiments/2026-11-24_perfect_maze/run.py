"""Three path methods on perfect mazes (walls between grid points, one way between any two cells; user's choice,
2026-11-24): baseline corridor smoothing, corridor + min snap, minimum curvature. The elastic band is left out (it
needed hand-set limits; decided not to tune it further).

Maze: experiments/common/bench.py perfect_maze(seed): 8 x 6 grid points, spacing 5 m, walls 0.5 m thick, outer ring
closed except a door on the left (start side) and on the right (goal side) at random rows; map height = grid height, so
there is no way around the maze. Obstacles are capsules (segment + radius).

The methods are imported from their own experiments unchanged, except the two parts that assumed round pillars:
  - min snap corridor: tangent half-plane of each CLEAR-inflated capsule at its point closest to the polyline segment
    (same construction; for a pillar it is the old one)
  - minimum curvature bounds: free interval along each normal found by sampling (every 0.02 m) instead of the
    circle formula
Shared back end unchanged (G3 fit -> speed plan -> flight).

    cd experiments/2026-11-24_perfect_maze
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [mazes...]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "common"))
SEEDS = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
import bench as B                           # noqa: E402


def _load(name, d):
    argv, sys.argv = sys.argv, sys.argv[:1]  # the experiment scripts read sys.argv when imported
    spec = importlib.util.spec_from_file_location(name, EXP / d / "run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


BL = _load("bl", "2026-11-24_baseline_corridor_smooth")
MS = _load("ms", "2026-11-24_corridor_minsnap")
MC = _load("mc", "2026-11-24_min_curvature")


def closest_pair(a, b, p, q):
    """Closest points of segments ab and pq (assumed not crossing): (on ab, on pq)."""
    def on(x, u, v):
        d = v - u
        return u + np.clip((x - u) @ d / max(d @ d, 1e-12), 0, 1) * d
    cands = [(a, on(a, p, q)), (b, on(b, p, q)), (on(p, a, b), p), (on(q, a, b), q)]
    return min(cands, key=lambda c: np.linalg.norm(c[0] - c[1]))


def ms_corridor(obs):
    occ, path = B.astar(obs, B.CLEAR + 0.75 * B.CELL)
    poly = B.shortcut(occ, path)
    polys, pts = [], [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        A, bb = [[-1, 0], [1, 0], [0, -1], [0, 1]], [0, B.W, 0, B.H]
        for x1, y1, x2, y2, r in obs:
            s, c = closest_pair(a, b, np.array([x1, y1]), np.array([x2, y2]))
            n = (s - c) / np.linalg.norm(s - c)
            A.append(-n); bb.append(-(n @ c + r + B.CLEAR))
        d = b - a
        k = int(np.ceil(np.linalg.norm(d) / MS.SEG_MAX))
        for j in range(1, k + 1):
            polys.append((np.array(A, float), np.array(bb, float)))
            pts.append(a + d * j / k)
    return polys, np.array(pts), path, poly


def mc_bounds(R, N, T, obs):
    al = np.arange(-MC.A_MAX, MC.A_MAX + 1e-9, 0.02)
    i0 = int(np.argmin(np.abs(al)))
    n = len(R)
    lo, hi = np.zeros(n), np.zeros(n)
    for i in range(n):
        Q = R[i] + al[:, None] * N[i]
        free = (B.seg_dist(Q, obs) - obs[None, :, 4] >= B.CLEAR).all(1)
        free &= (Q[:, 0] >= B.CLEAR) & (Q[:, 0] <= B.W - B.CLEAR) & (Q[:, 1] >= B.CLEAR) & (Q[:, 1] <= B.H - B.CLEAR)
        if not free[i0]:
            continue
        j = i0
        while j > 0 and free[j - 1]:
            j -= 1
        k = i0
        while k < len(al) - 1 and free[k + 1]:
            k += 1
        lo[i], hi[i] = al[j], al[k]
    th = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
    dth = np.gradient(th)
    rad = MC.DS / np.maximum(np.abs(dth), 1e-9)
    left = dth > 0
    hi[left] = np.minimum(hi[left], 0.8 * rad[left])
    lo[~left] = np.maximum(lo[~left], -0.8 * rad[~left])
    lo[0] = hi[0] = lo[-1] = hi[-1] = 0.0
    return lo, hi


def run_baseline(obs, log):
    S, info = BL.plan(obs)
    log(f"  A* {len(info['cells'])} cells -> {len(info['poly'])} vertices | smoothing {info['smooth']['status']}")
    return S, None


def run_minsnap(obs, log, a_lat):
    polys, joints, path, poly = ms_corridor(obs)
    Cs, Ts, info = MS.minsnap(polys, joints, a_lat)
    P2 = np.vstack([MS.bern(Cp, np.linspace(0, 1, 200, endpoint=False)) for Cp in Cs] + [Cs[-1][-1:]])
    log(f"  {len(poly)}-vertex polyline, {len(polys)} segments | stage 1 {info['stage1']}, stage 2 {info['status']} "
        f"({info['iters']} it, {info['t']:.1f} s), own time {Ts.sum():.2f} s")

    def extra(ax):
        ax.plot(poly[:, 0], poly[:, 1], "--", lw=0.8, color="tab:orange", label="shortcut polyline")
        for k, (A, bb) in enumerate(polys):
            try:
                V = MS.polygon_vertices(A, bb, (joints[k] + joints[k + 1]) / 2)
            except Exception:
                continue
            V = np.vstack([V, V[:1]])
            ax.plot(V[:, 0], V[:, 1], "-", lw=0.6, color="tab:green", alpha=0.6, label="corridor polygons" if k == 0 else None)
    return P2, extra


def run_mincurv(obs, log):
    R, poly = MC.reference(obs)
    R0 = R
    for it in range(MC.N_ITER):
        Nn, T = MC.normals(R, MC.SMOOTH if it == 0 else 0.0)
        lo, hi = mc_bounds(R, Nn, T, obs)
        P, info = MC.min_curv(R, Nn, lo, hi)
        log(f"  round {it + 1}: {info['status']} ({info['iters']} it, {info['t']:.1f} s) int k^2 {info['J']:.3f}, "
            f"{B.shape_stats(P)['length']:.1f} m, clearance {B.clearance(P, obs):.2f} m")
        R = B.resample(P, MC.DS)

    def extra(ax):
        ax.plot(R0[:, 0], R0[:, 1], ".", ms=2, color="0.6", label="reference (A* shortcut)")
    return P, extra


METHODS = ("baseline", "min snap", "min curvature")


def main():
    quad = B.qs.Quad()
    a_lat = B.a_lat(quad)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    log(f"perfect mazes {SEEDS}: {B.PM_NX} x {B.PM_NY} points, spacing {B.SP} m, walls {2 * B.WALL_R} m, "
        f"clearance {B.CLEAR} m")
    rows = []
    for seed in SEEDS:
        obs = B.perfect_maze(seed)
        log(f"maze {seed}: start {B.START}, goal {B.GOAL}, {len(obs)} wall segments")
        for name in METHODS:
            log(f" {name}:")
            d = HERE / "results" / name.replace(" ", "_")
            try:
                if name == "baseline":
                    P, extra = run_baseline(obs, log)
                elif name == "min snap":
                    P, extra = run_minsnap(obs, log, a_lat)
                else:
                    P, extra = run_mincurv(obs, log)
                res = B.finish(P, obs, quad, log=log)
            except Exception as e:
                log(f"  failed: {type(e).__name__}: {e}")
                continue
            rows.append((seed, name, res))
            B.save(None, seed, res, res_dir=d)
            cmp = None if name == "baseline" else HERE / "results" / "baseline"
            B.plot(d / f"maze{seed}.png", seed, obs, res, f"{name}, perfect maze {seed}", extra, compare=cmp)
    log("\nsummary:")
    log("maze | method | plan time s | flown s | length m | turns | k max 1/m | clearance flown m | |phi| max m | goal miss m")
    for seed, name, r in rows:
        log(f"{seed} | {name} | {r['time']:.2f} | {r.get('t_fly', float('nan')):.2f} | {r['shape']['length']:.1f} | "
            f"{r['shape']['turns']} | {r['shape']['kmax']:.2f} | {r.get('clear_fly', float('nan')):.2f} | "
            f"{r.get('phi_max', float('nan')):.3f} | {r.get('goal_miss', float('nan')):.2f}")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "run.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
