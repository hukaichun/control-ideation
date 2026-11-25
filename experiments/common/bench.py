"""Shared maze and back end for the path-planning experiments (not an experiment itself).

Maze (user's design, 2026-11-24): round pillars on the points of a uniform grid (NX x NY points, spacing SP); a maze
is one draw of random radii (uniform R_LO..R_HI, seed = maze number); start and goal outside the grid, left and right.
Flight at constant height Z, one way from start to goal, no stop at the goal.

A method only has to produce a dense open path P (n x 3) from start to goal; this module then does the same back end
for every method:
  P -> pieces_from (graph pieces, <= 60 deg, <= 6 m) -> smooth_path fit (G3) -> _plan speed (motor margin 0.9,
    V_START at the start, free at the goal, at most V_MAX) -> fly PolyELPolicy from the start until it passes the goal -> metrics.
Results are saved per maze (results/maze<k>.npz in the experiment's directory) so other experiments can overlay them.
"""
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
sys.path.insert(0, str(ROOT / "examples"))
_argv, sys.argv = sys.argv, sys.argv[:1]         # astar_demo reads sys.argv when imported
import astar_demo as ad          # noqa: E402
sys.argv = _argv
import flat_el_policy as fe      # noqa: E402
import poly_plan as pp           # noqa: E402
import quad_sim as qs            # noqa: E402

# maze
NX, NY, SP = 6, 4, 5.0
X0, Y0 = 10.0, 5.0               # first grid point; grid x 10..35, y 5..20
R_LO, R_HI = 0.5, 2.4                 # user, 2026-11-24 (0.3-1.8 left straight lanes between every two rows)
W, H = 45.0, 25.0
START, GOAL = np.array([3.0, 10.0]), np.array([42.0, 15.0])   # on the lines of grid rows 2 and 3: no straight way
Z, CELL = 2.0, 0.25
R_DRONE = 0.25
CLEAR = R_DRONE + 0.25           # required distance from a pillar surface to the path (drone radius + 0.25 m)
MARGIN = 0.9                     # motor margin of the speed plan
V_START = 1.0                    # m/s at the start (leaves from almost hovering)
V_MAX = 15.0                     # m/s speed ceiling (without it the one-way plan accelerates without bound on straights)
RESERVE = 0.3                    # share of the thrust kept for recovery when shaping the path


def maze(seed):
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(X0 + SP * np.arange(NX), Y0 + SP * np.arange(NY), indexing="ij")
    r = rng.uniform(R_LO, R_HI, gx.size)
    return np.column_stack([gx.ravel(), gy.ravel(), r])


# --- perfect maze (user's choice, 2026-11-24): walls between neighbouring grid points, carved as a perfect maze
# (exactly one way between any two cells, randomised depth-first search), outer ring closed except two doors.
# Obstacles are capsules (x1, y1, x2, y2, r): a wall is the segment between two grid points with radius WALL_R; a pillar
# of the round-pillar mazes is the capsule (x, y, x, y, r). Every helper below takes either form.
PM_NX, PM_NY = 8, 6              # grid points: 7 x 5 cells of SP
PM_X0 = 10.0                     # grid x 10..45; grid y 0..25 = the map's height, so there is no way around the maze
WALL_R = 0.25                    # walls 0.5 m thick


def perfect_maze(seed):
    """Walls of a perfect maze as capsules; sets this module's W, H, START, GOAL for the maze (doors at random rows)."""
    global W, H, START, GOAL
    rng = np.random.default_rng(seed)
    cx, cy = PM_NX - 1, PM_NY - 1
    h = np.ones((PM_NX - 1, PM_NY), bool)        # wall (i, j)-(i+1, j)
    v = np.ones((PM_NX, PM_NY - 1), bool)        # wall (i, j)-(i, j+1)
    seen = np.zeros((cx, cy), bool)
    st = [(0, int(rng.integers(cy)))]
    seen[st[0]] = True
    while st:
        x, y = st[-1]
        nb = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= x + dx < cx and 0 <= y + dy < cy and not seen[x + dx, y + dy]]
        if not nb:
            st.pop()
            continue
        n = nb[rng.integers(len(nb))]
        seen[n] = True
        st.append(n)
        if n[0] != x:
            v[max(x, n[0]), y] = False
        else:
            h[x, max(y, n[1])] = False
    ds, dg = int(rng.integers(cy)), int(rng.integers(cy))
    v[0, ds] = v[-1, dg] = False
    P = lambda i, j: (PM_X0 + SP * i, SP * j)
    caps = [(*P(i, j), *P(i + 1, j), WALL_R) for i, j in zip(*np.nonzero(h))] + \
           [(*P(i, j), *P(i, j + 1), WALL_R) for i, j in zip(*np.nonzero(v))]
    W, H = PM_X0 + SP * (PM_NX - 1) + PM_X0, SP * (PM_NY - 1)
    START = np.array([3.0, SP * (ds + 0.5)])
    GOAL = np.array([W - 3.0, SP * (dg + 0.5)])
    return np.array(caps, float)


def caps(obs):
    """Obstacles as capsules (n x 5) from pillars (n x 3) or capsules."""
    obs = np.asarray(obs, float)
    return obs if obs.shape[1] == 5 else np.column_stack([obs[:, :2], obs[:, :2], obs[:, 2]])


def seg_dist(Q, cp):
    """Distance of points Q (m x 2) to the axis of every capsule (m x n)."""
    a, b = cp[:, 0:2], cp[:, 2:4]
    d = b - a
    dd = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((Q[:, None, :2] - a[None]) * d[None]).sum(2) / dd[None], 0, 1)
    c = a[None] + t[..., None] * d[None]
    return np.linalg.norm(Q[:, None, :2] - c, axis=2)


def occupancy(pil, infl):
    xs, ys = np.arange(0, W, CELL) + CELL / 2, np.arange(0, H, CELL) + CELL / 2
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cp = caps(pil)
    Q = np.column_stack([X.ravel(), Y.ravel()])
    occ = (seg_dist(Q, cp) <= cp[None, :, 4] + infl).any(1).reshape(X.shape)
    occ[:, 0] = occ[:, -1] = occ[0, :] = occ[-1, :] = True
    return occ


def astar(pil, infl):
    occ = occupancy(pil, infl)
    return occ, ad.astar(occ, START, GOAL)


def free_line(occ, p, q):
    return ad.free_line(occ, p, q)


def shortcut(occ, pts):
    """Line-of-sight shortcut of a cell path (jump to the farthest cell still visible), vertices turning < 3 deg dropped;
    ends replaced by START and GOAL."""
    out, i, n = [pts[0]], 0, len(pts)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not free_line(occ, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    out = np.array(out)
    keep = [0]
    for i in range(1, len(out) - 1):
        d0, d1 = out[i] - out[keep[-1]], out[i + 1] - out[i]
        if np.arccos(np.clip(d0 @ d1 / np.linalg.norm(d0) / np.linalg.norm(d1), -1, 1)) > np.deg2rad(3):
            keep.append(i)
    keep.append(len(out) - 1)
    out = out[keep]
    out[0], out[-1] = START, GOAL
    return out


def clearance(Q, pil):
    cp = caps(pil)
    return float(np.min(seg_dist(np.asarray(Q)[:, :2], cp) - cp[None, :, 4]))


def a_lat(quad, reserve=RESERVE):
    """Horizontal acceleration available at constant height with `reserve` of the thrust kept back."""
    return float(np.sqrt(((1 - reserve) * quad.A) ** 2 - qs.G ** 2))


def curvature(P):
    """Signed curvature of an open planar polyline (x, y), per vertex (ends copied from their neighbours)."""
    d1 = np.gradient(P[:, :2], axis=0)
    d2 = np.zeros_like(d1)
    d2[1:-1] = P[2:, :2] - 2 * P[1:-1, :2] + P[:-2, :2]
    d2[0], d2[-1] = d2[1], d2[-2]
    ds = np.linalg.norm(d1, axis=1)
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(ds, 1e-9) ** 3


def shape_stats(P):
    """Length, max |curvature|, number of turns (sign changes of curvature where |k| > 0.05 1/m)."""
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    k = curvature(P)
    sig = np.sign(k[np.abs(k) > 0.05])
    turns = int(np.sum(sig[1:] != sig[:-1])) + (1 if len(sig) else 0)
    return dict(length=L, kmax=float(np.abs(k).max()), turns=turns)


def resample(P, ds):
    """Open polyline resampled at spacing ds (end point kept)."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    q = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / ds)) + 1))
    return np.column_stack([np.interp(q, s, P[:, k]) for k in range(P.shape[1])])


def finish(P, pil, quad=None, fly=True, log=print):
    """Back end shared by every method. P: dense open path (n x 2 or n x 3) from start to goal."""
    quad = quad or qs.Quad()
    if P.shape[1] == 2:
        P = np.column_stack([P, np.full(len(P), Z)])
    out, T = {}, {}
    P = resample(P, 0.02)
    t0 = time.perf_counter()
    res = pp.pieces_from(P, np.zeros((0, 3)), closed=False)
    if res is None:
        raise RuntimeError("pieces_from: a piece is not monotone along its chord")
    pcs, _ = res
    zp, bp = pp.smooth_path(pcs, [], np.zeros((0, 3)), 0.0, w_fit=1.0, w_bend=1e-4, closed=False)
    T["fit"] = time.perf_counter() - t0
    spec0 = pp.spec_of(pcs, zp)
    S0 = pp.sample(spec0, 300)
    from scipy.spatial import cKDTree
    out["fit_err"] = float(cKDTree(P[:, :2]).query(S0[:, :2])[0].max())
    sl = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(sl)])
    L = s[-1]
    # starting speed from the curvature of the fitted (smooth) path, not of the dense input (may be a polyline)
    kap = np.abs(curvature(S0))[cKDTree(S0[:, :2]).query(P[:, :2])[1]]
    sig0 = 0.5 * np.minimum(V_MAX, np.sqrt(0.5 * quad.A / np.maximum(kap, 1e-3)))
    motor = dict(m=quad.m, Ix=quad.J[0, 0], l=0.15, Tmax=quad.Tmax, Tmin=quad.Tmin, margin=MARGIN)
    t0 = time.perf_counter()
    r = pp._plan(pcs, [], np.zeros((0, 3)), types.SimpleNamespace(length=L + 1.0), P, s, motor,
                 lambda q: np.interp(q, s, sig0), 0.25, 0.0, 1e-6, 15.0, 0.9, 2.0, 1000, 1e-6, zp, bp,
                 fix_path=True, closed=False, v_max=V_MAX, v_start=V_START)
    T["speed"] = time.perf_counter() - t0
    spec, tt = r["spec"], r["lap"]
    S = pp.sample(spec, 300)
    S = np.vstack([S, pp.point(spec[-1], spec[-1]["l"])])
    out.update(spec=spec, time=tt, status=r["status"], pieces=len(spec), shape=shape_stats(S), clear_plan=clearance(S, pil),
               end_miss=float(np.linalg.norm(S[-1, :2] - GOAL)), times=T, S=S)
    log(f"  fit: {len(spec)} pieces, err {out['fit_err']:.3f} m ({T['fit']:.1f} s) | speed: {r['status']} time "
        f"{tt:.2f} s ({T['speed']:.1f} s) | length {out['shape']['length']:.1f} m, turns {out['shape']['turns']}, "
        f"k max {out['shape']['kmax']:.2f} 1/m, clearance {out['clear_plan']:.2f} m")
    if not fly or not r["status"].startswith("Solve_Succeeded"):
        return out
    pol = fe.PolyELPolicy(spec, wp=16, wH=12)
    p0, v0, a0, j0 = pp.state_at(spec[0], 0.0)
    lg = fe.fly(quad, pol, p0, v0, a0, j0, 1.3 * tt, mode="B")
    # passed the goal: first time the flown point is beyond the goal along the last piece's direction
    e_end = spec[-1]["e1"][:2]
    beyond = np.nonzero((lg[:, 4:6] - GOAL) @ e_end >= 0)[0]
    ok = len(beyond) > 0
    out["flight_ok"] = ok
    if ok:
        k = beyond[0]
        F = lg[:k + 1]
        out.update(t_fly=float(F[-1, 0]), goal_miss=float(np.linalg.norm(F[-1, 4:6] - GOAL)), phi_max=float(F[:, 1].max()),
                   clear_fly=clearance(F[:, 4:7], pil), sat=float(F[:, 8].mean()), wmax=float(F[:, 7].max()),
                   flown=F[:, [0, 4, 5, 6]])
        log(f"  flight: reached the goal line at {out['t_fly']:.2f} s, {out['goal_miss']:.2f} m from the goal | "
            f"|phi| max {out['phi_max']:.3f} m, clearance {out['clear_fly']:.2f} m, sat {100 * out['sat']:.1f}%, "
            f"max |w| {out['wmax']:.1f}")
    else:
        log(f"  flight: did not reach the goal (log ends at t = {lg[-1, 0]:.2f} s)")
    return out


def save(exp_dir, seed, res, res_dir=None):
    """Saved to <exp_dir>/results/maze<seed>.npz, or to res_dir/maze<seed>.npz."""
    keep = {k: v for k, v in res.items() if k in ("S", "time", "t_fly", "flown")}
    d = Path(res_dir) if res_dir else Path(exp_dir) / "results"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"maze{seed}.npz", spec=np.array(res["spec"], dtype=object), **keep)


def load(exp_name, seed):
    """exp_name: an experiment's directory name (its results/), or a Path to a directory of maze<k>.npz."""
    f = (exp_name if isinstance(exp_name, Path) else EXP / exp_name / "results") / f"maze{seed}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {k: d[k] for k in d.files}


BASELINE = "2026-11-24_baseline_corridor_smooth"


def plot(path_png, seed, pil, res, title, extra=None, compare=BASELINE):
    """Flown path coloured by speed (or the plan), with another experiment's path for comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6.5 * (H / W) / (25 / 45)))
    for x1, y1, x2, y2, rr in caps(pil):
        for r_, kw in ((rr, dict(color="0.45", lw=0)), (rr + CLEAR, dict(fill=False, ls=":", color="0.45"))):
            ang = np.arctan2(y2 - y1, x2 - x1)
            t1 = np.linspace(ang + np.pi / 2, ang + 3 * np.pi / 2, 20)
            t2 = t1 + np.pi
            O = np.vstack([np.column_stack([x1 + r_ * np.cos(t1), y1 + r_ * np.sin(t1)]),
                           np.column_stack([x2 + r_ * np.cos(t2), y2 + r_ * np.sin(t2)])])
            ax.add_patch(plt.Polygon(O, closed=True, **kw))
    if extra:
        extra(ax)
    b = load(compare, seed) if compare else None
    if b is not None:
        ax.plot(b["S"][:, 0], b["S"][:, 1], "-", color="tab:red", lw=1, alpha=0.7,
                label=f"baseline (plan {float(b['time']):.2f} s)")
    if "flown" in res:
        F = res["flown"]
        spd = np.linalg.norm(np.diff(F[:, 1:4], axis=0), axis=1) / np.diff(F[:, 0])
        sc = ax.scatter(F[1:, 1], F[1:, 2], c=spd, s=2, cmap="viridis", label=f"flown (plan {res['time']:.2f} s)")
        plt.colorbar(sc, label="speed m/s", shrink=0.8)
    else:
        ax.plot(res["S"][:, 0], res["S"][:, 1], "-", color="tab:blue", label="plan")
    ax.plot(*START, "g^", ms=10, label="start"); ax.plot(*GOAL, "r*", ms=12, label="goal")
    ax.set_aspect("equal"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8, ncol=4)
    ax.set_title(title)
    fig.savefig(path_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def summary(rows, log, compare=BASELINE):
    """rows: (seed, label, res). Table against the comparison experiment's saved results."""
    log(f"\nsummary (comparison: experiments/{compare}):")
    log("maze | variant | plan time s (baseline) | length m (baseline) | turns (baseline) | k max 1/m | "
        "clearance flown m | |phi| max m")
    for seed, label, r in rows:
        b = load(compare, seed) if compare else None
        bs = shape_stats(b["S"]) if b is not None else None
        bt = f"{float(b['time']):.2f}" if b is not None else "-"
        bl = f"{bs['length']:.1f}" if bs else "-"
        bn = f"{bs['turns']}" if bs else "-"
        log(f"{seed} | {label} | {r['time']:.2f} ({bt}) | {r['shape']['length']:.1f} ({bl}) | {r['shape']['turns']} ({bn}) | "
            f"{r['shape']['kmax']:.2f} | {r.get('clear_fly', float('nan')):.2f} | {r.get('phi_max', float('nan')):.3f}")
