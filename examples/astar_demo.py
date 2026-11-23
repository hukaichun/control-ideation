"""A* polyline -> corridor smoothing into G3 piecewise polynomials -> speed plan (V) -> fly the flat E-L law.

Map: round pillars on a plane, flight at constant height. Closed loop through waypoints (A* between consecutive ones).
  1. A* (8-connected) on a grid whose obstacles are inflated by corridor radius + drone radius
  2. shortcut: keep a vertex only when the straight line to a later one is not free (line of sight, inflated map)
     -> every segment's corridor of radius R_C is obstacle-free for the drone
  3. start curve: rounded corners (path_g2, lines + arcs), cut into graph pieces (<= 60 deg, <= 6 m), fitted
  4. smoothing: G3 joints, every sample inside the box around its nearest segment, least bending
  5. speed: poly_plan (motor margin), start from a curvature speed limit
  6. fly PolyELPolicy, distance to the real pillars

    PYTHONPATH=. uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python astar_demo.py [seed] [margin]
"""
import heapq
import sys
import time
import types

import numpy as np

import flat_el_policy as fe
import path_g2 as pg
import poly_plan as pp
import quad_sim as qs

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MARGIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
W, H, CELL, Z = 40.0, 30.0, 0.25, 2.0
R_C, R_DRONE = 1.0, 0.25                      # corridor half-width, drone radius
RZ = 0.3


def make_map(seed, n=22):
    """Pillars anywhere except next to the waypoints (so the A* legs have to weave)."""
    rng = np.random.default_rng(seed)
    wps = np.array([[5, 5], [35, 5], [35, 25], [5, 25]], float)
    pil = []
    while len(pil) < n:
        c, r = rng.uniform([3, 3], [37, 27]), rng.uniform(0.6, 1.6)
        if np.min(np.linalg.norm(wps - c, axis=1)) > r + 3:
            pil.append((c[0], c[1], r))
    return wps, np.array(pil)


def occupancy(pil, infl):
    xs, ys = np.arange(0, W, CELL) + CELL / 2, np.arange(0, H, CELL) + CELL / 2
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    occ = np.zeros(X.shape, bool)
    for cx, cy, r in pil:
        occ |= (X - cx) ** 2 + (Y - cy) ** 2 <= (r + infl) ** 2
    occ[:, 0] = occ[:, -1] = occ[0, :] = occ[-1, :] = True
    return occ


def astar(occ, a, b):
    to = lambda p: (int(p[0] / CELL), int(p[1] / CELL))
    s, g = to(a), to(b)
    h = lambda c: np.hypot(c[0] - g[0], c[1] - g[1])
    openq, came, cost = [(h(s), s)], {s: None}, {s: 0.0}
    while openq:
        _, c = heapq.heappop(openq)
        if c == g:
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (c[0] + dx, c[1] + dy)
                if (dx, dy) == (0, 0) or not (0 <= n[0] < occ.shape[0] and 0 <= n[1] < occ.shape[1]) or occ[n]:
                    continue
                nc = cost[c] + np.hypot(dx, dy)
                if nc < cost.get(n, np.inf):
                    cost[n], came[n] = nc, c
                    heapq.heappush(openq, (nc + h(n), n))
    out, c = [], g
    while c is not None:
        out.append(((c[0] + 0.5) * CELL, (c[1] + 0.5) * CELL))
        c = came[c]
    return np.array(out[::-1])


def free_line(occ, p, q):
    n = int(np.ceil(np.linalg.norm(q - p) / (CELL / 4))) + 1
    for t in np.linspace(0, 1, n):
        c = ((p + t * (q - p)) / CELL).astype(int)
        if occ[c[0], c[1]]:
            return False
    return True


def _wp_ok(poly, wps, tol=0.5):
    for w in wps:
        d = np.inf
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            lam = np.clip((w - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
            d = min(d, np.linalg.norm(w - a - lam * (b - a)))
        if d > tol:
            return False
    return True


def shortcut(occ, legs):
    """Each A* leg (waypoint to waypoint): from each kept vertex jump to the farthest vertex still in line of sight;
    the legs joined into one closed polyline."""
    out = []
    for pts in legs:
        i, n = 0, len(pts)
        out.append(pts[0])
        while i < n - 1:
            j = n - 1
            while j > i + 1 and not free_line(occ, pts[i], pts[j]):
                j -= 1
            if j < n - 1:
                out.append(pts[j])
            i = j
    out = np.array(out)
    changed = True                               # drop a vertex when its neighbours see each other and every
    while changed and len(out) > 3:              # waypoint stays within 0.5 m of the polyline
        changed = False
        for i in range(len(out)):
            cand = np.delete(out, i, 0)
            if free_line(occ, out[i - 1], out[(i + 1) % len(out)]) and _wp_ok(cand, WPS):
                out, changed = cand, True
                break
    keep = []                                    # drop vertices that (almost) do not turn
    for i in range(len(out)):
        d0, d1 = out[i] - out[i - 1], out[(i + 1) % len(out)] - out[i]
        if np.arccos(np.clip(d0 @ d1 / np.linalg.norm(d0) / np.linalg.norm(d1), -1, 1)) > np.deg2rad(3):
            keep.append(i)
    return out[keep]


def main():
    quad = qs.Quad()
    global WPS
    wps, pil = make_map(SEED)
    WPS = wps
    T = {}
    t0 = time.perf_counter()
    occ = occupancy(pil, R_C + R_DRONE)
    legs = [astar(occ, wps[i], wps[(i + 1) % len(wps)]) for i in range(len(wps))]
    raw = np.vstack([lg[:-1] for lg in legs])
    T["astar"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    poly = shortcut(occ, legs)
    T["shortcut"] = time.perf_counter() - t0
    verts = np.column_stack([poly, np.full(len(poly), Z)])
    boxes = []
    for i in range(len(verts)):
        a, b = verts[i], verts[(i + 1) % len(verts)]
        u = (b - a) / np.linalg.norm(b - a)
        n1 = np.cross([0, 0, 1.0], u); n1 /= np.linalg.norm(n1)
        boxes.append(dict(a=a, u=u, n1=n1, n2=np.cross(u, n1), len=np.linalg.norm(b - a), r=R_C, rz=RZ))
    t0 = time.perf_counter()
    # start curve: corner size so that the arc stays within R_C / 2 of the vertex (T tan-ratio of a fillet)
    fr = np.zeros(len(verts))
    for i in range(len(verts)):
        d0, d1 = verts[i] - verts[i - 1], verts[(i + 1) % len(verts)] - verts[i]
        L0, L1 = np.linalg.norm(d0), np.linalg.norm(d1)
        th = np.arccos(np.clip(d0 @ d1 / L0 / L1, -1, 1))
        T_max = 0.5 * R_C * np.tan(th / 2) / (1 / np.cos(th / 2) - 1)
        fr[i] = min(0.9, T_max / (0.5 * min(L0, L1)))
    path = pg.PathG2(verts, fr, 0.0)
    P, Tn, K, D, sl = path.sample(0.02)
    pcs, _ = pp.pieces_from(P, np.zeros((0, 3)))
    zp, bp = pp.smooth_path(pcs, [], np.zeros((0, 3)), 0.0, w_fit=1e-3, w_bend=1.0, boxes=boxes,
                            discs=[(v, R_C) for v in verts], w_box=1e3)
    print(f"  smoothing: {pp.SMOOTH_STATS}", flush=True)
    T["smooth"] = time.perf_counter() - t0
    s = np.concatenate([[0], np.cumsum(sl)[:-1]])
    kap = np.linalg.norm(K, axis=1)
    sig0 = 0.5 * np.minimum(15.0, np.sqrt(0.5 * quad.A / np.maximum(kap, 1e-3)))   # slow start: converges
    motor = dict(m=quad.m, Ix=quad.J[0, 0], l=0.15, Tmax=quad.Tmax, Tmin=quad.Tmin, margin=MARGIN)
    t0 = time.perf_counter()
    r = pp._plan(pcs, [], np.zeros((0, 3)), types.SimpleNamespace(length=sl.sum()), P, s, motor,
                 lambda q: np.interp(q, s, sig0, period=sl.sum()), 0.25, 0.0, 1e-6, 15.0, 0.9, 2.0, 1000, 1e-6, zp, bp,
                 fix_path=True)
    T["speed"] = time.perf_counter() - t0
    spec, lap = r["spec"], r["lap"]
    gv = np.array(r["g_samples"]); dtv = np.array(r["dt_samples"]).ravel()
    names = ["speed floor", "thrust floor", "body rate", "thrust ceiling", "thrust min", "motor ceiling", "motor floor"]
    act = [gv[0] < 0.25 * 1.05, gv[1] < 2.0 ** 2 * 1.05, gv[2] > 0.97, gv[3] < 0.03 * quad.Tmax * 4,
           gv[4] < 0.03 * quad.Tmax * 4, gv[5] < 0.03, gv[6] < 0.03]
    print("  time share at a limit: " + ", ".join(f"{nm} {100 * dtv[a_].sum() / dtv.sum():.0f}%"
                                                  for nm, a_ in zip(names, act)) + f", none {100 * dtv[~np.any(act, 0)].sum() / dtv.sum():.0f}%", flush=True)
    S = pp.sample(spec, 300)
    # corridor check of the plan and clearance to the real pillars
    box_out = max(min(max(0, np.abs((p - b["a"]) @ b["n1"]) - b["r"]) + max(0, -(p - b["a"]) @ b["u"])
                      + max(0, (p - b["a"]) @ b["u"] - b["len"]) for b in boxes) for p in S)
    clear = lambda Q: np.min(np.hypot(Q[:, None, 0] - pil[None, :, 0], Q[:, None, 1] - pil[None, :, 1]) - pil[None, :, 2])
    print(f"map {SEED}: A* {len(raw)} cells -> {len(poly)} vertices -> {len(spec)} pieces | plan {r['status']} lap {lap:.2f} s "
          f"| plan outside corridor by {box_out:.3f} m, clearance {clear(S):.2f} m | time: A* {T['astar']:.2f} "
          f"shortcut {T['shortcut']:.2f} smooth {T['smooth']:.2f} speed {T['speed']:.2f} s", flush=True)
    np.savez(f"../logs/astar_demo_{SEED}.npz", spec=np.array(spec, dtype=object), pil=pil, wps=wps, lap=lap)
    pol = fe.PolyELPolicy(spec, wp=16, wH=12)
    p0, v0, a0, j0 = pp.state_at(spec[0], 0.0)
    log = fe.fly(quad, pol, p0, v0, a0, j0, 2.2 * lap, mode="B")
    ok = log[-1, 0] > 2.19 * lap
    L2 = log[(log[:, 0] >= lap) & (log[:, 0] < 2 * lap)]
    if not ok:
        print(f"  flight: failed at t = {log[-1, 0]:.2f} s", flush=True)
        return
    print(f"  flight: ok {ok} | lap 2: |phi| max {L2[:, 1].max():.3f} mean {L2[:, 1].mean():.4f} m, clearance "
          f"{clear(L2[:, 4:7]):.2f} m (drone radius {R_DRONE}) | sat {100 * log[:, 8].mean():.0f}% max |w| {log[:, 7].max():.1f}",
          flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for cx, cy, rr in pil:
        ax.add_patch(plt.Circle((cx, cy), rr, color="0.45"))
        ax.add_patch(plt.Circle((cx, cy), rr + R_DRONE, fill=False, ls=":", color="0.45"))
    ax.plot(raw[:, 0], raw[:, 1], ".", ms=2, color="0.7", label="A* cells")
    pc_ = np.vstack([poly, poly[:1]])
    ax.plot(pc_[:, 0], pc_[:, 1], "--", color="tab:orange", lw=1, label="shortcut polyline (corridor ±1 m)")
    spd = np.linalg.norm(np.diff(L2[:, 4:7], axis=0), axis=1) / np.diff(L2[:, 0])
    sc = ax.scatter(L2[1:, 4], L2[1:, 5], c=spd, s=2, cmap="viridis", label="flown (lap 2), colour = speed")
    plt.colorbar(sc, label="speed m/s")
    ax.plot(wps[:, 0], wps[:, 1], "r*", ms=12, label="waypoints")
    ax.set_aspect("equal"); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.legend(loc="upper center", fontsize=8, ncol=2)
    ax.set_title(f"map {SEED}: lap {lap:.2f} s, margin {MARGIN}")
    fig.savefig(f"../logs/astar_demo_{SEED}.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
