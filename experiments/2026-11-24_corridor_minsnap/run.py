"""Corridor + minimum snap on the grid mazes: a chain of convex free polygons around the A* path (the "hollow tube"), one
degree-7 polynomial per box with its duration as a variable, minimum snap + time; only the geometry is kept and handed
to the shared back end (experiments/common/bench.py: G3 fit -> speed plan -> flight).

Corridor (convex polygons, as Liu et al. 2017 "safe flight corridor"): A* on the map inflated by CLEAR (+ a cell
margin), line-of-sight shortcut to a polyline; for each polyline segment a polygon = map bounds and, for every pillar,
the half-plane beyond the tangent line of its CLEAR-inflated circle at the point closest to the segment (this
separates the segment from the inflated pillar, so the polygon is free and contains the segment). Segments longer
than SEG_MAX are split into equal parts sharing the polygon. Consecutive polygons share the polyline vertex.
(First version: axis-aligned boxes of free cells grown along the A* path; between round pillars they were thin strips
and their small overlaps forced S-bends; see the record.)
Trajectory: segment i in polygon i, Bernstein control points c_i0..c_i7 (x, y; z = Z), duration T_i = exp(u_i).
  min  sum_i (1/T_i^7) int_0^1 |4th tau-derivative of c_i|^2 dtau  +  RHO sum_i T_i
  s.t. every control point of segment i inside polygon i              (curve inside the polygon: convex hull)
       p, v, a, j continuous at every joint; starts at START, ends at GOAL (derivatives free at both ends)
       |a| <= a_lat on every control point of the 2nd derivative        (thrust, RESERVE kept back)
       |v| <= V_MAX on every control point of the 1st derivative        (same ceiling as the speed plan)
RHO trades snap against time; at RHO >= 1e5 the thrust limit is active (checked on the earlier loop maps).
Solver: stage 1 durations fixed (3 m/s; convex in the control points), stage 2 durations free (exact Hessian); when
stage 2 does not converge, all stage-1 durations scaled by one factor, bisected to the smallest feasible one.

    cd experiments/2026-11-24_corridor_minsnap
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [rho] [mazes...]
"""
import sys
import time
from math import comb
from pathlib import Path

import casadi as ca
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
RHO = float(sys.argv[1]) if len(sys.argv) > 1 else 1e5
SEEDS = [int(a) for a in sys.argv[2:]] or [0, 1, 2, 3, 4]
import bench as B                           # noqa: E402

SEG_MAX = 8.0                               # m, longer polyline segments are split
DEG = 7


def corridor(pil):
    """Polyline from start to goal and one polygon (A, b: A x <= b) per trajectory segment; the segment end points."""
    occ, path = B.astar(pil, B.CLEAR + 0.75 * B.CELL)
    poly = B.shortcut(occ, path)
    polys, pts = [], [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        A, bb = [[-1, 0], [1, 0], [0, -1], [0, 1]], [0, B.W, 0, B.H]
        d = b - a
        for cx, cy, r in pil:
            c = np.array([cx, cy])
            q = a + np.clip((c - a) @ d / (d @ d), 0, 1) * d
            n = (q - c) / np.linalg.norm(q - c)
            A.append(-n); bb.append(-(n @ c + r + B.CLEAR))                # n.x >= n.c + r + CLEAR
        k = int(np.ceil(np.linalg.norm(d) / SEG_MAX))
        for j in range(1, k + 1):
            polys.append((np.array(A, float), np.array(bb, float)))
            pts.append(a + d * j / k)
    return polys, np.array(pts), path, poly


def polygon_vertices(A, b, inner):
    from scipy.spatial import HalfspaceIntersection, ConvexHull
    hs = HalfspaceIntersection(np.column_stack([A, -b]), inner)
    V = hs.intersections
    return V[ConvexHull(V).vertices]


def gram(n):
    return np.array([[comb(n, j) * comb(n, k) / (comb(2 * n, j + k) * (2 * n + 1)) for k in range(n + 1)] for j in range(n + 1)])


def diff_ops():
    """Matrices mapping the 8 control points to the control points of the k-th tau-derivative (k = 1..4)."""
    ops, M, n = [], np.eye(DEG + 1), DEG
    for k in range(1, 5):
        D = np.zeros((n, n + 1))
        for j in range(n):
            D[j, j], D[j, j + 1] = -n, n
        M = D @ M
        ops.append(M)
        n -= 1
    return ops


def minsnap(polys, joints, a_lat):
    m = len(polys)
    ops = diff_ops()
    G4 = gram(DEG - 4)
    c = ca.MX.sym("c", m * (DEG + 1) * 2)
    u = ca.MX.sym("u", m)                      # T = exp(u): durations stay positive, better scaled
    T = ca.exp(u)
    C = [ca.reshape(c[i * 16:(i + 1) * 16], DEG + 1, 2) for i in range(m)]
    J, g, lb, ub = 0, [], [], []
    for i in range(m):
        s4 = ca.mtimes(ca.DM(ops[3]), C[i])
        J += ca.trace(ca.mtimes(ca.mtimes(s4.T, ca.DM(G4)), s4)) / T[i] ** 7 / RHO + T[i]      # (snap + RHO time) / RHO
        a2 = ca.mtimes(ca.DM(ops[1]), C[i])
        g.append(ca.sum2(a2 ** 2) - a_lat ** 2 * T[i] ** 4); lb.append(np.full(DEG - 1, -np.inf)); ub.append(np.zeros(DEG - 1))
        v1 = ca.mtimes(ca.DM(ops[0]), C[i])
        g.append(ca.sum2(v1 ** 2) - B.V_MAX ** 2 * T[i] ** 2); lb.append(np.full(DEG, -np.inf)); ub.append(np.zeros(DEG))
        A, bb = polys[i]
        g.append(ca.vec(ca.mtimes(ca.DM(A), C[i].T))); lb.append(np.full(len(bb) * (DEG + 1), -np.inf))
        ub.append(np.tile(bb, DEG + 1))                  # vec() stacks per control point
        if i == m - 1:
            continue
        k = i + 1
        ends_i = [C[i][DEG, :]] + [ca.mtimes(ca.DM(ops[d - 1][-1:, :]), C[i]) for d in (1, 2, 3)]
        ends_k = [C[k][0, :]] + [ca.mtimes(ca.DM(ops[d - 1][:1, :]), C[k]) for d in (1, 2, 3)]
        for d in range(4):
            g.append((ends_i[d] * T[k] ** d - ends_k[d] * T[i] ** d).T); lb.append(np.zeros(2)); ub.append(np.zeros(2))
    lbx, ubx = np.full(m * 17, -np.inf), np.full(m * 17, np.inf)
    for idx, val in ((0, B.START), ((m - 1) * 16 + DEG, B.GOAL)):         # x and y of the first / last point
        lbx[idx] = ubx[idx] = val[0]
        lbx[idx + 8] = ubx[idx + 8] = val[1]
    lbx[m * 16:], ubx[m * 16:] = np.log(0.05), np.log(30.0)
    # start: straight segments between the corridor joints, 3 m/s
    x0v = np.zeros(m * 17)
    for i in range(m):
        a, b = joints[i], joints[i + 1]
        s = np.linspace(0, 1, DEG + 1)[:, None]
        P = a + s * (b - a)
        x0v[i * 16:i * 16 + 8], x0v[i * 16 + 8:i * 16 + 16] = P[:, 0], P[:, 1]
        x0v[m * 16 + i] = np.log(np.linalg.norm(b - a) / 3 + 0.3)
    S = ca.nlpsol("ms", "ipopt", {"x": ca.vertcat(c, u), "f": J, "g": ca.vertcat(*g)},
                  {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 3000, "ipopt.tol": 1e-6,
                   "ipopt.acceptable_tol": 1e-4, "ipopt.acceptable_iter": 10})
    t0 = time.perf_counter()
    # stage 1: durations fixed: convex in the control points, finds a feasible curve in the corridor
    l1, u1 = lbx.copy(), ubx.copy()
    l1[m * 16:] = u1[m * 16:] = x0v[m * 16:]
    sol = S(x0=x0v, lbx=l1, ubx=u1, lbg=np.concatenate(lb), ubg=np.concatenate(ub))
    st1 = S.stats()["return_status"]
    # stage 2: durations free (exact Hessian). When it does not converge: the stage-1 problem with all durations
    # scaled by one factor, bisected to the smallest feasible factor (uniform time scaling)
    ok = ("Solve_Succeeded", "Solved_To_Acceptable_Level")
    x1 = np.array(sol["x"]).ravel()
    S2 = ca.nlpsol("ms2", "ipopt", {"x": ca.vertcat(c, u), "f": J, "g": ca.vertcat(*g)},
                   {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 500, "ipopt.mu_strategy": "adaptive",
                    "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-4, "ipopt.acceptable_iter": 10})
    sol2 = S2(x0=x1, lbx=lbx, ubx=ubx, lbg=np.concatenate(lb), ubg=np.concatenate(ub))
    tries = [("free durations", S2.stats()["return_status"], S2.stats()["iter_count"])]
    if tries[0][1] in ok:
        sol = sol2
    else:
        lo_, hi_, best = 0.1, 1.0, sol
        if st1 not in ok:
            raise RuntimeError(f"min snap: stage 1 {st1}")
        for _ in range(7):
            mid = np.sqrt(lo_ * hi_)
            l1[m * 16:] = u1[m * 16:] = x0v[m * 16:] + np.log(mid)
            sm = S(x0=np.concatenate([x1[:m * 16], l1[m * 16:]]), lbx=l1, ubx=u1, lbg=np.concatenate(lb), ubg=np.concatenate(ub))
            if S.stats()["return_status"] in ok:
                hi_, best = mid, sm
            else:
                lo_ = mid
        sol = best
        tries.append(("uniform scaling", f"factor {hi_:.3f}", 7))
    xs = np.array(sol["x"]).ravel()
    Cs = [xs[i * 16:(i + 1) * 16].reshape(2, DEG + 1).T for i in range(m)]
    Ts = np.exp(xs[m * 16:])
    return Cs, Ts, dict(stage1=st1, status=f"{tries[-1][0]}: {tries[-1][1]}", iters=tries[-1][2], tries=tries,
                        t=time.perf_counter() - t0)


def bern(Cp, tau):
    n = len(Cp) - 1
    Bm = np.array([[comb(n, j) * t ** j * (1 - t) ** (n - j) for j in range(n + 1)] for t in tau])
    return Bm @ Cp


def main():
    quad = B.qs.Quad()
    a_lat = B.a_lat(quad)
    tag = f"rho{RHO:g}"
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    log(f"corridor + min snap: a_lat {a_lat:.1f} m/s^2 (reserve {B.RESERVE}), RHO {RHO:g}, segments <= {SEG_MAX} m, "
        f"clearance {B.CLEAR} m, mazes {SEEDS}")
    rows = []
    for seed in SEEDS:
        pil = B.maze(seed)
        polys, joints, path, poly = corridor(pil)
        Cs, Ts, info = minsnap(polys, joints, a_lat)
        P2 = np.vstack([bern(Cp, np.linspace(0, 1, 200, endpoint=False)) for Cp in Cs] + [Cs[-1][-1:]])
        log(f"maze {seed}: {len(poly)}-vertex polyline, {len(polys)} segments | min snap stage 1 {info['stage1']}, stage 2 {info['status']} "
            f"({info['iters']} it, {info['t']:.1f} s), own time {Ts.sum():.2f} s, segment T {Ts.min():.2f}..{Ts.max():.2f} s")
        try:
            res = B.finish(P2, pil, quad, log=log)
        except RuntimeError as e:
            log(f"  back end failed: {e}")
            continue
        res["minsnap_time"] = float(Ts.sum())
        rows.append((seed, "min snap", res))
        B.save(HERE, seed, res)

        def extra(ax, polys=polys, joints=joints, path=path, poly=poly):
            ax.plot(path[:, 0], path[:, 1], ".", ms=2, color="0.7", label="A* cells")
            ax.plot(poly[:, 0], poly[:, 1], "--", lw=0.8, color="tab:orange", label="shortcut polyline")
            for k, (A, bb) in enumerate(polys):
                V = polygon_vertices(A, bb, (joints[k] + joints[k + 1]) / 2)
                V = np.vstack([V, V[:1]])
                ax.plot(V[:, 0], V[:, 1], "-", lw=0.6, color="tab:green", alpha=0.6,
                        label="corridor polygons" if k == 0 else None)
        B.plot(out_dir / f"maze{seed}_{tag}.png", seed, pil, res, f"corridor + min snap, maze {seed}: RHO {RHO:g}", extra)
    B.summary(rows, log)
    (out_dir / f"run_{tag}.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
