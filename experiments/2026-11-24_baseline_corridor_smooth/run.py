"""Baseline: the current method (examples/astar_demo.py) on the grid mazes, one way from start to goal.

  1. A* (8-connected) on a grid whose pillars are inflated by corridor half-width R_C + drone radius
  2. shortcut: from each kept vertex jump to the farthest A* cell still in line of sight; drop vertices that turn < 3 deg
     -> every segment's corridor of half-width R_C is free for the drone
  3. start curve: the polyline with rounded corners (each arc within R_C / 2 of its vertex)
  4. smoothing (poly_plan.smooth_path): G3 pieces, every sample inside the box around its segment (or the disc of
     radius R_C at a vertex), least bending; corridor as a penalty (always solvable)
  5. shared back end (experiments/common/bench.py): fit, speed plan, flight
Same as astar_demo except: open path, and R_C = 0.5 m instead of 1 m (the grid's gaps are 5 m minus two radii of
0.5-2.4 m; a 1 m corridor would close most of them).

    cd experiments/2026-11-24_baseline_corridor_smooth
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [mazes...]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
SEEDS = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
import bench as B                           # noqa: E402

R_C, RZ = 0.5, 0.3


def rounded(V, ds=0.02):
    """Open polyline with circular corners; each arc stays within R_C / 2 of its vertex and uses < 45% of a side."""
    pts = [V[0]]
    for i in range(1, len(V) - 1):
        a, b, c = V[i - 1], V[i], V[i + 1]
        u0, u1 = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
        th = np.arccos(np.clip(u0 @ u1, -1, 1))
        if th < 1e-3:
            continue
        T = min(0.5 * R_C * np.tan(th / 2) / (1 / np.cos(th / 2) - 1),
                0.45 * min(np.linalg.norm(b - a), np.linalg.norm(c - b)))
        R = T / np.tan(th / 2)
        p0 = b - T * u0
        nrm = np.array([-u0[1], u0[0]]) * np.sign(u0[0] * u1[1] - u0[1] * u1[0])
        ctr = p0 + R * nrm
        ang0 = np.arctan2(*(p0 - ctr)[::-1])
        sgn = np.sign(u0[0] * u1[1] - u0[1] * u1[0])
        for t in np.linspace(0, th, max(3, int(R * th / ds))):
            pts.append(ctr + R * np.array([np.cos(ang0 + sgn * t), np.sin(ang0 + sgn * t)]))
    pts.append(V[-1])
    return B.resample(np.array(pts), ds)


def plan(pil):
    occ, cells = B.astar(pil, R_C + B.R_DRONE)
    poly = B.shortcut(occ, cells)
    verts = np.column_stack([poly, np.full(len(poly), B.Z)])
    boxes = []
    for i in range(len(verts) - 1):
        a, b = verts[i], verts[i + 1]
        u = (b - a) / np.linalg.norm(b - a)
        n1 = np.cross([0, 0, 1.0], u); n1 /= np.linalg.norm(n1)
        boxes.append(dict(a=a, u=u, n1=n1, n2=np.cross(u, n1), len=np.linalg.norm(b - a), r=R_C, rz=RZ))
    P = rounded(poly)
    P = np.column_stack([P, np.full(len(P), B.Z)])
    pcs, _ = B.pp.pieces_from(P, np.zeros((0, 3)), closed=False)
    zp, _ = B.pp.smooth_path(pcs, [], np.zeros((0, 3)), 0.0, w_fit=1e-3, w_bend=1.0, boxes=boxes,
                             discs=[(v, R_C) for v in verts[1:-1]] or None, w_box=1e3, closed=False)
    spec = B.pp.spec_of(pcs, zp)
    S = np.vstack([B.pp.sample(spec, 300), B.pp.point(spec[-1], spec[-1]["l"])])
    return S, dict(cells=cells, poly=poly, smooth=dict(B.pp.SMOOTH_STATS))


def main():
    quad = B.qs.Quad()
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    log(f"baseline corridor smoothing: R_C {R_C} m, clearance check {B.CLEAR} m, mazes {SEEDS}")
    rows = []
    for seed in SEEDS:
        pil = B.maze(seed)
        S, info = plan(pil)
        log(f"maze {seed}: A* {len(info['cells'])} cells -> {len(info['poly'])} vertices | smoothing {info['smooth']['status']}")
        res = B.finish(S, pil, quad, log=log)
        B.save(HERE, seed, res)
        rows.append((seed, "baseline", res))

        def extra(ax, info=info):
            ax.plot(info["cells"][:, 0], info["cells"][:, 1], ".", ms=2, color="0.7", label="A* cells")
            ax.plot(info["poly"][:, 0], info["poly"][:, 1], "--", lw=1, color="tab:orange", label=f"shortcut (corridor ±{R_C} m)")
        B.plot(out_dir / f"maze{seed}.png", seed, pil, res, f"baseline corridor smoothing, maze {seed}", extra, compare=None)
    B.summary(rows, log, compare=None)
    (out_dir / "run.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
