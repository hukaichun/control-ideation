"""Elastic band on the grid mazes: the A* path pulled tight like a rubber band between start and goal (stage 1,
"taut"), then made stiff only where it bends more than the drone can fly (stage 2, "stiff"); each result goes through
the shared back end (experiments/common/bench.py: G3 fit -> speed plan -> flight).

Band = open chain of points p_0..p_n (spacing ~ DS), ends fixed at start and goal. The equilibrium of the forces is
computed directly as an optimisation (IPOPT) instead of iterating the forces; same equilibrium, fewer steps.
  stage 1 (taut):  min  sum |p_{i+1} - p_i|^2 * n / L0          tension (zero-rest-length springs: shortest, even spacing)
                   s.t. |p_i - c_j| >= r_j + CLEAR  every pillar j  clearance (hard)
  stage 2 (stiff): same + W sum max(0, k_i - k_max)^2 h_i       stiffness, only where curvature k_i > k_max
                   s.t. |p_i - p_i(taut)| <= R_min / 2            stays near the taut band (otherwise it makes loops)
                        |p_{i+1} - p_i| <= 2 DS                   honest finite-difference curvature
  k_i = |p_{i-1} - 2 p_i + p_{i+1}| / h_i^2, h_i = mean of the two neighbouring spacings.
  k_max = 1 / R_min = a_lat / v_d^2: a_lat = horizontal acceleration with RESERVE of the thrust kept back (quad model),
  v_d = design corner speed (the one remaining knob).

    cd experiments/2026-11-24_elastic_band
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python run.py [v_d] [mazes...]
"""
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
V_D = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
SEEDS = [int(a) for a in sys.argv[2:]] or [0, 1, 2, 3, 4]
import bench as B                           # noqa: E402

DS = 0.4                                    # band point spacing (m)
W_STIFF = 100.0                             # stiffness penalty weight: large, i.e. "as hard as possible"


def start_chain(pil):
    occ, cells = B.astar(pil, B.CLEAR + 0.75 * B.CELL)               # start feasible: cells clear by CLEAR
    cells = np.vstack([B.START, cells[1:-1], B.GOAL])
    return B.resample(cells, DS), cells


def band(P0, pil, k_max=None, ref=None, reach=None):
    """k_max None: taut band (tension only). Otherwise stiffness above k_max, every point within `reach` of ref."""
    n = len(P0)
    L0 = float(np.linalg.norm(np.diff(P0, axis=0), axis=1).sum())
    x = ca.MX.sym("x", 2 * n)
    p = ca.reshape(x, 2, n)
    d = p[:, 1:] - p[:, :-1]
    J = ca.sum2(ca.sum1(d ** 2)) * n / L0
    g, lb, ub = [], [], []
    if k_max is not None:
        seg = ca.sqrt(ca.sum1(d ** 2) + 1e-12)
        h = (seg[:, 1:] + seg[:, :-1]) / 2
        k = ca.sqrt(ca.sum1((p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) ** 2) + 1e-12) / h ** 2
        J += W_STIFF * ca.sum2(ca.fmax(0, k - k_max) ** 2 * h)
        g.append(ca.sum1((p - ca.DM(ref.T)) ** 2).T); lb.append(np.zeros(n)); ub.append(np.full(n, reach ** 2))
        g.append(ca.sum1(d ** 2).T); lb.append(np.zeros(n - 1)); ub.append(np.full(n - 1, (2 * DS) ** 2))
    for cx, cy, r in pil:
        g.append(((p[0, :] - cx) ** 2 + (p[1, :] - cy) ** 2).T); lb.append(np.full(n, (r + B.CLEAR) ** 2)); ub.append(np.full(n, np.inf))
    lbx, ubx = np.full(2 * n, -np.inf), np.full(2 * n, np.inf)
    lbx[:2] = ubx[:2] = B.START
    lbx[-2:] = ubx[-2:] = B.GOAL
    S = ca.nlpsol("band", "ipopt", {"x": x, "f": J, "g": ca.vertcat(*g)},
                  {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 3000, "ipopt.mu_strategy": "adaptive"})
    t0 = time.perf_counter()
    sol = S(x0=P0.ravel(), lbx=lbx, ubx=ubx, lbg=np.concatenate(lb), ubg=np.concatenate(ub))
    P = np.array(sol["x"]).reshape(n, 2)
    info = dict(status=S.stats()["return_status"], iters=S.stats()["iter_count"], t=time.perf_counter() - t0, n=n)
    if k_max is not None:
        kk = np.abs(B.curvature(P))[1:-1]
        info["share_over"] = float(np.mean(kk > 1.05 * k_max))
        info["k_peak"] = float(kk.max())
    return P, info


def main():
    quad = B.qs.Quad()
    a_lat = B.a_lat(quad)
    k_max = a_lat / V_D ** 2
    tag = f"vd{V_D:g}"
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    log(f"elastic band: a_lat {a_lat:.1f} m/s^2 (reserve {B.RESERVE}), v_d {V_D:g} m/s -> R_min {1 / k_max:.2f} m, "
        f"clearance {B.CLEAR} m, stiffness weight {W_STIFF}, stiff band within R_min/2 of the taut one, mazes {SEEDS}")
    rows = []
    for seed in SEEDS:
        pil = B.maze(seed)
        P0, cells = start_chain(pil)
        Pt, it = band(P0, pil)
        Ps, is_ = band(Pt, pil, k_max, ref=Pt, reach=0.5 / k_max)
        log(f"maze {seed}: A* chain {B.shape_stats(P0)['length']:.1f} m ({len(P0)} points) | taut: {it['status']} "
            f"({it['iters']} it, {it['t']:.1f} s) {B.shape_stats(Pt)['length']:.1f} m | stiff: {is_['status']} "
            f"({is_['iters']} it, {is_['t']:.1f} s) {B.shape_stats(Ps)['length']:.1f} m, curvature over the limit at "
            f"{100 * is_['share_over']:.0f}% of points (peak {is_['k_peak']:.2f} 1/m, limit {k_max:.2f})")
        for name, P2 in (("taut", Pt), ("stiff", Ps)):
            log(f" {name}:")
            try:
                res = B.finish(P2, pil, quad, log=log)
            except RuntimeError as e:
                log(f"  back end failed: {e}")
                continue
            rows.append((seed, name, res))
            if name == "stiff":
                B.save(HERE, seed, res)

            def extra(ax, P0=P0, P2=P2, name=name):
                ax.plot(P0[:, 0], P0[:, 1], ".", ms=2, color="0.7", label="A* start of the band")
                ax.plot(P2[:, 0], P2[:, 1], "-", lw=0.8, color="k", label=f"band ({name})")
            B.plot(out_dir / f"maze{seed}_{name}_{tag}.png", seed, pil, res,
                   f"elastic band ({name}), maze {seed}" + (f": v_d {V_D:g} m/s, R_min {1 / k_max:.1f} m" if name == "stiff" else ""),
                   extra)
    B.summary(rows, log)
    (out_dir / f"run_{tag}.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
