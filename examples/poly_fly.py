"""Fly the flat E-L law on CPC paths fitted with G3 piecewise polynomials (poly_plan), speed planned at several motor
margins; no disturbance. Saves every flight for the margin analysis.

    PYTHONPATH=. uv run --with casadi --with pyyaml --with scipy --with numba python poly_fly.py <seed> [margins...]
"""
import sys
import time

import numpy as np

import flat_el_policy as fe
import poly_plan as pp
import quad_sim as qs

quad = qs.Quad()
seed = int(sys.argv[1])
MARGINS = [float(a) for a in sys.argv[2:]] or [1.0, 0.9, 0.8, 0.7]
d = np.load(f"../logs/cpc_traj_tol0.24_{seed}.npz")
gates = d["gates"]
P, V, lap_cpc = pp.dense_cpc(d)
for margin in MARGINS:
    motor = dict(m=quad.m, Ix=quad.J[0, 0], l=0.15, Tmax=quad.Tmax, Tmin=quad.Tmin, margin=margin)
    t0 = time.perf_counter()
    r = pp.plan_from_points(P, V, gates, motor)
    t_plan = time.perf_counter() - t0
    if r is None or not r["status"].startswith("Solve_Succeeded"):
        print(f"track {seed:2d} margin {margin}: plan failed ({None if r is None else r['status']})", flush=True)
        continue
    pol = fe.PolyELPolicy(r["spec"], wp=16, wH=12)
    p0, v0, a0, j0 = pp.state_at(r["spec"][0], 0.0)
    lap = r["lap"]
    log = fe.fly(quad, pol, p0, v0, a0, j0, 2.2 * lap, mode="B")
    ok = log[-1, 0] > 2.19 * lap
    L2 = log[(log[:, 0] >= lap) & (log[:, 0] < 2 * lap)]
    miss2 = max(np.min(np.linalg.norm(L2[:, 4:7] - g, axis=1)) for g in gates) if ok else np.nan
    d0 = np.linalg.norm(log[:, 4:7] - p0, axis=1)
    k = np.where((log[:, 0] > 0.5 * lap) & (d0 < 0.5))[0]
    lap_f = log[k[0], 0] if len(k) and ok else np.nan
    print(f"track {seed:2d} margin {margin}: CPC {lap_cpc:.3f} plan {lap:.3f} ({100 * (lap / lap_cpc - 1):+.1f}%) "
          f"flown {lap_f:.3f} | ok {ok} lap-2 gate miss {miss2:.2f} | |phi| max {L2[:, 1].max() if ok else np.nan:.3f} "
          f"mean {L2[:, 1].mean() if ok else np.nan:.3f} | sat {100 * log[:, 8].mean():.0f}% | max |w| {log[:, 7].max():.1f} "
          f"| requested motor max {log[:, 12].max():.2f} Tmax | plan {t_plan:.1f}s", flush=True)
    np.savez(f"../logs/poly_fly_{seed}_m{margin}.npz", log=log, spec=np.array(r["spec"], dtype=object), lap=lap,
             lap_cpc=lap_cpc, gates=gates, margin=margin)
