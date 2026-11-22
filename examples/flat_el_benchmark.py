"""Flat E-L policy (one L, V planned directly) on the 20 benchmark tracks, full quad model, vs CPC.
Path: lines + arcs (path_g2.fast_plan_g2_fit, frac 0.95, gates 0.24 m inside). Speed: v_plan.plan_V with
torque-first motor limits at the given margin. Flown 2.2 laps from the plan's start; lap = time of the first lap.

    PYTHONPATH=. uv run --with casadi --with pyyaml --with scipy --with numba python flat_el_benchmark.py [disturbed] [margins]
Mode B, the law's a = measured acceleration.
"""
import sys
import time

import numpy as np
from numpy.polynomial import chebyshev as Ch

import flat_el_policy as fe
import path_g2 as pg
import quad_sim as qs
import race_compare as rc
import track_utils as tu
import v_plan as vp

quad = qs.Quad()
DIST = "disturbed" in sys.argv[1:]     # wind (3, 2, 0) m/s^2 and 10 % heavier, unknown to us
MARGINS = [float(a) for a in sys.argv[1:] if a != "disturbed"] or [0.9]
plant = qs.Quad()
if DIST:
    plant.ext_acc = np.array([3.0, 2.0, 0.0])
    plant.m *= 1.1
CPC = np.loadtxt("../logs/cpc_benchmark.csv", delimiter=",", skiprows=1)
rows = []
for seed in range(20):
    g = rc.load_track() if seed == 0 else tu.random_track(seed)
    t0 = time.perf_counter()
    v_, f_ = pg.fast_plan_g2_fit(g, 0.95, 0.0, 0.24)
    path = pg.PathG2(v_, f_, 0.0)
    t_path = time.perf_counter() - t0
    gate_plan = rc.gate_dist(path.sample(0.02)[0], g).max()
    for margin in MARGINS:
        t0 = time.perf_counter()
        try:
            sp, r0, motor, S0 = vp.warm_sigma(path, quad, margin)
            t_warm = time.perf_counter() - t0
            r = vp.plan_V(path, motor, sp)
        except Exception as e:
            print(f"track {seed:2d} margin {margin}: plan failed {e}", flush=True)
            continue
        t0 = time.perf_counter()
        pol = fe.FlatELPolicy(path, V_spec=r["spec"], wp=16, wH=12)
        t_pol = time.perf_counter() - t0
        P0, q0 = pol.pcs[0], r["spec"][0]
        u = lambda l: -2 * Ch.chebval(2 * (l - q0["l0"]) / (q0["l1"] - q0["l0"]) - 1, q0["coef"])
        h = 1e-3
        u0, u1, u2 = u(0.0), (u(h) - u(-h)) / (2 * h), (u(h) - 2 * u(0.0) + u(-h)) / h ** 2
        sg, t0v = np.sqrt(max(u0, 1e-6)), P0["u"]
        for mode in ("B",):
            log = fe.fly(quad, pol, P0["a"], sg * t0v, 0.5 * u1 * t0v, sg * 0.5 * u2 * t0v, 2.2 * r["lap"], mode=mode,
                         plant=plant)
            d0 = np.linalg.norm(log[:, 4:7] - P0["a"], axis=1)
            k = np.where((log[:, 0] > 0.5 * r["lap"]) & (d0 < 0.5))[0]
            lap = log[k[0], 0] if len(k) else np.nan
            ok = log[-1, 0] > 2.19 * r["lap"]
            miss = max(np.min(np.linalg.norm(log[:, 4:7] - gg, axis=1)) for gg in g)
            L2 = log[(log[:, 0] >= r["lap"]) & (log[:, 0] < 2 * r["lap"])]           # second lap only (as for PX4)
            miss2 = max(np.min(np.linalg.norm(L2[:, 4:7] - gg, axis=1)) for gg in g) if ok else np.nan
            row = (seed, margin, mode == "B", CPC[seed, 1], r["lap"], lap if ok else np.nan, miss, log[:, 1].max(),
                   log[:, 1].mean(), log[:, 7].max(), log[:, 8].mean(), t_path, t_warm, r["t_build"], r["t_solve"],
                   t_pol, gate_plan, ok, miss2)
            rows.append(row)
            print(f"track {seed:2d} margin {margin} {mode}: CPC {CPC[seed, 1]:.3f} plan {r['lap']:.3f} ({r['status']}) "
                  f"flown {row[5]:.3f} ({100 * (row[5] / CPC[seed, 1] - 1):+.0f}%) gate miss {miss:.2f} "
                  f"max|phi| {row[7]:.2f} max w {row[9]:.0f} sat {100 * row[10]:.0f}% | plan time: path {t_path:.2f} "
                  f"warm {t_warm:.2f} build {r['t_build']:.2f} solve {r['t_solve']:.2f} s ({r['iters']} it) lap-2 gate miss {miss2:.2f}", flush=True)
np.savetxt(f"../logs/flat_el_benchmark{'_disturbed' if DIST else ''}_m{'_'.join(map(str, MARGINS))}.csv", np.array(rows, float), delimiter=",", comments="",
           header="seed,margin,modeB,lap_cpc,lap_plan,lap_flown,gate_miss,max_phi,mean_phi,max_w,sat,t_path,t_warm,"
                  "t_build,t_solve,t_policy,gate_plan,ok,gate_miss_lap2")
