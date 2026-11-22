"""Full-quadrotor time-optimal lap (home-made CPC-style, cold straight-line start) on 20 tracks: the UZH track
(seed 0) and 19 random gate tracks (track_utils.random_track). Table: logs/cpc_benchmark.csv (seed, lap_cpc, t_cpc).

    uv run --with casadi --with pyyaml --with scipy --with numba examples/cpc_benchmark.py
"""
import contextlib
import io
import time

import numpy as np

import quad_time_optimal as qto
import race_compare as rc
import track_utils as tu

N, N_TRACKS = 30, 20
rows = []
for seed in range(N_TRACKS):
    g = rc.load_track() if seed == 0 else tu.random_track(seed)
    M = len(g)
    # cold guess only needs segment placeholders (overwritten inside solve when cold=True)
    dummy_T = np.ones(M)
    dummy_p = np.repeat(g, N, axis=0)
    dummy_v = np.zeros_like(dummy_p)
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            T, x, u, P = qto.solve(g, tu.TOL, N, dummy_T, dummy_p, dummy_v, cold=True)
        lap = float(np.sum(T))
    except RuntimeError:
        lap = np.nan
    wall = time.perf_counter() - t0
    rows.append((seed, lap, wall))
    print(f"track {seed:2d}: CPC {lap:6.3f} s ({wall:5.1f} s)", flush=True)

C = np.array(rows)
ok = np.isfinite(C[:, 1])
print(f"\nCPC (full model): success {ok.sum()}/{len(C)}, solve time median {np.median(C[:, 2]):.1f} s "
      f"(max {C[:, 2].max():.1f} s)")
np.savetxt("../logs/cpc_benchmark.csv", C, delimiter=",", header="seed,lap_cpc,t_cpc", comments="")
print("table -> ../logs/cpc_benchmark.csv")
