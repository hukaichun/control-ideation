"""Solve the home-made CPC (full quad, cold start) on the 20 benchmark tracks and keep the trajectories
(logs/cpc_traj_<seed>.npz: T per segment, x (13 x N per segment), u) for the PX4-default comparison.

    PYTHONPATH=. uv run --with casadi --with pyyaml --with scipy --with numba python cpc_trajs.py
"""
import contextlib
import io
import time

import numpy as np

import quad_time_optimal as qto
import race_compare as rc
import track_utils as tu

import sys
N = 30
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else tu.TOL
SUF = "" if len(sys.argv) < 2 else f"_tol{sys.argv[1]}"
for seed in range(20):
    g = rc.load_track() if seed == 0 else tu.random_track(seed)
    M = len(g)
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            T, x, u, P = qto.solve(g, TOL, N, np.ones(M), np.repeat(g, N, axis=0), np.zeros((M * N, 3)), cold=True)
        np.savez(f"../logs/cpc_traj{SUF}_{seed}.npz", T=T, x=x, u=u, gates=g, N=N)
        print(f"track {seed:2d}: CPC {np.sum(T):.3f} s ({time.perf_counter() - t0:.0f} s)", flush=True)
    except RuntimeError as e:
        print(f"track {seed:2d}: FAILED {e}", flush=True)
