"""CPC with a restricted vehicle, as the starting point of our optimisation (user, 2026-11-24): its trajectory keeps
a margin to our 100% limits, so it is feasible in our model even after the speed is smoothed into a B-spline.
Restriction: rotor thrust <= THRUST x Tmax, |w_x|, |w_y| <= RATE x 15 / sqrt(2) (so |w_xy| <= RATE x 15, the
magnitude limit our model uses). Started from the saved full CPC solution (logs/cpc_traj_tol0.24_<k>.npz).

    cd experiments/2026-11-24_time_optimal_in_our_space
    [THRUST=0.8] uv run --with casadi --with pyyaml --with scipy --with numba python cpc_restricted.py [tracks...]
(THRUST also scales the body-rate limit; 0.8 gives the start of our 90% version)
"""
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import quad_time_optimal as qto   # noqa: E402

import os
THRUST = float(os.environ.get("THRUST", 0.9))        # 0.8: start for our 90% version
RATE, TOL = THRUST, 0.24
TAG = "" if THRUST == 0.9 else f"_{THRUST:g}"
TRACKS = [int(a) for a in sys.argv[1:]] or list(range(10))
for k in TRACKS:
    d = np.load(ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")
    g, N = d["gates"], int(d["N"])
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            T, x, u, P = qto.solve(g, TOL, N, d["T"], np.repeat(g, N, axis=0), np.zeros((len(g) * N, 3)), cold=True, thrust_scale=THRUST, wxy=RATE * 15 / np.sqrt(2),
                                   init=(d["T"], d["x"], d["u"]))
        np.savez(HERE / "results" / f"cpc_restricted{TAG}_{k}.npz", T=T, x=x, u=u, gates=g, N=N)
        print(f"track {k}: restricted CPC {np.sum(T):.3f} s (full {np.sum(d['T']):.3f} s, "
              f"{100 * (np.sum(T) / np.sum(d['T']) - 1):+.1f}%), max rotor {u.max():.2f} / {P['Tmax']:.2f} N, "
              f"max |w_x|,|w_y| {np.abs(x[:, 10:12]).max():.1f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    except RuntimeError as e:
        print(f"track {k}: FAILED {e} ({time.perf_counter() - t0:.0f} s)", flush=True)
