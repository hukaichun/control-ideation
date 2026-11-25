"""The original CPC (Foehn et al., Science Robotics 2021; code: third_party/rpg_time_optimal, unmodified) on our 10
tracks, as the time-optimal reference (user, 2026-11-25: compare against the original, not the simplified
examples/quad_time_optimal.py).

Original formulation: one total time T, N = nodes_per_gate x (number of waypoints) RK4 nodes, per-rotor thrust in
[T_min, T_max], |w_x|, |w_y| <= 15, |w_z| <= 0.3, z >= 0.5, thrust / body-rate ramp-up over the first 35 m
(quad.yaml), progress variables with complementarity constraints decide at which node each waypoint is passed
(within `tolerance`).

Setup (mine): hover start at the last gate, the 7 gates three times (21 waypoints), no end constraint. Lap time =
time from passing gate 0 in lap 2 to passing gate 0 in lap 3 (steady flight: the ramp-up is over, and no braking
for the end). Track 0 = UZH track (examples/race_compare.load_track), tracks 1..9 = track_utils.random_track(seed).

    cd experiments/2026-11-25_cpc_original
    uv run --with "casadi" --with pyyaml --with scipy --with numba --with matplotlib python run.py [tracks...]
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CPC = ROOT / "third_party" / "rpg_time_optimal"
sys.path += [str(CPC / "src"), str(ROOT / "examples")]
from track import Track            # noqa: E402
from quad import Quad              # noqa: E402
from integrator import RungeKutta4  # noqa: E402
from planner import Planner        # noqa: E402
from trajectory import Trajectory  # noqa: E402
import race_compare as rc          # noqa: E402
import track_utils as tu           # noqa: E402

TRACKS = [int(a) for a in sys.argv[1:]] or list(range(10))
TOL = 0.24          # same gate tolerance as the rest of our comparisons
NPW = 40            # nodes per gate (the original example)
LAPS = 3


def gates_of(k):
    return rc.load_track() if k == 0 else tu.random_track(k)


def main():
    (HERE / "results").mkdir(exist_ok=True)
    for k in TRACKS:
        g = np.asarray(gates_of(k), float)
        M = len(g)
        tr = Track()
        tr.gates = [list(map(float, p)) for p in np.tile(g, (LAPS, 1))]
        tr.init_pos = list(map(float, g[-1]))
        tr.init_att = [1.0, 0.0, 0.0, 0.0]
        tr.init_vel = [0.0, 0.0, 0.0]
        tr.init_omega = [0.0, 0.0, 0.0]
        quad = Quad(str(CPC / "quads" / "quad.yaml"))
        pl = Planner(quad, tr, RungeKutta4, {"tolerance": TOL, "nodes_per_gate": NPW, "vel_guess": 3.0})
        pl.setup()
        t0 = time.perf_counter()
        x = pl.solve()
        dt_solve = time.perf_counter() - t0
        status = pl.solver.stats()["return_status"]
        NW, N, NX, NU = pl.NW, pl.N, 13, 4
        T = float(x[0])
        n_start, n_slice = 1 + NX + NW, NX + NU + 3 * NW
        X = np.array([x[n_start + i * n_slice + NU: n_start + i * n_slice + NU + NX] for i in range(N)])
        U = np.array([x[n_start + i * n_slice: n_start + i * n_slice + NU] for i in range(N)])
        lam = np.array([x[n_start + i * n_slice + NU + NX: n_start + i * n_slice + NU + NX + NW] for i in range(N)])
        ipass = lam.argmax(0)                               # node at which each waypoint is passed
        dist = np.array([np.linalg.norm(X[ipass[j], :3] - np.array(tr.gates[j])) for j in range(NW)])
        lap = (ipass[2 * M] - ipass[M]) * T / N
        lap3 = (ipass[2 * M + M - 1] - ipass[2 * M - 1]) * T / N if LAPS >= 3 else np.nan
        line = (f"track {k}: {status}, {dt_solve:.0f} s, total {T:.3f} s ({N} nodes), lap 2 (gate 0 -> gate 0) "
                f"{lap:.3f} s, lap from gate {M - 1} -> gate {M - 1} {lap3:.3f} s, worst gate distance "
                f"{dist.max():.3f} m, max rotor {U.max():.2f} N, max |w_xy| {np.abs(X[:, 10:12]).max():.2f}, "
                f"max |w_z| {np.abs(X[:, 12]).max():.3f}")
        print(line, flush=True)
        with open(HERE / "results" / "run.log", "a") as f:
            f.write(line + "\n")
        np.savez(HERE / "results" / f"cpc_{k}.npz", x=X, u=U, T=T, N=N, ipass=ipass, gates=g, lap=lap,
                 status=status, t_solve=dt_solve)


if __name__ == "__main__":
    main()
