"""PX4 default cascade (px4_baseline.py) flying the CPC trajectory of each of the 20 tracks (logs/cpc_traj_<seed>.npz),
slowed down by a factor lam; per track the smallest lam (grid) whose second lap keeps every gate within 0.3 m.

    PYTHONPATH=. uv run --with casadi --with pyyaml --with scipy --with numba python px4_benchmark.py [disturbed] [tol]
"""
import sys
import time

import numpy as np

import px4_baseline as pb4
import quad_sim as qs

G = 9.81
EZ = np.array([0.0, 0.0, 1.0])
LAMS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)


def traj_from_npz(path):
    D = np.load(path)
    T, x, u, N = D["T"], D["x"], D["u"], int(D["N"])
    dts = np.repeat(T / N, N)
    t = np.concatenate([[0], np.cumsum(dts)])[:-1]
    R = [qs.rot(q) for q in x[:, 6:10]]
    m = qs.Quad().m
    A = np.array([Ri[:, 2] * ui.sum() / m for Ri, ui in zip(R, u)]) - G * EZ
    yaw = np.unwrap([np.arctan2(Ri[1, 0], Ri[0, 0]) for Ri in R])
    return dict(t=t, lap=float(T.sum()), P=x[:, 0:3], V=x[:, 3:6], A=A, yaw=yaw, X=x), D["gates"]


def plant_for(disturbed):
    plant = qs.Quad()
    if disturbed:
        plant.ext_acc = np.array([3.0, 2.0, 0.0])
        plant.m *= 1.1
    return plant


if __name__ == "__main__":
    disturbed = "disturbed" in sys.argv[1:]
    tol = next((a for a in sys.argv[1:] if a != "disturbed"), None)      # e.g. 0.24: CPC solved with that tolerance
    suf = "" if tol is None else f"_tol{tol}"
    quad = qs.Quad()
    rows = []
    for seed in range(20):
        try:
            tr, gates = traj_from_npz(f"../logs/cpc_traj{suf}_{seed}.npz")
        except FileNotFoundError:
            print(f"track {seed:2d}: no CPC trajectory", flush=True)
            continue
        best = None
        t0 = time.perf_counter()
        for lam in LAMS:
            sp = pb4.Setpoint(tr, lam)
            ctl = pb4.PX4(quad, "default")
            log = pb4.fly(plant_for(disturbed), ctl, sp, 2 * sp.T + 0.01) if disturbed else pb4.fly(quad, ctl, sp, 2 * sp.T + 0.01)
            miss, err = pb4.evaluate(log, sp, gates, sp.T)
            if miss <= 0.3:
                best = (lam, sp.T, miss, err)
                break
        if best is None:
            print(f"track {seed:2d}: CPC {tr['lap']:.3f} s | PX4 never within 0.3 m up to x{LAMS[-1]}", flush=True)
            rows.append((seed, tr["lap"], np.nan, np.nan, np.nan, np.nan))
        else:
            print(f"track {seed:2d}: CPC {tr['lap']:.3f} s | PX4 needs x{best[0]} -> lap {best[1]:.3f} s, gate miss "
                  f"{best[2]:.2f} m, max tracking error {best[3]:.2f} m ({time.perf_counter() - t0:.0f} s)", flush=True)
            rows.append((seed, tr["lap"], best[0], best[1], best[2], best[3]))
    np.savetxt(f"../logs/px4_benchmark{suf}{'_disturbed' if disturbed else ''}.csv", np.array(rows, float), delimiter=",",
               comments="", header="seed,lap_cpc,lam,lap_px4,gate_miss,max_track_err")
