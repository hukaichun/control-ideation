"""Is CPC's own speed profile (on its own path) inside our limits? Fit CPC's speed on the path as a q spline
(Q ~ speed^2) and as a w spline (w = dt/du), evaluate the physical check. Then start IPOPT from it."""
import sys

import numpy as np

import common as C
import q_version as QV
import w_version as WV


def cpc_speed(k):
    d = np.load(C.ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")
    x, T, N = d["x"], d["T"], int(d["N"])
    dts = np.repeat(T / N, N)
    P, Vv = [], []
    for i in range(len(x)):
        h = dts[i]
        p0, v0, p1, v1 = x[i, :3], x[i, 3:6], x[(i + 1) % len(x), :3], x[(i + 1) % len(x), 3:6]
        for s in np.arange(20) / 20:
            P.append((2 * s ** 3 - 3 * s ** 2 + 1) * p0 + (s ** 3 - 2 * s ** 2 + s) * h * v0
                     + (-2 * s ** 3 + 3 * s ** 2) * p1 + (s ** 3 - s ** 2) * h * v1)
            Vv.append(((6 * s ** 2 - 6 * s) * p0 + (3 * s ** 2 - 4 * s + 1) * h * v0
                       + (-6 * s ** 2 + 6 * s) * p1 + (3 * s ** 2 - 2 * s) * h * v1) / h)
    P, Vv = np.array(P), np.array(Vv)
    sl = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    return np.r_[0, np.cumsum(sl)[:-1]] / sl.sum(), np.linalg.norm(Vv, axis=1)


k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
Pa = C.Path_(k)
u, v = cpc_speed(k)
pu = np.linalg.norm((C.prows(u, Pa.nref, 5, 1) @ Pa.Cp), axis=1)       # |p_u|
B0 = C.prows(u, Pa.n, C.DEG, 0)
zq = np.linalg.lstsq(B0, Pa.L ** 2 * (v / pu) ** 2, rcond=None)[0]
zw = np.linalg.lstsq(B0, pu / v, rcond=None)[0]
for nm, V, z in (("q", QV, zq), ("w", WV, zw)):
    print(f"[{nm}] CPC speed profile fitted: lap {V.lap(Pa, z):.4f} s (CPC {Pa.lap_cpc:.3f})")
    print(C.report("   40x finer", *V.phys(Pa, z, 40)))
np.save("results/cpc_speed_q.npy", zq); np.save("results/cpc_speed_w.npy", zw)
