"""Path fit at the V-turn tip: uniform periodic spline vs one open spline cut at the tip (CUT=1). Curvature of the fitted
path against CPC's own curvature |v x a| / |v|^3 at its nodes, and position error.
    uv run --with casadi --with pyyaml --with scipy --with numpy python cut_check.py
"""
import os

import numpy as np

import common as C
from check import quat_R


def report(k, cut, hp=0.5, cont=3):
    os.environ["CUT"], os.environ["CUTC"] = ("1" if cut else "0"), str(cont)
    Pa = C.Path_(k, hp)
    d = np.load(C.ROOT / f"logs/cpc_traj_tol0.24_{k}.npz"); x, uu = d["x"], d["u"]
    x, uu = np.roll(x, -Pa.i0, 0), np.roll(uu, -Pa.i0, 0)
    kc = np.array([np.linalg.norm(np.cross(x[i, 3:6], quat_R(x[i, 6:10])[:, 2] * uu[i].sum() / C.MASS - [0, 0, C.G]))
                   / np.linalg.norm(x[i, 3:6]) ** 3 for i in range(len(x))])
    sl = np.linalg.norm(np.roll(Pa.P, -1, 0) - Pa.P, axis=1)
    ud = np.r_[0, np.cumsum(sl)[:-1]] / Pa.L
    un = ud[::20]
    p1, p2 = Pa.prow(un, 1) @ Pa.Cp, Pa.prow(un, 2) @ Pa.Cp
    kf = np.linalg.norm(np.cross(p1, p2), axis=1) / np.linalg.norm(p1, axis=1) ** 3
    pos = np.abs(Pa.prow(ud, 0) @ Pa.Cp - Pa.P).max()
    it = int(np.argmin(np.linalg.norm(x[:, 3:6], axis=1)))           # CPC's slowest node = the tip
    lab = f"cut at tip (C{cont})" if cut else "uniform periodic"
    print(f"track {k}, {lab:18s}, knots {hp} m: position error max {pos * 1000:6.2f} mm | curvature at the tip: CPC "
          f"{kc[it]:6.2f}, fit {kf[it]:6.2f} 1/m | curvature error max over CPC nodes {np.abs(kf - kc).max():6.2f} 1/m "
          f"(at node {(int(np.argmax(np.abs(kf - kc))) + Pa.i0) % len(x)})")


if __name__ == "__main__":
    for k in (5, 6):
        report(k, False)
        for cont in (3, 2):
            report(k, True, cont=cont)
        report(k, True, hp=0.1)
