import sys; sys.path.insert(0, "/home/kai/uav/experiments/2026-11-25_two_formulations")
import numpy as np, common as C
from check import quat_R
k = int(sys.argv[1]); Pa = C.Path_(k)
d = np.load(C.ROOT / f"logs/cpc_traj_tol0.24_{k}.npz"); x, uu = d["x"], d["u"]
# CPC curvature at its nodes: kappa = |v x a| / |v|^3, a = R e3 c - g e3
P, _ = C.cpc_lap(k)
sl = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1); s_ = np.r_[0, np.cumsum(sl)[:-1]]; L = sl.sum()
un = s_[::20] / L
kc = []
for i in range(len(x)):
    v = x[i, 3:6]; a = quat_R(x[i, 6:10])[:, 2] * uu[i].sum() / C.MASS - [0, 0, C.G]
    kc.append(np.linalg.norm(np.cross(v, a)) / np.linalg.norm(v) ** 3)
kc = np.array(kc)
Pd, _ = Pa.at(un)
p1, p2 = Pd[0:3].T, Pd[3:6].T
kf = np.linalg.norm(np.cross(p1, p2), axis=1) / np.linalg.norm(p1, axis=1) ** 3
# fitted path between nodes: max curvature on a fine grid
uf = np.linspace(0, 1, 200000, endpoint=False); Pf, _ = Pa.at(uf)
kff = np.linalg.norm(np.cross(Pf[0:3].T, Pf[3:6].T), axis=1) / np.linalg.norm(Pf[0:3].T, axis=1) ** 3
print(f"track {k}: CPC curvature at nodes max {kc.max():.2f} 1/m; fitted path at the same points max {kf.max():.2f}, "
      f"difference max {np.abs(kf - kc).max():.2f}; fitted path anywhere max {kff.max():.2f} 1/m; |p_u|/L range "
      f"[{np.linalg.norm(Pf[0:3], axis=0).min() / L:.3f}, {np.linalg.norm(Pf[0:3], axis=0).max() / L:.3f}]; CPC speed min {np.linalg.norm(x[:, 3:6], axis=1).min():.2f} m/s")
