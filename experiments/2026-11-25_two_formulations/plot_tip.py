"""Plot the fitted path splines at the V-turn tip (tracks 5, 6) against CPC: shape (projected on the turn plane) and
curvature along the path. Output results/tip_splines.png.
    uv run --with casadi --with pyyaml --with scipy --with numpy --with matplotlib python plot_tip.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C
from check import quat_R

FITS = [("uniform 0.5 m (original)", "0", 0.5, 3, "#2a78d6"), ("cut at tip, C3, 0.5 m", "1", 0.5, 3, "#eb6834"),
        ("cut at tip, C2, 0.5 m", "1", 0.5, 2, "#e87ba4"), ("cut at tip, C3, 0.1 m", "1", 0.1, 3, "#1baf7a")]
W = 0.5                                     # half window in m of arc length around the tip


def tip_data(k):
    d = np.load(C.ROOT / f"logs/cpc_traj_tol0.24_{k}.npz"); x, uu = d["x"], d["u"]
    it = int(np.argmin(np.linalg.norm(x[:, 3:6], axis=1)))
    kc = np.array([np.linalg.norm(np.cross(x[i, 3:6], quat_R(x[i, 6:10])[:, 2] * uu[i].sum() / C.MASS - [0, 0, C.G]))
                   / np.linalg.norm(x[i, 3:6]) ** 3 for i in range(len(x))])
    return x, kc, it


fig, ax = plt.subplots(2, 2, figsize=(12, 9.5), facecolor="#fcfcfb")
for r, k in enumerate((5, 6)):
    x, kc, it = tip_data(k)
    os.environ["CUT"] = "0"
    P, _ = C.cpc_lap(k)
    tip = x[it, :3]
    near = P[np.linalg.norm(P - tip, axis=1) < 0.6]
    ctr = near.mean(0)
    _, _, Vt = np.linalg.svd(near - ctr)
    E = Vt[:2]                                # turn plane
    pr = lambda Q: (Q - tip) @ E.T * 100     # cm
    # CPC arc length from the tip
    sl = np.linalg.norm(np.diff(x[:, :3], axis=0), axis=1); cum = np.r_[0, np.cumsum(sl)]
    sn = cum - cum[it]
    m = np.abs(sn) < W
    a0, a1 = ax[r, 0], ax[r, 1]
    dense = P[np.linalg.norm(P - tip, axis=1) < 0.35]
    a0.plot(*pr(dense).T, ".", color="#8a8a85", ms=3, label="CPC path", zorder=1)
    a0.plot(*pr(x[m, :3]).T, "o", color="#4a4a46", ms=4, label="CPC nodes", zorder=2)
    a1.plot(sn[m] * 100, kc[m], "o", color="#4a4a46", ms=5, label="CPC (|v x a| / |v|^3)", zorder=3)
    for lab, cut, hp, cont, col in FITS:
        os.environ["CUT"], os.environ["CUTC"] = cut, str(cont)
        Pa = C.Path_(k, hp)
        sl2 = np.linalg.norm(np.roll(Pa.P, -1, 0) - Pa.P, axis=1); ud = np.r_[0, np.cumsum(sl2)[:-1]] / Pa.L
        i_tip = int(np.argmin(np.linalg.norm(Pa.P - tip, axis=1)))
        ut = ud[i_tip]
        uu_ = (ut + np.linspace(-W, W, 4001) / Pa.L) % 1.0
        if Pa.cut:
            uu_ = np.sort(np.r_[np.linspace(0, W / Pa.L, 2001), np.linspace(1 - W / Pa.L, 1 - 1e-12, 2001)])
        p0, p1, p2 = (Pa.prow(uu_, dd) @ Pa.Cp for dd in range(3))
        kap = np.linalg.norm(np.cross(p1, p2), axis=1) / np.linalg.norm(p1, axis=1) ** 3
        s_rel = (((uu_ - ut + 0.5) % 1.0) - 0.5) * Pa.L
        o = np.argsort(s_rel)
        sel = np.linalg.norm(p0[o] - tip, axis=1) < 0.35
        a0.plot(*pr(p0[o][sel]).T, "-", color=col, lw=2, label=lab, zorder=4)
        a1.plot(s_rel[o] * 100, kap[o], "-", color=col, lw=2, label=lab, zorder=4)
    # parameter = CPC's time instead of arc length (same knot count as uniform 0.5 m), periodic, no cut
    d = np.load(C.ROOT / f"logs/cpc_traj_tol0.24_{k}.npz"); T, N = d["T"], int(d["N"])
    tp = np.r_[0, np.cumsum(np.repeat(np.repeat(T / N, N) / 20, 20))[:-1]] / T.sum()
    sl3 = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1); L3 = sl3.sum(); nref = int(round(L3 / 0.5))
    Cp = np.linalg.lstsq(C.prows(tp, nref, 5, 0), P, rcond=None)[0]
    t_tip = tp[20 * it]
    uu_ = (t_tip + np.linspace(-0.03, 0.03, 6001)) % 1.0
    p0, p1, p2 = (C.prows(uu_, nref, 5, dd) @ Cp for dd in range(3))
    kap = np.linalg.norm(np.cross(p1, p2), axis=1) / np.linalg.norm(p1, axis=1) ** 3
    ds = np.linalg.norm(np.diff(p0, axis=0), axis=1); s_rel = np.r_[0, np.cumsum(ds)]
    s_rel -= s_rel[np.argmin(np.linalg.norm(p0 - tip, axis=1))]
    sel = np.abs(s_rel) < W
    a1.plot(s_rel[sel] * 100, kap[sel], "--", color="#008300", lw=2, label="parameter = CPC time, 0.5 m knots, no cut", zorder=5)
    sel = np.linalg.norm(p0 - tip, axis=1) < 0.35
    a0.plot(*pr(p0[sel]).T, "--", color="#008300", lw=2, label="parameter = CPC time, no cut", zorder=5)
    a0.set_title(f"track {k}: path at the V-turn tip (turn plane, cm)", fontsize=11)
    a0.set_aspect("equal"); a0.set_xlabel("cm"); a0.set_ylabel("cm")
    a1.set_title(f"track {k}: curvature along the path (1/m)", fontsize=11)
    a1.set_xlabel("arc length from the tip (cm)"); a1.set_ylabel("curvature (1/m)")
    for a in (a0, a1):
        a.grid(color="#e4e3dd", lw=0.8); a.set_facecolor("#fcfcfb")
        for sp_ in a.spines.values():
            sp_.set_color("#c3c2b7")
    a1.legend(fontsize=8, frameon=False)
    a0.legend(fontsize=8, frameon=False, loc="best")
fig.tight_layout()
fig.savefig("results/tip_splines.png", dpi=130)
print("saved results/tip_splines.png")
