"""Which limit is active along the planned lap (user, 2026-11-25: "那應該是角速度被限制造成的").
Time-weighted fractions of the lap where: |w| >= 99% of 15 rad/s; worst-heading torque share >= 99% of what the rotors
allow; collective >= 99% of max; none of these (slack). Pieces plan (cores, this directory), one-spline q plan
(../2026-11-25_two_formulations), CPC (|w_x| or |w_y| >= 99% of 15, a rotor within 1% of a bound).
    uv run --with casadi --with pyyaml --with scipy --with numpy python active.py [tracks...]
"""
import os
import sys

import casadi as ca
import numpy as np

import pieces as PC
import run as R

C = R.C
CMAX = 4 * C.TMAX / C.MASS


def classify(w, tr, c, dt):
    rate, torque, coll = w >= 0.99 * C.WMAX, tr >= 0.99, c >= 0.99 * CMAX
    T = dt.sum()
    pct = lambda m: 100 * dt[m].sum() / T
    return (f"|w| at limit {pct(rate):4.0f}%, torque at limit {pct(torque):4.0f}%, collective at max {pct(coll):4.0f}%, "
            f"none {pct(~(rate | torque | coll)):4.0f}%")


def main(tracks):
    import q_version as QV
    out = []
    for k in tracks:
        d = np.load(C.ROOT / f"logs/cpc_traj_tol0.24_{k}.npz")
        dtc = np.repeat(d["T"] / int(d["N"]), int(d["N"]))
        wc = np.abs(d["x"][:, 10:12]).max(1)
        uu = d["u"]
        rot = ((uu >= 0.99 * C.TMAX) | (uu <= 0.01 * C.TMAX)).any(1)
        T = dtc.sum()
        out.append(f"track {k}:\n   CPC                : |w_x| or |w_y| at limit {100 * dtc[wc >= 0.99 * C.WMAX].sum() / T:4.0f}%, "
                   f"a rotor at a bound {100 * dtc[rot].sum() / T:4.0f}%, neither {100 * dtc[~(rot | (wc >= 0.99 * C.WMAX))].sum() / T:4.0f}%")
        P, _ = C.cpc_lap(k)
        pcs, junc, L = PC.build(P)
        z = np.load(R.HERE / "results" / f"track{k}.npz")
        cs = np.split(z["z"], np.cumsum(z["nr"])[:-1])
        W, TR, CC, DT = [], [], [], []
        for pc, c_ in zip(pcs, cs):
            x = np.linspace(*pc.xc, 4000)
            f, fu, fuu, r, ru = (np.array(v) for v in R.parts(pc, x, ca.DM(c_)))
            w, tr, cc = C.physical(f, np.sqrt(r) * fu, r * fuu + ru * fu / 2)
            W.append(w); TR.append(tr); CC.append(cc); DT.append(np.gradient(x) / np.sqrt(r.ravel()))
        out.append("   pieces plan        : " + classify(*(np.concatenate(a) for a in (W, TR, CC, DT))))
        os.environ.pop("CUT", None)
        Pa = C.Path_(k)
        zq = np.load(R.HERE.parent / "2026-11-25_two_formulations" / "results" / f"q_{k}.npy")
        u = np.linspace(0, 1, 40000, endpoint=False)
        Pd, B = Pa.at(u)
        f, fu, fuu, r, ru = (np.array(v) for v in QV.parts(Pa, Pd, B, zq))
        w, tr, cc = C.physical(f, np.sqrt(r) * fu, r * fuu + ru * fu / 2)
        out.append("   one-spline q plan  : " + classify(w, tr, cc, (1 / 40000) / np.sqrt(r.ravel())))
    print("\n".join(out))
    (R.HERE / "results" / f"active_{'_'.join(map(str, tracks))}.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [0])
