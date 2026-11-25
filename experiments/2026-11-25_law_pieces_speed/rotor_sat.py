"""Is it close to bang-bang? Per-rotor saturation (user, 2026-11-25: look at single rotors, not the collective).
CPC: rotor thrusts at its nodes (time-weighted). Ours: the rotor thrusts actually applied in the flight of the planned
law (fly.py, recorded every simulation step), on lap 2.
Fractions of time: each rotor within 1% of T_max or of 0, averaged over the 4 rotors; all 4 rotors at a bound at once.
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python rotor_sat.py [tracks...]
"""
import sys

import numpy as np

import fly as FL

TOL = 0.01


def frac(u, w, Tmax):
    at_hi, at_lo = u >= (1 - TOL) * Tmax, u <= TOL * Tmax
    per = ((at_hi | at_lo) * w[:, None]).sum(0) / w.sum()
    allb = ((at_hi | at_lo).all(1) * w).sum() / w.sum()
    return per.mean(), (at_hi * w[:, None]).sum() / w.sum() / 4, (at_lo * w[:, None]).sum() / w.sum() / 4, allb


class RecQuad(FL.B.qs.Quad):
    def __init__(self):
        super().__init__()
        self.U = []

    def step(self, x, u, dt):
        self.U.append(np.array(u, float))
        return super().step(x, u, dt)


def main(tracks):
    out = []
    for k in tracks:
        d = np.load(FL.R.C.ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")
        q = FL.B.qs.Quad()
        wc = np.repeat(d["T"] / int(d["N"]), int(d["N"]))
        a = frac(d["u"], wc, q.Tmax)
        P, lap_cpc, pcs, cs, law, gap = FL.build_law(k)
        lap_plan = float(open(FL.HERE / "results" / f"run_{k}.txt").read().split("lap ")[2].split(" s")[0])
        p0, v0, a0, j0 = FL.start_state(pcs[0], cs[0])
        rq = RecQuad()
        lg = FL.fe.fly(rq, FL.BL.BlendPolicy(law), p0, v0, a0, j0, 2.2 * lap_plan, mode="B", plant=rq)
        U = np.array(rq.U)[:len(lg)]
        side = (lg[:, 4:7] - p0) @ (v0 / np.linalg.norm(v0))
        near = np.linalg.norm(lg[:, 4:7] - p0, axis=1) < 3.0
        cr = np.nonzero((side[:-1] < 0) & (side[1:] >= 0) & near[1:])[0]
        sl = slice(cr[0], cr[1] + 1)
        b = frac(U[sl], np.ones(len(U[sl])), q.Tmax)
        line = (f"track {k}: rotor at a bound (mean over rotors) / at T_max / at 0 / all four at a bound:  "
                f"CPC {100 * a[0]:.0f}% / {100 * a[1]:.0f}% / {100 * a[2]:.0f}% / {100 * a[3]:.0f}%  |  "
                f"ours (flown, +{100 * (lap_plan / lap_cpc - 1):.1f}%) {100 * b[0]:.0f}% / {100 * b[1]:.0f}% / {100 * b[2]:.0f}% / {100 * b[3]:.0f}%")
        print(line, flush=True)
        out.append(line)
    (FL.HERE / "results" / f"rotor_sat_{'_'.join(map(str, tracks))}.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [0])
