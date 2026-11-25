"""Start IPOPT from CPC's own speed profile, slowed by a factor (default 0.85 of the speed), instead of 2 m/s."""
import sys

import numpy as np

import common as C
import q_version as QV
import w_version as WV

nm, fac = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
V = QV if nm == "q" else WV
Pa = C.Path_(0)
z0 = np.load(f"results/cpc_speed_{nm}.npy")
z0 = z0 * fac ** 2 if nm == "q" else z0 / fac
print(C.report(f"[{nm}] start {V.lap(Pa, z0):.3f} s, 40x finer", *V.phys(Pa, z0, 40)))
zs, st, it, t = V.solve(Pa, z0)
print(f"[{nm}] {st}, {it} it, {t:.1f} s | lap {V.lap(Pa, zs):.4f} s (CPC {Pa.lap_cpc:.3f})")
for f_, lab in ((1, "   samples"), (40, "   40x finer")):
    print(C.report(lab, *V.phys(Pa, zs, f_)))
np.save(f"results/{nm}_0_fromcpc.npy", zs)
