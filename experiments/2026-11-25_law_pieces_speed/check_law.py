"""Before flying: on the planned trajectory, each law piece must return exactly the planned snap (zero tracking error),
its energy must be H = 1/2 |v|^2 + h = 0, and the blend in the overlaps returns what each piece returns.
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python check_law.py [track]
"""
import sys

import casadi as ca
import numpy as np

import fly as FL

k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
P, lap_cpc, pcs, cs, law, gap = FL.build_law(k)
pol = FL.BL.BlendPolicy(law)
ew, eh, eb, n = 0.0, 0.0, 0.0, 0
for i, (pc, c, lp) in enumerate(zip(pcs, cs, law)):
    for x in np.linspace(pc.xa + 0.05, pc.xb - 0.05, 15):
        xa = np.array([x])
        f, fu, fuu, r, ru = (np.array(v).ravel() for v in FL.R.parts(pc, xa, ca.DM(c)))
        G = pc.geom(xa)[:, 0]
        p = pc.O + x * pc.e1 + pc.f(x) * pc.e2 + pc.g(x) * pc.e3
        v = G[0:3] * np.sqrt(r[0])
        a = f - [0, 0, FL.R.C.G]
        j = np.sqrt(r[0]) * fu
        s = r[0] * fuu + ru[0] * fu / 2
        w = np.array(lp.F(p, v, a, j)[0]).ravel()
        ew = max(ew, np.linalg.norm(w - s) / max(np.linalg.norm(s), 1))
        eh = max(eh, abs(0.5 * v @ v + float(lp.hfun(p))))
        wb = np.array(pol._w(p, v, a, j)[0]).ravel()
        eb = max(eb, np.linalg.norm(wb - s) / max(np.linalg.norm(s), 1))
        n += 1
print(f"track {k}, {n} points on the planned trajectory (cores and overlaps):")
print(f"   own piece's law snap vs planned snap: max relative error {ew:.1e}")
print(f"   energy 1/2|v|^2 + h: max |H| {eh:.1e} m^2/s^2")
print(f"   blended law (all pieces with weight at that point) vs planned snap: max relative error {eb:.1e}")
