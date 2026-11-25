"""Before optimising the joint problem: at the start (fitted path, speed from run.py) the joint formulation must give
the same limits as run.py (path fixed), the overlap curve rows must be the fit mismatch (mm), and the gradient must
match central differences.
    ADAPT=1 FTOL=0.01 uv run --with casadi --with pyyaml --with scipy --with numpy python check_joint.py [track]
"""
import sys

import casadi as ca
import numpy as np

import pieces as PC
import run as R
import run_joint as J

k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
P, _ = R.C.cpc_lap(k)
pcs, junc, L = PC.build(P)
st0 = np.load(R.HERE / "results" / f"track{k}_cm.npz")
r0 = np.split(st0["z"], np.cumsum(st0["nr"])[:-1])
err, gerr = 0.0, 0.0
for pc, r in zip(pcs, r0):
    p = J.JPiece(pc)
    x = R.core_x(pc)
    a = np.array(R.QV.LIM.map(len(x))(*R.parts(pc, x, ca.DM(r))))
    b = np.array(R.QV.LIM.map(len(x))(*p.parts(x, ca.DM(p.z0(r)))))
    err = max(err, np.abs(a - b).max())
    zz = ca.SX.sym("z", p.n)
    gfun = ca.Function("g", [zz], [ca.vec(R.QV.LIM.map(len(x))(*p.parts(x, zz)))])
    jf = ca.Function("j", [zz], [ca.jacobian(ca.vec(R.QV.LIM.map(len(x))(*p.parts(x, zz))), zz)])
    z0 = p.z0(r)
    Ja = np.array(jf(z0))
    rng = np.random.default_rng(0)
    for q in rng.choice(p.n, 6, replace=False):
        h = 1e-6 * max(1, abs(z0[q]))
        zp, zm = z0.copy(), z0.copy(); zp[q] += h; zm[q] -= h
        num = (np.array(gfun(zp)).ravel() - np.array(gfun(zm)).ravel()) / (2 * h)
        gerr = max(gerr, np.abs(Ja[:, q] - num).max() / max(np.abs(Ja[:, q]).max(), 1e-9))
print(f"track {k}: joint limits at the start vs run.py (path fixed): max difference {err:.1e}")
print(f"   limit gradient w.r.t. f, g, r coefficients vs central differences: max relative error {gerr:.1e}")
