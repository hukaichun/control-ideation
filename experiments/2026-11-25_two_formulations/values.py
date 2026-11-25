"""Print the limit-row values, their gradients and the objective gradient at a point (user: "把值列出來看一下")."""
import sys

import casadi as ca
import numpy as np

import common as C
import q_version as QV
import w_version as WV

name = sys.argv[1] if len(sys.argv) > 1 else "q"
V = QV if name == "q" else WV
Pa = C.Path_(0)
z = ca.SX.sym("z", Pa.n)
Pd, B = Pa.at(Pa.us)
if name == "q":
    g = QV.LIM.map(Pa.M)(*QV.parts(Pa, Pd, B, z)); T = QV.T_expr(Pa, z)
else:
    g = WV.LIM.map(Pa.M)(*WV.fjs(Pd, B, z)); T = ca.sum1(z) / Pa.n
F = ca.Function("F", [z], [g, ca.jacobian(ca.vec(g), z), T, ca.gradient(T, z)])
for lab, zz in (("start (2 m/s)", V.start(Pa)), ("IPOPT stop", np.load(f"results/{name}_0.npy"))):
    G, J, Tv, gT = (np.array(v) for v in F(zz))
    J = J.reshape(Pa.M, 4, Pa.n) if False else np.array(J).reshape(Pa.M, 4, Pa.n)   # vec: sample-major
    print(f"[{name}] {lab}: T {Tv.item():.3f} s, |dT/dz| {np.abs(gT).min():.2e}..{np.abs(gT).max():.2e}, "
          f"z {zz.min():.3g}..{zz.max():.3g}")
    for i, nm in enumerate(C.NAMES):
        gi = G[i]; Ji = np.linalg.norm(J[:, i, :], axis=1)
        k = int(np.argmin(gi))
        print(f"   {nm:16s} value {gi.min():+.2e} .. {gi.max():+.2e} | gradient norm {Ji.min():.2e} .. {Ji.max():.2e}"
              f" | smallest at sample {k}: value {gi[k]:+.2e}, value/gradient {gi[k] / Ji[k]:.2e}")

# along the uniform speed-up direction from the stop point: actual row vs its linear prediction
zz = np.load(f"results/{name}_0.npy")
d = -zz if name == "w" else 2 * zz            # w -> w (1 - t);  Q -> Q (1 + 2t) (first order of (1+t)^2)
G0, J0 = (np.array(v) for v in F(zz)[:2])
J0 = J0.reshape(Pa.M, 4, Pa.n)
k = int(np.argmin(G0[0]))
print(f"   along the speed-up line, sample {k} (smallest |w| row):  t, actual row, linear prediction, |w| rad/s")
for t in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
    Gt = np.array(F(zz + t * d)[0])
    w_, _, _ = V.phys(Pa, zz + t * d, 1)
    print(f"     t {t:5.3f}  {Gt[0, k]:+.3e}  {G0[0, k] + t * J0[k, 0] @ d:+.3e}  {w_[k]:.2f}")
