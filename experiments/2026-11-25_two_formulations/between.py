"""Where and how much the limits are broken between the samples, in physical units (N, rad/s), not as ratios."""
import sys

import casadi as ca
import numpy as np

import common as C
import q_version as QV
import w_version as WV

nm = sys.argv[1] if len(sys.argv) > 1 else "q"
V = QV if nm == "q" else WV
TR = int(sys.argv[2]) if len(sys.argv) > 2 else 0
Pa = C.Path_(TR)
z = np.load(f"results/{nm}_{TR}.npy")
F = 40
u = Pa.grid(F)
M = len(u)
Pd, B = Pa.at(u)
if nm == "q":
    g = np.array(QV.LIM.map(M)(*(np.array(v) for v in QV.parts(Pa, Pd, B, z))))
else:
    g = np.array(WV.LIM.map(M)(*(np.array(v) for v in WV.fjs(Pd, B, z))))
w, tr, c = V.phys(Pa, z, F)
samp = np.isin(np.round(u, 12), np.round(Pa.us, 12))   # fine-grid points that are also optimizer samples
print(f"[{nm}] lap {V.lap(Pa, z):.4f} s. Rows in physical units on the 40x grid: min over all / min at the samples")
for i, n_ in enumerate(C.NAMES):
    print(f"   {n_:16s} {g[i].min():+.4f} / {g[i][samp].min():+.4f}  ({['rad/s', 'm/s^2', 'N', 'N'][i]})")
k = int(np.argmin(g[2]))
print(f"   worst rotor<=Tmax at u {u[k]:.5f} (s = {u[k] * Pa.L:.2f} m): c {c[k]:.3f}, T_max - m c/4 = "
      f"{C.TMAX - C.MASS * c[k] / 4:.4f} N, torque share needed {C.TMAX - C.MASS * c[k] / 4 - g[2, k]:.4f} N")
print("   profile around it (S = optimizer sample):")
for i in range(k - 60, k + 61, 5):
    print(f"     u {u[i]:.5f} {'S' if samp[i] else ' '}  c {c[i]:7.3f}  T_max-mc/4 {C.TMAX - C.MASS * c[i] / 4:7.4f} N  "
          f"torque share {C.TMAX - C.MASS * c[i] / 4 - g[2, i]:7.4f} N  row {g[2, i]:+.4f} N  |w| {w[i]:6.2f}")

k = int(np.argmin(g[0]))
print(f"   worst |w| at u {u[k]:.5f} (s = {u[k] * Pa.L:.2f} m): |w| {w[k]:.2f}, c {c[k]:.2f}")
for i in range(k - 60, k + 61, 5):
    print(f"     u {u[i]:.5f} {'S' if samp[i] else ' '}  c {c[i]:7.3f}  |w| {w[i]:6.2f}")
print("   path knots near:", [round(i / Pa.nref, 5) for i in range(int(u[k] * Pa.nref) - 1, int(u[k] * Pa.nref) + 3)],
      " speed knots near:", [round(i / Pa.n, 5) for i in range(int(u[k] * Pa.n), int(u[k] * Pa.n) + 2)],
      " sample spacing in u", round(1 / Pa.M0, 5))
