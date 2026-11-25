"""Before optimising: (1) junction state (v, a, j) against finite differences in time along one piece (t(x) = int dx /
sqrt(r)); (2) block Jacobians and Lagrangian Hessian against central differences.
    ADAPT=1 uv run --with jax --with numpy --with scipy --with clarabel --with pyyaml python check_lawplan.py
"""
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid

import cyl2
import jprob as JP
import lawplan as LP

Pl = cyl2.build(2)
z = Pl.z0(3.0) + np.random.default_rng(0).normal(0, 1e-3, Pl.nz) * Pl.scale()
p = Pl.pp[0]
zi = jnp.array(z[Pl.off[0]:Pl.off[1]])
xs = np.linspace(p.pc.xa + 0.5, p.pc.xa + 2.5, 200001)
q, Qd, pos = Pl.geo(p, xs)(zi)
r = np.asarray(LP.jax.vmap(LP.r_derivs)(*q, *Qd)[0])
t = cumulative_trapezoid(1 / np.sqrt(r), xs, initial=0.0)
pos = np.asarray(pos)
k = 100000
h = 2e-3
tk = t[k] + h * np.arange(-3, 4)
Pt = np.array([np.interp(tt, t, pos[:, c]) for tt in tk for c in range(3)]).reshape(7, 3)
V = np.vander(h * np.arange(-3, 4), 7, increasing=True).T
fd = [np.linalg.solve(V, np.eye(7)[d] * [1, 1, 2, 6, 24, 120, 720][d]) @ Pt for d in (1, 2, 3)]
st = np.asarray(LP.state(*(v[k] for v in q), *(v[k] for v in Qd)))
print("1. junction state vs finite differences in time: v", np.round(st[:3], 4), np.round(fd[0], 4),
      "| a", np.round(st[3:6], 3), np.round(fd[1], 3), "| j", np.round(st[6:], 2), np.round(fd[2], 2))
rng = np.random.default_rng(1)
lam = rng.normal(size=len(Pl.bounds_g()[0]))
Ja, Ha = Pl.jac(z), Pl.hess_lag(z, lam)
gl = lambda zz: np.asarray(Pl.gradT(zz)) + Pl.jac(zz).T @ lam
ej, eh = 0.0, 0.0
for c in rng.choice(Pl.nz, 8, replace=False):
    hh = 1e-6 * Pl.scale()[c]
    zp, zm = z.copy(), z.copy(); zp[c] += hh; zm[c] -= hh
    ej = max(ej, np.abs((Pl.cons(zp) - Pl.cons(zm)) / (2 * hh) - Ja[:, c]).max() / max(np.abs(Ja[:, c]).max(), 1e-9))
    eh = max(eh, np.abs((gl(zp) - gl(zm)) / (2 * hh) - Ha[:, c]).max() / max(np.abs(Ha[:, c]).max(), 1e-9))
print(f"2. Jacobian vs central differences: {ej:.1e}; Lagrangian Hessian vs differences of its gradient: {eh:.1e}")
