"""Before optimising the cylinder toy: the quotient-rule derivatives of r = Q/m against JAX derivatives along x,
the objective against a direct arc-length / speed integral, and the start point's feasibility.
    ADAPT=1 uv run --with jax --with numpy --with scipy --with clarabel --with pyyaml python check_cyl.py
"""
import jax
import jax.numpy as jnp
import numpy as np

import jprob as JP
import toy_cylinder as TC

prob = TC.CylProblem(2)
p = prob.pp[0]
z0 = prob.z0()
zi = jnp.array(z0[prob.off[0]:prob.off[1]])
cf, cg, cq = p.split(zi)
# r(x) = Q(x) / |q_x(x)|^2 as a function of x (exact B-spline evaluation), derivatives by JAX
fspl = lambda d: (lambda x: jnp.array(JP.brows(p.tf, 5, [float(x)], d))[0])


def r_of_x(x):
    # piecewise polynomial around x0 via Taylor of the splines (exact on the span)
    return None


x0 = 0.5 * (p.pc.xa + p.pc.xb) + 0.123
Bf = [jnp.array(JP.brows(p.tf, 5, [x0], d))[0] for d in range(6)]
Bq = [jnp.array(JP.brows(p.tr, 4, [x0], d))[0] for d in range(5)]
E = jnp.array(p.E)


def r_t(t):
    fp = sum((Bf[d + 1] @ cf) * t ** d / JP.FACT[d] for d in range(5))
    gp = sum((Bf[d + 1] @ cg) * t ** d / JP.FACT[d] for d in range(5))
    Q = sum((Bq[d] @ cq) * t ** d / JP.FACT[d] for d in range(5))
    return Q / (1 + fp ** 2 + gp ** 2)


ad = [r_t(0.0), jax.grad(r_t)(0.0), jax.grad(jax.grad(r_t))(0.0), jax.grad(jax.grad(jax.grad(r_t)))(0.0)]
# the same through limits_Q's internal formulas
q = [(d == 1) * E[0] + (Bf[d] @ cf) * E[1] + (Bf[d] @ cg) * E[2] for d in range(1, 5)]
Qd = [Bq[d] @ cq for d in range(4)]
m0, m1 = q[0] @ q[0], 2 * q[0] @ q[1]
m2, m3 = 2 * (q[1] @ q[1] + q[0] @ q[2]), 2 * (3 * q[1] @ q[2] + q[0] @ q[3])
u0, u1 = 1 / m0, -m1 / m0 ** 2
u2 = (2 * m1 ** 2 - m0 * m2) / m0 ** 3
u3 = (-6 * m1 ** 3 + 6 * m0 * m1 * m2 - m0 ** 2 * m3) / m0 ** 4
mine = [Qd[0] * u0, Qd[1] * u0 + Qd[0] * u1, Qd[2] * u0 + 2 * Qd[1] * u1 + Qd[0] * u2,
        Qd[3] * u0 + 3 * Qd[2] * u1 + 3 * Qd[1] * u2 + Qd[0] * u3]
print("1. r, r', r'', r''' (quotient rule vs JAX):", np.round(np.array(mine, float), 6), np.round(np.array(ad, float), 6))
# objective: arc length of the start path / 3 m/s over the cores (start speed 3 m/s everywhere)
L = 0.0
for pp_ in prob.pp:
    xx = np.linspace(*pp_.pc.xc, 200001)
    pts = pp_.pc.O + np.outer(xx, pp_.pc.e1) + np.outer(pp_.pc.f(xx), pp_.pc.e2) + np.outer(pp_.pc.g(xx), pp_.pc.e3)
    L += np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
print(f"2. objective at the start {prob.T(z0):.6f} s vs core arc length / 3 m/s {L / 3:.6f} s")
g, lb, ub = prob.cons(z0, jac=False)
print(f"3. start: violation {np.maximum(0, lb - g).sum() + np.maximum(0, g - ub).sum():.1e}")
