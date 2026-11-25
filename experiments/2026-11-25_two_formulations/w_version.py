"""Version w (user, 2026-11-25): variable w(u) = dt/du (periodic spline), d/dt = (1/w) d/du.
Objective T = int_0^1 w du = mean of the control points (exact, linear). Bounds w >= 0 (natural domain only).
Limits: common.rows at the samples. Start: 2 m/s.
    uv run --with casadi --with pyyaml --with scipy --with numpy python w_version.py [track]
"""
import sys

import casadi as ca
import numpy as np

import common as C


def chain():
    """f, j, s from p_u..p_uuuu and w, w_u, w_uu, w_uuu (division by w)."""
    X = [ca.SX.sym(f"p{k}", 3) for k in range(5)]
    W = [ca.SX.sym(f"w{k}") for k in range(4)]
    base, rate = ca.vertcat(*X[:4], *W[:3]), ca.vertcat(*X[1:], *W[1:])

    def dt(E):
        return ca.jtimes(E, base, rate) / W[0]
    f = dt(dt(X[0])) + ca.DM([0, 0, C.G])
    j = dt(f)
    s = dt(j)
    return ca.Function("chain_w", [ca.vertcat(*X[1:], *W)], [f, j, s])


CHAIN = chain()


def fjs(Pd, B, z):
    """f, j, s at the points of (Pd, B) for coefficients z (casadi or numpy)."""
    W = [ca.mtimes(ca.DM(b), z).T for b in B]
    return CHAIN.map(Pd.shape[1])(ca.vertcat(ca.DM(Pd), *W))


def limits(f, j, s):
    A = ca.cross(f, j)
    N = ca.cross(f, s) * ca.dot(f, f) - 2 * A * ca.dot(f, j)
    return C.rows(f, ca.norm_2(A), N)


_f, _j, _s = (ca.SX.sym(nm, 3) for nm in "fjs")
LIM = ca.Function("lim_w", [_f, _j, _s], [limits(_f, _j, _s)])


def lap(Pa, z):
    return float(np.mean(z))


def start(Pa, v=2.0):
    return np.full(Pa.n, Pa.L / v)


def solve(Pa, z0):
    z = ca.SX.sym("z", Pa.n)
    Pd, B = Pa.at(Pa.us)
    f, j, s = fjs(Pd, B, z)
    g = LIM.map(Pa.M)(f, j, s)
    return C.ipopt(z, ca.sum1(z) / Pa.n, ca.vec(g), z0, np.zeros(Pa.n))


def phys(Pa, z, fac):
    Pd, B = Pa.at(Pa.grid(fac))
    return C.physical(*(np.array(v) for v in fjs(Pd, B, z)))


if __name__ == "__main__":
    import run
    run.main(sys.modules[__name__], "w", int(sys.argv[1]) if len(sys.argv) > 1 else 0)
