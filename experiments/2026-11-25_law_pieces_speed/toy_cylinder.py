"""Time-optimal turn around a cylinder (user, 2026-11-25: "用拋物線當初始，當作繞圓柱過彎的 time optimal").
Speed written as the user designed it: velocity = speed x tangent; the tangent comes from the path parametrisation, the
speed is its own variable: Q(x) = v^2 = -2 h (the law's potential, directly), per piece a degree-4 B-spline on the
same knots as f, g. dx/dt = v / |q_x|, |q_x|^2 = 1 + f'^2 + g'^2.
  min  T = sum_i int_core |q_x| / sqrt(Q) dx                (arc length / speed; depends on the path)
  s.t. the four physical limits (heading free) at samples on the whole piece
       cylinder: vertical axis at (CX, CY), radius RC: horizontal distance of every sample >= RC
       overlaps: same curve (PTOL), same speed (|dQ| <= EPS)
       ends: start and end points fixed, speed V0 there; Q coefficients >= 0
Initial path: the parabola y = A x^2, z = Z0, x in [-X, X] (vertex radius 2.5 m); the cylinder sits inside the turn.
Own SQP (sqp.py), derivatives by JAX.
    uv run --with jax --with numpy --with scipy --with clarabel --with pyyaml python toy_cylinder.py [pieces ...]
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import jprob as JP
import pieces as PC
import sqp

A, X, Z0, V0 = 0.2, 6.0, 2.0, 3.0
CX, CY, RC = 0.0, 2.5, 1.0
PTOL, EPS = JP.PTOL, JP.EPS


def limits_Q(q1, q2, q3, q4, Q0, Q1, Q2, Q3, phys=False):
    """limits at one point with the speed Q = v^2: r = Q / m, m = |q_x|^2 (>= 1), derivatives by the quotient rule."""
    m0, m1 = q1 @ q1, 2 * q1 @ q2
    m2, m3 = 2 * (q2 @ q2 + q1 @ q3), 2 * (3 * q2 @ q3 + q1 @ q4)
    u0 = 1 / m0
    u1 = -m1 / m0 ** 2
    u2 = (2 * m1 ** 2 - m0 * m2) / m0 ** 3
    u3 = (-6 * m1 ** 3 + 6 * m0 * m1 * m2 - m0 ** 2 * m3) / m0 ** 4
    r0, r1 = Q0 * u0, Q1 * u0 + Q0 * u1
    r2 = Q2 * u0 + 2 * Q1 * u1 + Q0 * u2
    r3 = Q3 * u0 + 3 * Q2 * u1 + 3 * Q1 * u2 + Q0 * u3
    return (JP.sample_phys if phys else JP.sample_limits)(q1, q2, q3, q4, r0, r1, r2, r3)


VL = jax.vmap(limits_Q)
VP = jax.vmap(lambda *a: limits_Q(*a, phys=True))


class CylProblem:
    def __init__(self, npieces):
        x = np.linspace(-X, X, 20001)
        P = np.c_[x, A * x ** 2, np.full(len(x), Z0)]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        cuts = [s[np.argmin(np.abs(P[:, 0] - xc))] for xc in np.linspace(-X, X, npieces + 1)]
        cuts[0], cuts[-1] = 0.0, s[-1]
        self.pcs = [PC.Piece(P, s, max(0.0, cuts[i] - (PC.D if i else 0)), min(s[-1], cuts[i + 1] + (PC.D if i < npieces - 1 else 0)),
                             (cuts[i], cuts[i + 1])) for i in range(npieces)]
        self.pp = [JP.Piece(pc) for pc in self.pcs]
        self.off = np.cumsum([0] + [p.n for p in self.pp])
        self.nz = self.off[-1]
        self.P0, self.P1 = P[0], P[-1]
        self.blocks, self.hblocks = [], []
        for i, p in enumerate(self.pp):
            self._piece_block(i)
        for i in range(npieces - 1):
            self._overlap(i, i + 1)
        self._ends()
        self._objective()

    def _add(self, idx, g, lb, ub, name):
        gj, jj = jax.jit(g), jax.jit(jax.jacfwd(g))
        hj = jax.jit(jax.hessian(lambda zi, lam: lam @ g(zi)))
        m = int(np.asarray(gj(jnp.zeros(len(idx)))).size)
        self.blocks.append((idx, gj, jj, np.broadcast_to(lb, m).astype(float), np.broadcast_to(ub, m).astype(float), name))
        self.hblocks.append(hj)

    def _geo(self, p, x):
        Bf, Br = p.rows(x)
        E = jnp.array(p.E)
        O = jnp.array(p.pc.O)
        xj = jnp.array(x)

        def geo(zi):
            cf, cg, cq = p.split(zi)
            q = [(d == 1) * E[0][None, :] + (Bf[d] @ cf)[:, None] * E[1][None, :] + (Bf[d] @ cg)[:, None] * E[2][None, :]
                 for d in range(1, 5)]
            pos = O[None, :] + xj[:, None] * E[0] + (Bf[0] @ cf)[:, None] * E[1] + (Bf[0] @ cg)[:, None] * E[2]
            return q, [Br[d] @ cq for d in range(4)], pos
        return geo

    def _piece_block(self, i):
        p = self.pp[i]
        x = p.samples()
        geo = self._geo(p, x)

        def g(zi):
            q, Qd, pos = geo(zi)
            lim = VL(*q, *Qd).ravel()
            cyl = (pos[:, 0] - CX) ** 2 + (pos[:, 1] - CY) ** 2 - RC ** 2
            return jnp.concatenate([lim, cyl])
        n = len(x)
        self._add(np.arange(self.off[i], self.off[i + 1]), g, np.zeros(5 * n), np.full(5 * n, np.inf), "limits+cylinder")

    def _overlap(self, i, j):
        a, b = self.pp[i], self.pp[j]
        pa, pb = a.pc, b.pc
        xa = np.arange(pa.xc[1] - PC.D, pa.xb, 0.1)
        pts = pa.O + np.outer(xa, pa.e1) + np.outer(pa.f(xa), pa.e2) + np.outer(pa.g(xa), pa.e3)
        xb0 = pb.xof(pts)
        ok = (xb0 >= pb.xa + 0.05) & (xb0 <= pb.xb - 0.05)
        xa, xb0 = xa[ok], xb0[ok]
        Bfa, Bra = a.rows(xa)
        Bfb = [jnp.array(JP.brows(b.tf, 5, xb0, m)) for m in range(6)]
        Brb = jnp.array(JP.brows(b.tr, 4, xb0, 0))
        Ea, Eb, Oa, Ob = jnp.array(a.E), jnp.array(b.E), jnp.array(pa.O), jnp.array(pb.O)
        xa_j, xb_j = jnp.array(xa), jnp.array(xb0)

        def g(zz):
            cfa, cga, cqa = a.split(zz[:a.n])
            cfb, cgb, cqb = b.split(zz[a.n:])
            qa = Oa[None, :] + xa_j[:, None] * Ea[0] + (Bfa[0] @ cfa)[:, None] * Ea[1] + (Bfa[0] @ cga)[:, None] * Ea[2]
            d = qa - Ob[None, :]
            dx = d @ Eb[0] - xb_j
            fb = sum((Bfb[m] @ cfb) * dx ** m / JP.FACT[m] for m in range(6))
            gb = sum((Bfb[m] @ cgb) * dx ** m / JP.FACT[m] for m in range(6))
            Qb = sum((jnp.array(JP.brows(b.tr, 4, xb0, m)) @ cqb) * dx ** m / JP.FACT[m] for m in range(5))
            return jnp.concatenate([d @ Eb[1] - fb, d @ Eb[2] - gb, Bra[0] @ cqa - Qb])
        n = len(xa)
        idx = np.r_[np.arange(self.off[i], self.off[i + 1]), np.arange(self.off[j], self.off[j + 1])]
        self._add(idx, g, np.r_[np.full(2 * n, -PTOL), np.full(n, -EPS)], np.r_[np.full(2 * n, PTOL), np.full(n, EPS)], "overlap")

    def _ends(self):
        for i, xe, Pe in ((0, self.pp[0].pc.xa, self.P0), (len(self.pp) - 1, self.pp[-1].pc.xb, self.P1)):
            p = self.pp[i]
            bf = jnp.array(JP.brows(p.tf, 5, [xe], 0))[0]
            bq = jnp.array(JP.brows(p.tr, 4, [xe], 0))[0]
            fy, gz = (Pe - p.pc.O) @ p.pc.e2, (Pe - p.pc.O) @ p.pc.e3

            def g(zi, p=p, bf=bf, bq=bq, fy=fy, gz=gz):
                cf, cg, cq = p.split(zi)
                return jnp.array([bf @ cf - fy, bf @ cg - gz, bq @ cq - V0 ** 2])
            self._add(np.arange(self.off[i], self.off[i + 1]), g, np.zeros(3), np.zeros(3), "ends")

    def _objective(self):
        xg_, wg_ = np.polynomial.legendre.leggauss(8)
        parts = []
        for i, p in enumerate(self.pp):
            a, b = p.pc.xc
            br = np.unique(np.r_[a, p.tr[(p.tr > a) & (p.tr < b)], b])
            xq = np.concatenate([(br[k] + br[k + 1]) / 2 + (br[k + 1] - br[k]) / 2 * xg_ for k in range(len(br) - 1)])
            wq = jnp.array(np.concatenate([(br[k + 1] - br[k]) / 2 * wg_ for k in range(len(br) - 1)]))
            B1 = jnp.array(JP.brows(p.tf, 5, xq, 1))
            B0q = jnp.array(JP.brows(p.tr, 4, xq, 0))
            parts.append((i, p, wq, B1, B0q))

        def T(z):
            tot = 0.0
            for i, p, wq, B1, B0q in parts:
                cf, cg, cq = p.split(z[self.off[i]:self.off[i + 1]])
                tot = tot + wq @ (jnp.sqrt(1 + (B1 @ cf) ** 2 + (B1 @ cg) ** 2) / jnp.sqrt(B0q @ cq))
            return tot
        self._T, self._gT, self._hT = jax.jit(T), jax.jit(jax.grad(T)), jax.jit(jax.hessian(T))

    T = lambda self, z: float(self._T(jnp.array(z)))
    gradT = lambda self, z: np.asarray(self._gT(jnp.array(z)))
    hessT = lambda self, z: np.asarray(self._hT(jnp.array(z)))
    cons = JP.Problem.cons
    hess_lag = JP.Problem.hess_lag

    def path_mask(self):
        m = np.zeros(self.nz, bool)
        for i, p in enumerate(self.pp):
            m[self.off[i]:self.off[i] + 2 * p.nf] = True
        return m

    def bounds(self):
        lo = np.concatenate([np.r_[np.full(2 * p.nf, -np.inf), np.zeros(p.nr)] for p in self.pp])
        return lo, np.full(self.nz, np.inf)

    def z0(self, v=V0):
        out = []
        for p in self.pp:
            xx = np.linspace(p.pc.xa, p.pc.xb, 400)
            cq = np.linalg.lstsq(JP.brows(p.tr, 4, xx, 0), np.full(len(xx), v ** 2), rcond=None)[0]
            out.append(np.r_[p.pc.f.c[:p.nf], p.pc.g.c[:p.nf], cq])
        return np.concatenate(out)

    def report(self, z):
        W, TR, CC, D = [], [], [], []
        for i, p in enumerate(self.pp):
            x = p.samples(40, core=True)
            q, Qd, pos = self._geo(p, x)(jnp.array(z[self.off[i]:self.off[i + 1]]))
            ph = np.asarray(VP(*q, *Qd))
            W.append(ph[:, 0]); TR.append(ph[:, 1]); CC.append(ph[:, 2])
            D.append(np.hypot(np.asarray(pos)[:, 0] - CX, np.asarray(pos)[:, 1] - CY))
        W, TR, CC, D = (np.concatenate(a) for a in (W, TR, CC, D))
        return (f"40x finer: |w| max {W.max():.2f}, torque ratio max {TR.max():.3f}, collective [{CC.min():.1f}, {CC.max():.2f}], "
                f"closest to the cylinder axis {D.min():.3f} m (radius {RC})")


def main(npcs):
    out = [f"cylinder turn: initial parabola y = {A} x^2, x in [-{X}, {X}]; cylinder axis ({CX}, {CY}), radius {RC} m; "
           f"ends fixed, speed {V0} m/s there; knots: ADAPT={PC.ADAPT} KMAX={PC.KMAX} FTOL={PC.FTOL}"]
    print(out[0], flush=True)
    for n in npcs:
        t0 = time.perf_counter()
        prob = CylProblem(n)
        z0 = prob.z0()
        lo, hi = prob.bounds()
        g, lb, ub = prob.cons(z0, jac=False)
        tb = time.perf_counter() - t0
        print(f"  {n} piece(s): {prob.nz} variables, {len(g)} constraints, build {tb:.1f} s, start T {prob.T(z0):.4f} s, "
              f"start violation {sqp.viol(g, lb, ub):.1e}", flush=True)
        z, info = sqp.solve(prob, z0, lo, hi, max_it=int(os.environ.get("MAXIT", 300)), log=lambda s: print(s, flush=True))
        line = (f"  {n} piece(s): T {info['T']:.4f} s | {info['it']} SQP steps, {info['t']:.1f} s | violation {info['viol']:.1e} | "
                + prob.report(z))
        print(line, flush=True)
        out.append(line)
        np.save(JP.Path(__file__).parent / "results" / f"cyl_{n}.npy", z)
    (JP.Path(__file__).parent / "results" / f"toy_cylinder_{'_'.join(map(str, npcs))}.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1, 2, 4])
