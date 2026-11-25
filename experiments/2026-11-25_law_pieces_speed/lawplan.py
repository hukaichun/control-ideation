"""Planning in the control law's pieces, written cleanly (user, 2026-11-25).

Variables   per piece i: f_i, g_i (quintic) and Q_i = v^2 = -2 h_i (degree 4) coefficients on the same knots.
            velocity = speed x tangent: dx/dt = sqrt(Q / m), m = |q_x|^2 = 1 + f'^2 + g'^2.
Objective   T = sum_i int_{core_i} sqrt(m) / sqrt(Q) dx      (arc length / speed), Gauss-Legendre per span.
Constraints (1) physical limits (heading free; 15 - |w|, c_max - c, rotors within [0, T_max] for every heading) on the
                core only (user: "現階段我要乾淨的、可控的"), at checkpoints that move with the core ends
            (2) junctions: at every cut, the two pieces meet with equal position, velocity, acceleration and jerk
                (12 equalities; the cut sits at the core end x of piece i and the core start x of piece i+1)
            (3) the task's own constraints (ends, obstacles, gates), given as functions of a piece's position
Everything is a small JAX function of the coefficients it touches; derivatives by JAX.
"""
import jax
import jax.numpy as jnp
import numpy as np

import jprob as JP

NS = 3                                        # interior check points per span


def r_derivs(q1, q2, q3, q4, Q0, Q1, Q2, Q3):
    """r = Q / m and its x-derivatives (quotient rule), m = q1.q1."""
    m0, m1 = q1 @ q1, 2 * q1 @ q2
    m2, m3 = 2 * (q2 @ q2 + q1 @ q3), 2 * (3 * q2 @ q3 + q1 @ q4)
    u0, u1 = 1 / m0, -m1 / m0 ** 2
    u2 = (2 * m1 ** 2 - m0 * m2) / m0 ** 3
    u3 = (-6 * m1 ** 3 + 6 * m0 * m1 * m2 - m0 ** 2 * m3) / m0 ** 4
    return Q0 * u0, Q1 * u0 + Q0 * u1, Q2 * u0 + 2 * Q1 * u1 + Q0 * u2, Q3 * u0 + 3 * Q2 * u1 + 3 * Q1 * u2 + Q0 * u3


def limits(*a):
    return JP.sample_limits(*a[:4], *r_derivs(*a))


def phys(*a):
    return JP.sample_phys(*a[:4], *r_derivs(*a))


def state(q1, q2, q3, q4, Q0, Q1, Q2, Q3):
    """velocity, acceleration, jerk at a point (for the junctions)."""
    r0, r1, r2, r3 = r_derivs(q1, q2, q3, q4, Q0, Q1, Q2, Q3)
    v = q1 * jnp.sqrt(r0)
    a = q2 * r0 + q1 * r1 / 2
    j = jnp.sqrt(r0) * (q3 * r0 + 1.5 * q2 * r1 + 0.5 * q1 * r2)
    return jnp.concatenate([v, a, j])


VLIM, VPHYS = jax.vmap(limits), jax.vmap(phys)


class Plan:
    """pcs: pieces (pieces.Piece) with cores; task(pos) -> extra rows >= 0 (or None); ends: list of
    (piece index, 'start'|'end', point, speed) equalities."""
    def __init__(self, pcs, task=None, ends=(), closed=False):
        self.pp = [JP.Piece(pc) for pc in pcs]
        self.off = np.cumsum([0] + [p.n for p in self.pp])
        n = len(self.pp)
        self.juncs = [(i, (i + 1) % n) for i in range(n if closed else n - 1)]
        # junction chord coordinates are variables (the meeting point must be free to move: with both fixed the point
        # was pinned to the intersection of two lines): x_end of piece i and x_start of piece j, each kept inside the
        # knot span that holds its start value (so the span polynomial is exact)
        self.xj0, self.xjb = [], []
        for i, j in self.juncs:
            for p, x0 in ((self.pp[i], self.pp[i].pc.xc[1]), (self.pp[j], self.pp[j].pc.xc[0])):
                br = np.unique(p.tr)
                k = min(int(np.searchsorted(br, x0, side="right")) - 1, len(br) - 2)
                self.xj0.append(x0)
                self.xjb.append((p.pc.xa + 0.05, p.pc.xb - 0.05))      # anywhere in the piece (exact evaluation)
        self.nzc = int(self.off[-1])
        self.nz = self.nzc + len(self.xj0)
        self.core_end = {i: self.nzc + 2 * k for k, (i, j) in enumerate(self.juncs)}
        self.core_start = {j: self.nzc + 2 * k + 1 for k, (i, j) in enumerate(self.juncs)}
        self.blocks = []
        for p in self.pp:                 # span tables built outside any JAX trace (cached constants)
            self.spans(p)
        for i, p in enumerate(self.pp):
            self._piece(i, task)
        for k, (i, j) in enumerate(self.juncs):
            self._junction(k, i, j)
        for e in ends:
            self._end(*e)
        self._objective()

    # ---- helpers ----
    def _add(self, idx, g, lb, ub, name):
        gj, jj = jax.jit(g), jax.jit(jax.jacfwd(g))
        hj = jax.jit(jax.hessian(lambda zi, lam: lam @ g(zi)))
        m = int(np.asarray(gj(jnp.zeros(len(idx)))).size)
        self.blocks.append(dict(idx=idx, g=gj, J=jj, H=hj, lb=np.broadcast_to(lb, m).astype(float),
                                ub=np.broadcast_to(ub, m).astype(float), name=name))

    def checkpoints(self, p, fac=1, core=False):
        a, b = p.pc.xc if core else (p.pc.xa, p.pc.xb)
        br = np.unique(np.r_[a, p.tr[(p.tr > a) & (p.tr < b)], b])
        k = NS * fac
        inner = [br[s] + (br[s + 1] - br[s]) * (np.arange(1, k + 1) / (k + 1)) for s in range(len(br) - 1)]
        return np.unique(np.r_[br, np.concatenate(inner)])

    def geo(self, p, x):
        Bf, Br = p.rows(x)
        Bq4 = jnp.array(JP.brows(p.tr, 4, x, 0))
        E, O, xj = jnp.array(p.E), jnp.array(p.pc.O), jnp.array(x)

        def fn(zi):
            cf, cg, cq = p.split(zi)
            q = [(d == 1) * E[0][None, :] + (Bf[d] @ cf)[:, None] * E[1][None, :] + (Bf[d] @ cg)[:, None] * E[2][None, :]
                 for d in range(1, 5)]
            pos = O[None, :] + xj[:, None] * E[0] + (Bf[0] @ cf)[:, None] * E[1] + (Bf[0] @ cg)[:, None] * E[2]
            return q, [Br[d] @ cq for d in range(4)], pos
        return fn

    def _piece(self, i, task):
        """limits (and the task rows) on the core only: checkpoints x_k = x_s + tau_k (x_e - x_s) with fixed tau_k
        (NS + 1 per span of the start core), so they move with the junction coordinates x_s, x_e; values exact (eval_at)."""
        p = self.pp[i]
        a0, b0 = p.pc.xc
        nsp = max(1, int(np.sum((np.unique(p.tr) > a0) & (np.unique(p.tr) < b0))) + 1)
        tau = jnp.array(np.linspace(0, 1, (NS + 1) * nsp + 1))
        js, je = self.core_start.get(i), self.core_end.get(i)
        n = p.n
        idx = np.r_[np.arange(self.off[i], self.off[i + 1]), [js] if js is not None else [], [je] if je is not None else []].astype(int)

        def g(zz):
            zi = zz[:n]
            xs = zz[n] if js is not None else a0
            xe = zz[n + (js is not None)] if je is not None else b0
            xk = xs + tau * (xe - xs)
            out = jax.vmap(lambda x: self._point(p, zi, x))(xk)
            lim = jax.vmap(lambda q1, q2, q3, q4, Q0, Q1, Q2, Q3: limits(q1, q2, q3, q4, Q0, Q1, Q2, Q3))(*out[0], *out[1])
            rows = [lim.ravel()]
            if task is not None:
                rows.append(task(out[2]))
            return jnp.concatenate(rows)
        zt = jnp.array(np.r_[np.zeros(n), [a0] if js is not None else [], [b0] if je is not None else []])
        m = int(np.asarray(jax.jit(g)(zt)).size)
        self._add(idx, g, np.zeros(m), np.full(m, np.inf), "limits")

    def _point(self, p, zi, x):
        """(q1..q4), (Q0..Q3), position at a variable x (exact)."""
        F, Gg, Qd, _ = self.eval_at(p, zi, x)
        E, O = jnp.array(p.E), jnp.array(p.pc.O)
        q = [(d == 1) * E[0] + F[d] * E[1] + Gg[d] * E[2] for d in range(1, 5)]
        return q, Qd[:4], O + x * E[0] + F[0] * E[1] + Gg[0] * E[2]

    def spans(self, p):
        """per span of the piece (unique knots t_k): Taylor rows at the left knot for f, g (derivatives 0..5) and Q
        (0..4), Gauss nodes / weights of the span, and the rows there for the objective's integrand."""
        if not hasattr(p, "_sp"):
            t = np.unique(p.tr)
            S = len(t) - 1
            xg_, wg_ = np.polynomial.legendre.leggauss(8)
            Rf = np.stack([JP.brows(p.tf, 5, t[:-1], d) for d in range(6)])              # 6 x S x nf
            Rq = np.stack([JP.brows(p.tr, 4, t[:-1], d) for d in range(5)])              # 5 x S x nr
            xq = (t[:-1, None] + t[1:, None]) / 2 + (t[1:, None] - t[:-1, None]) / 2 * xg_[None, :]
            wq = (t[1:, None] - t[:-1, None]) / 2 * wg_[None, :]
            B1 = JP.brows(p.tf, 5, xq.ravel(), 1).reshape(S, 8, -1)
            B0 = JP.brows(p.tr, 4, xq.ravel(), 0).reshape(S, 8, -1)
            p._sp = dict(t=jnp.array(t), Rf=jnp.array(Rf), Rq=jnp.array(Rq), wq=jnp.array(wq), B1=jnp.array(B1),
                         B0=jnp.array(B0), xg=jnp.array(xg_), wg=jnp.array(wg_), S=S)
        return p._sp

    def eval_at(self, p, zi, x):
        """exact spline values at a variable x anywhere in the piece: the polynomial of the span holding x."""
        sp_ = self.spans(p)
        k = jnp.clip(jnp.searchsorted(sp_["t"], x, side="right") - 1, 0, sp_["S"] - 1)
        cf, cg, cq = p.split(zi)
        u = x - sp_["t"][k]
        F = [sum((sp_["Rf"][d + m][k] @ cf) * u ** m / JP.FACT[m] for m in range(6 - d)) for d in range(6)]
        Gg = [sum((sp_["Rf"][d + m][k] @ cg) * u ** m / JP.FACT[m] for m in range(6 - d)) for d in range(6)]
        Qd = [sum((sp_["Rq"][d + m][k] @ cq) * u ** m / JP.FACT[m] for m in range(5 - d)) for d in range(5)]
        return F, Gg, Qd, k

    def F_time(self, p, zi, x):
        """int_{xa}^{x} sqrt(m)/sqrt(Q) dx, exact per span (Gauss 8), x a variable anywhere in the piece."""
        sp_ = self.spans(p)
        cf, cg, cq = p.split(zi)
        Ik = jnp.sum(sp_["wq"] * jnp.sqrt(1 + (sp_["B1"] @ cf) ** 2 + (sp_["B1"] @ cg) ** 2) / jnp.sqrt(sp_["B0"] @ cq), axis=1)
        C = jnp.concatenate([jnp.zeros(1), jnp.cumsum(Ik)])
        F, Gg, Qd, k = self.eval_at(p, zi, x)
        tk = sp_["t"][k]
        un = (x - tk) * (1 + sp_["xg"]) / 2
        fp = sum((sp_["Rf"][1 + m][k] @ cf) * un ** m / JP.FACT[m] for m in range(5))
        gp = sum((sp_["Rf"][1 + m][k] @ cg) * un ** m / JP.FACT[m] for m in range(5))
        Q = sum((sp_["Rq"][m][k] @ cq) * un ** m / JP.FACT[m] for m in range(5))
        return C[k] + (x - tk) / 2 * (sp_["wg"] @ (jnp.sqrt(1 + fp ** 2 + gp ** 2) / jnp.sqrt(Q)))

    def geo_at(self, p, x0):
        """position, path derivatives and Q derivatives at a VARIABLE x near x0: Taylor of the span polynomial around x0
        (exact while x stays in the span of x0)."""
        Bf = [jnp.array(JP.brows(p.tf, 5, [x0], d))[0] for d in range(6)]
        Bq = [jnp.array(JP.brows(p.tr, 4, [x0], d))[0] for d in range(5)]
        E, O = jnp.array(p.E), jnp.array(p.pc.O)
        fac = JP.FACT

        def fn(zi, x):
            F, Gg, Qd, _ = self.eval_at(p, zi, x)      # exact in every span (x may leave the span of x0)
            Qd = Qd[:4]
            q = [(d == 1) * E[0] + F[d] * E[1] + Gg[d] * E[2] for d in range(1, 5)]
            pos = O + x * E[0] + F[0] * E[1] + Gg[0] * E[2]
            return q, Qd, pos
        return fn

    def _junction(self, k, i, j):
        a, b = self.pp[i], self.pp[j]
        ga, gb = self.geo_at(a, self.xj0[2 * k]), self.geo_at(b, self.xj0[2 * k + 1])
        na, nb = a.n, b.n

        def g(zz):
            qa, Qa, pa = ga(zz[:na], zz[na + nb])
            qb, Qb, pb = gb(zz[na:na + nb], zz[na + nb + 1])
            return jnp.concatenate([pa - pb, state(*qa, *Qa) - state(*qb, *Qb)])
        idx = np.r_[np.arange(self.off[i], self.off[i + 1]), np.arange(self.off[j], self.off[j + 1]),
                    self.nzc + 2 * k, self.nzc + 2 * k + 1]
        self._add(idx, g, np.zeros(12), np.zeros(12), "junction")

    def _end(self, i, which, point, speed):
        p = self.pp[i]
        x = p.pc.xa if which == "start" else p.pc.xb
        geo = self.geo(p, np.array([x]))
        pt = jnp.array(point)

        def g(zi):
            q, Qd, pos = geo(zi)
            return jnp.concatenate([pos[0] - pt, jnp.array([Qd[0][0] - speed ** 2])])
        self._add(np.arange(self.off[i], self.off[i + 1]), g, np.zeros(4), np.zeros(4), "end")

    def _objective(self):
        """T = sum_i [F_i(core end) - F_i(core start)], F_i(x) = int_{xa}^{x} sqrt(m)/sqrt(Q) dx (exact per span); a core
        end at a junction is the junction coordinate (a variable)."""
        ends = []
        for i, p in enumerate(self.pp):
            self.spans(p)
            ends.append((i, p, self.core_start.get(i), p.pc.xc[0], self.core_end.get(i), p.pc.xc[1]))

        def T(z):
            tot = 0.0
            for i, p, js, xs0, je, xe0 in ends:
                zi = z[self.off[i]:self.off[i + 1]]
                xs = z[js] if js is not None else xs0
                xe = z[je] if je is not None else xe0
                tot = tot + self.F_time(p, zi, xe) - self.F_time(p, zi, xs)
            return tot
        self.T, self.gradT, self.hessT = jax.jit(T), jax.jit(jax.grad(T)), jax.jit(jax.hessian(T))

    # ---- what the solver needs ----
    def cons(self, z):
        g = [np.asarray(b["g"](jnp.array(z[b["idx"]]))) for b in self.blocks]
        return np.concatenate(g)

    def jac(self, z):
        J = np.zeros((sum(len(b["lb"]) for b in self.blocks), self.nz))
        r = 0
        for b in self.blocks:
            m = len(b["lb"])
            J[r:r + m, b["idx"]] = np.asarray(b["J"](jnp.array(z[b["idx"]])))
            r += m
        return J

    def hess_lag(self, z, lam):
        H = np.array(self.hessT(jnp.array(z)))
        r = 0
        for b in self.blocks:
            m = len(b["lb"])
            H[np.ix_(b["idx"], b["idx"])] += np.asarray(b["H"](jnp.array(z[b["idx"]]), jnp.array(lam[r:r + m])))
            r += m
        return H

    def bounds_g(self):
        return np.concatenate([b["lb"] for b in self.blocks]), np.concatenate([b["ub"] for b in self.blocks])

    def bounds_z(self, qmin=0.0):
        lo = np.concatenate([np.r_[np.full(2 * p.nf, -np.inf), np.full(p.nr, qmin)] for p in self.pp] + [[b[0] for b in self.xjb]])
        hi = np.r_[np.full(self.nzc, np.inf), [b[1] for b in self.xjb]]
        return lo, hi

    def scale(self):
        """variable scale: path coefficients and junction coordinates 1 m, speed coefficients 100 m^2/s^2."""
        return np.concatenate([np.r_[np.ones(2 * p.nf), np.full(p.nr, 100.0)] for p in self.pp] + [np.ones(len(self.xj0))])

    def z0(self, speed):
        return np.concatenate([np.r_[p.pc.f.c[:p.nf], p.pc.g.c[:p.nf],
                                     np.linalg.lstsq(JP.brows(p.tr, 4, xx := np.linspace(p.pc.xa, p.pc.xb, 400), 0),
                                                     np.full(400, speed ** 2), rcond=None)[0]] for p in self.pp] + [self.xj0])
