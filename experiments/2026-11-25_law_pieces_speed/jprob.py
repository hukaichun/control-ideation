"""The planning problem in the control law's pieces, written with JAX (user, 2026-11-25: "你還是不要用 CasADi 好了";
own optimiser, sqp.py). Same problem as run_joint.py:
  variables  per piece: f, g (quintic) and r = (dx/dt)^2 (degree 4) coefficients on the piece's knots; one chord
             coordinate per gate (the point of the path that passes the gate)
  objective  T = sum_i int_core dx / sqrt(r_i)  (Gauss-Legendre, 8 points per span)
  limits     at samples on the whole piece, heading free, physical units: 15 - |w| [rad/s], c_max - c [m/s^2],
             T_max - m c/4 - sqrt2 J |w'|/(4l) and m c/4 - sqrt2 J |w'|/(4l) [N]
  overlaps   piece i's points lie on piece i+1's graph within PTOL (e2, e3), speeds agree within EPS (v^2)
  gates      |q(x_g) - gate| <= TOL
Every constraint block is a JAX function of the coefficients it touches (one piece, a pair, or a piece and a gate
coordinate); its Jacobian is jax.jacfwd of that block (no global expression graph).
"""
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.interpolate import BSpline

jax.config.update("jax_enable_x64", True)

import pieces as PC                      # noqa: E402  (numpy / scipy only)

ROOT = Path(__file__).resolve().parents[2]
Q = yaml.safe_load((ROOT / "third_party/rpg_time_optimal/quads/quad.yaml").read_text())
MASS, ARM = Q["mass"], Q["arm_length"]
JXY = float(np.array(Q["inertia"], float)[0, 0])
G = 9.81
TMAX = Q["TWR_max"] * MASS * G / 4
WMAX = Q["omega_max_xy"]
CMAX = 4 * TMAX / MASS
TOL = 0.24
PTOL = float(os.environ.get("PTOL", 0.01))
EPS = float(os.environ.get("EPS", 1.0))
SP = 0.125
FACT = np.array([1.0, 1, 2, 6, 24, 120])


def cpc_lap(k):
    d = np.load(ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")
    x, T, N = d["x"], d["T"], int(d["N"])
    dts = np.repeat(T / N, N)
    P = []
    for i in range(len(x)):
        h = dts[i]
        p0, v0, p1, v1 = x[i, :3], x[i, 3:6], x[(i + 1) % len(x), :3], x[(i + 1) % len(x), 3:6]
        for s in np.arange(20) / 20:
            P.append((2 * s ** 3 - 3 * s ** 2 + 1) * p0 + (s ** 3 - 2 * s ** 2 + s) * h * v0
                     + (-2 * s ** 3 + 3 * s ** 2) * p1 + (s ** 3 - s ** 2) * h * v1)
    return np.array(P), float(T.sum()), d["gates"]


def brows(t, k, x, d):
    n = len(t) - k - 1
    b = BSpline(t, np.eye(n), k)
    return (b.derivative(d) if d else b)(np.clip(np.asarray(x, float), t[0], t[-1]))


def sample_limits(q1, q2, q3, q4, r0, r1, r2, r3):
    """the four limits at one point from the path derivatives in x (3-vectors) and r, r', r'', r'''."""
    f = q2 * r0 + q1 * r1 / 2 + jnp.array([0.0, 0, G])
    fx = q3 * r0 + 1.5 * q2 * r1 + 0.5 * q1 * r2
    fxx = q4 * r0 + 2.5 * q3 * r1 + 2 * q2 * r2 + 0.5 * q1 * r3
    s = r0 * fxx + r1 * fx / 2
    ff = f @ f
    c = jnp.sqrt(ff)
    A = jnp.cross(f, fx)
    N = jnp.cross(f, s) * ff - 2 * r0 * A * (f @ fx)
    w = jnp.sqrt(r0 * (A @ A)) / ff
    tq = np.sqrt(2) * JXY * jnp.sqrt(N @ N) / ff ** 2 / (4 * ARM)
    return jnp.array([WMAX - w, CMAX - c, TMAX - MASS * c / 4 - tq, MASS * c / 4 - tq])


def sample_phys(q1, q2, q3, q4, r0, r1, r2, r3):
    """|w| [rad/s], torque needed / allowed, collective (check only)."""
    f = q2 * r0 + q1 * r1 / 2 + jnp.array([0.0, 0, G])
    fx = q3 * r0 + 1.5 * q2 * r1 + 0.5 * q1 * r2
    fxx = q4 * r0 + 2.5 * q3 * r1 + 2 * q2 * r2 + 0.5 * q1 * r3
    s = r0 * fxx + r1 * fx / 2
    ff = f @ f
    c = jnp.sqrt(ff)
    A = jnp.cross(f, fx)
    N = jnp.cross(f, s) * ff - 2 * r0 * A * (f @ fx)
    tq = np.sqrt(2) * JXY * jnp.sqrt(N @ N) / ff ** 2 / (4 * ARM)
    return jnp.array([jnp.sqrt(r0 * (A @ A)) / ff, tq / jnp.minimum(TMAX - MASS * c / 4, MASS * c / 4), c])


VLIM = jax.vmap(sample_limits)
VPHYS = jax.vmap(sample_phys)


class Piece:
    """numeric data of one piece for the JAX blocks."""
    def __init__(self, pc):
        self.pc = pc
        self.tf, self.tr = pc.f.t, pc.tk
        self.nf, self.nr = len(self.tf) - 6, pc.nr
        self.n = 2 * self.nf + self.nr
        self.E = np.array([pc.e1, pc.e2, pc.e3])

    def samples(self, fac=1, core=False):
        a, b = self.pc.xc if core else (self.pc.xa, self.pc.xb)
        n = max(2, int(np.ceil((b - a) / (SP / fac))))
        kn = np.unique(np.r_[self.tf, self.tr])
        return np.unique(np.r_[a, a + (np.arange(n) + 0.5) * (b - a) / n, kn[(kn > a) & (kn < b)], b])

    def rows(self, x):
        return [jnp.array(brows(self.tf, 5, x, d)) for d in range(5)], [jnp.array(brows(self.tr, 4, x, d)) for d in range(4)]

    def split(self, zi):
        return zi[:self.nf], zi[self.nf:2 * self.nf], zi[2 * self.nf:]

    def limits_fn(self, x, phys=False):
        Bf, Br = self.rows(x)
        e1, e2, e3 = (jnp.array(v) for v in self.E)
        fun = VPHYS if phys else VLIM

        def g(zi):
            cf, cg, cr = self.split(zi)
            q = [(d == 1) * e1[None, :] + (Bf[d] @ cf)[:, None] * e2[None, :] + (Bf[d] @ cg)[:, None] * e3[None, :]
                 for d in range(1, 5)]
            return fun(*q, *(Br[d] @ cr for d in range(4))).ravel()
        return g


class Problem:
    def __init__(self, k, free_path=True):
        P, self.lap_cpc, self.gates = cpc_lap(k)
        self.pcs, _, self.L = PC.build(P)
        self.pp = [Piece(pc) for pc in self.pcs]
        self.free_path = free_path
        self.off = np.cumsum([0] + [p.n for p in self.pp])
        self.ng = len(self.gates)
        self.nz = self.off[-1] + self.ng
        self.blocks = []                       # (variable index array, jitted g, jitted jacobian, lb, ub, name)
        for i, p in enumerate(self.pp):
            g = p.limits_fn(p.samples())
            self._add(np.arange(self.off[i], self.off[i + 1]), g, 0.0, np.inf, "limits")
        for i in range(len(self.pp)):
            self._overlap(i, (i + 1) % len(self.pp))
        for gi, gt in enumerate(self.gates):
            self._gate(gi, gt, active=free_path)
        self._objective()

    def _add(self, idx, g, lb, ub, name):
        gj, jj = jax.jit(g), jax.jit(jax.jacfwd(g))
        hj = jax.jit(jax.hessian(lambda zi, lam: lam @ g(zi)))       # second derivatives of lam . g (block)
        m = int(np.asarray(gj(jnp.zeros(len(idx)))).size)
        self.blocks.append((idx, gj, jj, np.broadcast_to(lb, m).astype(float), np.broadcast_to(ub, m).astype(float), name))
        self.hblocks = getattr(self, "hblocks", []) + [hj]

    def _overlap(self, i, j):
        a, b = self.pp[i], self.pp[j]
        pa, pb = a.pc, b.pc
        xa = np.arange(pa.xc[1] - PC.D, pa.xb, 0.1)
        pts = pa.O + np.outer(xa, pa.e1) + np.outer(pa.f(xa), pa.e2) + np.outer(pa.g(xa), pa.e3)
        xb0 = pb.xof(pts)
        ok = (xb0 >= pb.xa + 0.05) & (xb0 <= pb.xb - 0.05)
        xa, xb0 = xa[ok], xb0[ok]
        Bfa, Bra = a.rows(xa)
        Bfb = [jnp.array(brows(b.tf, 5, xb0, m)) for m in range(6)]
        Brb = jnp.array(brows(b.tr, 4, xb0, 0))
        Ea, Eb, Oa, Ob = jnp.array(a.E), jnp.array(b.E), jnp.array(pa.O), jnp.array(pb.O)
        xa_j, xb_j = jnp.array(xa), jnp.array(xb0)

        def g(zz):
            cfa, cga, cra = a.split(zz[:a.n])
            cfb, cgb, crb = b.split(zz[a.n:])
            q = Oa[None, :] + xa_j[:, None] * Ea[0] + (Bfa[0] @ cfa)[:, None] * Ea[1] + (Bfa[0] @ cga)[:, None] * Ea[2]
            d = q - Ob[None, :]
            dx = d @ Eb[0] - xb_j
            fb = sum((Bfb[m] @ cfb) * dx ** m / FACT[m] for m in range(6))
            gb = sum((Bfb[m] @ cgb) * dx ** m / FACT[m] for m in range(6))
            va = (Bra[0] @ cra) * (1 + (Bfa[1] @ cfa) ** 2 + (Bfa[1] @ cga) ** 2)
            vb = (Brb @ crb) * (1 + (Bfb[1] @ cfb) ** 2 + (Bfb[1] @ cgb) ** 2)
            return jnp.concatenate([d @ Eb[1] - fb, d @ Eb[2] - gb, va - vb])
        n = len(xa)
        idx = np.r_[np.arange(self.off[i], self.off[i + 1]), np.arange(self.off[j], self.off[j + 1])]
        self._add(idx, g, np.r_[np.full(2 * n, -PTOL), np.full(n, -EPS)], np.r_[np.full(2 * n, PTOL), np.full(n, EPS)], "overlap")

    def _gate(self, gi, gt, active=True):
        best = None
        for i, p in enumerate(self.pp):
            xg = float(p.pc.xof(gt)[0])
            if p.pc.xa + 0.05 <= xg <= p.pc.xb - 0.05:
                xs = np.linspace(max(p.pc.xa, xg - 1), min(p.pc.xb, xg + 1), 4001)
                pts = p.pc.O + np.outer(xs, p.pc.e1) + np.outer(p.pc.f(xs), p.pc.e2) + np.outer(p.pc.g(xs), p.pc.e3)
                kk = int(np.argmin(np.linalg.norm(pts - gt, axis=1)))
                dd = np.linalg.norm(pts[kk] - gt)
                if best is None or dd < best[0]:
                    best = (dd, i, xs[kk])
        _, i, x0 = best
        p = self.pp[i]
        Bf = [jnp.array(brows(p.tf, 5, [x0], m))[0] for m in range(6)]
        E, O, gtj = jnp.array(p.E), jnp.array(p.pc.O), jnp.array(gt)

        def g(zz):
            cf, cg, _ = p.split(zz[:-1])
            dx = zz[-1] - x0
            f0 = sum((Bf[m] @ cf) * dx ** m / FACT[m] for m in range(6))
            g0 = sum((Bf[m] @ cg) * dx ** m / FACT[m] for m in range(6))
            dv = O + zz[-1] * E[0] + f0 * E[1] + g0 * E[2] - gtj
            return jnp.array([dv @ dv])
        self.gate_x0 = getattr(self, "gate_x0", []) + [(x0, i)]
        if active:      # path fixed: the gates are whatever the fixed path does (not constraints)
            self._add(np.r_[np.arange(self.off[i], self.off[i + 1]), self.off[-1] + gi], g, 0.0, TOL ** 2, "gate")

    def _objective(self):
        xg_, wg_ = np.polynomial.legendre.leggauss(8)
        rows, wts = [], []
        for p in self.pp:
            a, b = p.pc.xc
            br = np.unique(np.r_[a, p.tr[(p.tr > a) & (p.tr < b)], b])
            xq = np.concatenate([(br[i] + br[i + 1]) / 2 + (br[i + 1] - br[i]) / 2 * xg_ for i in range(len(br) - 1)])
            wq = np.concatenate([(br[i + 1] - br[i]) / 2 * wg_ for i in range(len(br) - 1)])
            Bz = np.zeros((len(xq), self.nz))
            off = self.off[self.pp.index(p)] + 2 * p.nf
            Bz[:, off:off + p.nr] = brows(p.tr, 4, xq, 0)
            rows.append(Bz); wts.append(wq)
        self.Bq, self.wq = np.vstack(rows), np.concatenate(wts)

    # ---- evaluation ----
    def T(self, z):
        return float(self.wq @ (self.Bq @ z) ** -0.5)

    def gradT(self, z):
        return self.Bq.T @ (-0.5 * self.wq * (self.Bq @ z) ** -1.5)

    def hessT(self, z):
        return (self.Bq * (0.75 * self.wq * (self.Bq @ z) ** -2.5)[:, None]).T @ self.Bq

    def cons(self, z, jac=True):
        import scipy.sparse as sp
        gs, lbs, ubs, R_, C_, V_ = [], [], [], [], [], []
        row = 0
        for idx, gj, jj, lb, ub, _ in self.blocks:
            zi = jnp.array(z[idx])
            gv = np.asarray(gj(zi))
            gs.append(gv); lbs.append(lb); ubs.append(ub)
            if jac:
                Jb = np.asarray(jj(zi))
                r, c = np.nonzero(Jb)
                R_.append(r + row); C_.append(idx[c]); V_.append(Jb[r, c])
            row += len(gv)
        g, lb, ub = np.concatenate(gs), np.concatenate(lbs), np.concatenate(ubs)
        if not jac:
            return g, lb, ub
        Jm = sp.csc_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))), shape=(row, self.nz))
        return g, lb, ub, Jm

    def hess_lag(self, z, lam):
        """sum_i lam_i Hess g_i (dense nz x nz), block by block."""
        H = np.zeros((self.nz, self.nz))
        row = 0
        for (idx, gj, jj, lb, ub, _), hj in zip(self.blocks, self.hblocks):
            m = len(lb)
            H[np.ix_(idx, idx)] += np.asarray(hj(jnp.array(z[idx]), jnp.array(lam[row:row + m])))
            row += m
        return H

    def z0(self, r0):
        z = np.concatenate([np.r_[p.pc.f.c[:p.nf], p.pc.g.c[:p.nf], r] for p, r in zip(self.pp, r0)])
        return np.r_[z, [x for x, _ in self.gate_x0]]

    def bounds(self):
        lo, hi = [], []
        for p in self.pp:
            lo += [-np.inf] * (2 * p.nf) + [0.0] * p.nr
            hi += [np.inf] * p.n
        lo += [self.pp[i].pc.xa for _, i in self.gate_x0]
        hi += [self.pp[i].pc.xb for _, i in self.gate_x0]
        return np.array(lo), np.array(hi)

    def path_mask(self):
        m = np.zeros(self.nz, bool)
        for i, p in enumerate(self.pp):
            m[self.off[i]:self.off[i] + 2 * p.nf] = True
        return m

    def phys(self, z, fac=40):
        out = []
        for i, p in enumerate(self.pp):
            g = p.limits_fn(p.samples(fac, core=True), phys=True)
            out.append(np.asarray(jax.jit(g)(jnp.array(z[self.off[i]:self.off[i + 1]]))).reshape(-1, 3))
        return np.vstack(out)
