"""Cylinder turn with lawplan.py + sqp2.py (clean version). Same task as toy_cylinder.py: initial parabola y = 0.2 x^2,
x in [-6, 6], z = 2 m; cylinder axis (0, 2.5), radius 1 m; ends fixed at 3 m/s.
    ADAPT=1 uv run --with jax --with numpy --with scipy --with clarabel --with pyyaml --with matplotlib python cyl2.py [pieces...]
"""
import sys
import time

import jax.numpy as jnp
import numpy as np

import jprob as JP
import lawplan as LP
import pieces as PC
import sqp2
import sqp_tr

A, X, Z0, V0 = 0.2, 6.0, 2.0, 3.0
CX, CY, RC = 0.0, 2.5, 1.0


def build(npieces):
    x = np.linspace(-X, X, 20001)
    P = np.c_[x, A * x ** 2, np.full(len(x), Z0)]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    cuts = [s[np.argmin(np.abs(P[:, 0] - xc))] for xc in np.linspace(-X, X, npieces + 1)]
    cuts[0], cuts[-1] = 0.0, s[-1]
    pcs = [PC.Piece(P, s, max(0.0, cuts[i] - (PC.D if i else 0)), min(s[-1], cuts[i + 1] + (PC.D if i < npieces - 1 else 0)),
                    (cuts[i], cuts[i + 1])) for i in range(npieces)]
    task = lambda pos: (pos[:, 0] - CX) ** 2 + (pos[:, 1] - CY) ** 2 - RC ** 2
    return LP.Plan(pcs, task=task, ends=[(0, "start", P[0], V0), (npieces - 1, "end", P[-1], V0)])


def report(Pl, z):
    W, TR, CC, D = [], [], [], []
    for i, p in enumerate(Pl.pp):
        x = np.linspace(z[Pl.core_start[i]] if i in Pl.core_start else p.pc.xc[0], z[Pl.core_end[i]] if i in Pl.core_end else p.pc.xc[1], 400)
        q, Qd, pos = Pl.geo(p, x)(jnp.array(z[Pl.off[i]:Pl.off[i + 1]]))
        ph = np.asarray(LP.VPHYS(*q, *Qd))
        W.append(ph[:, 0]); TR.append(ph[:, 1]); CC.append(ph[:, 2])
        D.append(np.hypot(np.asarray(pos)[:, 0] - CX, np.asarray(pos)[:, 1] - CY))
    W, TR, CC, D = (np.concatenate(a) for a in (W, TR, CC, D))
    return (f"10x finer: |w| max {W.max():.2f}, torque ratio max {TR.max():.3f}, collective [{CC.min():.1f}, {CC.max():.2f}], "
            f"closest to the cylinder axis {D.min():.3f} m")


def main(npcs):
    out = []
    for n in npcs:
        t0 = time.perf_counter()
        Pl = build(n)
        z0 = Pl.z0(V0)
        lb, ub = Pl.bounds_g()
        g0 = Pl.cons(z0)
        tb = time.perf_counter() - t0
        out.append(f"{n} piece(s): {Pl.nz} variables, {len(lb)} constraints ({sum(1 for b in Pl.blocks if b['name'] == 'junction')} junctions), "
                   f"build {tb:.1f} s | start T {float(Pl.T(z0)):.4f} s, violation {sqp2.theta(g0, lb, ub):.1e}")
        print(out[-1], flush=True)
        z, info = (sqp_tr if __import__("os").environ.get("TR", "1") == "1" else sqp2).solve(Pl, z0, log=lambda s: print(s, flush=True))
        out.append(f"   T {info['T']:.4f} s | {info['it']} SQP steps, {info['t']:.2f} s | violation {info['viol']:.1e} | " + report(Pl, z))
        print(out[-1], flush=True)
        np.save(JP.Path(__file__).parent / "results" / f"cyl2_{n}.npy", z)
    (JP.Path(__file__).parent / "results" / f"cyl2_{'_'.join(map(str, npcs))}.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1, 2, 4])
