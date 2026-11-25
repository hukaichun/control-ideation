"""Step 2 (user, 2026-11-25: "我想先放寬軌跡限制，讓軌跡也可以一起被最佳化"): path and speed of every law piece optimised
together. Pieces (frames O, e1, e2, e3, x ranges, cores) come from the initial lap (CPC's lap, test only) and stay fixed;
per piece the variables are the coefficients of f, g (quintic) and r (degree 4) on the same adaptive knots
(ADAPT, FTOL = 1 cm, KMAX = 2 m, decided by the initial fit).
  min  T = sum_i int_core dx / sqrt(r_i)
  s.t. the four physical limits (heading free) at samples on the whole piece (tails included)
       overlaps: the two pieces describe the same curve (lateral distance of piece i's points from piece i+1's graph
                 <= PTOL in e2 and in e3, every 0.1 m) and the same speed (|v_i^2 - v_(i+1)^2| <= EPS)
       gates: in the piece whose core contains the gate, at the gate's chord coordinate x_g the lateral distance
              (f(x_g) - y_g)^2 + (g(x_g) - z_g)^2 <= TOL^2 (TOL = 0.24 m, CPC's; distance at the same x, so >= the true
              nearest distance: conservative)
       r coefficients >= 0
  start: the fitted path and the speed solution of run.py with the same knots (results/track<k>_cm.npz).
My choices (to confirm with the user): PTOL = 1 cm (the user's fit tolerance), gate distance at the same x;
the speed agreement compares v^2 at the start chord coordinate of piece b (the move along the path is small).

    cd experiments/2026-11-25_law_pieces_speed
    ADAPT=1 FTOL=0.01 uv run --with casadi --with pyyaml --with scipy --with numpy python run_joint.py [tracks...]
"""
import os
import sys
import time

import casadi as ca
import numpy as np
from scipy.interpolate import BSpline

import pieces as PC
import run as R

C = R.C
TOL = 0.24
PTOL = float(os.environ.get("PTOL", 0.01))
EPS = float(os.environ.get("EPS", 1.0))
START = os.environ.get("START", "_cm")


def brows(t, k, x, d):
    n = len(t) - k - 1
    b = BSpline(t, np.eye(n), k)
    return (b.derivative(d) if d else b)(np.clip(x, t[0], t[-1]))


class JPiece:
    """symbolic view of a piece: f, g, r coefficients as variables."""
    def __init__(self, pc):
        self.pc = pc
        self.tf = pc.f.t
        self.nf = len(self.tf) - 6
        self.n = 2 * self.nf + pc.nr

    def split(self, z):
        return z[:self.nf], z[self.nf:2 * self.nf], z[2 * self.nf:]

    def z0(self, r0):
        return np.r_[self.pc.f.c[:self.nf], self.pc.g.c[:self.nf], r0]

    def fg(self, x, cf, cg, d):
        return ca.mtimes(ca.DM(brows(self.tf, 5, x, d)), cf), ca.mtimes(ca.DM(brows(self.tf, 5, x, d)), cg)

    def point(self, x, cf, cg):
        f0, g0 = self.fg(x, cf, cg, 0)
        pc = self.pc
        return [ca.DM(pc.O + xx * pc.e1).T + f0[i] * ca.DM(pc.e2).T + g0[i] * ca.DM(pc.e3).T for i, xx in enumerate(x)]

    def parts(self, x, z):
        cf, cg, cr = self.split(z)
        pc = self.pc
        geo = []
        for d in range(1, 5):
            fd, gd = self.fg(x, cf, cg, d)
            geo.append(ca.DM(np.outer((d == 1) * pc.e1, np.ones(len(x)))) + ca.mtimes(ca.DM(pc.e2), fd.T) + ca.mtimes(ca.DM(pc.e3), gd.T))
        Rr = [ca.mtimes(ca.DM(pc.rrows(x, d)), cr).T for d in range(4)]
        return R.CHAIN.map(len(x))(ca.vertcat(*geo, *Rr))

    def v2(self, x, z):
        cf, cg, cr = self.split(z)
        f1, g1 = self.fg(x, cf, cg, 1)
        return ca.mtimes(ca.DM(self.pc.rrows(x, 0)), cr) * (1 + f1 ** 2 + g1 ** 2)


def solve(k):
    P, lap_cpc = C.cpc_lap(k)
    pcs, junc, Llap = PC.build(P)
    jp = [JPiece(pc) for pc in pcs]
    ns = [p.n for p in jp]
    tb = time.perf_counter()
    z = ca.MX.sym("z", sum(ns))      # MX: the per-sample functions stay function calls (SX inlined everything and the
    # exact Hessian grew 38x per piece: 15 k -> 570 k nodes; user: "也太久，我覺得你又把東西寫錯了")
    zs_ = ca.vertsplit(z, list(np.cumsum([0] + ns)))
    g, lb, ub = [], [], []
    for p, zi in zip(jp, zs_):                                     # limits on the whole piece
        x = R.core_x(p.pc)
        gi = ca.vec(R.QV.LIM.map(len(x))(*p.parts(x, zi)))
        g.append(gi); lb.append(np.zeros(gi.numel())); ub.append(np.full(gi.numel(), np.inf))
    for i in range(len(jp)):                                       # overlaps: same curve, same speed
        a, b = jp[i], jp[(i + 1) % len(jp)]
        za, zb = zs_[i], zs_[(i + 1) % len(jp)]
        xa = np.arange(a.pc.xc[1] - PC.D, a.pc.xb, 0.1)
        xb0 = b.pc.xof(a.pc.O + np.outer(xa, a.pc.e1) + np.outer(a.pc.f(xa), a.pc.e2) + np.outer(a.pc.g(xa), a.pc.e3))
        ok = (xb0 >= b.pc.xa + 0.05) & (xb0 <= b.pc.xb - 0.05)
        xa = xa[ok]
        cfa, cga, _ = a.split(za)
        cfb, cgb, _ = b.split(zb)
        # piece b evaluated at the chord coordinate of the (moving) point of piece a, x_b = e1_b.(q_a - O_b): Taylor
        # expansion of b's quintic span around the start value (exact while x_b stays in that span)
        qa = a.point(xa, cfa, cga)
        xbf = xb0[ok]
        fact = [1, 1, 2, 6, 24, 120]
        for j in range(len(xa)):
            d = qa[j].T - ca.DM(b.pc.O)
            dx = ca.dot(ca.DM(b.pc.e1), d) - xbf[j]
            fbx = sum(b.fg(np.array([xbf[j]]), cfb, cgb, m)[0] * dx ** m / fact[m] for m in range(6))
            gbx = sum(b.fg(np.array([xbf[j]]), cfb, cgb, m)[1] * dx ** m / fact[m] for m in range(6))
            g.append(ca.vertcat(ca.dot(ca.DM(b.pc.e2), d) - fbx, ca.dot(ca.DM(b.pc.e3), d) - gbx))
            lb.append([-PTOL, -PTOL]); ub.append([PTOL, PTOL])
        dv = a.v2(xa, za) - b.v2(xbf, zb)
        g.append(dv); lb.append(np.full(len(xa), -EPS)); ub.append(np.full(len(xa), EPS))
    gate_x = []
    gates = np.load(C.ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")["gates"]
    for gt in gates:                                               # gates
        best = None
        for i, p in enumerate(jp):                                 # the piece closest to the gate whose range holds it
            xg = float(p.pc.xof(gt)[0])
            if p.pc.xa + 0.05 <= xg <= p.pc.xb - 0.05:
                dd = np.hypot((gt - p.pc.O) @ p.pc.e2 - p.pc.f(xg), (gt - p.pc.O) @ p.pc.e3 - p.pc.g(xg))
                if best is None or dd < best[0]:
                    best = (dd, i, xg)
        _, i, xg = best
        # some point of the path within TOL of the gate: the chord coordinate of that point is a variable (its own
        # Taylor expansion of the quintic around the start value), |q(x_g) - gate|^2 <= TOL^2
        # (first version measured at the gate's own x: larger than the true distance, the start already broke it)
        xv = ca.MX.sym(f"xg{len(gate_x)}")
        gate_x.append((xv, xg, i))
        cf, cg, _ = jp[i].split(zs_[i])
        fact = [1, 1, 2, 6, 24, 120]
        dxg = xv - xg
        f0 = sum(jp[i].fg(np.array([xg]), cf, cg, m)[0] * dxg ** m / fact[m] for m in range(6))
        g0 = sum(jp[i].fg(np.array([xg]), cf, cg, m)[1] * dxg ** m / fact[m] for m in range(6))
        pc = jp[i].pc
        dvec = ca.DM(pc.O - gt) + xv * ca.DM(pc.e1) + f0 * ca.DM(pc.e2) + g0 * ca.DM(pc.e3)
        g.append(ca.dot(dvec, dvec)); lb.append([0.0]); ub.append([TOL ** 2])
    T = R.T_expr(pcs, [p.split(zi)[2] for p, zi in zip(jp, zs_)])
    st0 = np.load(R.HERE / "results" / f"track{k}{START}.npz")
    r0 = np.split(st0["z"], np.cumsum(st0["nr"])[:-1])
    z0 = np.concatenate([p.z0(r) for p, r in zip(jp, r0)])
    lbx = np.concatenate([np.r_[np.full(2 * p.nf, -np.inf), np.zeros(p.pc.nr)] for p in jp])
    xg0 = []
    for _, xg, i in gate_x:                        # start: the nearest point of the fitted path to the gate
        pc = jp[i].pc
        xs = np.linspace(max(pc.xa, xg - 1), min(pc.xb, xg + 1), 4001)
        pts = pc.O + np.outer(xs, pc.e1) + np.outer(pc.f(xs), pc.e2) + np.outer(pc.g(xs), pc.e3)
        xg0.append(xs[np.argmin(np.linalg.norm(pts - gates[len(xg0)], axis=1))])
    z0 = np.r_[z0, xg0]
    lbx = np.r_[lbx, [jp[i].pc.xa for _, _, i in gate_x]]
    ubx = np.r_[np.full(sum(ns), np.inf), [jp[i].pc.xb for _, _, i in gate_x]]
    G = ca.vertcat(*g)
    t_build = time.perf_counter() - tb
    zx = ca.vertcat(z, *[v for v, _, _ in gate_x])
    S = ca.nlpsol("s", "ipopt", {"x": zx, "f": T, "g": G}, {"print_time": False, "ipopt": {"print_level": int(os.environ.get("PL", 0)), "max_iter": 3000, "print_frequency_iter": 10}})
    t_nlp = time.perf_counter() - tb - t_build
    print(f"   build {t_build:.1f} s, nlpsol {t_nlp:.1f} s", flush=True)
    LB, UB = np.concatenate(lb), np.concatenate(ub)
    g0v = np.array(ca.Function("g", [zx], [G])(z0)).ravel()
    viol0 = max(0.0, (LB - g0v).max(), (g0v - UB).max())
    if os.environ.get("DIAG"):
        kinds = []
        for gi, l_ in zip(g, lb):
            kinds += [len(l_)]
        off = np.cumsum([0] + kinds)
        names = ["limits"] * len(jp) + ["overlap"] * (len(g) - len(jp) - len(gates)) + ["gate"] * len(gates)
        for nm in ("limits", "overlap", "gate"):
            idx = np.concatenate([np.arange(off[i], off[i + 1]) for i, n_ in enumerate(names) if n_ == nm])
            v = np.maximum(LB[idx] - g0v[idx], g0v[idx] - UB[idx]).max()
            print(f"   start violation, {nm}: {v:.2e}", flush=True)
        return "diag" 
    t0 = time.perf_counter()
    sol = S(x0=z0, lbx=lbx, ubx=ubx, lbg=LB, ubg=UB)
    t = time.perf_counter() - t0
    zs = np.array(sol["x"]).ravel()[:sum(ns)]
    st = S.stats()
    T0 = float(ca.Function("T", [z], [T])(z0[:sum(ns)]))
    gs = np.array(ca.Function("g", [zx], [G])(np.array(sol["x"]).ravel())).ravel()
    viol = max(0.0, (LB - gs).max(), (gs - UB).max())
    # how far the path moved
    mv = 0.0
    for p, zi in zip(jp, np.split(zs, np.cumsum(ns)[:-1])):
        x = np.linspace(p.pc.xa, p.pc.xb, 200)
        cf, cg, _ = zi[:p.nf], zi[p.nf:2 * p.nf], zi[2 * p.nf:]
        mv = max(mv, np.hypot(brows(p.tf, 5, x, 0) @ cf - p.pc.f(x), brows(p.tf, 5, x, 0) @ cg - p.pc.g(x)).max())
    lines = [f"[joint] track {k}: {len(pcs)} pieces, {sum(ns)} variables ({sum(p.nr for p in jp)} speed), {G.numel()} constraints | "
             f"build {t_build:.1f} s + derivatives {t_nlp:.1f} s | start {T0:.4f} s (worst start violation {viol0:.1e}) | {st['return_status']}, {st['iter_count']} it, {t:.1f} s | "
             f"lap {float(sol['f']):.4f} s ({100 * (float(sol['f']) / lap_cpc - 1):+.1f}% vs CPC {lap_cpc:.3f}) | worst violation {viol:.1e} | "
             f"path moved up to {mv:.3f} m"]
    for fac, lab in ((1, "solver samples"), (40, "40x finer")):
        W, TR, CC = [], [], []
        for p, zi in zip(jp, np.split(zs, np.cumsum(ns)[:-1])):
            x = R.core_x(p.pc, fac)
            f, fu, fuu, r, ru = (np.array(v) for v in p.parts(x, ca.DM(zi)))
            w, tr, cc = C.physical(f, np.sqrt(r) * fu, r * fuu + ru * fu / 2)
            W.append(w); TR.append(tr); CC.append(cc)
        lines.append(C.report(f"   {lab}", np.concatenate(W), np.concatenate(TR), np.concatenate(CC)))
    np.savez(R.HERE / "results" / f"joint{k}.npz", z=zs, ns=ns, nf=[p.nf for p in jp], nr=[p.pc.nr for p in jp])
    return "\n".join(lines)


if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:]] or [0]:
        out = solve(k)
        print(out, flush=True)
        (R.HERE / "results" / f"joint_{k}.txt").write_text(out + "\n")
