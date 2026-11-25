"""Parabola toy (user, 2026-11-25: "你找一個拋物線型的路線，然後截成兩段，跑我們的最佳化看看").

Path: horizontal U-turn y = A x^2, z = Z0, x in [-X, X] (vertex radius 1 / (2A)); start and end speed V0.
Our optimisation (run.py): speed r(x) = (dx/dt)^2 per piece, degree-4 B-spline (knots KR), the four physical limits
(heading free) at samples, objective sum of core integrals of dx / sqrt(r).
Variants:
  1 piece          : the whole parabola as one piece (frame = world x axis: the parabola is exact, quintic spline of a
                     quadratic)
  2 pieces, x frame: cut at the vertex, cores [-X, 0] and [0, X], extended by D = 1 m; both frames = world x axis, so the
                     path is still exact; junction (v^2 and 3 arc-length derivatives equal), overlap speeds agree
                     (|dv^2| <= EPS), limits on the whole piece (tails included) -- exactly as run.py
  2 pieces, chord  : as above but each piece in its own chord frame (as the law cuts): rotated parabola fitted
  2 pieces, x frame, core limits only: limits on the cores only (tails not checked)
    uv run --with casadi --with pyyaml --with scipy --with numpy python toy_parabola.py
"""
import time

import casadi as ca
import numpy as np

import pieces as PC
import run as R

C = R.C
A, X, Z0, V0 = 0.2, 6.0, 2.0, 3.0
EPS = 1.0


def path(n=20001):
    x = np.linspace(-X, X, n)
    P = np.c_[x, A * x ** 2, np.full(n, Z0)]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return P, s


def make_pieces(npieces, frame):
    P, s = path()
    xf = (np.array([0.0, 0, Z0]), np.array([1.0, 0, 0])) if frame == "x" else None
    cuts = [s[np.argmin(np.abs(P[:, 0] - xc))] for xc in np.linspace(-X, X, npieces + 1)]   # equal steps in world x
    cuts[0], cuts[-1] = 0.0, s[-1]
    return [PC.Piece(P, s, max(0.0, cuts[i] - (PC.D if i else 0)), min(s[-1], cuts[i + 1] + (PC.D if i < npieces - 1 else 0)),
                     (cuts[i], cuts[i + 1]), frame=xf) for i in range(npieces)], s


def solve(pcs, limw=True, eps=EPS, junc=True):
    nr = [pc.nr for pc in pcs]
    z = ca.SX.sym("z", sum(nr))
    cs = ca.vertsplit(z, list(np.cumsum([0] + nr)))
    g, lb, ub = [], [], []
    for pc, c in zip(pcs, cs):
        a, b = (pc.xa, pc.xb) if limw else pc.xc
        n = int(np.ceil((b - a) / R.SP))
        x = np.unique(np.r_[a + (np.arange(n) + 0.5) * (b - a) / n, pc.knots[(pc.knots > a) & (pc.knots < b)], a, b])
        gi = ca.vec(R.QV.LIM.map(len(x))(*R.parts(pc, x, c)))
        g.append(gi); lb.append(np.zeros(gi.numel())); ub.append(np.full(gi.numel(), np.inf))
    # ends: speed V0 (v^2 = r m)
    for pc, c, xe in ((pcs[0], cs[0], pcs[0].xa), (pcs[-1], cs[-1], pcs[-1].xb)):
        g.append(ca.mtimes(ca.DM(pc.rrows(np.array([xe]), 0) * pc.metric(np.array([xe]))[:, None]), c) - V0 ** 2)
        lb.append([0.0]); ub.append([0.0])
    for ip in range(len(pcs) - 1):
        a, b = pcs[ip], pcs[ip + 1]
        ca_, cb_ = cs[ip], cs[ip + 1]
        J = a.O + a.xc[1] * a.e1 + a.f(a.xc[1]) * a.e2 + a.g(a.xc[1]) * a.e3
        e = R.junction_derivs(a, a.xc[1], ca_) - R.junction_derivs(b, float(b.xof(J)[0]), cb_)
        if junc:
            g.append(e); lb.append(np.zeros(4)); ub.append(np.zeros(4))
        xa = np.arange(a.xc[1] - PC.D, a.xb, 0.1)
        pts = a.O + xa[:, None] * a.e1 + a.f(xa)[:, None] * a.e2 + a.g(xa)[:, None] * a.e3
        xb = b.xof(pts)
        ok = (xb >= b.xa) & (xb <= b.xb)
        g.append(ca.mtimes(ca.DM(a.rrows(xa[ok], 0) * a.metric(xa[ok])[:, None]), ca_)
                 - ca.mtimes(ca.DM(b.rrows(xb[ok], 0) * b.metric(xb[ok])[:, None]), cb_))
        lb.append(np.full(ok.sum(), -eps)); ub.append(np.full(ok.sum(), eps))
    T = R.T_expr(pcs, cs)
    z0 = np.concatenate([np.linalg.lstsq(pc.rrows(xx := np.linspace(pc.xa, pc.xb, 400), 0), V0 ** 2 / pc.metric(xx), rcond=None)[0]
                         for pc in pcs])
    S = ca.nlpsol("s", "ipopt", {"x": z, "f": T, "g": ca.vertcat(*g)}, {"print_time": False, "ipopt": {"print_level": int(__import__("os").environ.get("PL", 0)), "max_iter": 3000, "print_frequency_iter": 250}})
    t0 = time.perf_counter()
    sol = S(x0=z0, lbx=0, lbg=np.concatenate(lb), ubg=np.concatenate(ub))
    zs = np.array(sol["x"]).ravel()
    return float(sol["f"]), S.stats()["return_status"], S.stats()["iter_count"], time.perf_counter() - t0, np.split(zs, np.cumsum(nr)[:-1])


def active(pcs, cs):
    """time-weighted fractions along the cores: |w| at limit, torque at limit, collective at max, none."""
    W, TR, CC, DT = [], [], [], []
    for pc, c in zip(pcs, cs):
        x = np.linspace(*pc.xc, 6000)
        f, fu, fuu, r, ru = (np.array(v) for v in R.parts(pc, x, ca.DM(c)))
        w, tr, cc = C.physical(f, np.sqrt(r) * fu, r * fuu + ru * fu / 2)
        W.append(w); TR.append(tr); CC.append(cc); DT.append(np.gradient(x) / np.sqrt(r.ravel()))
    w, tr, cc, dt = (np.concatenate(a) for a in (W, TR, CC, DT))
    rate, torque, coll = w >= 0.99 * C.WMAX, tr >= 0.99, cc >= 0.99 * 4 * C.TMAX / C.MASS
    pct = lambda m: 100 * dt[m].sum() / dt.sum()
    return (f"|w| {pct(rate):3.0f}%, torque {pct(torque):3.0f}%, collective {pct(coll):3.0f}%, none {pct(~(rate | torque | coll)):3.0f}% "
            f"| max over the cores: |w| {w.max():.2f}, torque {tr.max():.3f}")


def main_knots():
    """How close can the speed spline get to the optimum? (user: "剩下的問題就是速度 spline 是不是能貼著最佳的速度")
    Pieces 1, 2, 4 (chord frames, as the law cuts; 1 piece: x frame), no junction-derivative equality, speed knots
    2 .. 0.125 m."""
    out = ["speed-spline resolution on the parabola (no junction-derivative equality; overlap speeds agree; limits on whole pieces)"]
    for kr in (2.0, 1.0, 0.5, 0.25, 0.125):
        PC.KR = kr
        row = []
        for npc in (1, 2, 4):
            pcs, s = make_pieces(npc, "x" if npc == 1 else "chord")
            T, st, it, t, cs = solve(pcs, junc=False)
            row.append(f"{npc} piece{'s' if npc > 1 else ''}: {T:.4f} s ({st[:5]}, {it} it, {t:.1f} s)")
        out.append(f"  knots {kr:5.3f} m | " + " | ".join(row))
        print(out[-1], flush=True)
    (R.HERE / "results" / "toy_parabola_knots.txt").write_text("\n".join(out) + "\n")


def main():
    out = [f"parabola y = {A} x^2, x in [-{X}, {X}] (vertex radius {1 / (2 * A):.1f} m), start / end speed {V0} m/s, "
           f"speed knots {PC.KR} m"]
    for lab, npc, fr, limw in (("1 piece", 1, "x", True), ("2 pieces, x frame", 2, "x", True),
                               ("2 pieces, chord frames", 2, "chord", True), ("2 pieces, x frame, core limits only", 2, "x", False)):
        pcs, s = make_pieces(npc, fr)
        T, st, it, t, cs = solve(pcs, limw)
        out.append(f"  {lab:36s}: {st}, {it} it, {t:.1f} s | T {T:.4f} s | path fit {max(pc.fit_err for pc in pcs) * 1000:.3f} mm | "
                   + active(pcs, cs))
        print(out[-1], flush=True)
    (R.HERE / "results" / f"toy_parabola_kr{PC.KR}.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    import sys
    main_knots() if "knots" in sys.argv else main()
