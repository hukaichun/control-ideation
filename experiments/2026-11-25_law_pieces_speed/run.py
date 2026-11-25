"""Step 1 of planning in the law's representation (user, 2026-11-25): path pieces fixed, speed r_i(x) per piece.

  min  T = sum_i int_{core_i} dx / sqrt(r_i(x))          (Gauss-Legendre, 8 points per span between knots)
  s.t. per piece, at core samples: the four physical limits of ../2026-11-25_two_formulations (heading free,
       |w| <= 15 rad/s, collective <= max, rotors within [0, T_max] for every heading), with
       f = q_xx r + q_x r_x / 2 + g e3, j = sqrt(r) f_x, s = r f_xx + r_x f_x / 2  (q_version's chain, u = x, L = 1)
       at each junction (cut point, where the core of piece i ends and the core of piece i+1 starts):
       v^2 and its first three arc-length derivatives equal on both sides (speed, and with it a, j, snap along the path,
       continuous; my choice for the overlap question, to confirm with the user: the junction is matched, the rest of
       the overlap is only measured, not constrained)
       r coefficients >= 0
  start: 2 m/s.  IPOPT.
Checks: path pieces at the junctions (position, tangent, curvature mismatch), limits on a 40x finer core grid, speed
disagreement between neighbouring pieces over the whole overlap.

    cd experiments/2026-11-25_law_pieces_speed
    uv run --with casadi --with pyyaml --with scipy --with numpy python run.py [tracks...]
"""
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-11-25_two_formulations"))
import common as C            # noqa: E402
import q_version as QV        # noqa: E402
import pieces as PC           # noqa: E402

CHAIN = QV.chain(1.0)
import os  # noqa: E402
TAG = os.environ.get("TAG", "" if os.environ.get("OVL", "1") == "1" else "_free")
SP = 0.125                    # core sample spacing (m of x), plus every knot


OVL = __import__("os").environ.get("OVL", "1") == "1"    # overlaps in the problem (limits + speed agreement)
EPS = float(__import__("os").environ.get("EPS", 1.0))   # |v_i^2 - v_(i+1)^2| <= EPS m^2/s^2 in the overlaps (~0.05 m/s at 10 m/s)


JUNC = __import__("os").environ.get("JUNC", "1") == "1"  # junction: v^2 and 3 arc-length derivatives equal
LIMW = __import__("os").environ.get("LIMW", "1") == "1"  # limits on the whole piece (else cores only)


def core_x(pc, fac=1):
    a, b = (pc.xa, pc.xb) if OVL and LIMW else pc.xc
    n = max(2, int(np.ceil((b - a) / (SP / fac))))
    xs = a + (np.arange(n) + 0.5) * (b - a) / n
    kn = pc.knots[(pc.knots > a) & (pc.knots < b)]
    return np.unique(np.r_[a, xs, kn, b])


def parts(pc, x, c):
    Rr = [ca.mtimes(ca.DM(pc.rrows(x, d)), c).T for d in range(4)]
    return CHAIN.map(len(x))(ca.vertcat(ca.DM(pc.geom(x)), *Rr))


def junction_derivs(pc, x0, c):
    """v^2 and d/ds, d2/ds2, d3/ds3 of it at x0 (Taylor in t = x - x0, exact on the span; one-sided at a knot is fine:
    r is C3, f, g are C4)."""
    t = ca.SX.sym("t")
    rd = [ca.mtimes(ca.DM(pc.rrows(np.array([x0]), d)), c) for d in range(5)]
    r = sum(rd[d] * t ** d / [1, 1, 2, 6, 24][d] for d in range(5))
    fd = [float(pc.f.derivative(d)(x0)) if d else float(pc.f(x0)) for d in range(6)]
    gd = [float(pc.g.derivative(d)(x0)) if d else float(pc.g(x0)) for d in range(6)]
    fp = sum(fd[d + 1] * t ** d / [1, 1, 2, 6, 24][d] for d in range(5))
    gp = sum(gd[d + 1] * t ** d / [1, 1, 2, 6, 24][d] for d in range(5))
    m = 1 + fp ** 2 + gp ** 2
    E = r * m
    out = [E]
    for _ in range(3):
        E = ca.jacobian(E, t) / ca.sqrt(m)
        out.append(E)
    return ca.substitute(ca.vertcat(*out), t, 0)


def T_expr(pcs, cs, ng=8):
    x_, w_ = np.polynomial.legendre.leggauss(ng)
    T = 0
    for pc, c in zip(pcs, cs):
        a, b = pc.xc
        br = np.unique(np.r_[a, pc.tk[(pc.tk > a) & (pc.tk < b)], b])
        xq = np.concatenate([(br[i] + br[i + 1]) / 2 + (br[i + 1] - br[i]) / 2 * x_ for i in range(len(br) - 1)])
        wq = np.concatenate([(br[i + 1] - br[i]) / 2 * w_ for i in range(len(br) - 1)])
        T = T + ca.dot(ca.DM(wq), 1 / ca.sqrt(ca.mtimes(ca.DM(pc.rrows(xq, 0)), c)))
    return T


def solve(k):
    P, lap_cpc = C.cpc_lap(k)
    try:
        pcs, junc, Llap = PC.build(P)
    except RuntimeError as e:
        return f"track {k}: cutting failed: {e}"
    nr = [pc.nr for pc in pcs]
    z = ca.SX.sym("z", sum(nr))
    cs = ca.vertsplit(z, list(np.cumsum([0] + nr)))
    g, lbg, ubg = [], [], []
    for pc, c in zip(pcs, cs):
        x = core_x(pc)
        gi = QV.LIM.map(len(x))(*parts(pc, x, c))
        g.append(ca.vec(gi)); lbg.append(np.zeros(gi.numel())); ubg.append(np.full(gi.numel(), np.inf))
    jm = []
    for i, J in enumerate(junc):
        a, b = pcs[i], pcs[(i + 1) % len(pcs)]
        xa, xb = float(a.xof(J)[0]), float(b.xof(J)[0])
        if JUNC:
            e = junction_derivs(a, xa, cs[i]) - junction_derivs(b, xb, cs[(i + 1) % len(pcs)])
            g.append(e); lbg.append(np.zeros(4)); ubg.append(np.zeros(4))
        # path mismatch at the junction (fixed path, measured only)
        qa = a.O + xa * a.e1 + a.f(xa) * a.e2 + a.g(xa) * a.e3
        qb = b.O + xb * b.e1 + b.f(xb) * b.e2 + b.g(xb) * b.e3
        ta, tb = a.geom(np.array([xa]))[:3, 0], b.geom(np.array([xb]))[:3, 0]
        ka = np.linalg.norm(np.cross(ta, a.geom(np.array([xa]))[3:6, 0])) / np.linalg.norm(ta) ** 3
        kb = np.linalg.norm(np.cross(tb, b.geom(np.array([xb]))[3:6, 0])) / np.linalg.norm(tb) ** 3
        jm.append((np.linalg.norm(qa - qb), np.degrees(np.arccos(np.clip(ta @ tb / np.linalg.norm(ta) / np.linalg.norm(tb), -1, 1))), abs(ka - kb)))
    if OVL:   # the law blends both pieces in the overlap: same speed there (v^2 linear in r: linear rows)
        for i in range(len(pcs)):
            a, b = pcs[i], pcs[(i + 1) % len(pcs)]
            xa = np.arange(a.xc[1] - PC.D, a.xb, 0.1)
            pts = a.O + xa[:, None] * a.e1 + a.f(xa)[:, None] * a.e2 + a.g(xa)[:, None] * a.e3
            xb = b.xof(pts)
            ok = (xa >= a.xa) & (xb >= b.xa) & (xb <= b.xb)
            Ra = a.rrows(xa[ok], 0) * a.metric(xa[ok])[:, None]
            Rb = b.rrows(xb[ok], 0) * b.metric(xb[ok])[:, None]
            g.append(ca.mtimes(ca.DM(Ra), cs[i]) - ca.mtimes(ca.DM(Rb), cs[(i + 1) % len(pcs)]))
            lbg.append(np.full(ok.sum(), -EPS)); ubg.append(np.full(ok.sum(), EPS))
    T = T_expr(pcs, cs)
    z0 = np.concatenate([np.linalg.lstsq(pc.rrows(xx := np.linspace(pc.xa, pc.xb, 400), 0), 4.0 / pc.metric(xx), rcond=None)[0]
                         for pc in pcs])
    S = ca.nlpsol("s", "ipopt", {"x": z, "f": T, "g": ca.vertcat(*g)},
                  {"print_time": False, "ipopt": {"print_level": 0, "max_iter": 3000}})
    t0 = time.perf_counter()
    sol = S(x0=z0, lbx=0, lbg=np.concatenate(lbg), ubg=np.concatenate(ubg))
    t = time.perf_counter() - t0
    zs = np.array(sol["x"]).ravel()
    csn = np.split(zs, np.cumsum(nr)[:-1])
    st = S.stats()
    T0 = float(ca.Function("T", [z], [T])(z0))
    lines = [f"[overlaps {'in the problem (limits on the whole piece, |dv^2| <= ' + str(EPS) + ')' if OVL else 'free'}] "
             f"track {k}: lap {Llap:.1f} m, {len(pcs)} pieces, {sum(nr)} speed variables, path fit max "
             f"{max(pc.fit_err for pc in pcs) * 1000:.2f} mm, junction mismatch max: position {max(m[0] for m in jm) * 1000:.2f} mm, "
             f"tangent {max(m[1] for m in jm):.3f} deg, curvature {max(m[2] for m in jm):.3f} 1/m",
             f"   start {T0:.3f} s | {st['return_status']}, {st['iter_count']} it, {t:.1f} s | lap {float(sol['f']):.4f} s "
             f"({100 * (float(sol['f']) / lap_cpc - 1):+.1f}% vs CPC {lap_cpc:.3f})"]
    # physical check: cores on the solver samples and 40x finer
    for fac, lab in ((1, "solver samples"), (40, "40x finer")):
        W, TR, CC = [], [], []
        for pc, c in zip(pcs, csn):
            x = core_x(pc, fac)
            f, fu, fuu, r, ru = (np.array(v) for v in parts(pc, x, ca.DM(c)))
            w, tr, cc = C.physical(f, np.sqrt(r) * fu, r * fuu + ru * fu / 2)
            W.append(w); TR.append(tr); CC.append(cc)
        lines.append(C.report(f"   {lab}", np.concatenate(W), np.concatenate(TR), np.concatenate(CC)))
    # overlap: speed of piece i vs piece i+1 at the same physical points (whole overlap, 1 cm)
    dv = 0.0
    for i in range(len(pcs)):
        a, b = pcs[i], pcs[(i + 1) % len(pcs)]
        xa = np.linspace(a.xc[1] - PC.D, a.xb, 200)
        xa = xa[(xa >= a.xa) & (xa <= a.xb)]
        pts = a.O + xa[:, None] * a.e1 + a.f(xa)[:, None] * a.e2 + a.g(xa)[:, None] * a.e3
        xb = b.xof(pts)
        ok = (xb >= b.xa) & (xb <= b.xb)
        va = np.sqrt(np.maximum(a.rrows(xa[ok], 0) @ csn[i], 0) * a.metric(xa[ok]))
        vb = np.sqrt(np.maximum(b.rrows(xb[ok], 0) @ csn[(i + 1) % len(pcs)], 0) * b.metric(xb[ok]))
        dv = max(dv, np.abs(va - vb).max())
    lines.append(f"   overlaps: speed difference between neighbouring pieces max {dv:.3f} m/s")
    np.savez(HERE / "results" / f"track{k}{TAG}.npz", z=zs, nr=nr)
    return "\n".join(lines)


if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:]] or [0]:
        out = solve(k)
        print(out, flush=True)
        with open(HERE / "results" / f"run_{k}{TAG}.txt", "w") as fh:
            fh.write(out + "\n")
