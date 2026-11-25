"""Usability (user, 2026-11-25: "最佳化效率以及可用性"): fly the planned pieces as the blended control law, with no
second fit. Each planned piece IS a law piece: phi_i = (e2.q - f_i(x), e3.q - g_i(x)), V_i = h_i(x) =
-1/2 r_i(x) (1 + f_i'^2 + g_i'^2) (exact, from the planned r_i), x = e1.(p - O_i); blending as
../2026-11-24_blended_bspline_pieces (rise / fall over cut -+ D, lateral window from the lap's leg gap), E-L law with the
same gains (WP 16, WH 12), full quad model, mode B, measured acceleration, no disturbance, heading free (yaw rate damped).
Start on the planned state at the first cut; 2.2 laps; lap time = time between the 2nd and 3rd crossing of the start
plane; gates and path deviation on that lap.

    cd experiments/2026-11-25_law_pieces_speed
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python fly.py [tracks...]
"""
import importlib.util
import sys
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


BL = _load("blend", EXP / "2026-11-24_blended_bspline_pieces" / "run.py")
B, fe = BL.B, BL.fe
import run as R               # noqa: E402
import pieces as PC           # noqa: E402
from scipy.interpolate import BSpline   # noqa: E402


def leg_gap(P, sep=6.0):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    Lp = s[-1] + np.linalg.norm(P[0] - P[-1])
    pr = np.array(list(cKDTree(P).query_pairs(6.0)))
    if not len(pr):
        return 12.0
    ds = np.abs(s[pr[:, 0]] - s[pr[:, 1]])
    ds = np.minimum(ds, Lp - ds)
    far = pr[ds > sep]
    return float(np.linalg.norm(P[far[:, 0]] - P[far[:, 1]], axis=1).min()) if len(far) else 12.0


class LawPiece:
    """BL.Piece interface (O, e1..e3, xa, xb, f, g, rise, fall, weight, F) built from a planned piece."""
    def __init__(self, pc, c, cut0, cut1):
        self.O, self.e1, self.e2, self.e3 = pc.O, pc.e1, pc.e2, pc.e3
        self.xa, self.xb, self.f, self.g = pc.xa, pc.xb, pc.f, pc.g
        xs = lambda sv: float(np.interp(sv, pc.s_, pc.x_))
        self.rise = (xs(cut0 - PC.D), xs(cut0 + PC.D))
        self.fall = (xs(cut1 - PC.D), xs(cut1 + PC.D))
        self.r = BSpline(pc.tk, c, PC.DEGR)
        p, v, a_, j = (ca.SX.sym(nm, 3) for nm in "pvaj")
        qq = p - ca.DM(self.O)
        xx = ca.dot(ca.DM(self.e1), qq)
        phi = ca.vertcat(ca.dot(ca.DM(self.e2), qq) - BL.spline_expr(self.f, xx), ca.dot(ca.DM(self.e3), qq) - BL.spline_expr(self.g, xx))
        h = -0.5 * BL.spline_expr(self.r, xx) * (1 + BL.spline_expr(self.f.derivative(), xx) ** 2 + BL.spline_expr(self.g.derivative(), xx) ** 2)
        w, info = fe._el_w(p, v, a_, j, phi, h, 0.0, BL.WP, BL.WH)
        self.F = ca.Function("w", [p, v, a_, j], [w, info])
        self.hfun = ca.Function("h", [p], [h])

    weight = BL.Piece.weight


def build_law(k, tag=None):
    tag = __import__("os").environ.get("TAG", "") if tag is None else tag
    P, lap_cpc = R.C.cpc_lap(k)
    pcs, junc, Llap = PC.build(P)
    d = np.load(HERE / "results" / f"track{k}{tag}.npz")
    cs = np.split(d["z"], np.cumsum(d["nr"])[:-1])
    gap = leg_gap(P)
    BL.R_OUT = min(2.0, 0.5 * gap)
    BL.R_IN = 0.5 * BL.R_OUT
    law = []
    for i, (pc, c) in enumerate(zip(pcs, cs)):
        cut0, cut1 = pc.s_[0] + PC.D, pc.s_[-1] - PC.D            # pieces were built from cut -+ D
        law.append(LawPiece(pc, c, cut0, cut1))
    return P, lap_cpc, pcs, cs, law, gap


def start_state(pc, c):
    x = np.array([pc.xc[0]])
    f, fu, fuu, r, ru = (np.array(v).ravel() for v in R.parts(pc, x, ca.DM(c)))
    G = pc.geom(x)[:, 0]
    p = pc.O + x[0] * pc.e1 + pc.f(x[0]) * pc.e2 + pc.g(x[0]) * pc.e3
    return p, G[0:3] * np.sqrt(r[0]), f - [0, 0, R.C.G], np.sqrt(r[0]) * fu


def main(tracks):
    quad = B.qs.Quad()
    out = []
    for k in tracks:
        P, lap_cpc, pcs, cs, law, gap = build_law(k)
        tag = __import__("os").environ.get("TAG", "")
        lap_plan = float(open(HERE / "results" / f"run_{k}{tag}.txt").read().split("lap ")[2].split(" s")[0])
        p0, v0, a0, j0 = start_state(pcs[0], cs[0])
        pol = BL.BlendPolicy(law)
        lg = fe.fly(quad, pol, p0, v0, a0, j0, 2.2 * lap_plan, mode="B")
        t0v = v0 / np.linalg.norm(v0)
        side = (lg[:, 4:7] - p0) @ t0v
        near = np.linalg.norm(lg[:, 4:7] - p0, axis=1) < 3.0
        cross = np.nonzero((side[:-1] < 0) & (side[1:] >= 0) & near[1:])[0]
        lap2 = float(lg[cross[1] + 1, 0] - lg[cross[0] + 1, 0]) if len(cross) >= 2 else np.nan
        F = lg[cross[0]:cross[1] + 1] if len(cross) >= 2 else lg
        gates = np.load(R.C.ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")["gates"]
        gerr = max(np.linalg.norm(F[:, 4:7] - g, axis=1).min() for g in gates)
        dev = cKDTree(P).query(F[:, 4:7])[0]
        mix = np.array(pol.nmix)
        line = (f"track {k}: planned {lap_plan:.3f} s | flown {'lap 2 ' + format(lap2, '.3f') + ' s (' + format(100 * (lap2 / lap_plan - 1), '+.1f') + '% vs plan)' if np.isfinite(lap2) else 'DID NOT COMPLETE (' + format(lg[-1, 0], '.2f') + ' s flown)'}"
                f" | worst gate {gerr:.3f} m (CPC tol 0.24) | off path max {dev.max():.3f} m, rms {np.sqrt(np.mean(dev ** 2)):.3f} m"
                f" | motor saturation {100 * F[:, 8].mean():.1f}% | no-weight fallback {100 * np.mean(mix == 0):.1f}% | "
                f"{len(law)} pieces, lateral window {BL.R_IN:.2f}..{BL.R_OUT:.2f} m (leg gap {gap:.2f})")
        print(line, flush=True)
        out.append(line)
        np.save(HERE / "results" / f"fly_{k}{tag}.npy", lg)
    (HERE / "results" / f"fly_{'_'.join(map(str, tracks))}{__import__('os').environ.get('TAG', '')}.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [0])
