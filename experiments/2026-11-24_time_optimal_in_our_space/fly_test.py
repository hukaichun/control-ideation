"""Fly the laps optimised in our space with the blended B-spline law (experiments/2026-11-24_blended_bspline_pieces),
to check that the planned lap can actually be flown (full quad model, mode B, measured acceleration, no disturbance).

Closed laps: the lap is unrolled three times and the pieces are cut on the middle copy, so every piece has a rise
and a fall (overlaps across the start line too). B-spline knots of the pieces every KNOT m (V changes fast on these
laps). Flight: 2.2 laps from the planned state at the start; lap time = time between the 2nd and 3rd crossing of
the start plane; gates checked on lap 2.

    cd experiments/2026-11-24_time_optimal_in_our_space
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python fly_test.py [variant] [V spacing] [tracks...]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "ours_100pct"
HQ = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
TRACKS = [int(a) for a in sys.argv[3:]] or list(range(10))


def _load(name, path):
    argv, sys.argv = sys.argv, sys.argv[:1]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


R = _load("opt", HERE / "run.py")
BL = _load("blend", EXP / "2026-11-24_blended_bspline_pieces" / "run.py")
BL.KNOT = 0.5
B = BL.B
fe = BL.fe


def dense(d, Ca, Cq, n=6000):
    nref, na, nq, L = len(d["Cref"]), len(Ca), len(Cq), float(d["L"])
    u = np.arange(n) / n
    al, be = R.rows(u, na, 0) @ Ca[:, 0], R.rows(u, na, 0) @ Ca[:, 1]
    rd = []
    for k in range(4):
        e = R.rows(u, nref, k) @ d["Cref"]
        from math import comb
        for j in range(k + 1):
            e = e + comb(k, j) * ((R.rows(u, na, j) @ Ca[:, 0])[:, None] * (R.rows(u, nref, k - j) @ d["CN1"])
                                  + (R.rows(u, na, j) @ Ca[:, 1])[:, None] * (R.rows(u, nref, k - j) @ d["CN2"]))
        rd.append(e)
    q = [R.rows(u, nq, k) @ Cq / L ** 2 for k in range(3)]
    V = -0.5 * np.sum(rd[1] ** 2, 1) * q[0]
    sq = np.sqrt(q[0])
    v0 = rd[1][0] * sq[0]
    a0 = rd[2][0] * q[0][0] + rd[1][0] * q[1][0] / 2
    j0 = sq[0] * (rd[3][0] * q[0][0] + 1.5 * rd[2][0] * q[1][0] + 0.5 * rd[1][0] * q[2][0])
    dt = 1 / n / sq
    return rd[0], V, (rd[0][0], v0, a0, j0), float(dt.sum())


def leg_gap(P, sep=6.0):
    """Smallest distance between two points of the lap that are more than `sep` m apart along it."""
    from scipy.spatial import cKDTree
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    Lp = s[-1] + np.linalg.norm(P[0] - P[-1])
    pr = np.array(list(cKDTree(P).query_pairs(6.0)))
    if not len(pr):
        return 12.0
    ds = np.abs(s[pr[:, 0]] - s[pr[:, 1]])
    ds = np.minimum(ds, Lp - ds)
    far = pr[ds > sep]
    return float(np.linalg.norm(P[far[:, 0]] - P[far[:, 1]], axis=1).min()) if len(far) else 12.0


def build_closed(P, V):
    # lateral window from the geometry: outer edge half the smallest gap between non-neighbouring parts of the lap
    # (at most 2 m), inner edge half of that (a piece of another leg passing nearby must get no weight)
    gap = leg_gap(P)
    BL.R_OUT = min(2.0, 0.5 * gap)
    BL.R_IN = 0.5 * BL.R_OUT
    L1 = len(P)
    P3, V3 = np.vstack([P, P, P]), np.concatenate([V, V, V])
    s3 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P3, axis=0), axis=1))]
    Llap = s3[L1]
    s_, cuts = BL.cut(P3[L1:2 * L1 + 1])
    cuts = cuts[:-1] + Llap                       # cut positions on the middle copy (start line included)
    pieces = []
    nc = len(cuts)
    for i in range(nc):
        c0, c1 = cuts[i], cuts[(i + 1) % nc] + (Llap if i == nc - 1 else 0)
        pieces.append(BL.Piece(P3, V3, s3, c0 - BL.D, c1 + BL.D, (c0 - BL.D, c0 + BL.D), (c1 - BL.D, c1 + BL.D)))
    return pieces


def main():
    quad = B.qs.Quad()
    out = []
    for k in TRACKS:
        d = np.load(HERE / "results" / f"track{k}{'_90' if '90' in VARIANT else ''}.npz", allow_pickle=True)
        key = f"{VARIANT}_{HQ}"
        if f"{key}_Ca" not in d:
            print(f"track {k}: no {key}"); continue
        P, V, (p0, v0, a0, j0), lap_plan = dense(d, d[f"{key}_Ca"], d[f"{key}_Cq"])
        pieces = build_closed(P, V)
        pol = BL.BlendPolicy(pieces)
        lg = fe.fly(quad, pol, p0, v0, a0, j0, 2.2 * lap_plan, mode="B")
        t0v = v0 / np.linalg.norm(v0)
        side = (lg[:, 4:7] - p0) @ t0v
        near = np.linalg.norm(lg[:, 4:7] - p0, axis=1) < 3.0
        cross = np.nonzero((side[:-1] < 0) & (side[1:] >= 0) & near[1:])[0]
        times = lg[cross + 1, 0]
        lap2 = float(times[1] - times[0]) if len(times) >= 2 else np.nan
        k0, k1 = (cross[0], cross[1]) if len(cross) >= 2 else (0, len(lg) - 1)
        F = lg[k0:k1 + 1]
        gates = d["gates"]
        gerr = max(np.linalg.norm(F[:, 4:7] - g, axis=1).min() for g in gates) if len(F) else np.nan
        lap_cpc = float(d["lap_cpc"])
        from scipy.spatial import cKDTree
        dev = cKDTree(P).query(F[:, 4:7])[0] if len(F) else np.array([np.nan])
        mix = np.array(pol.nmix)
        fb = float(np.mean(mix == 0)) if len(mix) else np.nan
        n_par = sum(6 + 2 + 4 + 3 * pc.ncoef for pc in pieces)
        line = (f"track {k}: CPC {lap_cpc:.3f} | planned {lap_plan:.3f} ({100 * (lap_plan / lap_cpc - 1):+.1f}%) | flown lap 2 "
                f"{lap2:.3f} ({100 * (lap2 / lap_cpc - 1):+.1f}%) | worst gate on lap 2 {gerr:.3f} m | off path max {dev.max():.3f} m, rms {np.sqrt(np.mean(dev ** 2)):.3f} m | "
                f"no-weight fallback {100 * fb:.1f}% of steps | "
                f"motor saturation {100 * F[:, 8].mean() if len(F) else np.nan:.1f}% | {len(pieces)} pieces, {n_par} numbers, "
                f"fit err {max(pc.fit_err for pc in pieces) * 1000:.1f} mm, V err {max(pc.V_err for pc in pieces):.2f} | "
                f"leg gap {leg_gap(P):.2f} m -> window {BL.R_IN:.2f}..{BL.R_OUT:.2f} m")
        print(line, flush=True); out.append(line)
    (HERE / "results" / f"fly_{VARIANT}_{HQ}.log").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
