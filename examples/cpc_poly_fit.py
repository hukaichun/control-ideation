"""CPC path -> piecewise polynomials (implicit graph form, G3 joints) -> speed only on it (poly_plan); compare lap with CPC.
    PYTHONPATH=. uv run --with casadi --with pyyaml --with scipy --with numba python cpc_poly_fit.py 60 7 <seeds>   (max turn per piece in deg, path degree)
"""
import sys, time, types, numpy as np
from numpy.polynomial import chebyshev as C
sys.path.insert(0, '.')
import poly_plan as pp, quad_sim as qs, race_compare as rc
quad = qs.Quad()
MAXANG = np.deg2rad(float(sys.argv[1])); MAXLEN = 6.0; pp.DEG_P = int(sys.argv[2]); seeds = [int(a) for a in sys.argv[3:]]

dense_cpc, pieces_from, spec_of = pp.dense_cpc, pp.pieces_from, pp.spec_of

for seed in seeds:
    d = np.load(f"../logs/cpc_traj_tol0.24_{seed}.npz"); gates = d["gates"]
    P, V, lap_cpc = dense_cpc(d)
    sl = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1); s = np.concatenate([[0], np.cumsum(sl)[:-1]]); L = sl.sum()
    spd = np.linalg.norm(V, axis=1)
    sig0 = lambda q: np.interp(q, s, spd, period=L)
    res = pieces_from(P, gates, MAXANG, MAXLEN)
    if res is None: print(f"seed {seed}: piece not monotone"); continue
    pcs, gate_at = res
    raw = pp.sample(spec_of(pcs, np.concatenate([np.concatenate([pc['f'], pc['g']]) for pc in pcs])), 300)
    err_raw = np.linalg.norm(raw[:, None] - P[None], axis=2).min(1).max()
    t0 = time.perf_counter(); zp, bp = pp.smooth_path(pcs, gate_at, gates, 0.24); t_s = time.perf_counter() - t0
    S = pp.sample(spec_of(pcs, zp), 300)
    err_s = np.linalg.norm(S[:, None] - P[None], axis=2).min(1).max()
    out = []
    nf = pp.DEG_P + 1
    zraw = np.concatenate([np.concatenate([pc['f'], pc['g']]) for pc in pcs])
    def b_of(zz):
        bb = np.zeros(3 * len(pcs))
        for i in range(len(pcs)):
            k = (i + 1) % len(pcs)
            d1 = lambda pc, tt, f, g: pc["e1"] + C.chebval(tt, C.chebder(f)) * 2 / pc["l"] * pc["e2"] + C.chebval(tt, C.chebder(g)) * 2 / pc["l"] * pc["e3"]
            bb[3*i] = np.linalg.norm(d1(pcs[i], 1.0, zz[i*2*nf:i*2*nf+nf], zz[i*2*nf+nf:(i+1)*2*nf])) / np.linalg.norm(d1(pcs[k], -1.0, zz[k*2*nf:k*2*nf+nf], zz[k*2*nf+nf:(k+1)*2*nf]))
        return bb
    motor = dict(m=quad.m, Ix=quad.J[0, 0], l=0.15, Tmax=quad.Tmax, Tmin=quad.Tmin, margin=1.0)
    for name, zz, bb, eta, wmax, dv, mg in (("raw, CPC-like limits", zraw, b_of(zraw), 1.0, 1e3, 8, 1.0),
                                           ("smooth, CPC-like limits", zp, bp, 1.0, 1e3, 8, 1.0),
                                           ("smooth, our limits m1.0", zp, bp, 0.9, 15.0, 8, 1.0),
                                           ("smooth, our limits m0.9", zp, bp, 0.9, 15.0, 8, 0.9)):
        motor["margin"] = mg
        pp.DEG_V = dv
        r = pp._plan(pcs, gate_at, gates, types.SimpleNamespace(length=L), P, s, motor, sig0, 0.25, 0.24, 1e-6, wmax, eta, 2.0,
                     1000, 1e-6, zz, bb, fix_path=True)
        out.append(f"{name}: {r['lap']:.3f} ({100*(r['lap']/lap_cpc-1):+.1f}%) {r['status'][:8]}")
    pp.DEG_V = 8
    print(f"seed {seed}: CPC {lap_cpc:.3f} | pieces {len(pcs)} fit err raw {err_raw:.3f} smooth {err_s:.3f} m, gates {rc.gate_dist(S, gates).max():.2f} | " + "\n   " + "\n   ".join(out), flush=True)
