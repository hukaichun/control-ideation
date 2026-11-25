"""Checks before optimising (user: check first).
1. Cutting on every track: pieces, worst chord tilt, path fit.
2. Piece geometry q_x..q_xxxx against finite differences of q(x).
3. Junction derivatives d^k(v^2)/ds^k (k = 0..3) against finite differences along the arc length of the piece curve.
    uv run --with casadi --with pyyaml --with scipy --with numpy python check.py
"""
import casadi as ca
import numpy as np

import run as R
import pieces as PC

C = R.C
rng = np.random.default_rng(0)
for k in range(10):
    P, _ = C.cpc_lap(k)
    try:
        pcs, junc, L = PC.build(P)
    except RuntimeError as e:
        print(f"1. track {k}: cutting failed ({e})")
        continue
    tilt = max(np.degrees(np.arctan(np.sqrt(pc.metric(np.linspace(pc.xa, pc.xb, 2000)) - 1).max())) for pc in pcs)
    print(f"1. track {k}: {len(pcs)} pieces, lap {L:.1f} m, path fit max {max(pc.fit_err for pc in pcs) * 1000:.2f} mm, "
          f"max tangent angle to the chord {tilt:.1f} deg")
    if k:
        continue
    pc = pcs[2]
    q = lambda x: pc.O + x * pc.e1 + pc.f(x) * pc.e2 + pc.g(x) * pc.e3
    x0 = rng.uniform(pc.xa + 0.3, pc.xb - 0.3, 5)
    err = 0
    for xx in x0:
        h = 2e-2
        X = np.array([q(xx + h * i) for i in range(-3, 4)])
        V = np.vander(h * np.arange(-3, 4), 7, increasing=True).T
        fd = [np.linalg.solve(V, np.eye(7)[d] * [1, 1, 2, 6, 24, 120, 720][d]) @ X for d in range(1, 5)]
        G = pc.geom(np.array([xx]))[:, 0]
        err = max(err, max(np.abs(G[3 * (d - 1):3 * d] - fd[d - 1]).max() / max(np.abs(fd[d - 1]).max(), 1) for d in range(1, 5)))
    print(f"2. piece geometry vs finite differences (track 0, piece 2): max relative error {err:.1e}")
    c = np.abs(rng.normal(20, 5, pc.nr))
    xx = float(x0[0])
    Dk = np.array(ca.Function("d", [], [R.junction_derivs(pc, xx, ca.DM(c))])()["o0"]).ravel()
    # v^2 along arc length: s(x) = int sqrt(m) dx
    xs = np.linspace(xx - 0.05, xx + 0.05, 20001)
    sm = np.sqrt(pc.metric(xs))
    s = np.r_[0, np.cumsum((sm[1:] + sm[:-1]) / 2 * np.diff(xs))]
    v2 = (pc.rrows(xs, 0) @ c) * pc.metric(xs)
    i0 = 10000
    s0 = s[i0]
    sel = np.abs(s - s0) < 0.02
    fit = np.polynomial.polynomial.polyfit(s[sel] - s0, v2[sel], 7)
    num = np.array([fit[d] * [1, 1, 2, 6][d] for d in range(4)])
    print(f"3. junction derivatives vs a fit along the arc length: {np.round(Dk, 4)} vs {np.round(num, 4)}")
