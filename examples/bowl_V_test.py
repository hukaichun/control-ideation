"""Experiment: V is a bowl around a point p* (V = -V0 + k |p - p*|^2) instead of the planned V, on the A* map-1 path.
Speed on the path sqrt(2 (V0 - k d^2)): 10 m/s where the path is closest to p*, 4 m/s where farthest. grad V also
points off the path: does the drone leave the path or flip to go down V?

    PYTHONPATH=. uv run --with casadi --with pyyaml --with scipy --with numba python bowl_V_test.py
"""
import casadi as ca
import numpy as np

import flat_el_policy as fe
import poly_plan as pp
import quad_sim as qs

quad = qs.Quad()
D = np.load("../logs/astar_demo_1.npz", allow_pickle=True)
spec = list(D["spec"])
S = pp.sample(spec, 300)
sl = np.linalg.norm(np.roll(S, -1, 0) - S, axis=1)
CASES = {"map centre": np.array([20.0, 15.0, 2.0]), "on the path": S[len(S) // 3].copy(),
         "3 m beside the path (outside)": S[len(S) // 3] + 3 * (S[len(S) // 3] - [20.0, 15.0, 2.0]) / np.linalg.norm(S[len(S) // 3] - [20.0, 15.0, 2.0]),
         "far outside the map": np.array([80.0, 15.0, 2.0]), "10 m above map centre": np.array([20.0, 15.0, 12.0]),
         "10 m below map centre": np.array([20.0, 15.0, -8.0])}
for name, ps in CASES.items():
    d2 = np.sum((S - ps) ** 2, axis=1)
    k = (100.0 - 16.0) / (2 * (d2.max() - d2.min()))
    V0 = (100.0 + 2 * k * d2.min()) / 2
    sig = np.sqrt(2 * (V0 - k * d2))
    lap = np.sum(sl / sig)
    gperp = []                                   # size of grad V across the path (what could pull it off)
    for i in range(0, len(S), 10):
        t_ = S[(i + 1) % len(S)] - S[i]; t_ /= np.linalg.norm(t_)
        g = 2 * k * (S[i] - ps)
        gperp.append(np.linalg.norm(g - (g @ t_) * t_))
    pol = fe.PolyELPolicy(spec, wp=16, wH=12, V_global=lambda p, ps=ps, k=k, V0=V0: -V0 + k * ca.sumsqr(p - ps))
    p0, v0, a0, j0 = pp.state_at(spec[0], 0.0)
    s0 = np.sqrt(2 * (V0 - k * np.sum((p0 - ps) ** 2)))
    v0 = v0 / np.linalg.norm(v0) * s0
    log = fe.fly(quad, pol, p0, v0, a0 * (s0 / np.linalg.norm(pp.state_at(spec[0], 0.0)[1])) ** 2, 0 * j0, 2.2 * lap, mode="B")
    ok = log[-1, 0] > 2.19 * lap
    L = log[log[:, 0] > 1.0]
    tilt = np.degrees(np.arccos(np.clip(L[:, 14], -1, 1)))
    print(f"{name:30s}: speed 4-10 m/s, k {k:.3f}, |grad V| across path up to {max(gperp):5.1f} m/s^2 | ok {ok} "
          f"| |phi| max {L[:, 1].max():.3f} mean {L[:, 1].mean():.4f} m | |H-H*| mean {np.abs(L[:, 2]).mean():.2f} "
          f"| tilt max {tilt.max():.0f} deg, upside down {100 * (L[:, 14] < 0).mean():.1f}% | sat {100 * L[:, 8].mean():.0f}% "
          f"max |w| {L[:, 7].max():.1f}", flush=True)
