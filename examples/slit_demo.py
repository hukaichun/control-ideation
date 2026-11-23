"""Knife-edge through a vertical slit (a train window narrower than the drone): at the slit the drone's disc must lie in
the wall's plane, i.e. body z (= direction of a + g) along the wall normal's perpendicular y: a = (0, c, -g).

Reference (one closed lap, designed in time): a horizontal circle of radius R at speed vh (centripetal c = vh^2 / R
towards +y at the slit) with a small hop at the slit whose top has z'' = -g (z = z0 + A exp(-t^2 / 2 s^2), A = g s^2, s = SIG_HOP / vh).
So at the slit: velocity along x (through the wall), a + g = c y -> disc in the x-z plane.
Pipeline as everywhere else: dense points -> G3 piecewise polynomials (path) -> V = -|v|^2 / 2 fitted per piece
(speed taken from the reference, not re-optimised) -> the same law (PolyELPolicy) -> flown 3 laps.

    PYTHONPATH=. uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python slit_demo.py [vh ...]
"""
import sys

import numpy as np
from numpy.polynomial import chebyshev as C

import flat_el_policy as fe
import poly_plan as pp
import quad_sim as qs

G = 9.81
R, Z0, SIG_HOP = 6.5, 3.0, 1.2         # circle radius, height, hop width along the path (m)
MAXLEN = 4.0                            # piece length; the slit sits in the middle of a piece
R_DRONE, THICK, WALL = 0.25, 0.08, 0.10  # disc radius incl. props, drone thickness, wall thickness
quad = qs.Quad()
motor = dict(m=quad.m, Ix=quad.J[0, 0], l=0.15, Tmax=quad.Tmax, Tmin=quad.Tmin, margin=1.0)


def reference(vh, n=3000):
    w = vh / R
    T = 2 * np.pi / w
    S_HOP = SIG_HOP / vh
    t = (np.arange(n) / n - MAXLEN / 2 / (2 * np.pi * R)) * T     # start half a piece before the slit
    tau = (t + T / 2) % T - T / 2                       # time from the slit, in (-T/2, T/2]
    A = G * S_HOP ** 2
    e = A * np.exp(-tau ** 2 / (2 * S_HOP ** 2))
    P = np.stack([R * np.sin(w * t), R - R * np.cos(w * t), Z0 + e], 1)
    V = np.stack([vh * np.cos(w * t), vh * np.sin(w * t), -tau / S_HOP ** 2 * e], 1)
    Acc = np.stack([-vh * w * np.sin(w * t), vh * w * np.cos(w * t), (tau ** 2 / S_HOP ** 4 - 1 / S_HOP ** 2) * e], 1)
    J = np.stack([-vh * w * w * np.cos(w * t), -vh * w * w * np.sin(w * t),
                  (3 * tau / S_HOP ** 4 - tau ** 3 / S_HOP ** 6) * e], 1)
    return P, V, Acc, J, T


def build(vh):
    P, V, Acc, J, T = reference(vh)
    slit = np.array([0.0, 0.0, Z0 + G * (SIG_HOP / vh) ** 2])
    pcs, gate_at = pp.pieces_from(P, [slit], maxlen=MAXLEN)
    zp, _ = pp.smooth_path(pcs, gate_at, [slit], 0.01)
    spec = pp.spec_of(pcs, zp)
    for i, (sp, pc) in enumerate(zip(spec, pcs)):          # V = -|v|^2 / 2 from the reference, per piece
        Q = P[pc["ii"]]
        t = 2 * ((Q - sp["O"]) @ sp["e1"]) / sp["l"] - 1
        sp["h"] = C.chebfit(t, -0.5 * (V[pc["ii"]] ** 2).sum(1), pp.DEG_V)
    fv = Acc + G * qs.EZ
    cc = np.linalg.norm(fv, axis=1)
    jp = J - (np.einsum("ij,ij->i", J, fv / cc[:, None]))[:, None] * fv / cc[:, None]
    ref = dict(c_max=cc.max(), c_min=cc.min(), w_max=(np.linalg.norm(jp, axis=1) / cc).max(),
               tilt_slit=np.degrees(np.arccos(abs(fv[np.argmin(np.linalg.norm(P - slit, axis=1)), 1]) / cc[np.argmin(np.linalg.norm(P - slit, axis=1))])), T=T, P=P)
    return spec, slit, ref


def slit_passes(log, slit):
    """Samples inside the wall (|x| < WALL / 2) near the slit; required slit width (y) and height (z) per pass."""
    t, p, nz = log[:, 0], log[:, 4:7], log[:, 15:18]
    inside = (np.abs(p[:, 0] - slit[0]) < WALL / 2) & (np.abs(p[:, 1] - slit[1]) < 2.0)
    k = np.where(inside)[0]
    if not len(k):
        return []
    groups = np.split(k, np.where(np.diff(k) > 1)[0] + 1)
    out = []
    for g in groups:
        dy, dz = p[g, 1] - slit[1], p[g, 2] - slit[2]
        ny, nzz = np.abs(nz[g, 1]), np.abs(nz[g, 2])
        half_y = np.abs(dy) + R_DRONE * np.sqrt(1 - ny ** 2) + THICK / 2 * ny
        half_z = np.abs(dz) + R_DRONE * np.sqrt(1 - nzz ** 2) + THICK / 2 * nzz
        out.append(dict(t=t[g[0]], dev=np.degrees(np.arccos(ny)).max(), dy=np.abs(dy).max(), dz=np.abs(dz).max(),
                        width=2 * half_y.max(), height=2 * half_z.max()))
    return out


if __name__ == "__main__":
    speeds = [float(a) for a in sys.argv[1:]] or [6.0, 8.0, 10.0]
    res = []
    for vh in speeds:
        spec, slit, ref = build(vh)
        pol = fe.PolyELPolicy(spec, wp=16, wH=12)
        k0 = int(np.argmin([abs(np.linalg.norm(sp["O"] - ref["P"][len(ref["P"]) // 2])) for sp in spec]))
        p0, v0, a0, j0 = pp.state_at(spec[k0], 0.0)            # start half a lap before the slit
        log = fe.fly(quad, pol, p0, v0, a0, j0, 3 * ref["T"], mode="B")
        ok = log[-1, 0] > 3 * ref["T"] - 0.01
        ps = slit_passes(log, slit)
        print(f"vh {vh:4.1f} m/s (c at slit {vh ** 2 / R:4.1f} m/s^2, lap {ref['T']:.2f} s) | plan: tilt from knife-edge "
              f"{ref['tilt_slit']:.1f} deg, thrust {ref['c_min']:.1f}..{ref['c_max']:.1f} m/s^2, body rate max "
              f"{ref['w_max']:.1f} rad/s | flown ok {ok}, |phi| max {log[:, 1].max() * 1000:.1f} mm, sat "
              f"{100 * log[:, 8].mean():.1f}%, |w| max {log[:, 7].max():.1f}", flush=True)
        for q in ps:
            print(f"    pass t={q['t']:5.2f}s: knife-edge error {q['dev']:4.1f} deg, offset y {100 * q['dy']:.1f} cm "
                  f"z {100 * q['dz']:.1f} cm -> slit needed {100 * q['width']:.1f} cm wide x {100 * q['height']:.1f} cm tall")
        res.append((vh, spec, slit, ref, log))
        np.savez(f"../logs/slit_demo_v{vh:g}.npz", log=log, slit=slit, vh=vh, refP=ref["P"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 5))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    vh, spec, slit, ref, log = res[len(res) // 2]
    p, nz = log[:, 4:7], log[:, 15:18]
    ax3.plot(*p.T, lw=0.8, color="tab:blue")
    near = np.where(np.abs(p[:, 1]) < 1.5)[0][::25]
    near = near[(p[near, 0] > -3) & (p[near, 0] < 3)]
    for k in near:                                          # drone disc as a short bar perpendicular to body z
        d = np.cross(nz[k], log[k, 4:7] - log[max(k - 1, 0), 4:7]); d /= np.linalg.norm(d) + 1e-12
        ax3.plot(*np.stack([p[k] - R_DRONE * d, p[k] + R_DRONE * d]).T, color="tab:red", lw=1.0)
    yy, zz = np.meshgrid(np.linspace(-2, 2, 2), np.linspace(1.5, 4.5, 2))
    ax3.plot_surface(np.zeros_like(yy), yy, zz, alpha=0.15, color="gray")
    ax3.set_title(f"lap at vh {vh:g} m/s; red bars = drone disc (seen edge-on at the wall)")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    for vh, spec, slit, ref, log in res:                    # front view at the wall: the drone's cross-section
        q = slit_passes(log, slit)
        p, nz = log[:, 4:7], log[:, 15:18]
        k = np.where((np.abs(p[:, 0]) < WALL / 2) & (np.abs(p[:, 1]) < 2))[0]
        k = k[log[k, 0] > ref["T"]]
        if not len(k):
            continue
        k = k[len(k) // 2]
        n = nz[k]
        u = np.cross(n, [1.0, 0, 0]); u /= np.linalg.norm(u)
        dy_, dz_ = p[k, 1] - slit[1], p[k, 2] - slit[2]
        ax2.plot([dy_ - R_DRONE * u[1], dy_ + R_DRONE * u[1]], [dz_ - R_DRONE * u[2], dz_ + R_DRONE * u[2]],
                 lw=4, label=f"vh {vh:g} m/s, needs {100 * max(x['width'] for x in q):.0f} x {100 * max(x['height'] for x in q):.0f} cm")
    ax2.add_patch(plt.Rectangle((-0.075, -0.3), 0.15, 0.6, fill=False, ls="--", color="k"))
    ax2.set_aspect("equal"); ax2.set_xlim(-0.5, 0.5); ax2.set_ylim(-0.45, 0.45)
    ax2.set_xlabel("y from slit centre, m"); ax2.set_ylabel("z from slit centre, m"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    ax2.set_title("front view at the wall (dashed: a 15 x 60 cm window)")
    fig.savefig("../logs/slit_demo.png", dpi=120, bbox_inches="tight")
