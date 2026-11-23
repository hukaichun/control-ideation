"""Through a train carriage: in through a window on one side, out through the window opposite, knife-edge (disc
edge-on to both walls) the whole way across, then a vertical loop over the roof back to the first window.

Carriage: walls x = 0 and x = W (thickness WALL), roof z = Z_ROOF; windows (narrow in y, tall in z) at z = Z_WIN.
Inside (x in [-EXT, W + EXT]) one polynomial piece, exact: constant a = (0, c, -g) -> body z = +y. It is the parabola
p(t) = (W/2 + v t, c t^2 / 2, Z_WIN + g (tw^2 - t^2) / 2) (tw = W / 2v: both windows at Z_WIN, same y), with
V = -|v(t)|^2 / 2 (exact: degree 2 in the piece's coordinate). Path and V of this piece are fixed.
The rest: the parabola continued 1.5 m beyond each wall, and a minimum-snap curve back over the roof (geometry only), cut and
fitted with G3 polynomials (the inside piece fixed), speed planned for minimum time with the inside V fixed.
Then the same law (PolyELPolicy) flies 3 laps.

    PYTHONPATH=. uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python train_demo.py [v c]
"""
import sys
import time
import types

import numpy as np

import flat_el_policy as fe
import poly_plan as pp
import quad_sim as qs

G = 9.81
W, WALL, EXT = 2.8, 0.10, 0.25
Z_WIN, Z_ROOF = 2.0, 4.0
R_DRONE, THICK = 0.25, 0.08
quad = qs.Quad()


def inside_piece(v, c):
    th, tw = (W / 2 + EXT) / v, W / 2 / v
    k, kz = 0.25 * c * th ** 2, 0.25 * G * th ** 2
    O = np.array([-EXT, 0.5 * c * th ** 2, Z_WIN + 0.5 * G * (tw ** 2 - th ** 2)])
    f = np.zeros(pp.DEG_P + 1); f[[0, 2]] = (-k, k)
    g = np.zeros(pp.DEG_P + 1); g[[0, 2]] = (kz, -kz)
    h = np.zeros(pp.DEG_V + 1)
    s2 = (c ** 2 + G ** 2) * th ** 2
    h[[0, 2]] = (-0.5 * v ** 2 - 0.25 * s2, -0.25 * s2)
    pc = dict(O=O, e1=np.array([1.0, 0, 0]), e2=np.array([0, 1.0, 0]), e3=np.array([0, 0, 1.0]), l=W + 2 * EXT, f=f, g=g)
    return pc, h, th


def dense_lap(v, c, a=1.5, Tr=2.0, z_top=6.0, deg=11, ds=0.01):
    """Geometry only. The inside parabola continued a metres beyond each wall, then a return over the roof: a degree-11
    minimum-snap polynomial in a parameter tau in [0, Tr] matching position, velocity, acceleration and jerk of the
    parabola at both ends and passing (x = W/2, z = z_top) half-way. Starts at x = -EXT (start of the inside piece)."""
    th, ta, tw = (W / 2 + EXT) / v, (W / 2 + a) / v, W / 2 / v
    A = np.array([0.0, c, -G])
    par = lambda t: (np.stack([W / 2 + v * t, 0.5 * c * t ** 2, Z_WIN + 0.5 * G * (tw ** 2 - t ** 2)], -1),
                     np.stack([v + 0 * t, c * t, -G * t], -1))

    def B(s_, d):
        out = np.zeros((len(s_), deg + 1))
        for k in range(d, deg + 1):
            out[:, k] = np.prod(np.arange(k - d + 1, k + 1)) * s_ ** (k - d) / Tr ** d
        return out
    sq = np.linspace(0, 1, 400)
    H = B(sq, 4).T @ B(sq, 4) / len(sq)
    (pe, ve), (ps, vs) = par(np.array([ta])), par(np.array([-ta]))
    s0, s1, sm = np.array([0.0]), np.array([1.0]), np.array([0.5])
    tau = np.linspace(0, 1, 4000)
    ret = np.zeros((len(tau), 3))
    for ax in range(3):
        rows = [B(s0, 0), B(s0, 1), B(s0, 2), B(s0, 3), B(s1, 0), B(s1, 1), B(s1, 2), B(s1, 3)]
        vals = [pe[0, ax], ve[0, ax], A[ax], 0, ps[0, ax], vs[0, ax], A[ax], 0]
        if ax in (0, 2):
            rows.append(B(sm, 0)); vals.append(W / 2 if ax == 0 else z_top)
        Aq, b = np.vstack(rows), np.array(vals, float)
        K = np.block([[H, Aq.T], [Aq, np.zeros((len(b), len(b)))]])
        ret[:, ax] = B(tau, 0) @ np.linalg.solve(K, np.concatenate([np.zeros(deg + 1), b]))[:deg + 1]
    t1 = np.arange(-th, ta, ds / v)
    t2 = np.arange(-ta, -th, ds / v)
    P = np.vstack([par(t1)[0], ret[1:-1], par(t2)[0]])
    keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-3])
    return P[keep]


def build(v=8.0, c=8.0, a=1.5, Tr=2.0, z_top=6.0, margin=0.9):
    pc_in, h_in, th = inside_piece(v, c)
    P = dense_lap(v, c, a, Tr, z_top)
    pcs, _ = pp.pieces_from(P, [], maxang=np.deg2rad(60.0), maxlen=W + 2 * EXT - 0.02)
    ii0 = pcs[0]["ii"]
    pcs[0] = dict(pc_in, ii=ii0)
    t0 = time.perf_counter()
    zp, bp = pp.smooth_path(pcs, [], [], 0.1, fixed=(0,))
    t_smooth = time.perf_counter() - t0
    sl = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0], np.cumsum(sl)[:-1]])
    motor = dict(m=quad.m, Ix=quad.J[0, 0], l=0.15, Tmax=quad.Tmax, Tmin=quad.Tmin, margin=margin)
    t0 = time.perf_counter()
    r_ = pp._plan(pcs, [], [], types.SimpleNamespace(length=sl.sum()), P, s, motor, lambda q: np.full_like(q, v),
                  0.1, 0.1, 1e-6, 15.0, 0.9, 2.0, 1000, 1e-6, zp, bp, fix_path=True, fix_h={0: h_in})
    r_.update(t_smooth=t_smooth, t_speed=time.perf_counter() - t0, P=P, th=th)
    return r_


def window_passes(log, y_win):
    """Each crossing of each wall (|x - wall| < WALL / 2): knife-edge error and the window size it needs."""
    t, p, nz = log[:, 0], log[:, 4:7], log[:, 15:18]
    out = []
    for xw in (0.0, W):
        k = np.where((np.abs(p[:, 0] - xw) < WALL / 2) & (p[:, 2] < Z_ROOF))[0]
        if not len(k):
            continue
        for g in np.split(k, np.where(np.diff(k) > 1)[0] + 1):
            dy, dz = p[g, 1] - y_win, p[g, 2] - Z_WIN
            ny, nzz = np.abs(nz[g, 1]), np.abs(nz[g, 2])
            out.append(dict(wall=xw, t=t[g[0]], dev=np.degrees(np.arccos(np.clip(ny, -1, 1))).max(),
                            dy=np.abs(dy).max(), dz=np.abs(dz).max(),
                            width=2 * (np.abs(dy) + R_DRONE * np.sqrt(1 - ny ** 2) + THICK / 2 * ny).max(),
                            height=2 * (np.abs(dz) + R_DRONE * np.sqrt(1 - nzz ** 2) + THICK / 2 * nzz).max()))
    return sorted(out, key=lambda q: q["t"])


if __name__ == "__main__":
    v, c = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (8.0, 8.0)
    res = build(v, c)
    spec, lap = res["spec"], res["lap"]
    print(f"v {v} c {c}: plan {res['status']} lap {lap:.2f} s | smooth {res['t_smooth']:.1f}s speed "
          f"{res['t_speed']:.1f}s | pieces {len(spec)}", flush=True)
    y_win = 0.5 * c * (W / 2 / v) ** 2
    pol = fe.PolyELPolicy(spec, wp=16, wH=12)
    p0, v0, a0, j0 = pp.state_at(spec[len(spec) // 2], 0.0)       # start over the roof
    log = fe.fly(quad, pol, p0, v0, a0, j0, 3 * lap + 0.5, mode="B")
    ok = log[-1, 0] > 3 * lap + 0.49
    p = log[:, 4:7]
    # clearance: outside the carriage's x-band, or above the roof; inside the band below the roof only on the inside
    # piece (between the walls)
    band = (p[:, 0] > -WALL / 2 - R_DRONE) & (p[:, 0] < W + WALL / 2 + R_DRONE)
    roof_clear = (p[band & (p[:, 2] > Z_ROOF - 1), 2] - Z_ROOF).min() if np.any(band & (p[:, 2] > Z_ROOF - 1)) else np.nan
    print(f"flown ok {ok} | |phi| max {1000 * log[:, 1].max():.1f} mm | sat {100 * log[:, 8].mean():.1f}% | "
          f"|w| max {log[:, 7].max():.1f} rad/s | requested motor max {log[:, 12].max():.2f} Tmax | "
          f"upside down {100 * (log[:, 14] < 0).mean():.0f}% | lowest point over the roof {roof_clear:.2f} m above it", flush=True)
    for q in window_passes(log, y_win):
        print(f"    t={q['t']:5.2f}s wall x={q['wall']:.1f}: knife-edge error {q['dev']:4.1f} deg, offset y "
              f"{100 * q['dy']:.1f} cm z {100 * q['dz']:.1f} cm -> window needed {100 * q['width']:.1f} x "
              f"{100 * q['height']:.1f} cm", flush=True)
    np.savez(f"../logs/train_demo_v{v:g}_c{c:g}.npz", log=log, spec=np.array(spec, dtype=object), lap=lap)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw=dict(width_ratios=[2.2, 1]))
    L = log[log[:, 0] > lap]
    ax.add_patch(plt.Rectangle((0 - WALL / 2, 0.8), W + WALL, Z_ROOF - 0.8, color="0.85"))
    ax.add_patch(plt.Rectangle((WALL / 2, 0.8), W - WALL, Z_ROOF - 0.8 - WALL, color="white"))
    for xw in (0.0, W):
        ax.add_patch(plt.Rectangle((xw - WALL / 2, Z_WIN - 0.3), WALL, 0.6, color="skyblue"))
    sc = ax.scatter(L[::20, 4], L[::20, 6], c=np.linalg.norm(np.diff(L[:, 4:7], axis=0), axis=1)[::20] / 0.0005, s=3,
                    cmap="viridis")
    plt.colorbar(sc, ax=ax, label="speed m/s")
    for k in range(0, len(L), 60):                         # drone disc, side view: bar perpendicular to body z
        n = L[k, 15:18]
        d = np.cross(n, [0, 1.0, 0])
        if np.linalg.norm(d) < 0.2:                        # edge-on to this view's normal: draw the full disc
            ax.plot(L[k, 4] + R_DRONE * np.array([-1, 1]), [L[k, 6]] * 2, color="tab:red", lw=1.2)
            ax.plot([L[k, 4]] * 2, L[k, 6] + R_DRONE * np.array([-1, 1]), color="tab:red", lw=1.2)
            continue
        d /= np.linalg.norm(d)
        ax.plot(L[k, 4] + R_DRONE * np.array([-d[0], d[0]]), L[k, 6] + R_DRONE * np.array([-d[2], d[2]]), color="k", lw=1)
    ax.set_aspect("equal"); ax.set_xlabel("x m"); ax.set_ylabel("z m"); ax.grid(alpha=0.3)
    ax.set_title("side view: carriage grey, windows blue; black bar = disc from the side, red cross = knife-edge",
                 fontsize=9)
    for xw, col in ((0.0, "tab:blue"), (W, "tab:orange")):
        kk = np.where((np.abs(log[:, 4] - xw) < 0.005) & (log[:, 6] < Z_ROOF) & (log[:, 0] > lap))[0]
        if not len(kk):
            continue
        kk = kk[0]
        n = log[kk, 15:18]
        u = np.cross(n, [1.0, 0, 0]); u /= np.linalg.norm(u)
        ax2.plot(log[kk, 5] - y_win + R_DRONE * np.array([-u[1], u[1]]), log[kk, 6] - Z_WIN + R_DRONE * np.array([-u[2], u[2]]),
                 lw=4, color=col, label=f"wall x = {xw:.1f}")
    ax2.add_patch(plt.Rectangle((-0.075, -0.3), 0.15, 0.6, fill=False, ls="--", color="k"))
    ax2.set_aspect("equal"); ax2.set_xlim(-0.4, 0.4); ax2.set_ylim(-0.4, 0.4); ax2.grid(alpha=0.3)
    ax2.set_xlabel("y from window centre, m"); ax2.set_ylabel("z from window centre, m"); ax2.legend(fontsize=8)
    ax2.set_title("looking through the windows (dashed: 15 x 60 cm)", fontsize=9)
    fig.savefig(f"../logs/train_demo.png", dpi=120, bbox_inches="tight")
