"""Plot results/cyl2_<n>.npy (lawplan + sqp2): top view, speed, collective. Output results/cyl2.png.
    ADAPT=1 uv run --with jax --with numpy --with scipy --with clarabel --with pyyaml --with matplotlib python plot_cyl2.py
"""
import matplotlib
matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import cyl2
import jprob as JP
import lawplan as LP

COL = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a"}
fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), facecolor="#fcfcfb")
x = np.linspace(-cyl2.X, cyl2.X, 400)
ax[0].plot(x, cyl2.A * x ** 2, "--", color="#8a8a85", lw=1.5, label="initial parabola")
th = np.linspace(0, 2 * np.pi, 200)
ax[0].fill(cyl2.CX + cyl2.RC * np.cos(th), cyl2.CY + cyl2.RC * np.sin(th), color="#c3c2b7", label="cylinder (r = 1 m)")
for n in (1, 2, 4):
    Pl = cyl2.build(n)
    z = np.load(JP.Path(__file__).parent / "results" / f"cyl2_{n}.npy")
    T = float(Pl.T(z))
    for i, p in enumerate(Pl.pp):
        xs = np.linspace(p.pc.xa, p.pc.xb, 800)
        q, Qd, pos = Pl.geo(p, xs)(jnp.array(z[Pl.off[i]:Pl.off[i + 1]]))
        pos, v = np.asarray(pos), np.sqrt(np.maximum(np.asarray(Qd[0]), 0))
        ph = np.asarray(LP.VPHYS(*q, *Qd))
        a = z[Pl.core_start[i]] if i in Pl.core_start else p.pc.xc[0]
        b = z[Pl.core_end[i]] if i in Pl.core_end else p.pc.xc[1]
        core = (xs >= a) & (xs <= b)
        lab = f"{n} piece{'s' if n > 1 else ''}: {T:.3f} s" if i == 0 else None
        ax[0].plot(pos[core, 0], pos[core, 1], color=COL[n], lw=2.2, label=lab)
        ax[0].plot(pos[~core, 0], pos[~core, 1], ":", color=COL[n], lw=1.3)
        ax[1].plot(pos[core, 0], v[core], color=COL[n], lw=2, label=lab)
        ax[2].plot(pos[core, 0], ph[core, 2], color=COL[n], lw=2, label=lab)
        if i in Pl.core_end:
            ax[0].plot(*pos[np.argmin(np.abs(xs - b)), :2], "o", color=COL[n], ms=5)
ax[0].set_aspect("equal"); ax[0].set_title("top view (m); dotted: tails; dots: junctions")
ax[1].set_title("speed (m/s)"); ax[2].set_title("collective thrust (m/s^2)")
ax[2].axhline(JP.CMAX, color="#8a8a85", lw=1, ls="--")
for a in ax:
    a.set_xlabel("x (m)"); a.grid(color="#e4e3dd"); a.set_facecolor("#fcfcfb")
    for sp_ in a.spines.values():
        sp_.set_color("#c3c2b7")
    a.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(JP.Path(__file__).parent / "results" / "cyl2.png", dpi=120)
