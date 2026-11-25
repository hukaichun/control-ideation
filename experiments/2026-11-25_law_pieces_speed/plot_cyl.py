"""Plot the cylinder-turn results (results/cyl_<n>.npy): top view of the paths (cores solid, overlap tails dotted),
speed and collective thrust along the world x. Output results/toy_cylinder.png.
    ADAPT=1 uv run --with jax --with numpy --with scipy --with clarabel --with pyyaml --with matplotlib python plot_cyl.py
"""
import matplotlib
matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jprob as JP
import toy_cylinder as TC

COL = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a"}
fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), facecolor="#fcfcfb")
x = np.linspace(-TC.X, TC.X, 400)
ax[0].plot(x, TC.A * x ** 2, "--", color="#8a8a85", lw=1.5, label="initial parabola")
th = np.linspace(0, 2 * np.pi, 200)
ax[0].fill(TC.CX + TC.RC * np.cos(th), TC.CY + TC.RC * np.sin(th), color="#c3c2b7", label="cylinder (r = 1 m)")
for n in (1, 2, 4):
    prob = TC.CylProblem(n)
    z = np.load(JP.Path(__file__).parent / "results" / f"cyl_{n}.npy")
    T = prob.T(z)
    for i, p in enumerate(prob.pp):
        xs = np.linspace(p.pc.xa, p.pc.xb, 600)
        q, Qd, pos = prob._geo(p, xs)(jnp.array(z[prob.off[i]:prob.off[i + 1]]))
        pos = np.asarray(pos)
        v = np.sqrt(np.maximum(np.asarray(Qd[0]), 0))
        ph = np.asarray(TC.VP(*q, *Qd))
        core = (xs >= p.pc.xc[0]) & (xs <= p.pc.xc[1])
        lab = f"{n} piece{'s' if n > 1 else ''}: {T:.3f} s" if i == 0 else None
        ax[0].plot(pos[core, 0], pos[core, 1], color=COL[n], lw=2.2, label=lab)
        ax[0].plot(pos[~core, 0], pos[~core, 1], ":", color=COL[n], lw=1.5)
        ax[1].plot(pos[core, 0], v[core], color=COL[n], lw=2, label=lab)
        ax[2].plot(pos[core, 0], ph[core, 2], color=COL[n], lw=2, label=lab)
ax[0].set_aspect("equal"); ax[0].set_title("top view (m); dotted: overlap tails"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
ax[1].set_title("speed along the path (m/s)"); ax[1].set_xlabel("x (m)")
ax[2].set_title("collective thrust (m/s^2)"); ax[2].set_xlabel("x (m)")
ax[2].axhline(JP.CMAX, color="#8a8a85", lw=1, ls="--")
for a in ax:
    a.grid(color="#e4e3dd"); a.set_facecolor("#fcfcfb")
    for sp_ in a.spines.values():
        sp_.set_color("#c3c2b7")
    a.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(JP.Path(__file__).parent / "results" / "toy_cylinder.png", dpi=120)
print("saved")
