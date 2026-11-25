"""Own SQP on the law pieces (jprob.py + sqp.py, no CasADi). FREE=0: path fixed (check against run.py's IPOPT result),
FREE=1: path and speed together. Start: fitted path + run.py's speed (results/track<k>_cm.npz).
    ADAPT=1 FTOL=0.01 [FREE=1] uv run --with jax --with numpy --with scipy --with osqp --with pyyaml python run_sqp.py [tracks]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import jprob as JP     # noqa: E402
import sqp            # noqa: E402

FREE = os.environ.get("FREE", "0") == "1"


def main(k):
    t0 = time.perf_counter()
    prob = JP.Problem(k, free_path=FREE)
    st0 = np.load(os.path.join(HERE, "results", f"track{k}_cm.npz"))
    r0 = np.split(st0["z"], np.cumsum(st0["nr"])[:-1])
    z0 = prob.z0(r0)
    if os.environ.get("START") == "slow":             # 2 m/s instead of the IPOPT speed solution
        for i, p in enumerate(prob.pp):
            o = prob.off[i] + 2 * p.nf
            xx = np.linspace(p.pc.xa, p.pc.xb, 400)
            z0[o:o + p.nr] = np.linalg.lstsq(JP.brows(p.tr, 4, xx, 0), 4.0 / p.pc.metric(xx), rcond=None)[0]
    lo, hi = prob.bounds()
    fixed = None if FREE else prob.path_mask()
    if not FREE:
        fixed[prob.off[-1]:] = True          # gate coordinates: unused when the path is fixed
    g, lb, ub = prob.cons(z0, jac=False)
    tb = time.perf_counter() - t0
    print(f"track {k} ({'path + speed' if FREE else 'path fixed'}): {len(prob.pp)} pieces, {prob.nz} variables, {len(g)} "
          f"constraints, build {tb:.1f} s | start T {prob.T(z0):.4f} s, violation {sqp.viol(g, lb, ub):.1e}", flush=True)
    z, info = sqp.solve(prob, z0, lo, hi, fixed=fixed, log=lambda s: print(s, flush=True))
    ph = prob.phys(z)
    line = (f"track {k} ({'path + speed' if FREE else 'path fixed'}): T {info['T']:.4f} s ({100 * (info['T'] / prob.lap_cpc - 1):+.1f}% vs CPC) | "
            f"{info['it']} SQP steps, {info['t']:.1f} s | violation {info['viol']:.1e} | 40x finer: |w| max {ph[:, 0].max():.2f}, "
            f"torque ratio max {ph[:, 1].max():.3f}, collective [{ph[:, 2].min():.1f}, {ph[:, 2].max():.2f}]")
    print(line, flush=True)
    tag = "free" if FREE else "fixed"
    np.savez(os.path.join(HERE, "results", f"sqp_{tag}_{k}.npz"), z=z)
    with open(os.path.join(HERE, "results", f"sqp_{tag}_{k}.txt"), "w") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:]] or [1]:
        main(k)
