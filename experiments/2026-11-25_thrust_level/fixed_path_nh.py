"""Fixed path, heading left free (limits.LIM_NH: w_z = 0, limits valid for every heading), variables: w only.
Same path, knots, samples and objective as fixed_path.py. IPOPT.

    cd experiments/2026-11-25_thrust_level
    uv run --with casadi --with pyyaml --with scipy --with numpy python fixed_path_nh.py [track]
"""
import sys
import time

import casadi as ca
import numpy as np

import fixed_path as FP
from pathlib import Path
import limits as LM

HERE = Path(__file__).resolve().parent
TRACK = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def main():
    Pb = FP.Problem(TRACK)
    n = Pb.n

    def build(M):
        Pb.set_samples(M)
        z = ca.SX.sym("z", n)
        W = [ca.mtimes(ca.DM(B), z).T for B in Pb.Bw]
        fjs = FP.CHAIN.map(Pb.Ms)(ca.vertcat(ca.DM(Pb.Pd), *W, ca.DM.zeros(3, Pb.Ms)))[0:9, :]
        return z, LM.LIM_NH.map(Pb.Ms)(fjs), LM.CHK_NH.map(Pb.Ms)(fjs)
    z, g, chk = build(Pb.M)
    S = ca.nlpsol("s", "ipopt", {"x": z, "f": ca.sum1(z) / n, "g": ca.vec(g)},
                  {"print_time": False, "ipopt": {"print_level": int(__import__("os").environ.get("PL", 0)), "max_iter": int(__import__("os").environ.get("MAXIT", 3000)), "print_frequency_iter": 50, "nlp_scaling_method": __import__("os").environ.get("NLPSC", "gradient-based")}})
    t0 = time.perf_counter()
    sol = S(x0=np.full(n, Pb.L / 2.0), lbx=Pb.wmin, lbg=0, ubg=np.inf)
    t = time.perf_counter() - t0
    zs = np.array(sol["x"]).ravel()
    st = S.stats()
    print(f"track {TRACK}: {n} variables, {Pb.M} samples x {g.shape[0]} limits | {st['return_status']}, {st['iter_count']} it, "
          f"{t:.1f} s | lap {zs.mean():.4f} s ({100 * (zs.mean() / Pb.lap_cpc - 1):+.1f}% vs CPC {Pb.lap_cpc:.3f}) | "
          f"speed {Pb.L / zs.max():.1f}-{Pb.L / zs.min():.1f} m/s")
    np.save(HERE / "results" / f"nh_{TRACK}.npy", zs)
    for fac, lab in ((1, "optimizer samples"), (40, "40x finer")):
        zf, gf, chkf = build(fac * Pb.M)
        w, tn, ta, c = (np.array(v).ravel() for v in ca.Function("c", [zf], list(chkf))(zs))
        print(f"  {lab}: |w| max {w.max():.2f} rad/s (15) | torque needed / allowed max {np.max(tn / ta):.3f} "
              f"| collective [{c.min():.2f}, {c.max():.2f}] m/s^2")


if __name__ == "__main__":
    main()
