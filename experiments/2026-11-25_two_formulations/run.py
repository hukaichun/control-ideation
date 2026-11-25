"""Run one version on one track and check it in physical units at the samples and on a 40x finer grid."""
import os

import numpy as np

import common as C


def main(V, name, k):
    Pa = C.Path_(k)
    z0 = V.start(Pa)
    zs, st, it, t = V.solve(Pa, z0)
    T = V.lap(Pa, zs)
    out = [f"[{name}] track {k}: path {Pa.L:.1f} m, path knots {Pa.L / Pa.nref:.2f} m{", cut at the tip (node " + str(Pa.i0) + ")" if Pa.cut else ""}, {Pa.n} variables, {Pa.M} samples x 4 limits, start {V.lap(Pa, z0):.2f} s"
           f" | {st}, {it} it, {t:.1f} s | lap {T:.4f} s ({100 * (T / Pa.lap_cpc - 1):+.1f}% vs CPC {Pa.lap_cpc:.3f})"]
    for fac, lab in ((1, "  optimizer samples"), (40, "  40x finer")):
        out.append(C.report(lab, *V.phys(Pa, zs, fac)))
    # stall test: whole lap 1% faster (same shape) -- still within the limits?
    out.append(C.report("  1% faster everywhere, 40x finer", *V.phys(Pa, faster(name, zs), 40)))
    if name == "q":
        out.append(f"  objective quadrature: {V.NG} points {T:.6f} s, 8 points {V.lap(Pa, zs, 8):.6f} s, "
                   f"32 points {V.lap(Pa, zs, 32):.6f} s")
    print("\n".join(out), flush=True)
    tag = f"{name}_{k}" + (f"_hp{os.environ['HP']}" if "HP" in os.environ else "") + \
        (f"_cut{os.environ.get('CUTC', 3)}" if os.environ.get("CUT") == "1" else "")
    np.save(C.Path(__file__).parent / "results" / f"{tag}.npy", zs)
    with open(C.Path(__file__).parent / "results" / f"{tag}.txt", "w") as fh:
        fh.write("\n".join(out) + "\n")


def faster(name, z):
    return z / 1.01 if name == "w" else z * 1.01 ** 2
