"""Shared adaptive knots (ADAPT=1): per track, pieces, knots per piece, span lengths, fit error, speed variables.
    ADAPT=1 uv run --with casadi --with pyyaml --with scipy --with numpy python knots_check.py
"""
import numpy as np

import pieces as PC
import run as R

for k in (0, 1, 4, 7, 8, 9):
    P, _ = R.C.cpc_lap(k)
    pcs, junc, L = PC.build(P)
    spans = np.concatenate([np.diff(np.unique(pc.tk)) for pc in pcs])
    print(f"track {k}: {len(pcs)} pieces, lap {L:.1f} m | spans per piece {[len(np.unique(pc.tk)) - 1 for pc in pcs]} | "
          f"span length min {spans.min():.2f} m, median {np.median(spans):.2f} m, max {spans.max():.2f} m | fit max "
          f"{max(pc.fit_err for pc in pcs) * 1000:.2f} mm | speed variables {sum(pc.nr for pc in pcs)} (was ~{int(L / 1.0) + 4 * len(pcs)} with 1 m)")
