"""Test tracks: random gate tracks (same generator as the CPC benchmark, logs/cpc_benchmark.csv) and a rectangle
loop of lines + quarter circles."""
import numpy as np

import path_g2 as pg

TOL = 0.3   # gate tolerance [m]


def random_track(seed, M=7):
    rng = np.random.default_rng(seed)
    while True:
        ang = np.sort(rng.uniform(0, 2 * np.pi, M))
        if np.min(np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))) < 0.35:
            continue
        r = rng.uniform(5, 11, M)
        g = np.column_stack([r * np.cos(ang), r * np.sin(ang), rng.uniform(0.8, 3.6, M)])
        g[:, :2] += rng.normal(0, 1.5, (M, 2))
        if np.min(np.linalg.norm(np.roll(g, -1, 0) - g, axis=1)) > 3.0:
            return g


def rect_path(R, ell):
    """Rectangle: 4 straights of length ell joined by quarter circles of radius R (z = 0)."""
    side = ell + 2 * R
    verts = np.array([[0, 0, 0], [side, 0, 0], [side, side, 0], [0, side, 0]], float)
    return pg.PathG2(verts, np.full(4, 2 * R / side), 0.0)
