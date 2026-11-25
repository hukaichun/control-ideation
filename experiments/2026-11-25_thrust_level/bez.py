"""Thrust-level formulation without division (user, 2026-11-25).

Path p(u) and slowness w(u) = dt/du > 0 are B-splines in u. Then
    v = p'/w,   a = (p'' w - p' w') / w^3,   T = int w du   (linear)
and the collective thrust limit |a + g| <= c_max is, since w > 0,
    | p'' w - p' w' + g e3 w^3 | <= c_max w^3        (polynomial both sides, no division)
On every knot interval both sides are converted to Bernstein (Bezier) form; the cone {|F| <= C} is convex, so if every
Bezier control point (F_k, C_k) lies in the cone, the whole curve does (everywhere, no sampling).

Helpers here: B-spline -> per-interval Bernstein coefficients (a linear map), Bernstein product and degree elevation.
"""
from math import comb

import casadi as ca
import numpy as np
from scipy.interpolate import BSpline


def bernstein_matrix(deg):
    """V[i, k] = B_k^deg(x_i) at x_i = i / deg (invertible)."""
    x = np.linspace(0, 1, deg + 1)
    return np.array([[comb(deg, k) * xi ** k * (1 - xi) ** (deg - k) for k in range(deg + 1)] for xi in x])


def to_bezier(knots, deg, n_coef, d=0):
    """Per knot interval [knots[j], knots[j+1]]: matrix (deg-d+1) x n_coef giving the Bernstein coefficients (in the local
    variable x in [0, 1]) of the d-th u-derivative of the spline with coefficients c (so Bez_j @ c)."""
    t = knots
    brk = np.unique(t[deg:len(t) - deg])
    out = []
    dd = deg - d
    Vinv = np.linalg.inv(bernstein_matrix(dd))
    for a, b in zip(brk[:-1], brk[1:]):
        x = np.linspace(0, 1, dd + 1)
        u = a + (b - a) * x
        u = np.clip(u, a + 1e-12 * (b - a), b - 1e-12 * (b - a))
        E = np.zeros((dd + 1, n_coef))
        for i in range(n_coef):
            c = np.zeros(n_coef)
            c[i] = 1
            s = BSpline(t, c, deg)
            E[:, i] = (s.derivative(d) if d else s)(u)
        out.append(Vinv @ E)
    return out, brk


def bmul(a, b):
    """Bernstein product: coefficients (casadi or numpy, last axis = coefficient) of degrees len(a)-1, len(b)-1."""
    m, n = len(a) - 1, len(b) - 1
    out = []
    for k in range(m + n + 1):
        s = 0
        for i in range(max(0, k - n), min(m, k) + 1):
            s = s + comb(m, i) * comb(n, k - i) / comb(m + n, k) * a[i] * b[k - i]
        out.append(s)
    return out


def belevate(a, N):
    m = len(a) - 1
    return [sum(comb(m, i) * comb(N - m, k - i) / comb(N, k) * a[i] for i in range(max(0, k - N + m), min(m, k) + 1))
            for k in range(N + 1)]


def cone_rows(p1, p2, w, w1, g=9.81):
    """Per interval: p1, p2 = lists (Bernstein coefficients) of 3-vectors of p' and p'', w, w1 = of w and w'.
    Returns (F_k, W3_k) lists of the elevated control points of  F = p'' w - p' w' + g e3 w^3  and  w^3."""
    A = bmul(p2, w)
    B = bmul(p1, w1)
    w3 = bmul(bmul(w, w), w)
    N = len(w3) - 1
    A = belevate(A, N)
    B = belevate(B, N)
    e3 = ca.DM([0, 0, g]) if isinstance(w3[0], (ca.SX, ca.MX)) else np.array([0, 0, g])
    F = [A[k] - B[k] + e3 * w3[k] for k in range(N + 1)]
    return F, w3
