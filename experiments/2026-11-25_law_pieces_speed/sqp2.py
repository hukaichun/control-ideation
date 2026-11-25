"""SQP with a filter line search (user, 2026-11-25: own optimiser, written cleanly).
  min T(z) s.t. lb <= g(z) <= ub, lo <= z <= hi
Step: QP (Clarabel)  min grad T.d + 1/2 d'Hd  s.t.  lb <= g + J d <= ub,  lo <= z + d <= hi,
      H = Hessian of the Lagrangian (exact, JAX) with eigenvalues clipped to be positive, in scaled variables.
Line search: filter on (violation, T) with backtracking; one second-order correction when the full step is rejected.
Multipliers: the QP's. Stops when the step and the violation are below tol.
"""
import time

import clarabel
import numpy as np
import scipy.sparse as sp


def qp(H, c, A, l, u):
    """min 1/2 x'Hx + c'x s.t. l <= Ax <= u (dense A); returns x, y (y > 0: upper side active, < 0: lower)."""
    eq = np.isfinite(l) & np.isfinite(u) & (np.abs(u - l) < 1e-12)
    up, lo = np.isfinite(u) & ~eq, np.isfinite(l) & ~eq
    Ac = sp.csc_matrix(np.vstack([A[eq], A[up], -A[lo]]))
    b = np.r_[u[eq], u[up], -l[lo]]
    cones = ([clarabel.ZeroConeT(int(eq.sum()))] if eq.any() else []) + [clarabel.NonnegativeConeT(int(up.sum() + lo.sum()))]
    st = clarabel.DefaultSettings()
    st.verbose = False
    s = clarabel.DefaultSolver(sp.triu(sp.csc_matrix(H)).tocsc(), c, Ac, b, cones, st).solve()
    y = np.zeros(A.shape[0])
    z = np.array(s.z)
    ne, nu = int(eq.sum()), int(up.sum())
    y[eq], y[up], y[lo] = z[:ne], z[ne:ne + nu], -z[ne + nu:]
    return np.array(s.x), y, str(s.status)


def theta(g, lb, ub):
    return float(np.maximum(0, lb - g).sum() + np.maximum(0, g - ub).sum())


def solve(P, z, max_it=200, tol=1e-8, log=print):
    lb, ub = P.bounds_g()
    lo, hi = P.bounds_z()
    S = P.scale()                                 # z = S * y
    m = len(lb)
    lam = np.zeros(m)
    t0 = time.perf_counter()
    g = P.cons(z)
    T = float(P.T(z))
    filt = [(max(1.0, 10 * theta(g, lb, ub)), -np.inf)]
    gam = 1e-5
    for it in range(max_it):
        J = P.jac(z) * S[None, :]
        gT = np.asarray(P.gradT(z)) * S
        H = P.hess_lag(z, lam) * np.outer(S, S)
        H = 0.5 * (H + H.T)
        ev, V = np.linalg.eigh(H)
        H = (V * np.maximum(ev, 1e-8 * max(1.0, np.abs(ev).max()))) @ V.T
        A = np.vstack([J, np.eye(len(z))])
        l = np.r_[lb - g, (lo - z) / S]
        u = np.r_[ub - g, (hi - z) / S]
        d, y, st = qp(H, gT, A, l, u)
        if not st.startswith("Solved") and not st.startswith("AlmostSolved"):
            log(f"      it {it}: QP {st}")
            break
        lam_new = y[:m]
        th0 = theta(g, lb, ub)
        alpha, acc = 1.0, False
        while alpha > 1e-6:
            zt = z + alpha * S * d
            gt = P.cons(zt)
            tht, Tt = theta(gt, lb, ub), float(P.T(zt))
            ok_f = all(tht < tf or Tt < Tf for tf, Tf in filt)
            ok_d = tht <= (1 - gam) * th0 or Tt <= T - gam * th0 or (th0 < tol and Tt <= T + 1e-4 * alpha * gT @ d)
            if ok_f and ok_d:
                acc = True
                break
            if alpha == 1.0:                      # second-order correction once
                d2, _, st2 = qp(H, gT, A, np.r_[lb - gt + J @ d, (lo - z) / S], np.r_[ub - gt + J @ d, (hi - z) / S])
                if st2.startswith("Solved"):
                    z2 = z + S * d2
                    g2 = P.cons(z2)
                    th2, T2 = theta(g2, lb, ub), float(P.T(z2))
                    if all(th2 < tf or T2 < Tf for tf, Tf in filt) and (th2 <= (1 - gam) * th0 or T2 <= T - gam * th0):
                        zt, gt, tht, Tt, d, acc = z2, g2, th2, T2, d2, True
                        break
            alpha *= 0.5
        if not acc:
            log(f"      it {it}: line search failed (theta {th0:.1e})")
            break
        if not (Tt <= T + 1e-4 * alpha * gT @ d and th0 < tol):
            filt.append(((1 - gam) * th0, T - gam * th0))
        step = np.abs(alpha * d).max()
        z, g, T, lam = zt, gt, Tt, lam_new
        if it % 10 == 0:
            log(f"      it {it:3d}: T {T:.5f} s, violation {tht:.1e}, alpha {alpha:.3f}, |step| {step:.1e}, {time.perf_counter() - t0:.2f} s")
        if step < tol and tht < 1e-7:
            break
    return z, dict(it=it + 1, t=time.perf_counter() - t0, T=T, viol=theta(g, lb, ub))
