"""Trust-region filter SQP (Fletcher-Leyffer style; user, 2026-11-25: "試試看").
  min T(z) s.t. lb <= g(z) <= ub, lo <= z <= hi                (variables scaled: z = S y)
Step:  QP(Delta)   min grad T.d + 1/2 d'Hd  s.t.  lb <= g + J d <= ub, bounds, |d|_inf <= Delta
       H = exact Hessian of the Lagrangian (JAX), eigenvalues clipped to be positive (the QP must be convex).
       If QP(Delta) has no solution (linearised constraints and the box do not meet): restoration step
       min sum(s) s.t. lb - s <= g + J d <= ub + s, s >= 0, bounds, |d|_inf <= Delta   (reduces the violation only).
Accept the trial point if no filter entry dominates it and, when the QP predicted a decrease of T (dq > 0, an
"f-type" step), T actually fell by at least sigma * dq; otherwise reject and halve Delta. Accepted theta-type steps
enter the filter (f-type: the QP predicts dq >= theta^2). A rejected QP step gets one second-order correction (the
QP again with the constraints evaluated at the trial point). Delta doubles after an accepted step that reached the box.
"""
import time

import numpy as np

from sqp2 import qp, theta


def solve(P, z, max_it=300, tol=1e-8, log=print, delta0=1.0, dmax=10.0):
    lb, ub = P.bounds_g()
    lo, hi = P.bounds_z()
    S = P.scale()
    m = len(lb)
    lam = np.zeros(m)
    t0 = time.perf_counter()
    g = P.cons(z)
    T = float(P.T(z))
    filt = []
    beta, sigma = 0.99, 0.1
    D = delta0
    nrest = 0
    for it in range(max_it):
        J = P.jac(z) * S[None, :]
        gT = np.asarray(P.gradT(z)) * S
        H = P.hess_lag(z, lam) * np.outer(S, S)
        H = 0.5 * (H + H.T)
        ev, V = np.linalg.eigh(H)
        H = (V * np.maximum(ev, 1e-8 * max(1.0, np.abs(ev).max()))) @ V.T
        nz = len(z)
        bl, bu = np.maximum((lo - z) / S, -D), np.minimum((hi - z) / S, D)
        A = np.vstack([J, np.eye(nz)])
        d, y, st = qp(H, gT, A, np.r_[lb - g, bl], np.r_[ub - g, bu])
        kind = "QP"
        th0 = theta(g, lb, ub)
        if not (st.startswith("Solved") or st.startswith("AlmostSolved")):
            # restoration: least violation of the linearised constraints inside the box (LP in (d, s))
            nrest += 1
            Hs = np.zeros((nz + m, nz + m))
            Hs[:nz, :nz] = 1e-8 * np.eye(nz)
            c = np.r_[np.zeros(nz), np.ones(m)]
            Ar = np.vstack([np.hstack([J, np.eye(m)]), np.hstack([J, -np.eye(m)]), np.hstack([np.eye(nz), np.zeros((nz, m))]),
                            np.hstack([np.zeros((m, nz)), np.eye(m)])])
            lr = np.r_[lb - g, np.full(m, -np.inf), bl, np.zeros(m)]
            ur = np.r_[np.full(m, np.inf), ub - g, bu, np.full(m, np.inf)]
            dd, _, st = qp(Hs, c, Ar, lr, ur)
            d, y, kind = dd[:nz], np.zeros(nz + 2 * m), "restoration"
        dq = -(gT @ d + 0.5 * d @ H @ d) if kind == "QP" else 0.0
        zt = z + S * d
        gt = P.cons(zt)
        tht, Tt = theta(gt, lb, ub), float(P.T(zt))
        ftype = kind == "QP" and dq > 0 and dq >= th0 ** 2           # switching condition (Fletcher-Leyffer)

        def acceptable(tht, Tt):
            dom = any(tht >= beta * tf and Tt >= Tf - (1 - beta) * tf for tf, Tf in filt)
            if ftype:
                return (not dom) and (T - Tt >= sigma * dq)
            return (not dom) and (tht < beta * th0 or (Tt < T - (1 - beta) * th0 and tht <= th0))
        ok = acceptable(tht, Tt)
        if not ok and kind == "QP":
            # second-order correction: same QP with the constraints evaluated at the trial point (Maratos effect)
            d2, y2, st2 = qp(H, gT, A, np.r_[lb - gt + J @ d, bl], np.r_[ub - gt + J @ d, bu])
            if st2.startswith("Solved") or st2.startswith("AlmostSolved"):
                z2 = z + S * d2
                g2 = P.cons(z2)
                th2, T2 = theta(g2, lb, ub), float(P.T(z2))
                if acceptable(th2, T2):
                    zt, gt, tht, Tt, d, y, ok = z2, g2, th2, T2, d2, y2, True
        step = np.abs(d).max()
        if ok:
            if not ftype:
                filt.append((th0, T))
            z, g, T = zt, gt, Tt
            if kind == "QP":
                lam = y[:m]
            if step >= 0.99 * D:
                D = min(2 * D, dmax)
        else:
            D = D / 2
        if it % 10 == 0 or not ok:
            log(f"      it {it:3d}: T {T:.5f} s, violation {theta(g, lb, ub):.1e}, {kind}, {'accepted' if ok else 'rejected'}, "
                f"|d| {step:.1e}, Delta {D:.1e}, {time.perf_counter() - t0:.2f} s")
        if (ok and step < tol and theta(g, lb, ub) < 1e-7) or D < 1e-10:
            break
    return z, dict(it=it + 1, t=time.perf_counter() - t0, T=T, viol=theta(g, lb, ub), restorations=nrest)
