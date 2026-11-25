"""Own SQP (user, 2026-11-25: own optimiser, no CasADi; derivatives from JAX in jprob.py).

  min T(z)  s.t.  lb <= g(z) <= ub,  lo <= z <= hi
Each step solves the QP (OSQP)
  min  grad T . d + 1/2 d' H d + nu * sum(s)
  s.t. lb - g - s <= J d <= ub - g + s,  s >= 0,  bounds on z + d,  |d_j| <= Delta_j   (trust region)
H = Hessian of T (exact, T is explicit) + a small multiple of the identity on the path coefficients (T does not depend
on them). The slacks s keep the QP feasible when the linearised constraints and the trust region do not meet.
Acceptance by the merit phi = T + nu * total violation (actual vs predicted decrease, trust-region ratio); nu is raised
above the largest QP multiplier. Trust region per variable type: path coefficients (m) and speed / gate variables
(relative). Stops when the step is tiny and the violation is below tol.
"""
import time

import numpy as np
import clarabel
import scipy.sparse as sp


def qp_clarabel(P, q, A, l, u):
    """min 1/2 x'Px + q'x s.t. l <= Ax <= u with Clarabel (interior point): rows split into A x <= u and -A x <= -l."""
    fin_u, fin_l = np.isfinite(u), np.isfinite(l)
    eq = fin_u & fin_l & (np.abs(u - l) < 1e-12)
    up, lo_ = fin_u & ~eq, fin_l & ~eq
    Aeq, Au, Al = A[eq], A[up], -A[lo_]
    Ac = sp.vstack([Aeq, Au, Al]).tocsc()
    b = np.r_[u[eq], u[up], -l[lo_]]
    cones = ([clarabel.ZeroConeT(int(eq.sum()))] if eq.any() else []) + [clarabel.NonnegativeConeT(int(up.sum() + lo_.sum()))]
    st = clarabel.DefaultSettings()
    st.verbose = False
    sol = clarabel.DefaultSolver(sp.triu(P).tocsc(), q, Ac, b, cones, st).solve()
    z = np.zeros(A.shape[0])
    yv = np.array(sol.z)
    ne, nu_ = int(eq.sum()), int(up.sum())
    z[eq], z[up], z[lo_] = yv[:ne], yv[ne:ne + nu_], -yv[ne + nu_:]
    return np.array(sol.x)[:P.shape[0]], z, str(sol.status)


def viol(g, lb, ub):
    return np.maximum(0, lb - g).sum() + np.maximum(0, g - ub).sum()


def solve(prob, z, lo, hi, fixed=None, max_it=200, tol=1e-6, log=print, dpath=0.02, drel=0.2, reg_path=1e-2):
    nz = len(z)
    fixed = np.zeros(nz, bool) if fixed is None else fixed
    pathm = prob.path_mask()
    nu = 10.0
    Dp, Dr = dpath, drel
    t0 = time.perf_counter()
    g, lb, ub, J = prob.cons(z)
    T = prob.T(z)
    hist = []
    lamg = np.zeros(len(g))
    for it in range(max_it):
        gT = prob.gradT(z)
        # Hessian of the Lagrangian (objective + constraint curvature with the last QP multipliers), made positive
        # definite by clipping its eigenvalues (dense: a few hundred variables)
        HL = prob.hessT(z) + prob.hess_lag(z, lamg)
        HL = 0.5 * (HL + HL.T)
        ev, V = np.linalg.eigh(HL)
        H = (V * np.maximum(ev, 1e-6 * max(1.0, np.abs(ev).max()))) @ V.T
        H[np.ix_(fixed, fixed)] = np.eye(int(fixed.sum())) if fixed.any() else H[np.ix_(fixed, fixed)]
        m = len(g)
        # trust region box per variable
        tr = np.where(pathm, Dp, Dr * np.maximum(np.abs(z), 1e-3))
        dlo = np.maximum(lo - z, -tr)
        dhi = np.minimum(hi - z, tr)
        dlo[fixed], dhi[fixed] = 0.0, 0.0
        # variables (d, s): QP
        P = sp.block_diag([sp.csc_matrix(H), sp.csc_matrix((m, m))]).tocsc()
        q = np.r_[gT, np.full(m, nu)]
        A = sp.vstack([sp.hstack([J, sp.identity(m)]), sp.hstack([J, -sp.identity(m)]),
                       sp.hstack([sp.identity(nz), sp.csc_matrix((nz, m))]),
                       sp.hstack([sp.csc_matrix((m, nz)), sp.identity(m)])]).tocsc()

        def qp(gc):
            l = np.r_[lb - gc, np.full(m, -np.inf), dlo, np.zeros(m)]
            u = np.r_[np.full(m, np.inf), ub - gc, dhi, np.full(m, np.inf)]
            dd, yy, st_ = qp_clarabel(P, q, A, l, u)
            return dd[:nz], yy, st_
        d, y, qst = qp(g)
        lam = np.abs(y[:2 * m]).max()
        lam_new = y[:m] + y[m:2 * m]
        nu = max(nu, 1.5 * lam)
        lin = g + J @ d
        pred = -(gT @ d + 0.5 * d @ H @ d) + nu * (viol(g, lb, ub) - viol(lin, lb, ub))
        zn = z + d
        gn, lbn, ubn = prob.cons(zn, jac=False)
        Tn = prob.T(zn)
        ared = (T + nu * viol(g, lb, ub)) - (Tn + nu * viol(gn, lbn, ubn))
        rho = ared / pred if pred > 1e-14 else (1.0 if ared >= 0 else -1.0)
        if rho <= 0.1 and pred > 1e-14:
            # second-order correction (Maratos effect): same QP with the constraints taken at z + d, linearised at z
            d2, y2, _ = qp(gn - J @ d)
            z2 = z + d2
            g2, lb2, ub2 = prob.cons(z2, jac=False)
            T2 = prob.T(z2)
            ared2 = (T + nu * viol(g, lb, ub)) - (T2 + nu * viol(g2, lb2, ub2))
            if ared2 / pred > 0.1:
                d, zn, gn, Tn, ared, y = d2, z2, g2, T2, ared2, y2
                lam_new = y[:m] + y[m:2 * m]
                rho = ared / pred
        step = np.abs(d).max()
        if rho > 0.1:
            z, g, T = zn, gn, Tn
            lamg = lam_new
            g, lb, ub, J = prob.cons(z)
            if rho > 0.75:
                Dp, Dr = min(2 * Dp, 0.5), min(2 * Dr, 1.0)
        else:
            Dp, Dr = Dp / 4, Dr / 4
        v = viol(g, lb, ub)
        hist.append((it, T, v, rho, Dp, Dr))
        if it % 5 == 0 or rho <= 0.1:
            log(f"      it {it:3d}: T {T:.5f} s, violation {v:.1e}, rho {rho:+.2f}, step {step:.1e}, trust path {Dp:.1e} m / "
                f"rel {Dr:.1e}, nu {nu:.1e}, QP {qst}, {time.perf_counter() - t0:.1f} s")
        if (step < 1e-7 and v < tol) or Dr < 1e-9:
            break
    return z, dict(it=it + 1, t=time.perf_counter() - t0, T=T, viol=viol(g, lb, ub), hist=hist)
