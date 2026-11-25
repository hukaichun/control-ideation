"""Is the lifted formulation / its derivatives written correctly? (user, 2026-11-25: no higher-order optimiser,
plain gradient descent + line search)

MODE=speed, one track. Three checks on the same problem (run.py: Lifted):
 1. finite differences vs the assembled gradient of the penalty function
      f(v) = T(v) + MU/2 |e(v)|^2 + MU/2 |max(0, -g(v))|^2
 2. plain gradient descent with Armijo backtracking on f (MU fixed, then raised), Q and S kept positive by the step
 3. IPOPT on the same lifted problem (equalities and bounds as constraints), as the reference optimum

    cd experiments/2026-11-25_lifted_own_optimizer
    uv run --with casadi --with pyyaml --with scipy --with numba --with matplotlib python verify.py [track]
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

HERE = Path(__file__).resolve().parent
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(os.environ.get("STEPS", 20000))
argv, sys.argv = sys.argv, sys.argv[:1]
os.environ["MODE"] = "speed"
spec = importlib.util.spec_from_file_location("run", HERE / "run.py")
L_ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L_)
sys.argv = argv
R, K, NYV = L_.R, L_.K, L_.NYV


def build(k):
    Pth, V, lap_ref, gates = R.load_cpc(k)
    Ln = np.linalg.norm(np.roll(Pth, -1, 0) - Pth, axis=1).sum()
    n = int(round(Ln / L_.HK))
    sl = np.linalg.norm(np.roll(Pth, -1, 0) - Pth, axis=1)
    s_ = np.r_[0, np.cumsum(sl)[:-1]]
    C = np.linalg.lstsq(L_.rows(s_ / Ln, n, 0), Pth, rcond=None)[0]
    P = L_.Lifted(n, Ln, "speed", Cpath=C)
    Kn = np.column_stack([np.full(n, L_.V0 ** 2), np.tile([1.0, 0, 0], (n, 1))])
    return P, L_.start_values(P, Kn)


def pen(P, v, mu):
    T, gT, _ = P.obj(v)
    e, Je = P.eq_jac(v)
    g, Jg = P.ineqs(v)
    c = np.maximum(0, -g)
    f = T + 0.5 * mu * (e @ e) + 0.5 * mu * (c @ c)
    grad = gT + mu * (Je.T @ e) - mu * (Jg.T @ c)
    return f, grad, T, float(np.abs(e).max()), float(c.max())


def fval(P, v, mu):
    T = P.obj(v)[0]
    e = P.eqs(v)
    g, _ = P.ineqs(v)
    c = np.maximum(0, -g)
    return T + 0.5 * mu * (e @ e) + 0.5 * mu * (c @ c)


def main():
    P, v0 = build(k)
    print(f"track {k}: {P.nv} variables ({P.nk} knot + {NYV * P.M} sample), start |eq| {np.abs(P.eqs(v0)).max():.1e}")
    # 1. finite differences at a perturbed point (so that bounds and equations are active)
    rng = np.random.default_rng(0)
    v = v0 + 1e-2 * rng.standard_normal(P.nv) * np.maximum(np.abs(v0), 1e-2)
    v[P.nk + 19::NYV] = np.abs(v[P.nk + 19::NYV])
    mu = 10.0
    f, grad, *_ = pen(P, v, mu)
    idx = rng.choice(P.nv, 40, replace=False)
    errs = []
    for i in idx:
        h = 1e-6 * max(1.0, abs(v[i]))
        vp, vm = v.copy(), v.copy()
        vp[i] += h
        vm[i] -= h
        fd = (fval(P, vp, mu) - fval(P, vm, mu)) / (2 * h)
        errs.append(abs(fd - grad[i]) / max(1e-8, abs(fd), abs(grad[i])))
    print(f"1. gradient vs central differences at 40 random variables: max relative error {max(errs):.1e}, "
          f"median {np.median(errs):.1e}")
    # 2. plain gradient descent + Armijo backtracking
    pos = np.r_[np.arange(P.Qi, P.nk, P.kd), P.nk + 19 + NYV * np.arange(P.M)]
    v = v0.copy()
    t0 = time.perf_counter()
    alpha = 1e-3
    for mu in (1e1, 1e2, 1e3, 1e4):
        for it in range(STEPS // 4):
            f, grad, T, ve, vi = pen(P, v, mu)
            d = -grad
            neg = d[pos] < 0
            amax = 0.95 * np.min(v[pos][neg] / -d[pos][neg]) if neg.any() else np.inf
            a = min(alpha * 2, amax)
            gg = grad @ grad
            while a > 1e-20 and fval(P, v + a * d, mu) > f - 1e-4 * a * gg:
                a *= 0.5
            if a <= 1e-20:
                break
            v = v + a * d
            alpha = a
        print(f"2. gradient descent, mu {mu:.0e}: {it + 1} steps, T {T:.4f} s, |eq| {ve:.1e}, bound viol {vi:.1e}, "
              f"|grad| {np.sqrt(gg):.1e}, {time.perf_counter() - t0:.0f} s")
    # 3. IPOPT on the same lifted problem
    z = ca.MX.sym("v", P.nv)
    Kn_ = ca.reshape(z[:P.nk], P.kd, P.n).T
    Y_ = ca.reshape(z[P.nk:], NYV, P.M)
    Q_ = Kn_[:, P.Qi]
    Hc = Kn_[:, P.kd - 3:]
    r = [ca.DM((P.Bs[d] @ P.Cpath).T) for d in range(1, 5)]
    q = [ca.mtimes(ca.DM(P.Bs[d].toarray()), Q_).T * P.qs for d in range(4)]
    Hm = [ca.mtimes(ca.DM(P.Bs[d].toarray()), Hc).T for d in range(3)]
    X = ca.vertcat(*r, *q, *Hm, Y_)
    E = P.Fe(X)
    T = ca.sum2(P.L / Y_[19, :]) / P.M
    lbx = np.full(P.nv, -np.inf)
    ubx = np.full(P.nv, np.inf)
    lbx[P.Qi:P.nk:P.kd] = 1e-6
    for kk in range(NYV):
        lbx[P.nk + kk::NYV] = P.blb[kk]
        ubx[P.nk + kk::NYV] = P.bub[kk]
    lbx[P.nk + 19::NYV] = 1e-3
    S = ca.nlpsol("ip", "ipopt", {"x": z, "f": T, "g": ca.vec(E)},
                  {"expand": True, "print_time": False, "ipopt": {"print_level": 0, "max_iter": 3000}})
    t0 = time.perf_counter()
    sol = S(x0=v0, lbx=lbx, ubx=ubx, lbg=0, ubg=0)
    vi = np.array(sol["x"]).ravel()
    print(f"3. IPOPT (same lifted problem): {S.stats()['return_status']}, {S.stats()['iter_count']} it, T {float(sol['f']):.4f} s, "
          f"|eq| {np.abs(P.eqs(vi)).max():.1e}, {time.perf_counter() - t0:.0f} s")
    print(f"   gradient descent vs IPOPT: T {P.obj(v)[0]:.4f} vs {float(sol['f']):.4f}")


if __name__ == "__main__":
    main()
