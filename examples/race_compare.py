"""Track and point-mass planning utilities: UZH track loader (rpg_time_optimal tracks/track.yaml), point-mass
free-path periodic optimum (CasADi/IPOPT; used as the warm start of the CPC baseline in quad_time_optimal.py),
lines + arcs path sampling, TOPP forward/backward passes, gate distance.
"""
from pathlib import Path


import numpy as np
import yaml
from numba import njit
from scipy.optimize import minimize

G = 9.81
EZ = np.array([0.0, 0.0, 1.0])
TRACK = Path(__file__).resolve().parents[1] / "third_party/rpg_time_optimal/tracks/track.yaml"


def load_track():
    gates = np.array(yaml.safe_load(TRACK.read_text())["gates"], float)
    return gates[:7]  # one lap; the file repeats the 7 gates for multi-lap flight


# ------------------------------------------------------------------ baseline: free path ----------
def free_path_optimum(gates, A, tol, N=30, return_segments=False):
    import casadi as ca
    M = len(gates)
    opti = ca.Opti()
    T = opti.variable(M)
    P = [opti.variable(3, N + 1) for _ in range(M)]
    V = [opti.variable(3, N + 1) for _ in range(M)]
    U = [opti.variable(3, N) for _ in range(M)]
    for i in range(M):
        dt = T[i] / N
        for k in range(N):
            a = U[i][:, k]
            opti.subject_to(P[i][:, k + 1] == P[i][:, k] + V[i][:, k] * dt + 0.5 * a * dt ** 2)
            opti.subject_to(V[i][:, k + 1] == V[i][:, k] + a * dt)
            opti.subject_to(ca.sumsqr(a + G * EZ) <= A ** 2)
        j = (i + 1) % M
        opti.subject_to(P[j][:, 0] == P[i][:, N])
        opti.subject_to(V[j][:, 0] == V[i][:, N])
        opti.subject_to(ca.sumsqr(P[i][:, 0] - gates[i]) <= tol ** 2)
        opti.subject_to(T[i] >= 0.05)
        # initial guess: straight line at 8 m/s
        g0, g1 = gates[i], gates[j]
        L = np.linalg.norm(g1 - g0)
        opti.set_initial(T[i], L / 8.0)
        opti.set_initial(P[i], np.linspace(g0, g1, N + 1).T)
        opti.set_initial(V[i], np.tile((g1 - g0) / (L / 8.0), (N + 1, 1)).T)
    opti.minimize(ca.sum1(T))
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 3000})
    sol = opti.solve()
    p = np.hstack([sol.value(P[i])[:, :-1] for i in range(M)]).T
    v = np.hstack([sol.value(V[i])[:, :-1] for i in range(M)]).T
    if return_segments:
        return float(np.sum(sol.value(T))), p, v, np.array(sol.value(T))
    return float(np.sum(sol.value(T))), p, v


# ------------------------------------------------------------------ ours: lines + arcs -----------
def build_path(verts, frac, ds=0.02):
    """Closed polyline with circular fillets. Returns samples: pos, tangent, curvature vector, kappa."""
    M = len(verts)
    e = np.roll(verts, -1, 0) - verts                     # edge i: v_i -> v_{i+1}
    L = np.linalg.norm(e, axis=1)
    d = e / L[:, None]
    arcs = []
    for i in range(M):
        din, dout = d[i - 1], d[i]
        th = np.arccos(np.clip(din @ dout, -1, 1))
        if th < 1e-6:
            arcs.append(None)
            continue
        lt = frac[i] * 0.5 * min(L[i - 1], L[i])         # tangent length
        r = lt / np.tan(th / 2)
        n = dout - (dout @ din) * din
        n /= np.linalg.norm(n)
        A0 = verts[i] - din * lt
        C = A0 + r * n
        arcs.append((A0, verts[i] + dout * lt, C, r, din, n, th, lt))
    pos, tan, kv = [], [], []
    for i in range(M):
        j = (i + 1) % M
        a_i, a_j = arcs[i], arcs[j]
        start = a_i[1] if a_i else verts[i]
        end = a_j[0] if a_j else verts[j]
        seg = end - start
        ls = np.linalg.norm(seg)
        if ls > 1e-9:
            m = max(int(ls / ds), 1)
            s = (np.arange(m) + 0.5) / m
            pos.append(start + s[:, None] * seg)
            tan.append(np.tile(seg / ls, (m, 1)))
            kv.append(np.zeros((m, 3)))
        if a_j:
            A0, B0, C, r, din, n, th, lt = a_j
            m = max(int(r * th / ds), 1)
            ph = (np.arange(m) + 0.5) / m * th
            pos.append(C + r * (-np.outer(np.cos(ph), n) + np.outer(np.sin(ph), din)))
            tan.append(np.outer(np.cos(ph), din) + np.outer(np.sin(ph), n))
            kv.append((np.outer(np.cos(ph), n) - np.outer(np.sin(ph), din)) / r)
    pos, tan, kv = np.vstack(pos), np.vstack(tan), np.vstack(kv)
    seglen = np.linalg.norm(np.roll(pos, -1, 0) - pos, axis=1)
    return pos, tan, kv, seglen


@njit(cache=True)
def build_path_fast(verts, frac, ds):
    """Same as build_path (numba); returns pos, tan, kv, seglen."""
    M = verts.shape[0]
    d = np.empty((M, 3))
    L = np.empty(M)
    for i in range(M):
        e = verts[(i + 1) % M] - verts[i]
        L[i] = np.sqrt(e @ e)
        d[i] = e / L[i]
    has = np.zeros(M, np.bool_)
    A0s, B0s, Cs, ns = np.zeros((M, 3)), np.zeros((M, 3)), np.zeros((M, 3)), np.zeros((M, 3))
    rs, ths = np.zeros(M), np.zeros(M)
    for i in range(M):
        din, dout = d[i - 1], d[i]
        th = np.arccos(min(max(din @ dout, -1.0), 1.0))
        if th < 1e-6:
            continue
        lt = frac[i] * 0.5 * min(L[i - 1], L[i])
        r = lt / np.tan(th / 2)
        n = dout - (dout @ din) * din
        n /= np.sqrt(n @ n)
        has[i] = True
        A0s[i] = verts[i] - din * lt
        B0s[i] = verts[i] + dout * lt
        Cs[i] = A0s[i] + r * n
        ns[i], rs[i], ths[i] = n, r, th
    # count samples
    cnt = 0
    for i in range(M):
        j = (i + 1) % M
        start = B0s[i] if has[i] else verts[i]
        end = A0s[j] if has[j] else verts[j]
        seg = end - start
        ls = np.sqrt(seg @ seg)
        if ls > 1e-9:
            cnt += max(int(ls / ds), 1)
        if has[j]:
            cnt += max(int(rs[j] * ths[j] / ds), 1)
    pos, tan, kv = np.zeros((cnt, 3)), np.zeros((cnt, 3)), np.zeros((cnt, 3))
    k = 0
    for i in range(M):
        j = (i + 1) % M
        start = B0s[i] if has[i] else verts[i]
        end = A0s[j] if has[j] else verts[j]
        seg = end - start
        ls = np.sqrt(seg @ seg)
        if ls > 1e-9:
            m = max(int(ls / ds), 1)
            for q in range(m):
                pos[k] = start + (q + 0.5) / m * seg
                tan[k] = seg / ls
                k += 1
        if has[j]:
            m = max(int(rs[j] * ths[j] / ds), 1)
            din, n, r = d[j - 1], ns[j], rs[j]
            for q in range(m):
                ph = (q + 0.5) / m * ths[j]
                c, s = np.cos(ph), np.sin(ph)
                pos[k] = Cs[j] + r * (-c * n + s * din)
                tan[k] = c * din + s * n
                kv[k] = (c * n - s * din) / r
                k += 1
    seglen = np.empty(cnt)
    for q in range(cnt):
        e = pos[(q + 1) % cnt] - pos[q]
        seglen[q] = np.sqrt(e @ e)
    return pos, tan, kv, seglen


@njit(cache=True)
def gate_dist_fast(pos, gates):
    out = np.empty(len(gates))
    for g in range(len(gates)):
        best = 1e18
        gx, gy, gz = gates[g, 0], gates[g, 1], gates[g, 2]
        for q in range(len(pos)):
            dx, dy, dz = pos[q, 0] - gx, pos[q, 1] - gy, pos[q, 2] - gz
            best = min(best, dx * dx + dy * dy + dz * dz)
        out[g] = np.sqrt(best)
    return out


@njit(cache=True)
def _passes(tan, kv, ds, umax, laps, A):
    n = len(tan)
    m = n * laps
    u = np.empty(m)
    for k in range(m):
        u[k] = umax[k % n]
    for _ in range(2):
        for k in range(m - 1):                           # forward: accelerate
            i = k % n
            wx, wy, wz = u[k] * kv[i, 0], u[k] * kv[i, 1], u[k] * kv[i, 2] + G
            tw = tan[i, 0] * wx + tan[i, 1] * wy + tan[i, 2] * wz
            D = max(tw * tw - (wx * wx + wy * wy + wz * wz) + A * A, 0.0)
            u[k + 1] = min(u[k + 1], u[k] + 2 * ds[i] * (-tw + np.sqrt(D)))
        for k in range(m - 1, 0, -1):                    # backward: brake
            i = k % n
            j = (k - 1) % n
            wx, wy, wz = u[k] * kv[i, 0], u[k] * kv[i, 1], u[k] * kv[i, 2] + G
            tw = tan[i, 0] * wx + tan[i, 1] * wy + tan[i, 2] * wz
            D = max(tw * tw - (wx * wx + wy * wy + wz * wz) + A * A, 0.0)
            u[k - 1] = min(u[k - 1], u[k] + 2 * ds[j] * (tw + np.sqrt(D)))
    return u


def topp(tan, kv, seglen, A, laps=3):
    """Time-optimal u = sigma^2 along a closed path under |a + g z| <= A (forward/backward passes)."""
    n = len(tan)
    tz = tan[:, 2]
    kap = np.linalg.norm(kv, axis=1)
    nz = np.where(kap > 0, kv[:, 2] / np.maximum(kap, 1e-12), 0.0)
    # u_max from D(u) = g^2 tz^2 - u^2 k^2 - 2 u k g nz - g^2 + A^2 >= 0
    c0 = G ** 2 * tz ** 2 - G ** 2 + A ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        umax = np.where(kap > 1e-9, (-kap * G * nz + np.sqrt((kap * G * nz) ** 2 + kap ** 2 * c0)) / kap ** 2, 1e9)
    u = _passes(np.ascontiguousarray(tan), np.ascontiguousarray(kv), seglen, umax, laps, A)
    u_lap = np.maximum(u[n:2 * n], 1e-4)
    lap = np.sum(seglen / np.sqrt(0.5 * (u_lap + np.roll(u_lap, -1))))
    return lap, np.sqrt(u_lap)


def gate_dist(pos, gates):
    return np.array([np.min(np.linalg.norm(pos - g, axis=1)) for g in gates])


def primitive_optimum(gates, A, tol, iters):
    M = len(gates)

    def unpack(z):
        return gates + z[:3 * M].reshape(M, 3), 1 / (1 + np.exp(-z[3 * M:]))

    def cost(z):
        verts, frac = unpack(z)
        pos, tan, kv, sl = build_path_fast(verts, frac, 0.05)
        lap, _ = topp(tan, kv, sl, A, laps=2)
        viol = np.maximum(gate_dist_fast(pos, gates) - tol, 0)
        return lap + 200 * np.sum(viol ** 2)

    z0 = np.concatenate([np.zeros(3 * M), np.full(M, 1.5)])   # vertices at the gates
    best = minimize(cost, z0, method="Powell", options={"maxiter": iters, "xtol": 1e-3, "ftol": 1e-5})
    verts, frac = unpack(best.x)
    pos, tan, kv, sl = build_path(verts, frac, ds=0.02)
    lap, sigma = topp(tan, kv, sl, A)
    return lap, verts, (pos, tan, kv, sl, sigma)
