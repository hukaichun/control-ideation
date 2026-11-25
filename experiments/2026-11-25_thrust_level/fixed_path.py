"""Speed (and heading) on a fixed path, thrust-level limits without division by the thrust (limits.py).

Path: the saved CPC lap of track k (test path), periodic quintic B-spline p(u), u in [0, 1) ~ arc length.
Variables: w(u) = dt/du (periodic degree-4 B-spline, knots every HK m) and heading angle psi(u) (same knots).
d/dt = (1/w) d/du (w >= w_min > 0: w = 1/speed in time per u, never near 0):
    f = d2p/dt2 + g e3,  j = df/dt,  s = dj/dt,  psi_dot, psi_ddot   (casadi chain rule, exact)
Objective: T = int w du = mean of w's control points (periodic uniform B-spline), linear and exact.
Constraints at the samples: the 14 division-free limits (rotors in [T_min, T_max], |w_x|, |w_y| <= 15, |w_z| <= 0.3);
    w control points >= w_min.
Solver: IPOPT (my own gradient-projection solver was removed, user 2026-11-25: stalled at 20.6 s, see RECORD).
Start: constant 2 m/s, heading 0. Checked afterwards on a 4x finer grid (same limits).

    cd experiments/2026-11-25_thrust_level
    SOLVER=ipopt uv run --with casadi --with pyyaml --with scipy --with numpy python fixed_path.py [track]
"""
import os
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.interpolate import BSpline

import limits as LM

HERE = Path(__file__).resolve().parent
ROOT = LM.ROOT
TRACK = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SOLVER = os.environ.get("SOLVER", "ipopt")
HK = float(os.environ.get("HK", 1.0))
SPS = int(os.environ.get("SPS", 4))
DEGW = 4
V_START = 2.0
VMAX = 40.0
MAX_IT = int(os.environ.get("MAX_IT", 2000))
WIND = int(os.environ.get("WIND", 0))          # whole yaw turns per lap: psi(u) = periodic spline + 2 pi WIND u


def prows(u, n, deg, d):
    """Periodic uniform B-spline (n control points, degree deg): d-th derivative rows at u."""
    card = BSpline.basis_element(np.arange(deg + 2), extrapolate=False)
    fn = card.derivative(d) if d else card
    x = np.asarray(u) * n
    out = np.zeros((len(x), n))
    i0 = np.floor(x).astype(int)
    for k in range(deg + 1):
        jj = i0 - k
        out[np.arange(len(x)), jj % n] += np.nan_to_num(fn(x - jj)) * n ** d
    return out


def dense_cpc(k):
    d = np.load(ROOT / "logs" / f"cpc_traj_tol0.24_{k}.npz")
    x, T, N = d["x"], d["T"], int(d["N"])
    dts = np.repeat(T / N, N)
    P = []
    M = len(x)
    for i in range(M):
        p0, v0, p1, v1, h = x[i, :3], x[i, 3:6], x[(i + 1) % M, :3], x[(i + 1) % M, 3:6], dts[i]
        for s in np.arange(20) / 20:
            P.append((2 * s ** 3 - 3 * s ** 2 + 1) * p0 + (s ** 3 - 2 * s ** 2 + s) * h * v0
                     + (-2 * s ** 3 + 3 * s ** 2) * p1 + (s ** 3 - s ** 2) * h * v1)
    return np.array(P), float(T.sum())


def chain():
    """f, j, s, psi_dot, psi_ddot from u-derivatives of p (0..4), w (0..3), psi (0..2)."""
    Xs = [ca.SX.sym(f"p{k}", 3) for k in range(5)]
    Ws = [ca.SX.sym(f"w{k}") for k in range(4)]
    Ps = [ca.SX.sym(f"s{k}") for k in range(4)]
    base = ca.vertcat(*Xs[:4], *Ws[:3], *Ps[:3])
    rate = ca.vertcat(*Xs[1:5], *Ws[1:4], *Ps[1:4])

    def dt(E):
        return ca.jtimes(E, base, rate) / Ws[0]
    vel = dt(Xs[0])
    acc = dt(vel)
    f = acc + ca.DM([0, 0, LM.G])
    j = dt(f)
    s = dt(j)
    psd = dt(Ps[0])
    psdd = dt(psd)
    inp = ca.vertcat(*Xs[1:5], *Ws, *Ps[:3])
    out = ca.vertcat(f, j, s, Ps[0], psd, psdd)
    return ca.Function("chain", [inp], [ca.substitute(out, ca.vertcat(Ps[3]), ca.DM(0))])


CHAIN = chain()


class Problem:
    def __init__(self, k):
        P, self.lap_cpc = dense_cpc(k)
        sl = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        s_ = np.r_[0, np.cumsum(sl)[:-1]]
        self.L = sl.sum()
        nref = int(round(self.L / 0.5))
        self.Cp = np.linalg.lstsq(prows(s_ / self.L, nref, 5, 0), P, rcond=None)[0]
        self.fit = float(np.abs(prows(s_ / self.L, nref, 5, 0) @ self.Cp - P).max())
        self.n = int(round(self.L / HK))
        self.M = SPS * max(nref, self.n)
        self.wmin = self.L / VMAX
        self.set_samples(self.M)

    def set_samples(self, M):
        u = (np.arange(M) + 0.5) / M
        nref = len(self.Cp)
        self.Pd = np.vstack([(prows(u, nref, 5, d) @ self.Cp).T for d in range(1, 5)])      # 12 x M
        self.Bw = [prows(u, self.n, DEGW, d) for d in range(4)]
        self.u = u
        self.Ms = M

    def expr(self, z):
        """Limits (14 x Ms) and objective for decision vector z = [w coefficients, psi coefficients] (casadi)."""
        n = self.n
        wc, pc = z[:n], z[n:]
        W = [ca.mtimes(ca.DM(B), wc).T for B in self.Bw]
        Ps = [ca.mtimes(ca.DM(B), pc).T for B in self.Bw[:3]]
        Ps[0] = Ps[0] + ca.DM(2 * np.pi * WIND * self.u).T            # heading may turn WIND whole times per lap
        Ps[1] = Ps[1] + 2 * np.pi * WIND
        inp = ca.vertcat(ca.DM(self.Pd), *W, *Ps)
        fjs = CHAIN.map(self.Ms)(inp)
        g = LM.LIM.map(self.Ms)(fjs)
        T = ca.sum1(wc) / n
        return g, T


def solve_ipopt(Pb, z0):
    z = ca.SX.sym("z", 2 * Pb.n)
    g, T = Pb.expr(z)
    S = ca.nlpsol("s", "ipopt", {"x": z, "f": T, "g": ca.vec(g)},
                  {"print_time": False, "ipopt": {"print_level": 5, "max_iter": MAX_IT, "print_frequency_iter": 20}})
    lbx = np.r_[np.full(Pb.n, Pb.wmin), np.full(Pb.n, -np.inf)]
    t0 = time.perf_counter()
    sol = S(x0=z0, lbx=lbx, ubx=np.inf, lbg=0, ubg=np.inf)
    return np.array(sol["x"]).ravel(), dict(status=S.stats()["return_status"], it=S.stats()["iter_count"],
                                              t=time.perf_counter() - t0)


def check(Pb, z, factor=4):
    M0 = Pb.Ms
    Pb.set_samples(factor * M0)
    zz = ca.SX.sym("z", 2 * Pb.n)
    g, T = Pb.expr(zz)
    gv = np.array(ca.Function("c", [zz], [g])(z))
    Pb.set_samples(M0)
    # rotor thrusts and rates (division, check only) on the fine grid
    i = int(np.argmin(gv.min(0)))
    return float(gv.min()), LM.NAMES[int(np.argmin(gv.min(1)))]


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    Pb = Problem(TRACK)
    z0 = np.r_[np.full(Pb.n, Pb.L / V_START), np.zeros(Pb.n)]
    log(f"track {TRACK}, solver {SOLVER}: path {Pb.L:.1f} m (fit {Pb.fit * 1000:.2f} mm), {Pb.n} knots -> {2 * Pb.n} "
        f"variables, {Pb.Ms} samples x 14 limits, start {Pb.L / V_START:.2f} s (2 m/s), CPC {Pb.lap_cpc:.3f} s")
    zs, st = solve_ipopt(Pb, z0)
    T = float(np.mean(zs[:Pb.n]))
    gmin, worst = check(Pb, zs)
    log(f"  result: {st['status']}, {st['it']} iterations, {st['t']:.1f} s | lap {T:.4f} s ({100 * (T / Pb.lap_cpc - 1):+.1f}% vs CPC) "
        f"| speed range {Pb.L / zs[:Pb.n].max():.1f}-{Pb.L / zs[:Pb.n].min():.1f} m/s (control points) | "
        f"4x finer grid: min scaled limit {gmin:+.2e} ({worst})")
    (HERE / "results").mkdir(exist_ok=True)
    np.savez(HERE / "results" / f"fixed_{SOLVER}_{TRACK}.npz", z=zs, Cp=Pb.Cp, n=Pb.n, L=Pb.L)
    with open(HERE / "results" / f"fixed_{SOLVER}.log", "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
