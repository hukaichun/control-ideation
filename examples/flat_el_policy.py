"""Control law from one Lagrangian on the drone's flat state (p, v, a, j); input w = snap.

Design (agreed with the user):
  path   : lines and circles, each as two implicit functions phi(p) = 0
           line   (a, u):        phi = (n1.(p - a), n2.(p - a))
           circle (c, n, R):     phi = (n.(p - c), (|p - c|^2 - R^2) / (2R))
  speed  : energy H = 1/2 |v|^2 + V(p);  H = H* on the flight, so speed = sqrt(2 (H* - V(p)))
  L      = 1/2|phi''|^2 - 3 wp^2 |phi'|^2 + 1/2 wp^4 |phi|^2 + 1/2 mu^2,   mu = v_hat.(a + grad V)  (H' = |v| mu)
  Q      = grad(phi)^T (-4 wp phi''' - 4 wp^3 phi') + v_hat (-3 wH mu' - (3 wH^2 - V_tt) mu - wH^3 (H - H*)/|v|)
           (contains j; 0 on the flight; energy error poles (l + wH)^3)
  E-L    : L_p - D L_v + D^2 L_a = Q   ->   w = M^-1 [Q - L_p + D L_v - D(D L_a)],  M = L_aa
  input  : c'' = z.w + c |w x z|^2 ; angular acceleration from the part of w perpendicular to z; torque = J alpha + w x J w
Experiment-only choice (mine, not the user's design): V(p) on each piece is a polynomial in u.(p - a) (lines) or a
harmonic polynomial Re/Im (x + i y)^k in the circle plane (circles), fitted to a speed plan.
"""
import casadi as ca
import numpy as np

import path_pieces as pp

G = 9.81
EZ = np.array([0.0, 0.0, 1.0])


def _perp_basis(u):
    t = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    n1 = np.cross(u, t)
    n1 /= np.linalg.norm(n1)
    return n1, np.cross(u, n1)


def _harmonic(x, y, K):
    """1, Re z^k / r^k, Im z^k / r^k (z = x + i y, r = |z|), k = 1..K: cos k th, sin k th written without an angle;
    independent on the circle and constant along the radius."""
    r = (x * x + y * y) ** 0.5
    xn, yn = x / r, y / r                                   # divide by r^k: V constant along the radius
    out, re, im = [x * 0 + 1], x * 0 + 1, x * 0
    for _ in range(K):
        re, im = re * xn - im * yn, re * yn + im * xn
        out += [re, im]
    return out


def fit_V(pcs, sigma_fn, Hs, K=6, n=200, ridge=0.3):
    """Per piece, V(p) = polynomial in p (lines: in u.(p - a)/len; circles: in-plane (x, y)/R), least squares on
    V = Hs - 1/2 sigma^2 along the piece; end values pinned so V is continuous at joints."""
    coefs = []
    for P in pcs:
        sl = np.linspace(0, P["len"], n)
        Vt = Hs - 0.5 * sigma_fn(P["s0"] + sl) ** 2
        if P["kind"] == "line":
            xi = sl / P["len"]
            B = np.stack([xi ** k for k in range(K + 1)], 1)
        else:
            al = sl / P["R"]
            x, y = np.sin(al), -np.cos(al)                      # in-plane coords of the arc point / R
            B = np.stack(_harmonic(x, y, K), 1)
        w = np.ones(n)
        w[0] = w[-1] = 1e3                                      # pin the ends (continuity at joints)
        Aw = np.vstack([B * w[:, None], ridge * np.eye(B.shape[1])])   # small ridge: no huge cancelling coefficients
        coefs.append(np.linalg.lstsq(Aw, np.concatenate([Vt * w, np.zeros(B.shape[1])]), rcond=None)[0])
    return coefs


def _cheb(coef, t):
    Tk_1, Tk = 1, t
    out = coef[0] + coef[1] * t
    for k in range(2, coef.shape[0]):
        Tk_1, Tk = Tk, 2 * t * Tk - Tk_1
        out = out + coef[k] * Tk
    return out


def _el_w(p, v, a, j, phi, V, Hs, wp, wH, energy=True):
    """w = snap from L and Q for the given phi(p), V(p) (shared by every piece type).
    energy=False: only the phi terms (needs phi with 3 components, e.g. a point phi = p - p*)."""
    zero3 = ca.DM.zeros(3)
    pv, pva, pvaj = ca.vertcat(p, v), ca.vertcat(p, v, a), ca.vertcat(p, v, a, j)
    d1, d2 = ca.vertcat(v, a), ca.vertcat(v, a, j)
    H = 0.5 * ca.dot(v, v) + V
    ph1 = ca.jtimes(phi, p, v)
    ph2 = ca.jtimes(ph1, pv, d1)
    ph3 = ca.jtimes(ph2, pva, d2)
    eH = H - Hs
    sp_ = ca.sqrt(ca.dot(v, v))
    vh = v / sp_
    gV = ca.gradient(V, p)
    mu = ca.dot(vh, a + gV)                    # H' = |v| mu
    mu1 = ca.jtimes(mu, pva, d2)
    Vtt = ca.dot(vh, ca.mtimes(ca.hessian(V, p)[0], vh))
    Lag = (0.5 * ca.dot(ph2, ph2) - 3 * wp ** 2 * ca.dot(ph1, ph1) + 0.5 * wp ** 4 * ca.dot(phi, phi)
           + (0.5 * mu ** 2 if energy else 0))
    QH = -3 * wH * mu1 - (3 * wH ** 2 - Vtt) * mu - wH ** 3 * eH / sp_
    Q = ca.mtimes(ca.jacobian(phi, p).T, -4 * wp * ph3 - 4 * wp ** 3 * ph1) + (vh * QH if energy else 0)
    La, Lv, Lp = ca.gradient(Lag, a), ca.gradient(Lag, v), ca.gradient(Lag, p)
    M = ca.jacobian(La, a)
    DLv = ca.jtimes(Lv, pva, d2)
    DLa = ca.jtimes(La, pva, d2)
    DDLa = ca.jtimes(DLa, pvaj, ca.vertcat(v, a, j, zero3))
    w = ca.solve(M, Q - Lp + DLv - DDLa)
    return w, ca.vertcat(ca.norm_2(phi), eH, H)


class PointELPolicy:
    """Experiment: phi = p - p* (the path is one point), no energy term. Each axis of e = p - p* follows
    e4 + 4w e3 + 6w^2 e2 + 4w^3 e1 + w^4 e = 0 (e_k = k-th time derivative)."""

    def __init__(self, p_star, wp=16.0):
        p, v, a, j = (ca.SX.sym(nm, 3) for nm in "pvaj")
        w, info = _el_w(p, v, a, j, p - ca.DM(p_star), ca.SX(0), 0.0, wp, 1.0, energy=False)
        self.fun = [ca.Function("w_point", [p, v, a, j], [w, info])]

    def piece(self, p):
        return 0


class PolyELPolicy:
    """Same law on polynomial pieces (poly_plan spec): one function for every piece, the piece's numbers as input.
    Piece i: x = e1.(p - O), t = 2 x / l - 1, phi = (e2.(p - O) - f(t), e3.(p - O) - g(t)), V = h(t), H* = 0."""

    def __init__(self, spec, wp=16.0, wH=12.0, V_global=None):
        """V_global: optional casadi function V(p) used instead of the pieces' h (experiments)."""
        self.spec = spec
        nf, nh = len(spec[0]["f"]), len(spec[0]["h"])
        p, v, a, j = (ca.SX.sym(nm, 3) for nm in "pvaj")
        prm = ca.SX.sym("prm", 13 + 2 * nf + nh)
        O, e1, e2, e3, l = prm[0:3], prm[3:6], prm[6:9], prm[9:12], prm[12]
        fc, gc, hc = prm[13:13 + nf], prm[13 + nf:13 + 2 * nf], prm[13 + 2 * nf:]
        q = p - O
        t = 2 * ca.dot(e1, q) / l - 1
        phi = ca.vertcat(ca.dot(e2, q) - _cheb(fc, t), ca.dot(e3, q) - _cheb(gc, t))
        w, info = _el_w(p, v, a, j, phi, _cheb(hc, t) if V_global is None else V_global(p), 0.0, wp, wH)
        self.F = ca.Function("w_poly", [p, v, a, j, prm], [w, info])
        self.prm = [np.concatenate([s["O"], s["e1"], s["e2"], s["e3"], [s["l"]], s["f"], s["g"], s["h"]]) for s in spec]
        self.fun = [(lambda pr: (lambda p_, v_, a_, j_: self.F(p_, v_, a_, j_, pr)))(pr) for pr in self.prm]

    def piece(self, p):
        """Smallest |phi| among the pieces whose x-window (0 <= x <= l) contains p; if none is within 1 m (the outside
        of a joint lies beyond both neighbours' windows), the same with windows widened by 0.5 m; none: nearest piece
        start. No memory."""
        for slack in (0.0, 0.5):
            best, bd = None, np.inf
            for i, s in enumerate(self.spec):
                q = p - s["O"]
                x = q @ s["e1"]
                if not (-slack <= x <= s["l"] + slack):
                    continue
                t = 2 * x / s["l"] - 1
                dd = np.hypot(q @ s["e2"] - np.polynomial.chebyshev.chebval(t, s["f"]),
                              q @ s["e3"] - np.polynomial.chebyshev.chebval(t, s["g"]))
                if dd < bd:
                    best, bd = i, dd
            if best is not None and bd < 1.0:
                return best
        if best is None:
            best = int(np.argmin([np.linalg.norm(p - s["O"]) for s in self.spec]))
        return best


class FlatELPolicy:
    def __init__(self, path, sigma_fn=None, wp=8.0, wH=4.0, K=6, Hs=None, V_spec=None):
        """V_spec (from v_plan.plan_V): per piece V = sum c_k T_k(x), x = 2 (e.(p - o) - l0)/(l1 - l0) - 1, H* = 0.
        Without it, V is fitted to sigma_fn (experiment-only polynomial / harmonic space)."""
        self.L = path.length
        self.pcs = pp.pieces_of(path)
        if V_spec is None:
            sg = sigma_fn(np.linspace(0, self.L, 2000))
            self.Hs = 0.5 * sg.max() ** 2 + 1.0 if Hs is None else Hs
            self.Vc = fit_V(self.pcs, sigma_fn, self.Hs, K)
        else:
            self.Hs = 0.0
            self.Vc = [None] * len(self.pcs)
        p, v, a, j = (ca.SX.sym(nm, 3) for nm in "pvaj")
        self.fun = []
        for ip, (P, cf) in enumerate(zip(self.pcs, self.Vc)):
            if P["kind"] == "line":
                n1, n2 = _perp_basis(P["u"])
                phi = ca.vertcat(ca.dot(n1, p - P["a"]), ca.dot(n2, p - P["a"]))
                if cf is not None:
                    xi = ca.dot(P["u"], p - P["a"]) / P["len"]
                    V = sum(cf[k] * xi ** k for k in range(K + 1))
            else:
                q = p - P["c"]
                phi = ca.vertcat(ca.dot(P["n"], q), (ca.dot(q, q) - P["R"] ** 2) / (2 * P["R"]))
                if cf is not None:
                    x, y = ca.dot(q, P["e1"]) / P["R"], ca.dot(q, P["e2"]) / P["R"]
                    mons = _harmonic(x, y, K)
                    V = sum(cf[k] * mons[k] for k in range(len(mons)))
            if V_spec is not None:                                 # V = g(l(p)), Chebyshev in the projection
                sp_ = V_spec[ip]
                xx = 2 * (ca.dot(sp_["e"], p - sp_["o"]) - sp_["l0"]) / (sp_["l1"] - sp_["l0"]) - 1
                Tk_1, Tk = 1, xx
                V = sp_["coef"][0] + sp_["coef"][1] * xx
                for k in range(2, len(sp_["coef"])):
                    Tk_1, Tk = Tk, 2 * xx * Tk - Tk_1
                    V = V + sp_["coef"][k] * Tk
            w, info = _el_w(p, v, a, j, phi, V, self.Hs, wp, wH)
            self.fun.append(ca.Function("w", [p, v, a, j], [w, info]))

    def piece(self, p):
        return pp.piece_index(self.pcs, p)


class ThrustPlant:
    """Option B: the vehicle's collective thrust c has its own dynamics, c'' = u_c (states c, c'); torques direct.
    The controller reads c, c' from here."""

    def __init__(self, quad):
        self.quad = quad
        self.c, self.cd = 0.0, 0.0

    def apply(self, cdd, tau, dt):
        q = self.quad
        self.c += self.cd * dt + 0.5 * cdd * dt * dt
        self.cd += cdd * dt
        lo, hi = 4 * q.Tmin / q.m, q.A
        if self.c > hi or self.c < lo:
            self.c = float(np.clip(self.c, lo, hi))
            self.cd = 0.0
        u = np.clip(q.Mxinv @ np.concatenate([[q.m * self.c], tau]), q.Tmin, q.Tmax)
        self.c = u.sum() / q.m                               # what the motors really give
        return u


def fly(quad, pol, p0, v0, a0, j0, t_end, mode="A", dt=0.0005, ctrl_every=2, w_lim=15.0, plant=None, meas_a=True):
    """mode A: the controller integrates c, c' itself; mode B: c, c' are states of the vehicle (ThrustPlant).
    meas_a: the law's a is the measured acceleration (accelerometer: a = c z - g + d, external force d included);
    otherwise a = c z - g from the thrust model.
    Log: t, |phi|, H - H*, H, p(3), |w_xy|, motor_sat, c, c', piece, requested motor max / Tmax, min / Tmax
    (before clipping), body z-axis vertical component (1 level, < 0 upside down), body z-axis in world (3)."""
    import quad_sim as qs
    plant = quad if plant is None else plant
    fv = a0 + G * EZ
    c = np.linalg.norm(fv)
    z = fv / c
    cd = z @ j0
    q = qs.quat_from_z(z)
    x = np.concatenate([p0, v0, q, qs.rot(q).T @ (np.cross(z, j0) / c)])
    tp = ThrustPlant(quad)
    tp.c, tp.cd = c, cd
    n = int(t_end / dt)
    log = np.zeros((n, 18))
    ureq = np.full(4, quad.m * c / 4)
    u = np.full(4, quad.m * c / 4)
    info, i = np.zeros(3), pol.piece(p0)
    dtc = dt * ctrl_every
    for k in range(n):
        p, v, qq, wb = x[0:3], x[3:6], x[6:10], x[10:13]
        if k % ctrl_every == 0:
            if mode == "B":
                c, cd = tp.c, tp.cd
            R = qs.rot(qq)
            z = R[:, 2]
            w_w = R @ wb
            wz = np.cross(w_w, z)
            a = plant.deriv(x, u)[3:6] if meas_a else c * z - G * EZ
            j = cd * z + c * wz
            i = pol.piece(p)
            w, info = (np.array(y).ravel() for y in pol.fun[i](p, v, a, j))
            cdd = z @ w + c * (wz @ wz)
            rest = w - 2 * cd * wz - c * np.cross(w_w, wz)
            perp = rest - (z @ rest) * z
            alpha_b = R.T @ (np.cross(z, perp) / max(c, 1.0))
            alpha_b[2] = -20.0 * wb[2]
            if w_lim is not None:
                nw = np.linalg.norm(wb[:2])
                if nw > w_lim and alpha_b[:2] @ wb[:2] > 0:
                    alpha_b[:2] -= (alpha_b[:2] @ wb[:2]) / nw ** 2 * wb[:2]
            tau = quad.J @ alpha_b + np.cross(wb, quad.J @ wb)
            if mode == "B":
                c_req = tp.c + tp.cd * dtc + 0.5 * cdd * dtc * dtc
                ureq = quad.Mxinv @ np.concatenate([[quad.m * c_req], tau])
                u = tp.apply(cdd, tau, dtc)
            else:                                            # A: controller's own integrator (does not see clipping)
                c += cd * dtc + 0.5 * cdd * dtc * dtc
                cd += cdd * dtc
                c = float(np.clip(c, 0.0, quad.A))
                u = np.clip(quad.Mxinv @ np.concatenate([[quad.m * c], tau]), quad.Tmin, quad.Tmax)
        x = plant.step(x, u, dt)
        sat = np.any(u >= quad.Tmax - 1e-9) or np.any(u <= quad.Tmin + 1e-9)
        log[k] = (k * dt, info[0], info[1], info[2], *x[0:3], np.linalg.norm(x[10:12]), sat, c, cd, i,
                  ureq.max() / quad.Tmax, ureq.min() / quad.Tmax, qs.rot(x[6:10])[2, 2], *qs.rot(x[6:10])[:, 2])
        if not np.all(np.isfinite(x)) or info[0] > 20:
            return log[:k + 1]
    return log
