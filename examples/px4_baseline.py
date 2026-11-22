"""PX4's default multicopter cascade (v1.17 ext-policy fork: PositionControl.cpp, AttitudeControl.cpp,
rate_control.cpp) ported to the same full quad model (quad_sim.py), flying the CPC time-optimal trajectory
(logs/quad_time_optimal.csv) given as offboard trajectory setpoints (position + velocity + acceleration feed-forward,
yaw from the CPC attitude).

Kept from PX4 defaults: MPC_XY_P 0.95, MPC_Z_P 1, MPC_XY_VEL_P/I/D_ACC 1.8/0.4/0.2, MPC_Z_VEL_P/I/D_ACC 4/2/0,
MPC_ACC_DECOUPLE 1, MPC_THR_MIN 0.12, MPC_THR_MAX 1, MPC_THR_XY_MARG 0.3, MPC_VELD_LP 5 Hz, MC_ROLL/PITCH/YAW_P 4/4/2.8,
MC_YAW_WEIGHT 0.4, thrust prioritisation and tilt limiting as in the source; hover thrust = true value (MPC_USE_HTE).
Limits: 'default' = MPC_TILTMAX_AIR 45 deg, MPC_XY_VEL_MAX 12, MPC_Z_VEL_MAX_UP/DN 3/1.5 m/s, MC_*RATE_MAX 220/220/200
deg/s; 'opened' = tilt 89 deg, no velocity limit, rates at the airframe limit (15 rad/s xy, 0.3 z).
Rate loop: PX4 PID structure, but its gains are airframe tuning (normalised torque), so it is set to the same
fixed bandwidth (40 rad/s rate-error -> angular acceleration).
"""
import numpy as np

import quad_sim as qs
import quad_time_optimal as qto

G = 9.81
EZ = np.array([0.0, 0.0, 1.0])


def cpc_trajectory(path="../logs/quad_time_optimal.csv", n_seg=7):
    """CPC nodes + per-segment step recovered from the dynamics (x_{k+1} = RK4(x_k, u_k, dt))."""
    D = np.loadtxt(path, delimiter=",", skiprows=1)
    f, _ = qto.quad_model()
    N = len(D) // n_seg
    dts = []
    for i in range(n_seg):
        seg = D[i * N:(i + 1) * N + 1] if i < n_seg - 1 else np.vstack([D[i * N:], D[:1]])
        best = None
        for dt in np.linspace(0.005, 0.08, 301):
            err = 0.0
            for k in range(len(seg) - 1):
                x = seg[k, :13]
                xn = np.array(qto.rk4(f, x, seg[k, 13:], dt)).ravel()
                err += np.sum((xn[:6] - seg[k + 1, :6]) ** 2)
            if best is None or err < best[0]:
                best = (err, dt)
        lo, hi = best[1] - 0.00025, best[1] + 0.00025              # refine
        for dt in np.linspace(lo, hi, 51):
            err = sum(np.sum((np.array(qto.rk4(f, seg[k, :13], seg[k, 13:], dt)).ravel()[:6] - seg[k + 1, :6]) ** 2)
                      for k in range(len(seg) - 1))
            if err < best[0]:
                best = (err, dt)
        dts += [best[1]] * N
    t = np.concatenate([[0], np.cumsum(dts)])[:-1]
    lap = float(np.sum(dts))
    P, V = D[:, 0:3], D[:, 3:6]
    R = [qs.rot(q) for q in D[:, 6:10]]
    Acc = np.array([Ri[:, 2] * Di[13:].sum() / 0.85 for Ri, Di in zip(R, D)]) - G * EZ
    yaw = np.array([np.arctan2(Ri[1, 0], Ri[0, 0]) for Ri in R])
    return dict(t=t, lap=lap, P=P, V=V, A=Acc, yaw=np.unwrap(yaw), X=D)


class Setpoint:
    """Time-scaled CPC trajectory (scale lam >= 1 slows it down): cubic Hermite in p, linear in a, periodic."""

    def __init__(self, tr, lam):
        self.tr, self.lam, self.T = tr, lam, tr["lap"] * lam

    def __call__(self, t):
        tr, lam = self.tr, self.lam
        tau = (t / lam) % tr["lap"]
        tt = np.append(tr["t"], tr["lap"])
        k = int(np.searchsorted(tt, tau, side="right") - 1)
        k1 = (k + 1) % len(tr["t"])
        h = tt[k + 1] - tt[k]
        u = (tau - tt[k]) / h
        p0, p1, v0, v1 = tr["P"][k], tr["P"][k1], tr["V"][k], tr["V"][k1]
        h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
        p = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
        v = (1 - u) * v0 + u * v1
        a = (1 - u) * tr["A"][k] + u * tr["A"][k1]
        dyaw = tr["yaw"][k1] - tr["yaw"][k]
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        yaw = tr["yaw"][k] + u * dyaw
        return p, v / lam, a / lam ** 2, yaw


def quat_from_R(R):
    w = np.sqrt(max(1 + R[0, 0] + R[1, 1] + R[2, 2], 1e-12)) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def quat_between(a, b):
    """Shortest rotation taking unit vector a to b."""
    c = np.cross(a, b)
    d = a @ b
    q = np.array([1 + d, *c])
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([0.0, 1.0, 0, 0])


def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


class PX4:
    def __init__(self, quad, limits="default"):
        self.quad = quad
        self.h = quad.m * G / (4 * quad.Tmax)                       # hover thrust (normalised)
        self.Kp = np.array([0.95, 0.95, 1.0])
        self.Kv = np.array([1.8, 1.8, 4.0])
        self.Ki = np.array([0.4, 0.4, 2.0])
        self.Kd = np.array([0.2, 0.2, 0.0])
        self.att_P = np.array([4.0, 4.0, 2.8])
        self.yaw_w = 0.4
        self.thr_min, self.thr_max, self.xy_marg = 0.12, 1.0, 0.3
        if limits == "default":
            self.tilt, self.vxy, self.vup, self.vdn = np.radians(45), 12.0, 3.0, 1.5
            self.rate_lim = np.radians([220.0, 220.0, 200.0])
        else:
            self.tilt, self.vxy, self.vup, self.vdn = np.radians(89), 1e9, 1e9, 1e9
            self.rate_lim = np.array([quad.wxy, quad.wxy, quad.wz])
        self.k_rate = 40.0
        self.reset()

    def reset(self):
        self.vint = np.zeros(3)
        self.acc_lp = np.zeros(3)
        self.v_prev = None
        self.thr = np.array([0, 0, self.h])
        self.qd = np.array([1.0, 0, 0, 0])

    def position_velocity(self, p, v, sp, dt):
        ps, vs, as_, yaw = sp
        vel_p = (ps - p) * self.Kp
        vsp = vs + vel_p
        # constrainXY: prioritise the position-error part over the feed-forward
        ff = (vsp - vel_p)[:2]
        if np.linalg.norm(vsp[:2]) > self.vxy:
            if np.linalg.norm(vel_p[:2]) >= self.vxy:
                vsp[:2] = vel_p[:2] / np.linalg.norm(vel_p[:2]) * self.vxy
            else:
                a_, b_ = ff @ ff, 2 * vel_p[:2] @ ff
                c_ = vel_p[:2] @ vel_p[:2] - self.vxy ** 2
                sc = (-b_ + np.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
                vsp[:2] = vel_p[:2] + sc * ff
        vsp[2] = np.clip(vsp[2], -self.vdn, self.vup)                # ENU: up positive
        # velocity PID (D on the low-passed acceleration, MPC_VELD_LP = 5 Hz)
        if self.v_prev is not None:
            a_meas = (v - self.v_prev) / dt
            al = dt / (dt + 1 / (2 * np.pi * 5.0))
            self.acc_lp += al * (a_meas - self.acc_lp)
        self.v_prev = v.copy()
        verr = vsp - v
        acc_sp = as_ + verr * self.Kv + self.vint - self.acc_lp * self.Kd
        # acceleration -> thrust (MPC_ACC_DECOUPLE = 1: tilt from horizontal acceleration and g only)
        bz = np.array([acc_sp[0], acc_sp[1], G])
        bz /= np.linalg.norm(bz)
        ang = min(np.arccos(np.clip(bz[2], -1, 1)), self.tilt)
        rej = bz - bz[2] * EZ
        rej = rej / np.linalg.norm(rej) if np.linalg.norm(rej) > 1e-6 else np.array([1.0, 0, 0])
        bz = np.cos(ang) * EZ + np.sin(ang) * rej
        thr_up = (acc_sp[2] + G) * self.h / G
        coll = max(thr_up / bz[2], self.thr_min)
        thr = bz * coll
        # anti-windup (vertical) and vertical-priority saturation with horizontal margin
        if (thr[2] <= self.thr_min and verr[2] <= 0) or (thr[2] >= self.thr_max and verr[2] >= 0):
            verr[2] = 0.0
        txy = np.linalg.norm(thr[:2])
        thr[2] = min(thr[2], np.sqrt(self.thr_max ** 2 - min(txy, self.xy_marg) ** 2))
        txy_max = np.sqrt(max(self.thr_max ** 2 - thr[2] ** 2, 0.0))
        if txy > txy_max:
            thr[:2] *= txy_max / txy
        prod = thr[:2] * G / self.h
        if acc_sp[:2] @ acc_sp[:2] > prod @ prod:
            verr[:2] -= 2 / self.Kv[0] * (acc_sp[:2] - prod)
        self.vint += verr * self.Ki * dt
        self.vint[2] = np.clip(self.vint[2], -G, G)
        self.thr = thr
        # thrustToAttitude: body z along thrust, yaw setpoint
        z = thr / max(np.linalg.norm(thr), 1e-6)
        yc = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        x = np.cross(yc, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self.qd = quat_from_R(np.column_stack([x, y, z]))

    def attitude(self, q):
        R = qs.rot(q)
        ez, ezd = R[:, 2], qs.rot(self.qd)[:, 2]
        qd_red = qs.quat_mul(quat_between(ez, ezd), q)
        qdy = qs.quat_mul(qinv(qd_red), self.qd)
        if qdy[0] < 0:
            qdy = -qdy
        qd = qs.quat_mul(qd_red, np.array([np.cos(self.yaw_w * np.arccos(np.clip(qdy[0], -1, 1))), 0, 0,
                                           np.sin(self.yaw_w * np.arcsin(np.clip(qdy[3], -1, 1)))]))
        qe = qs.quat_mul(qinv(q), qd)
        if qe[0] < 0:
            qe = -qe
        rate_sp = 2 * qe[1:] * self.att_P
        return np.clip(rate_sp, -self.rate_lim, self.rate_lim)

    def rate(self, w, rate_sp):
        quad = self.quad
        tau = quad.J @ (self.k_rate * (rate_sp - w)) + np.cross(w, quad.J @ w)
        T = np.linalg.norm(self.thr) * 4 * quad.Tmax
        return np.clip(quad.Mxinv @ np.concatenate([[T], tau]), quad.Tmin, quad.Tmax)


def fly(quad, ctl, sp, t_end, dt=0.0005):
    """Rate loop 1 kHz, attitude 250 Hz, position/velocity 100 Hz. Log: t, p(3), p_sp(3), |w_xy|, sat."""
    p0, v0, a0, yaw0 = sp(0.0)
    z = (a0 + G * EZ) / np.linalg.norm(a0 + G * EZ)
    x = np.concatenate([p0, v0, qs.quat_from_z(z), np.zeros(3)])
    ctl.reset()
    n = int(t_end / dt)
    log = np.zeros((n, 9))
    u = np.full(4, quad.m * G / 4)
    rate_sp = np.zeros(3)
    for k in range(n):
        t = k * dt
        p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
        if k % 20 == 0:
            s = sp(t)
            ctl.position_velocity(p, v, s, 20 * dt)
        if k % 8 == 0:
            rate_sp = ctl.attitude(q)
        if k % 2 == 0:
            u = ctl.rate(w, rate_sp)
        x = quad.step(x, u, dt)
        sat = np.any(u >= quad.Tmax - 1e-9) or np.any(u <= quad.Tmin + 1e-9)
        log[k] = (t, *x[0:3], *s[0], np.linalg.norm(x[10:12]), sat)
        if not np.all(np.isfinite(x)) or np.linalg.norm(x[0:3] - s[0]) > 30:
            return log[:k + 1]
    return log


def evaluate(log, sp, gates, lap_T):
    """Second lap: worst gate miss and max tracking error."""
    m = (log[:, 0] >= lap_T) & (log[:, 0] < 2 * lap_T)
    if log[-1, 0] < 2 * lap_T - 1e-3:
        return np.inf, np.inf
    L = log[m]
    miss = max(np.min(np.linalg.norm(L[:, 1:4] - g, axis=1)) for g in gates)
    return miss, np.linalg.norm(L[:, 1:4] - L[:, 4:7], axis=1).max()
