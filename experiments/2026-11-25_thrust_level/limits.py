"""Rotor-thrust and body-rate limits written without dividing by the collective thrust (user, 2026-11-25).

Inputs (per point): thrust vector f = a + g e3 and its time derivatives j, s; heading angle psi and its time
derivatives (h = (cos psi, sin psi, 0): body y is perpendicular to h, body x = y x z).
Let c = |f|, Y = f x h, n = |Y|, X = Y x f (so z = f/c, y = Y/n, x = X/(n c)), A = f x j. Then (derived in the record)
    w_x = A.X / (n c^3),  w_y = A.Y / (n c^2),  w_z = [ (A.X)(f.h) + (f.f)^2 (Y.hd) ] / (n^2 c^3)
Common denominator D = n^2 c^3:  w = Om / D,  Om = (A.X n, A.Y n c, (A.X)(f.h) + (f.f)^2 (Y.hd)).
Time derivative without division: Dt*(E) = c n dE/dt, with dc/dt = (f.j)/c and dn/dt = (Y.Yd)/n multiplied out:
    Dt*(E) = c n (dE/d(f, j, h, hd) . (j, s, hd, hdd)) + n (f.j) dE/dc + c (Y.Yd) dE/dn
    w' = Wd / (c n D^2),   Wd = Dt*(Om) D - Om Dt*(D)
    tau = J w' + w x J w = [ J Wd + c n (Om x J Om) ] / (c n D^2)  =: Tn / Den,   Den = c n D^2 = c^7 n^5
    rotors u = Minv [m c; tau] = U / Den,  U = Minv [m c Den; Tn]
Limits (g >= 0), all polynomial in (f, j, s, h, hd, hdd, c, n) with c = sqrt(f.f), n = sqrt(Y.Y) (no division):
    U_i - T_min Den,  T_max Den - U_i  (i = 1..4);   w_max D -+ Om_x, w_max D -+ Om_y, w_z,max D -+ Om_z
Scaled by constants (reference sizes) only.
"""
import casadi as ca
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q = yaml.safe_load((ROOT / "third_party/rpg_time_optimal/quads/quad.yaml").read_text())
MASS, ARM, CQ = Q["mass"], Q["arm_length"], Q["torque_coeff"]
JD = np.array(Q["inertia"], float)
G = 9.81
TMAX, TMIN = Q["TWR_max"] * MASS * G / 4, Q["thrust_min"]
WXY, WZ = Q["omega_max_xy"], Q["omega_max_z"]
MIX = np.array([[1, 1, 1, 1], [ARM, -ARM, -ARM, ARM], [-ARM, -ARM, ARM, ARM], [CQ, -CQ, CQ, -CQ]], float)
MINV = np.linalg.inv(MIX)
CREF, NREF = 20.0, 20.0           # reference sizes for scaling the polynomial limits (constants)


def build():
    f, j, s = (ca.SX.sym(nm, 3) for nm in ("f", "j", "s"))
    ps = ca.SX.sym("psi", 3)                   # psi, psi_dot, psi_ddot
    c, n = ca.SX.sym("c"), ca.SX.sym("n")
    h = ca.vertcat(ca.cos(ps[0]), ca.sin(ps[0]), 0)
    hd = ca.vertcat(-ca.sin(ps[0]), ca.cos(ps[0]), 0) * ps[1]
    hdd = ca.vertcat(-ca.cos(ps[0]), -ca.sin(ps[0]), 0) * ps[1] ** 2 + ca.vertcat(-ca.sin(ps[0]), ca.cos(ps[0]), 0) * ps[2]
    Y = ca.cross(f, h)
    Yd = ca.cross(j, h) + ca.cross(f, hd)
    X = ca.cross(Y, f)
    A = ca.cross(f, j)
    AX = ca.dot(A, X)
    Om = ca.vertcat(AX * n, ca.dot(A, Y) * n * c, AX * ca.dot(f, h) + ca.dot(f, f) ** 2 * ca.dot(Y, hd))
    D = n ** 2 * c ** 3
    base = ca.vertcat(f, j, ps[0], ps[1])                     # variables whose time derivatives are known
    rate = ca.vertcat(j, s, ps[1], ps[2])

    def dts(E):                                               # c n dE/dt, multiplied out
        return c * n * ca.jtimes(E, base, rate) + n * ca.dot(f, j) * ca.jacobian(E, c) + c * ca.dot(Y, Yd) * ca.jacobian(E, n)
    Wd = dts(Om) * D - Om * dts(D)
    Jm = ca.DM(JD)
    Tn = ca.mtimes(Jm, Wd) + c * n * ca.cross(Om, ca.mtimes(Jm, Om))
    Den = c * n * D ** 2
    U = ca.mtimes(ca.DM(MINV), ca.vertcat(MASS * c * Den, Tn))
    sU = 1.0 / (TMAX * CREF ** 7 * NREF ** 5)
    sW = 1.0 / (WXY * NREF ** 2 * CREF ** 3)
    g = ca.vertcat((U - TMIN * Den) * sU, (TMAX * Den - U) * sU,
                   (WXY * D - Om[0]) * sW, (WXY * D + Om[0]) * sW,
                   (WXY * D - Om[1]) * sW, (WXY * D + Om[1]) * sW,
                   (WZ * D - Om[2]) * sW, (WZ * D + Om[2]) * sW)
    inp = ca.vertcat(f, j, s, ps)
    cn = ca.sqrt(ca.dot(f, f))
    nn = ca.sqrt(ca.dot(ca.cross(f, h), ca.cross(f, h)))
    gfun = ca.Function("limits", [inp], [ca.substitute(g, ca.vertcat(c, n), ca.vertcat(cn, nn))])
    # for checks only (divides): rotor thrusts and body rates
    chk = ca.Function("u_w", [inp], [ca.substitute(ca.vertcat(U / Den, Om / D), ca.vertcat(c, n), ca.vertcat(cn, nn))])
    return gfun, chk


LIM, UW = build()
NAMES = [f"rotor {i + 1} >= Tmin" for i in range(4)] + [f"rotor {i + 1} <= Tmax" for i in range(4)] + \
    ["w_x <= max", "w_x >= -max", "w_y <= max", "w_y >= -max", "w_z <= max", "w_z >= -max"]


def build_noheading():
    """Heading left free (user, 2026-11-25): body yaw rate w_z = 0; limits that hold for every heading.
        w = (f x j)/(f.f),  w' = N/(f.f)^2,  N = (f x s)(f.f) - 2 (f x j)(f.j),  tau = J_xy w'
        X rotors, worst heading: rotor i = m c/4 +- (tau_x +- tau_y)/(4 l), worst case sqrt(2)|tau|/(4 l).
    Written in physical units, no squaring, no scaling constants (user, 2026-11-25: the earlier squared form times
    1/(T_max^2 20^8) made a 60x torque violation read as -1e-8):
        rate:   w_max (f.f) - |f x j|                                     = (f.f)   (w_max - |w|)           [rad/s m^2/s^4]
        upper:  4 l (T_max - m c/4) (f.f)^2 - sqrt(2) J_xy |N|           = (f.f)^2 (4l * margin - sqrt2|tau|) [N m (m/s^2)^4]
        lower:  4 l (m c/4 - T_min) (f.f)^2 - sqrt(2) J_xy |N|
    The margins T_max - m c/4 >= 0 and m c/4 - T_min >= 0 follow (|N| >= 0), so they are not separate rows."""
    f, j, s = (ca.SX.sym(nm, 3) for nm in ("f", "j", "s"))
    ff = ca.dot(f, f)
    c = ca.sqrt(ff)
    A = ca.cross(f, j)
    N = ca.cross(f, s) * ff - 2 * A * ca.dot(f, j)
    jxy = JD[0, 0]
    tn = np.sqrt(2) * jxy * ca.norm_2(N)
    if FORM == "norm":
        g = ca.vertcat(WXY * ff - ca.norm_2(A),
                       4 * ARM * (TMAX - MASS * c / 4) * ff ** 2 - tn,
                       4 * ARM * (MASS * c / 4 - TMIN) * ff ** 2 - tn)
    else:   # squared (smooth where f x j = 0 or N = 0), no scaling constants; margins >= 0 then need their own rows
        up, lo = 4 * ARM * (TMAX - MASS * c / 4), 4 * ARM * (MASS * c / 4 - TMIN)
        t2 = 2 * jxy ** 2 * ca.dot(N, N)
        g = ca.vertcat((WXY * ff) ** 2 - ca.dot(A, A), up, (up * ff ** 2) ** 2 - t2, lo, (lo * ff ** 2) ** 2 - t2)
    inp = ca.vertcat(f, j, s)
    # check only (divides): |w| [rad/s], worst-heading torque needed and allowed [N m], c
    chk = ca.Function("w_tau", [inp], [ca.norm_2(A) / ff, tn / ff ** 2,
                                        4 * ARM * ca.fmin(TMAX - MASS * c / 4, MASS * c / 4 - TMIN), c])
    return ca.Function("limits_nh", [inp], [g]), chk


import os
FORM = os.environ.get("FORM", "squared")
LIM_NH, CHK_NH = build_noheading()
NAMES_NH = ["|w| <= 15", "rotors <= T_max (any heading)", "rotors >= T_min (any heading)"]
