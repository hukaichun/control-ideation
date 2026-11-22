"""Policy interface: observation in, velocity setpoint out.

All vectors are in the local NED frame (north, east, down), SI units. Yaw in degrees.
Swap `PPositionPolicy` for your own class to try a different controller.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Observation:
    t: float              # seconds since the policy took over
    pos: np.ndarray       # [n, e, d] m
    vel: np.ndarray       # [vn, ve, vd] m/s
    rpy_deg: np.ndarray   # [roll, pitch, yaw] deg
    ref_pos: np.ndarray   # reference position [n, e, d] m


@dataclass
class Action:
    vel: np.ndarray       # commanded velocity [vn, ve, vd] m/s
    yaw_deg: float = 0.0


class Policy(Protocol):
    def act(self, obs: Observation) -> Action: ...


class PPositionPolicy:
    """Baseline: proportional position controller with a speed limit."""

    def __init__(self, kp: float = 1.0, v_max: float = 3.0):
        self.kp = kp
        self.v_max = v_max

    def act(self, obs: Observation) -> Action:
        v = self.kp * (obs.ref_pos - obs.pos)
        speed = np.linalg.norm(v)
        if speed > self.v_max:
            v *= self.v_max / speed
        return Action(vel=v)
