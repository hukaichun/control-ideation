"""Run a Policy against PX4 SITL in offboard mode and log every control step to CSV."""
import asyncio
import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

from .policy import Observation, Policy

Reference = Callable[[float], np.ndarray]  # t -> reference position [n, e, d]

LOG_FIELDS = [
    "t", "n", "e", "d", "vn", "ve", "vd", "roll", "pitch", "yaw",
    "ref_n", "ref_e", "ref_d", "cmd_vn", "cmd_ve", "cmd_vd", "cmd_yaw",
]


class _State:
    """Latest telemetry, filled in by background stream tasks."""

    def __init__(self):
        self.pos = None
        self.vel = None
        self.rpy = None

    def ready(self) -> bool:
        return self.pos is not None and self.rpy is not None


async def _watch_posvel(drone: System, st: _State):
    async for pv in drone.telemetry.position_velocity_ned():
        p, v = pv.position, pv.velocity
        st.pos = np.array([p.north_m, p.east_m, p.down_m])
        st.vel = np.array([v.north_m_s, v.east_m_s, v.down_m_s])


async def _watch_attitude(drone: System, st: _State):
    async for a in drone.telemetry.attitude_euler():
        st.rpy = np.array([a.roll_deg, a.pitch_deg, a.yaw_deg])


async def run_episode(
    policy: Policy,
    reference: Reference,
    duration: float,
    log_path: Path,
    rate_hz: float = 50.0,
    address: str = "udpin://0.0.0.0:14540",
) -> Path:
    drone = System()
    await drone.connect(system_address=address)

    print("waiting for vehicle...")
    async for cs in drone.core.connection_state():
        if cs.is_connected:
            break
    async for h in drone.telemetry.health():
        if h.is_local_position_ok and h.is_home_position_ok and h.is_armable:
            break
    print("vehicle ready")

    for setter in (
        drone.telemetry.set_rate_position_velocity_ned,
        drone.telemetry.set_rate_attitude_euler,
    ):
        try:
            await setter(rate_hz)
        except Exception as e:  # rate requests are best-effort
            print(f"warning: could not set telemetry rate: {e}")

    st = _State()
    tasks = [
        asyncio.create_task(_watch_posvel(drone, st)),
        asyncio.create_task(_watch_attitude(drone, st)),
    ]
    while not st.ready():
        await asyncio.sleep(0.05)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        await drone.action.arm()
        # PX4 requires a setpoint stream before it will accept offboard mode.
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, float(st.rpy[2])))
        try:
            await drone.offboard.start()
        except OffboardError as e:
            await drone.action.disarm()
            raise RuntimeError(f"offboard start failed: {e._result.result}") from e

        print(f"policy in control for {duration:.0f}s -> {log_path}")
        dt = 1.0 / rate_hz
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(LOG_FIELDS)
            t0 = time.monotonic()
            k = 0
            while (t := time.monotonic() - t0) < duration:
                ref = np.asarray(reference(t), dtype=float)
                obs = Observation(t=t, pos=st.pos, vel=st.vel, rpy_deg=st.rpy, ref_pos=ref)
                act = policy.act(obs)
                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(*(float(x) for x in act.vel), float(act.yaw_deg))
                )
                w.writerow([f"{x:.4f}" for x in (t, *obs.pos, *obs.vel, *obs.rpy_deg, *ref, *act.vel, act.yaw_deg)])
                k += 1
                await asyncio.sleep(max(0.0, t0 + k * dt - time.monotonic()))
    finally:
        print("landing")
        try:
            await drone.offboard.stop()
        except Exception:
            pass
        await drone.action.land()
        async for in_air in drone.telemetry.in_air():
            if not in_air:
                break
        for task in tasks:
            task.cancel()
    print("done")
    return log_path
