"""SITL experiment for the PX4 ext_policy module: upload policies over MAVLink FTP, switch between
PX4 and the uploaded policy, hot-swap policies in flight, and check that a corrupted image is rejected.

Needs the simulator running from the fork:  PX4_SRC=PX4-Autopilot scripts/sim_up.sh

    uv run examples/ext_policy_flight.py build/policies/hold.xpol build/policies/circle.xpol
"""
import argparse
import asyncio
import csv
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mavsdk import System

SIM_NAME = "px4-sitl"
PX4_CLIENT = "build/px4_sitl_default/bin/px4-ext_policy"


async def px4_cmd(*args) -> str:
    """Run an ext_policy shell command on the vehicle (the shell's cwd is the FTP root)."""
    p = await asyncio.create_subprocess_exec(
        "docker", "exec", SIM_NAME, PX4_CLIENT, *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out = (await p.communicate())[0].decode().strip()
    print(f"  $ ext_policy {' '.join(args)}\n    " + out.replace("\n", "\n    "))
    return out


async def upload(drone: System, path: Path):
    async for _ in drone.ftp.upload(str(path), "/"):
        pass
    print(f"  uploaded {path.name} ({path.stat().st_size} bytes) over MAVLink FTP")


async def main(hold_img: Path, circle_img: Path, out: Path):
    bad_img = circle_img.with_name("circle_bad.xpol")
    data = bytearray(circle_img.read_bytes())
    data[100] ^= 0xFF                         # one corrupted byte in the code
    bad_img.write_bytes(data)

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    async for cs in drone.core.connection_state():
        if cs.is_connected:
            break
    async for h in drone.telemetry.health():
        if h.is_local_position_ok and h.is_home_position_ok and h.is_armable:
            break
    print("vehicle ready")

    rows, phase = [], {"name": "ground"}
    t0 = time.monotonic()

    async def log_pos():
        mode = {"m": ""}

        async def watch_mode():
            async for m in drone.telemetry.flight_mode():
                mode["m"] = str(m)

        mode_task = asyncio.create_task(watch_mode())
        try:
            await _log_posvel(mode)
        finally:
            mode_task.cancel()

    async def _log_posvel(mode):
        async for pv in drone.telemetry.position_velocity_ned():
            p = pv.position
            rows.append((time.monotonic() - t0, p.north_m, p.east_m, p.down_m, mode["m"], phase["name"]))

    logger = asyncio.create_task(log_pos())

    async def hold_phase(name, seconds):
        phase["name"] = name
        print(f"[{time.monotonic() - t0:5.1f}s] {name} for {seconds}s")
        await asyncio.sleep(seconds)

    for img in (hold_img, circle_img, bad_img):
        await upload(drone, img)

    await drone.action.set_takeoff_altitude(5.0)
    await drone.action.arm()
    await drone.action.takeoff()
    phase["name"] = "takeoff"
    async for pos in drone.telemetry.position():
        if pos.relative_altitude_m > 4.7:
            break
    await asyncio.sleep(3)

    await hold_phase("PX4 Hold", 6)

    await px4_cmd("load", hold_img.name)
    await px4_cmd("on")
    await hold_phase("policy: hold", 15)

    await px4_cmd("load", circle_img.name)          # hot swap while engaged
    await hold_phase("policy: circle", 35)

    await px4_cmd("load", bad_img.name)             # must be rejected, circle keeps flying
    await hold_phase("policy: circle (bad image rejected)", 8)

    await px4_cmd("off")
    await hold_phase("PX4 Hold", 10)

    phase["name"] = "land"
    await drone.action.land()
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            break
    logger.cancel()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "n", "e", "d", "flight_mode", "phase"])
        w.writerows(rows)
    print(f"log -> {out}")
    plot(rows, out.with_suffix(".png"))
    await asyncio.sleep(0)


def plot(rows, png: Path):
    t = np.array([r[0] for r in rows])
    p = np.array([r[1:4] for r in rows])
    phases = [r[5] for r in rows]
    names = list(dict.fromkeys(phases))
    colors = {n: f"C{i}" for i, n in enumerate(names)}

    fig = plt.figure(figsize=(14, 6))
    ax0 = fig.add_subplot(1, 2, 1)
    for n in names:
        idx = [i for i, ph in enumerate(phases) if ph == n]
        ax0.plot(p[idx, 1], p[idx, 0], ".", ms=2, color=colors[n], label=n)
    ax0.set_xlabel("east [m]")
    ax0.set_ylabel("north [m]")
    ax0.set_aspect("equal")
    ax0.legend(fontsize=8, markerscale=5)
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(1, 2, 2)
    for k, lab in enumerate(("north", "east", "down")):
        ax1.plot(t, p[:, k], label=lab)
    start = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            ax1.axvspan(t[start], t[i - 1], color=colors[phases[start]], alpha=0.12)
            start = i
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("[m]")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    print(f"plot -> {png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("hold", type=Path)
    ap.add_argument("circle", type=Path)
    ap.add_argument("--plot-only", action="store_true", help="re-plot an existing log")
    ap.add_argument("-o", "--out", type=Path, default=Path("logs/ext_policy_flight.csv"))
    a = ap.parse_args()
    if a.plot_only:
        with open(a.out) as f:
            r = list(csv.reader(f))[1:]
        plot([(float(x[0]), float(x[1]), float(x[2]), float(x[3]), x[4], x[5]) for x in r], a.out.with_suffix(".png"))
    else:
        asyncio.run(main(a.hold, a.circle, a.out))
        os._exit(0)  # mavsdk_server background threads otherwise keep the process alive
