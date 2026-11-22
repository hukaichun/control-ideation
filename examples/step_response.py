"""Take off to 5 m, then step 5 m north and back. Logs to logs/<name>.csv.

    uv run examples/step_response.py --kp 1.0
"""
import argparse
import asyncio
import time
from pathlib import Path

import numpy as np

from uavlab.policy import PPositionPolicy
from uavlab.runner import run_episode


def reference(t: float) -> np.ndarray:
    alt = -5.0  # NED: negative down = up
    if t < 10:
        return np.array([0.0, 0.0, alt])
    if t < 20:
        return np.array([5.0, 0.0, alt])
    return np.array([0.0, 0.0, alt])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--name", default=time.strftime("step_%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    asyncio.run(
        run_episode(
            PPositionPolicy(kp=args.kp),
            reference,
            duration=args.duration,
            log_path=Path("logs") / f"{args.name}.csv",
        )
    )
