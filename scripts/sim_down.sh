#!/usr/bin/env bash
# Stop the simulator. Pass a directory to also copy PX4's .ulg flight logs out first.
#   scripts/sim_down.sh [logs/ulog]
set -euo pipefail
SIM_NAME="${SIM_NAME:-px4-sitl}"
if [ $# -ge 1 ]; then
    mkdir -p "$1"
    # PX4 runs from /src when started with PX4_SRC, otherwise from the copy in the image
    src="$(docker exec "$SIM_NAME" sh -c 'pwd')"
    docker cp "$SIM_NAME":"$src"/build/px4_sitl_default/rootfs/log/. "$1" && echo "ulogs copied to $1"
fi
docker rm -f "$SIM_NAME" >/dev/null && echo "stopped $SIM_NAME"
