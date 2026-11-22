#!/usr/bin/env bash
# Start PX4 SITL + Gazebo (headless) in Docker. MAVLink offboard API on udp 14540.
#   scripts/sim_up.sh                 # x500, default world
#   PX4_GZ_WORLD=windy scripts/sim_up.sh
#   PX4_SIM_SPEED_FACTOR=2 scripts/sim_up.sh
#   PX4_SRC=PX4-Autopilot scripts/sim_up.sh   # run a local PX4 checkout (built with scripts/px4_build.sh)
set -euo pipefail

IMAGE="${IMAGE:-uav-px4-sitl:v1.17.0}"
SIM_NAME="${SIM_NAME:-px4-sitl}"

# With PX4_SRC, mount the checkout and run it as the host user instead of the copy baked into the image.
SRC_ARGS=()
if [ -n "${PX4_SRC:-}" ]; then
    SRC_ARGS=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$(realpath "$PX4_SRC")":/src -w /src)
fi

docker rm -f "$SIM_NAME" >/dev/null 2>&1 || true
# No RC in this setup: allow offboard without it. (MAVSDK counts as the GCS once it connects;
# until then PX4 logs "Preflight Fail: No connection to the GCS", which is expected.)
# -t: the PX4 shell needs a tty to stay alive. --network host: MAVLink UDP reaches the host directly.
docker run -d -t --name "$SIM_NAME" --network host "${SRC_ARGS[@]}" \
    -e PX4_SIM_MODEL="${PX4_SIM_MODEL:-gz_x500}" \
    -e PX4_PARAM_COM_RCL_EXCEPT=4 \
    ${PX4_GZ_WORLD:+-e PX4_GZ_WORLD="$PX4_GZ_WORLD"} \
    ${PX4_SIM_SPEED_FACTOR:+-e PX4_SIM_SPEED_FACTOR="$PX4_SIM_SPEED_FACTOR"} \
    "$IMAGE" >/dev/null

echo "waiting for PX4 to boot..."
for _ in $(seq 120); do
    if docker logs "$SIM_NAME" 2>&1 | grep -q "Startup script returned successfully"; then
        echo "PX4 SITL ready (container: $SIM_NAME). Logs: docker logs -f $SIM_NAME"
        exit 0
    fi
    if [ -z "$(docker ps -q -f name="^${SIM_NAME}$")" ]; then
        echo "container exited:" >&2
        docker logs --tail 40 "$SIM_NAME" >&2
        exit 1
    fi
    sleep 1
done
echo "timed out waiting for PX4 startup; last log lines:" >&2
docker logs --tail 40 "$SIM_NAME" >&2
exit 1
