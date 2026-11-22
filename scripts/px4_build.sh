#!/usr/bin/env bash
# Build a local PX4 checkout for SITL inside the toolchain image, as the host user.
#   scripts/px4_build.sh [PX4-Autopilot] [make target]
set -euo pipefail
SRC="$(realpath "${1:-PX4-Autopilot}")"
TARGET="${2:-px4_sitl_default}"
IMAGE="${IMAGE:-uav-px4-sitl:v1.17.0}"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e CCACHE_DIR=/src/build/.ccache \
    -v "$SRC":/src -w /src "$IMAGE" make "$TARGET"
