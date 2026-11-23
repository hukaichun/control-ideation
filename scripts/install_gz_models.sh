#!/usr/bin/env bash
# Copy the repo's Gazebo models (models/gz/<name>/) and PX4 SITL airframes (models/gz/4xxx_gz_*) into a PX4 checkout,
# and register the airframes in its ROMFS CMakeLists. Then rebuild: scripts/px4_build.sh
#   scripts/install_gz_models.sh [PX4-Autopilot]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PX4="$(realpath "${1:-$ROOT/PX4-Autopilot}")"
SRC="$ROOT/models/gz"
AF_DIR="$PX4/ROMFS/px4fmu_common/init.d-posix/airframes"
for d in "$SRC"/*/; do
    name="$(basename "$d")"
    rm -rf "$PX4/Tools/simulation/gz/models/$name"
    cp -r "$d" "$PX4/Tools/simulation/gz/models/$name"
    echo "model    $name"
done
for f in "$SRC"/[0-9]*_gz_*; do
    af="$(basename "$f")"
    install -m 755 "$f" "$AF_DIR/$af"
    if ! grep -q "^\s*$af\$" "$AF_DIR/CMakeLists.txt"; then
        sed -i "/^\s*4001_gz_x500\$/a\\	$af" "$AF_DIR/CMakeLists.txt"
    fi
    echo "airframe $af"
done
