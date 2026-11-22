# PX4 SITL + Gazebo Harmonic (headless by default)
FROM ubuntu:24.04

ARG PX4_VERSION=v1.17.0
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        git sudo lsb-release ca-certificates wget curl gnupg tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${PX4_VERSION} \
        --recurse-submodules --shallow-submodules \
        https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot

WORKDIR /opt/PX4-Autopilot

# Installs the build toolchain and Gazebo Harmonic (gz-sim8)
RUN bash Tools/setup/ubuntu.sh --no-nuttx && rm -rf /var/lib/apt/lists/*

RUN make px4_sitl_default

ENV HEADLESS=1 \
    PX4_SIM_MODEL=gz_x500
# World / model can be overridden at run time: -e PX4_GZ_WORLD=windy -e PX4_SIM_MODEL=gz_x500
CMD ["bash", "-c", "make px4_sitl ${PX4_SIM_MODEL}"]
