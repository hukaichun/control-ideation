"""Generate the Gazebo (gz sim) model and the PX4 SITL airframe for the Starling 2 Max from models/starling_2_max.yaml.

    uv run --with pyyaml python models/gz/gen_starling_2_max.py   # writes models/gz/starling_2_max/ and models/gz/4022_gz_starling_2_max
    scripts/install_gz_models.sh                       # copies them into PX4-Autopilot (then scripts/px4_build.sh)

Physics (mass, inertia, rotor positions, motor constants) from the yaml; the shape is primitives (boxes and cylinders)
sized from the datasheet, not the STEP. The PX4 side (airframe) uses ModalAI's own parameters, including their
CA_ROTOR geometry, so the simulated firmware is configured like the real one.
"""
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
Y = yaml.safe_load((HERE.parent / "starling_2_max.yaml").read_text())
NAME = "starling_2_max"
AIRFRAME = "4022_gz_starling_2_max"

M_PROP = 0.005                       # kg per rotor link (prop only; motors are in the base link)
R_PROP = Y["propeller"]["diameter"] / 2
Z_PROP = 0.030                       # prop plane above the centre of mass
Z_BOTTOM = -0.060                    # lowest point (height 120 mm prop to prop: 60 mm below the CoM, a guess)
RPM = Y["motor"]


def inertia_xml(ixx, iyy, izz, ind="          "):
    return (f"{ind}<ixx>{ixx:.6g}</ixx><ixy>0</ixy><ixz>0</ixz>\n"
            f"{ind}<iyy>{iyy:.6g}</iyy><iyz>0</iyz>\n{ind}<izz>{izz:.6g}</izz>")


def surface():
    return ("<surface><contact><ode><min_depth>0.001</min_depth><max_vel>0</max_vel></ode></contact>"
            "<friction><ode/></friction></surface>")


def box(name, pose, size, kind="visual", color=None):
    mat = f"<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>" if color else ""
    extra = surface() if kind == "collision" else mat
    return (f'      <{kind} name="{name}"><pose>{" ".join(f"{v:.4f}" for v in pose)}</pose>'
            f'<geometry><box><size>{" ".join(f"{v:.4f}" for v in size)}</size></box></geometry>{extra}</{kind}>')


def cyl(name, pose, r, h, color, kind="visual"):
    mat = f"<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>" if kind == "visual" else surface()
    return (f'      <{kind} name="{name}"><pose>{" ".join(f"{v:.4f}" for v in pose)}</pose>'
            f'<geometry><cylinder><radius>{r:.4f}</radius><length>{h:.4f}</length></cylinder></geometry>{mat}</{kind}>')


def model_sdf():
    rot = np.array(Y["rotors"], float)
    m = Y["mass"]
    J = np.array(Y["inertia"], float)
    # the yaml inertia is the whole vehicle: take the prop links (point masses at the rotors) out of the base link
    Jb = np.diag(J).copy()
    for x, y, _ in rot:
        Jb -= M_PROP * np.array([y ** 2 + Z_PROP ** 2, x ** 2 + Z_PROP ** 2, x ** 2 + y ** 2])
    mb = m - 4 * M_PROP
    base_pose_z = -Z_BOTTOM + 0.005
    vis = [box("body", (0, 0, 0.005, 0, 0, 0), (0.14, 0.09, 0.05), color="0.15 0.15 0.15 1"),
           box("battery", (0, 0, -0.03, 0, 0, 0), (0.07, 0.07, 0.04), color="0.1 0.3 0.6 1")]
    for i, (x, y, _) in enumerate(rot):
        L, yaw = np.hypot(x, y), np.arctan2(y, x)
        vis.append(box(f"arm_{i}", (x / 2, y / 2, 0.0, 0, 0, yaw), (L, 0.018, 0.003), color="0.1 0.1 0.1 1"))
        vis.append(cyl(f"motor_{i}", (x, y, Z_PROP - 0.012, 0, 0, 0), 0.014, 0.020, "0.6 0.6 0.6 1"))
        vis.append(box(f"leg_{i}", (0.6 * x, 0.6 * y, (Z_BOTTOM + 0.0) / 2, 0, 0, 0), (0.008, 0.008, -Z_BOTTOM),
                       color="0.1 0.1 0.1 1"))
    col = [box("body_collision", (0, 0, 0.005, 0, 0, 0), (0.14, 0.09, 0.05), "collision")]
    for i, (x, y, _) in enumerate(rot):
        col.append(box(f"leg_{i}_collision", (0.6 * x, 0.6 * y, Z_BOTTOM / 2, 0, 0, 0), (0.01, 0.01, -Z_BOTTOM),
                       "collision"))
    sensors = (HERE / "sensors.sdf.part").read_text()
    links, joints, plugins = [], [], []
    for i, (x, y, km) in enumerate(rot):
        # FLU yaw torque / thrust = -KM (PX4, FRD); PX4 KM > 0 = CCW rotor. gz: turningDirection ccw gives the
        # reaction torque -momentConstant * thrust about +z, i.e. km < 0 here.
        direction = "ccw" if km < 0 else "cw"
        Ip = M_PROP * R_PROP ** 2
        links.append(f"""    <link name="rotor_{i}">
      <pose>{x:.4f} {y:.4f} {Z_PROP:.4f} 0 0 0</pose>
      <inertial><mass>{M_PROP}</mass><inertia>
{inertia_xml(Ip / 4, Ip / 4, Ip / 2)}
      </inertia></inertial>
{cyl(f"rotor_{i}_visual", (0, 0, 0, 0, 0, 0), R_PROP, 0.002, "0.3 0.3 0.3 0.6" if direction == "ccw" else "0.6 0.3 0.3 0.6")}
{box(f"rotor_{i}_blade", (0, 0, 0.002, 0, 0, 0), (2 * R_PROP, 0.012, 0.002), color="0.05 0.05 0.05 1")}
{cyl(f"rotor_{i}_collision", (0, 0, 0, 0, 0, 0), R_PROP, 0.002, None, "collision")}
    </link>""")
        joints.append(f"""    <joint name="rotor_{i}_joint" type="revolute">
      <parent>base_link</parent><child>rotor_{i}</child>
      <axis><xyz>0 0 1</xyz><limit><lower>-1e+16</lower><upper>1e+16</upper></limit>
        <dynamics><spring_reference>0</spring_reference><spring_stiffness>0</spring_stiffness></dynamics></axis>
    </joint>""")
        plugins.append(f"""    <plugin filename="gz-sim-multicopter-motor-model-system" name="gz::sim::systems::MulticopterMotorModel">
      <jointName>rotor_{i}_joint</jointName>
      <linkName>rotor_{i}</linkName>
      <turningDirection>{direction}</turningDirection>
      <timeConstantUp>{Y["time_constant_up"]}</timeConstantUp>
      <timeConstantDown>{Y["time_constant_down"]}</timeConstantDown>
      <maxRotVelocity>{RPM["rpm_max_esc"] * 2 * np.pi / 60:.1f}</maxRotVelocity>
      <motorConstant>{Y["motor_constant"]}</motorConstant>
      <momentConstant>{Y["moment_constant"]}</momentConstant>
      <commandSubTopic>command/motor_speed</commandSubTopic>
      <motorNumber>{i}</motorNumber>
      <rotorDragCoefficient>{Y["rotor_drag_coefficient"]}</rotorDragCoefficient>
      <rollingMomentCoefficient>{Y["rolling_moment_coefficient"]}</rollingMomentCoefficient>
      <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
      <motorType>velocity</motorType>
    </plugin>""")
    nl = "\n"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by models/gz/gen_starling_2_max.py from models/starling_2_max.yaml. Do not edit by hand. -->
<sdf version='1.9'>
  <model name='{NAME}'>
    <pose>0 0 {base_pose_z:.3f} 0 0 0</pose>
    <self_collide>false</self_collide>
    <static>false</static>
    <link name="base_link">
      <inertial>
        <mass>{mb:.4f}</mass>
        <inertia>
{inertia_xml(*Jb)}
        </inertia>
      </inertial>
      <gravity>true</gravity>
      <velocity_decay/>
{nl.join(vis)}
{nl.join(col)}
{sensors}
    </link>
{nl.join(links)}
{nl.join(joints)}
{nl.join(plugins)}
  </model>
</sdf>
"""


def model_config():
    return f"""<?xml version="1.0"?>
<model>
  <name>{NAME}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>ModalAI Starling 2 Max (D0012), first rough model: physics from models/starling_2_max.yaml (uav repo),
  shape from primitives. Generated by models/gz/gen_starling_2_max.py.</description>
</model>
"""


def airframe():
    p = Y["px4"]
    pos = dict(Y["position"]["both"], **Y["position"]["outdoor"])
    w_min, w_max = (RPM[k] * 2 * np.pi / 60 for k in ("rpm_min", "rpm_max_px4"))
    lines = ["#!/bin/sh", "#", "# @name Gazebo Starling 2 Max", "#", "# @type Quadrotor", "#",
             "# Generated by models/gz/gen_starling_2_max.py (uav repo). ModalAI parameters (D0012_Starling_2_Max.params,",
             "# starling_2_max_outdoor_position.params); motor command range = the ESC rpm range.", "",
             ". ${R}etc/init.d/rc.mc_defaults", "",
             "PX4_SIMULATOR=${PX4_SIMULATOR:=gz}", "PX4_GZ_WORLD=${PX4_GZ_WORLD:=default}",
             f"PX4_SIM_MODEL=${{PX4_SIM_MODEL:={NAME}}}", "", "param set-default SIM_GZ_EN 1", "",
             "param set-default CA_AIRFRAME 0", f"param set-default CA_ROTOR_COUNT {p['CA_ROTOR_COUNT']}"]
    for i in range(4):
        lines += [f"param set-default CA_ROTOR{i}_PX {p['CA_ROTOR_PX'][i]}", f"param set-default CA_ROTOR{i}_PY {p['CA_ROTOR_PY'][i]}",
                  f"param set-default CA_ROTOR{i}_KM {p['CA_ROTOR_KM'][i]}"]
    lines.append("")
    for i in range(1, 5):
        lines += [f"param set-default SIM_GZ_EC_FUNC{i} {100 + i}", f"param set-default SIM_GZ_EC_MIN{i} {w_min:.0f}",
                  f"param set-default SIM_GZ_EC_MAX{i} {w_max:.0f}"]
    lines.append("")
    skip = {"CA_ROTOR_COUNT", "CA_ROTOR_PX", "CA_ROTOR_PY", "CA_ROTOR_KM", "VOXL_ESC_RPM_MIN", "VOXL_ESC_RPM_MAX",
            "MC_ROLL_CUTOFF", "MC_PITCH_CUTOFF", "MC_YAW_CUTOFF"}       # not in PX4 v1.17
    for k, v in p.items():
        if k not in skip:
            lines.append(f"param set-default {k} {v}")
    lines.append("")
    for k, v in pos.items():
        lines.append(f"param set-default {k} {v}")
    lines += ["", "param set-default NAV_DLL_ACT 2", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    d = HERE / NAME
    d.mkdir(exist_ok=True)
    (d / "model.sdf").write_text(model_sdf())
    (d / "model.config").write_text(model_config())
    (HERE / AIRFRAME).write_text(airframe())
    print("wrote", d / "model.sdf", d / "model.config", HERE / AIRFRAME)
