# uavlab

PX4 SITL + Gazebo Harmonic（Docker、headless）搭配主機端 MAVSDK-Python 的 control policy 實驗環境。

## 用法

```bash
docker build -t uav-px4-sitl:v1.17.0 docker   # 只需一次
scripts/sim_up.sh                             # 啟動模擬器（x500 四旋翼）
uv run examples/step_response.py --kp 1.0     # 跑一個 episode，寫入 logs/*.csv
uv run tools/plot_log.py logs/*.csv           # 畫圖並印出追蹤誤差
scripts/sim_down.sh logs/ulog                 # 關閉模擬器（可選：先把 PX4 .ulg 複製出來）
```

環境變數：`PX4_GZ_WORLD`（如 `windy`）、`PX4_SIM_MODEL`（預設 `gz_x500`）、`PX4_SIM_SPEED_FACTOR`。

## 結構

- `uavlab/policy.py` — `Policy` 介面（`Observation` → `Action`）與 P 控制器 baseline。寫新的 policy 就實作 `act()`。
- `uavlab/runner.py` — 連線、arm、進 offboard、以 50 Hz 呼叫 policy、記錄 CSV、降落。
- `examples/step_response.py` — 位置步階響應實驗。

目前 action 是 NED 速度設定點（PX4 內建的速度／姿態／角速度迴路仍在運作）。要研究更內層的迴路，把
`runner.py` 裡的 `set_velocity_ned` 換成 `set_attitude`（姿態 + 推力）或 `set_attitude_rate`（角速度 + 推力）即可。

座標皆為 local NED，向上為負 `d`。

## 機上 policy（PX4 `ext_policy` 模組）

`PX4-Autopilot/` 是 fork（分支 `ext-policy`，基於 v1.17.0），新增 `src/modules/ext_policy`：在飛控上執行上傳的
policy `u = f(x)`（平坦狀態 `x = [p, v, a, j]`，輸出 snap），可在飛行中熱抽換，並在 PX4 控制器與 policy 之間切換。

```bash
scripts/px4_build.sh                                   # 建置 fork 的 SITL
PX4_SRC=PX4-Autopilot scripts/sim_up.sh                # 用 fork 啟動模擬器
uv run tools/build_policy.py policies/circle.c         # C -> build/policies/circle.xpol（含 CRC）
uv run examples/xpol_offline_sim.py build/policies/*.xpol   # 不開模擬器先驗證
uv run examples/ext_policy_flight.py build/policies/hold.xpol build/policies/circle.xpol
```

PX4 shell 指令：`ext_policy load <file>`（驗證後下一個週期生效）、`ext_policy on`（切到 Offboard 由 policy 控制）、
`ext_policy off`（切回 Hold 由 PX4 控制）、`ext_policy status`。

policy 規則：只定義 `xpol_step`（可選 `xpol_init`、`xpol_state_size`），不能呼叫 libc、不能有可寫的全域變數，
數學函式由 `xpol_math_t` 提供；`build_policy.py` 會檢查外部符號、絕對位址重定位與資料段。ABI 定義在
`PX4-Autopilot/src/modules/ext_policy/xpol_abi.h`。
