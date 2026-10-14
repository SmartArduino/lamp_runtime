import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from pynput import keyboard

import calib

EPISODE_DIR = "demos"

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

try:
    cal = calib.load()
    problems = cal.problems()
    if problems:
        raise SystemExit("校准文件自检失败:\n  " + "\n  ".join(problems))
    print(f"关节限位(bubble.json, homing 单位: {cal.homing_unit}):")
    print(cal.table())
except (FileNotFoundError, ValueError) as e:
    print(f"[warn] 未加载 bubble.json ({e}), 录制数据只用 ±1.57 限位")
    cal = None

CUP_XR = (-0.07, 0.07)
CUP_YR = (-0.26, -0.21)
CUP_Z = 0.04
cup_jid = model.joint("cup_joint").id
CUP_Q = model.jnt_qposadr[cup_jid]
CUP_V = model.jnt_dofadr[cup_jid]

pressed = set()


def on_press(key):
    try:
        pressed.add(key.char.lower())
    except AttributeError:
        pass


def on_release(key):
    try:
        pressed.discard(key.char.lower())
    except AttributeError:
        pass


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


def reset_scene(rng, arm="home"):
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    if arm == "random":
        data.qpos[:5] = [rng.uniform(-0.8, 0.8), rng.uniform(-0.5, 0.5),
                         rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3),
                         rng.uniform(-0.3, 0.3)]
    else:
        data.qpos[:5] = 0.0
    data.qpos[CUP_Q + 0] = rng.uniform(*CUP_XR)
    data.qpos[CUP_Q + 1] = rng.uniform(*CUP_YR)
    data.qpos[CUP_Q + 2] = CUP_Z
    data.qpos[CUP_Q + 3] = 1.0
    data.qpos[CUP_Q + 4: CUP_Q + 7] = 0.0
    mujoco.mj_forward(model, data)
    print(f"reset arm={np.round(data.qpos[:5], 2)} cup=({data.qpos[CUP_Q]:.3f},{data.qpos[CUP_Q + 1]:.3f})")


rng = np.random.default_rng()
reset_scene(rng)

os.makedirs(EPISODE_DIR, exist_ok=True)
existing = [f for f in os.listdir(EPISODE_DIR) if f.startswith("episode_") and f.endswith(".npz")]
out_path = os.path.join(EPISODE_DIR, f"episode_{len(existing):03d}.npz")

target = np.zeros(model.nu)
log_t, log_qpos, log_qvel, log_ctrl = [], [], [], []

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_dt = model.opt.timestep
    next_step = time.perf_counter()
    prev_pressed = set()
    while viewer.is_running():
        SPEED = 0.004

        if 'a' in pressed: target[0] -= SPEED
        if 'd' in pressed: target[0] += SPEED
        if 'w' in pressed: target[1] += SPEED
        if 's' in pressed: target[1] -= SPEED
        if 'q' in pressed: target[2] -= SPEED
        if 'e' in pressed: target[2] += SPEED
        if 'z' in pressed: target[3] -= SPEED
        if 'x' in pressed: target[4] += SPEED
        if 'n' in pressed: target[4] -= SPEED
        if 'm' in pressed: target[4] += SPEED
        if 'r' in pressed and 'r' not in prev_pressed:
            target[:] = 0.0
            reset_scene(rng)
        if 'p' in pressed and 'p' not in prev_pressed:
            reset_scene(rng, arm="random")
            target[:] = data.qpos[:5]
        prev_pressed = set(pressed)

        target[:] = np.clip(target, -1.57, 1.57)
        if cal is not None:
            target[:] = cal.clip_radians(target)
        data.ctrl[:] = target

        mujoco.mj_step(model, data)
        viewer.sync()

        log_t.append(data.time)
        log_qpos.append(data.qpos.copy())
        log_qvel.append(data.qvel.copy())
        log_ctrl.append(data.ctrl.copy())

        next_step += step_dt
        sleep_time = next_step - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_step = time.perf_counter()

listener.stop()

if log_t:
    np.savez(out_path,
             t=np.array(log_t),
             qpos=np.array(log_qpos),
             qvel=np.array(log_qvel),
             ctrl=np.array(log_ctrl))
    print(f"\nsaved {len(log_t)} samples -> {out_path}")
else:
    print("\nno data recorded")
