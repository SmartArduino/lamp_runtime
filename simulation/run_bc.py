import time
import numpy as np
import mujoco
import mujoco.viewer
from pynput import keyboard

m = np.load("bc_model.npz")
W1, b1, W2, b2, W3, b3 = m["W1"], m["b1"], m["W2"], m["b2"], m["W3"], m["b3"]
mean_s, std_s, mean_a, std_a = m["mean_s"], m["std_s"], m["mean_a"], m["std_a"]

CUP_XR = (-0.07, 0.07)
CUP_YR = (-0.26, -0.21)
CUP_Z = 0.04

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
cup_jid = model.joint("cup_joint").id
CUP_Q = model.jnt_qposadr[cup_jid]
CUP_V = model.jnt_dofadr[cup_jid]
HEAD_ID = model.body("head").id

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


def predict(qpos, qvel, head_pos, cup_pos):
    s = (np.concatenate([qpos[:5], qvel[:5], head_pos, cup_pos]) - mean_s) / std_s
    h1 = np.maximum(s @ W1 + b1, 0)
    h2 = np.maximum(h1 @ W2 + b2, 0)
    an = h2 @ W3 + b3
    delta = an * std_a + mean_a
    return np.clip(qpos[:5] + delta, -1.57, 1.57)


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


rng = np.random.default_rng()
reset_scene(rng)
mujoco.mj_forward(model, data)
cup_start = data.qpos[CUP_Q:CUP_Q + 3].copy()
print(f"cup start: {np.round(cup_start, 3)}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_dt = model.opt.timestep
    next_step = time.perf_counter()
    last_print = 0.0
    ctrl_smooth = predict(data.qpos, data.qvel, data.xpos[HEAD_ID], data.qpos[CUP_Q:CUP_Q + 3])
    prev_pressed = set()
    while viewer.is_running():
        if 'r' in pressed and 'r' not in prev_pressed:
            reset_scene(rng)
            cup_start = data.qpos[CUP_Q:CUP_Q + 3].copy()
            ctrl_smooth = predict(data.qpos, data.qvel, data.xpos[HEAD_ID], data.qpos[CUP_Q:CUP_Q + 3])
            print(f"reset -> cup {np.round(cup_start, 3)}")
        if 'p' in pressed and 'p' not in prev_pressed:
            reset_scene(rng, arm="random")
            cup_start = data.qpos[CUP_Q:CUP_Q + 3].copy()
            ctrl_smooth = predict(data.qpos, data.qvel, data.xpos[HEAD_ID], data.qpos[CUP_Q:CUP_Q + 3])
            print(f"random pose -> qpos {np.round(data.qpos[:5], 2)} cup {np.round(cup_start, 3)}")
        prev_pressed = set(pressed)

        pred = predict(data.qpos, data.qvel, data.xpos[HEAD_ID], data.qpos[CUP_Q:CUP_Q + 3])
        ctrl_smooth = 0.9 * ctrl_smooth + 0.1 * pred
        data.ctrl[:] = ctrl_smooth
        mujoco.mj_step(model, data)
        viewer.sync()

        next_step += step_dt
        sleep_time = next_step - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_step = time.perf_counter()

        if data.time - last_print >= 1.0:
            last_print = data.time
            cup_now = data.qpos[CUP_Q:CUP_Q + 3]
            disp = np.linalg.norm(cup_now[:2] - cup_start[:2])
            print(f"\rt={data.time:6.1f}s  cup_disp={disp:.3f}m  qpos={np.round(data.qpos[:5], 2)}", end="")

listener.stop()
