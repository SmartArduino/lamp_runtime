# simulate_keyboard_hold.py
import mujoco
import mujoco.viewer
from pynput import keyboard
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# 记录当前被按下的键
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

target = np.zeros(model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_print_time = 0.0
    PRINT_INTERVAL = 0.2  # 每 0.2 秒打印一次（约 5Hz），可自行调整
    step_dt = model.opt.timestep
    next_step = time.perf_counter()

    while viewer.is_running():
        SPEED = 0.002

        # 键盘控制目标角度
        if 'a' in pressed: target[0] -= SPEED
        if 'd' in pressed: target[0] += SPEED
        if 'w' in pressed: target[1] += SPEED
        if 's' in pressed: target[1] -= SPEED
        if 'q' in pressed: target[2] -= SPEED
        if 'e' in pressed: target[2] += SPEED
        if 'z' in pressed: target[3] -= SPEED
        if 'x' in pressed: target[3] += SPEED
        if 'n' in pressed: target[4] -= SPEED
        if 'm' in pressed: target[4] += SPEED

        target[:] = np.clip(target, -1.57, 1.57)
        data.ctrl[:] = target

        mujoco.mj_step(model, data)
        viewer.sync()

        next_step += step_dt
        sleep_time = next_step - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_step = time.perf_counter()

        # ---------- 实时打印关节角度（限频） ----------
        now = time.time()
        if now - last_print_time >= PRINT_INTERVAL:
            last_print_time = now
            # data.qpos 是关节位置数组（对应 joint1, joint2 ...）
            angles_deg = np.degrees(data.qpos[:model.njnt])  # 转成角度更直观
            angle_str = ", ".join(f"J{i+1}: {a:+7.2f}° ({q:+.3f} rad)"
                                  for i, (a, q) in enumerate(zip(angles_deg, data.qpos[:model.njnt])))
            #print(f"\r{angle_str}   ", end="", flush=True)
            print(f"\r目标: {np.round(target,2)} | 实际: {np.round(data.qpos[:model.njnt],2)}", end="")


listener.stop()
