# simulate_keyboard.py
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# 目标关节位置（位置控制）
target = np.zeros(model.nu)

# 键盘回调：key 是按键的 ASCII/键码，只在这个窗口获得焦点时触发
def key_callback(keycode):
    global target
    STEP = 0.1  # 每按一次键，目标角度的变化量

    if keycode == 65:      # A 键 → joint1 负方向
        target[0] -= STEP
    elif keycode == 68:    # D 键 → joint1 正方向
        target[0] += STEP
    elif keycode == 87:    # W 键 → joint2 正方向
        target[1] += STEP
    elif keycode == 83:    # S 键 → joint2 负方向
        target[1] -= STEP
    elif keycode == 82:    # R 锓 → 处位到零
        target[:] = 0.0
        mujoco.mj_resetData(model, data)

    # 限位保护
    target[:] = np.clip(target, -1.57, 1.57)
    print(f"目标角度: {np.round(target, 2)}")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        # 位置控制：目标角度平滑跟踪（简单的低通滤波，避免突变）
        data.ctrl[:] = 0.9 * data.ctrl[:] + 0.1 * target

        mujoco.mj_step(model, data)
        viewer.sync()
