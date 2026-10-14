import mujoco
import mujoco.viewer
import numpy as np
import time

# 加载模型
model_path = "robot.xml"  # 你的MJCF文件路径
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 创建一个渲染器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 仿真循环
    start_time = time.time()
    while viewer.is_running():
        # --- 你的控制逻辑放在这里 ---
        # 例如，我们可以给关节施加一个简单的正弦波控制信号
        t = time.time() - start_time
        data.ctrl[0] = 0.5 * np.sin(2 * t)  # 控制关节1
        data.ctrl[1] = 0.5 * np.cos(2 * t)  # 控制关节2
        
        # --- 控制逻辑结束 ---

        # 推进仿真一步
        mujoco.mj_step(model, data)
        
        # 同步渲染器
        viewer.sync()
        
        # 控制仿真速度，使其接近实时
        # time_until_next_step = model.opt.timestep - (time.time() - start_time)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)
