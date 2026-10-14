import numpy as np
import mujoco

m = np.load("bc_model.npz")
W1, b1, W2, b2, W3, b3 = (m[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"])
mean_s, std_s, mean_a, std_a = m["mean_s"], m["std_s"], m["mean_a"], m["std_a"]

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
HEAD_ID = model.body("head").id


def head_pos(pose):
    data.qpos[:5] = pose
    data.qvel[:] = 0
    mujoco.mj_kinematics(model, data)
    return data.xpos[HEAD_ID].copy()


def predict(pose, cup):
    s = np.concatenate([pose, np.zeros(5), head_pos(pose), cup]) - mean_s
    sn = s / std_s
    h1 = np.maximum(sn @ W1 + b1, 0)
    h2 = np.maximum(h1 @ W2 + b2, 0)
    delta = (h2 @ W3 + b3) * std_a + mean_a
    return np.clip(pose + delta, -1.57, 1.57)


print("delta response to cup position (values = qpos + delta):")
for pose, label in [(np.zeros(5), "home"), (np.array([0, 0.3, 0.5, 0, 0]), "pitched")]:
    print(f"\n--- arm pose {label}: {pose[:3]} ---")
    for x in [-0.08, -0.04, 0.0, 0.04, 0.08]:
        row = []
        for y in [-0.26, -0.23, -0.21]:
            p = predict(pose, np.array([x, y, 0.04]))
            row.append(f"cup({x:+.2f},{y:+.2f}):j1={p[0]:+.2f},j2={p[1]:+.2f},j3={p[2]:+.2f}")
        print("  ".join(row))
