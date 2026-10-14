import re
import numpy as np
import mujoco

h = open("bc_model.h").read()


def arr(name):
    body = re.search(r"float %s\[\d+\] = \{(.*?)\};" % name, h, re.S).group(1)
    return np.array([float(v) for v in body.replace("\n", "").split(",") if v.strip()])


W1, b1, W2, b2, W3, b3 = (arr(n) for n in ["W1", "b1", "W2", "b2", "W3", "b3"])
IN_DIM = int(re.search(r"#define BC_IN_DIM (\d+)", h).group(1))
H1 = int(re.search(r"#define BC_H1 (\d+)", h).group(1))
H2 = int(re.search(r"#define BC_H2 (\d+)", h).group(1))
OUT = int(re.search(r"#define BC_OUT (\d+)", h).group(1))
print("dims:", IN_DIM, H1, H2, OUT)


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def head_fk(q):
    p = np.array([0, 0, 0.01])
    R = np.eye(3)
    p = p + R @ np.array([0, 0, 0.041]); R = R @ Rz(q[0])
    p = p + R @ np.array([0.002, 0, 0.04]); R = R @ Rx(q[1])
    p = p + R @ np.array([0.005, -0.005, 0.177]); R = R @ Rx(q[2])
    p = p + R @ np.array([0, 0, 0.139]); R = R @ Rz(q[3])
    p = p + R @ np.array([0, 0, 0.033])
    return p


def fc_c(x, W, b, n_in, relu):
    W2d = W.reshape(len(b), n_in)
    y = np.empty(len(b))
    for j in range(len(b)):
        s = b[j]
        for i in range(n_in):
            s += x[i] * W2d[j, i]
        y[j] = max(s, 0.0) if relu else s
    return y


def bc_predict_c(qpos, qvel, cup):
    state = np.concatenate([qpos, qvel, head_fk(qpos), cup])
    h1 = fc_c(state, W1, b1, IN_DIM, True)
    h2 = fc_c(h1, W2, b2, H1, True)
    delta = fc_c(h2, W3, b3, H2, False)
    return np.clip(qpos + delta, -1.57, 1.57)


m = np.load("bc_model.npz")
mw = {k: m[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3", "mean_s", "std_s", "mean_a", "std_a"]}


def predict_ref(qpos, qvel, head, cup):
    s = (np.concatenate([qpos, qvel, head, cup]) - mw["mean_s"]) / mw["std_s"]
    h1 = np.maximum(s @ mw["W1"] + mw["b1"], 0)
    h2 = np.maximum(h1 @ mw["W2"] + mw["b2"], 0)
    delta = (h2 @ mw["W3"] + mw["b3"]) * mw["std_a"] + mw["mean_a"]
    return np.clip(qpos + delta, -1.57, 1.57)


model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
hid = model.body("head").id
rng = np.random.default_rng(0)
err = 0.0
for _ in range(200):
    qpos = rng.uniform(-1.5, 1.5, 5)
    qvel = rng.normal(0, 0.5, 5)
    cup = np.array([rng.uniform(-0.07, 0.07), rng.uniform(-0.26, -0.21), 0.04])
    data.qpos[:5] = qpos
    mujoco.mj_kinematics(model, data)
    out_c = bc_predict_c(qpos, qvel, cup)
    out_ref = predict_ref(qpos, qvel, data.xpos[hid], cup)
    err = max(err, np.abs(out_c - out_ref).max())
print("C-path vs run_bc predict max diff:", err)
