import glob
import numpy as np

DOWNSAMPLE = 10
HIDDEN = 64
EPOCHS = 2000
BATCH = 256
LR = 1e-3
VAL_RATIO = 0.1
SEED = 0

files = sorted(glob.glob("demos/episode_*.npz"))
if not files:
    raise SystemExit("demos/ 下没有 episode_*.npz，先运行 record_demo.py 录制数据")

qpos_list, qvel_list, act_list = [], [], []
for f in files:
    d = np.load(f)
    q, v, a = d["qpos"], d["qvel"], d["ctrl"]
    arm_move = (np.abs(a).max(1) > 0.05) | (np.abs(q[:, :5]).max(1) > 0.05)
    i0 = int(arm_move.argmax()) if arm_move.any() else 0
    cup_d = np.linalg.norm(np.diff(q[:, 5:7], axis=0), axis=1)
    i1 = len(q) - 1
    if (cup_d > 1e-4).any():
        i1 = min(len(q) - 1, int(np.where(cup_d > 1e-4)[0][-1]) + 500)
    qpos_list.append(q[i0:i1 + 1])
    qvel_list.append(v[i0:i1 + 1])
    act_list.append(a[i0:i1 + 1])
    print(f"{f}: keep [{i0}:{i1 + 1}] / {len(q)}  ({d['t'][i1] - d['t'][i0]:.1f}s)")

qpos = np.concatenate(qpos_list, axis=0)
qvel = np.concatenate(qvel_list, axis=0)
act = np.concatenate(act_list, axis=0)

if qpos.shape[1] != 12 or qvel.shape[1] != 11:
    raise SystemExit("qpos/qvel 维度不对，需要用新版 robot.xml(含杯子) 重新录制数据")

qpos, qvel, act = qpos[::DOWNSAMPLE], qvel[::DOWNSAMPLE], act[::DOWNSAMPLE]
act = act - qpos[:, :5]

moving = np.abs(act).max(1) > 0.005
keep = np.sort(np.concatenate([np.where(moving)[0], np.where(~moving)[0][::20]]))
qpos, qvel, act = qpos[keep], qvel[keep], act[keep]

import mujoco
fk_model = mujoco.MjModel.from_xml_path("robot.xml")
fk_data = mujoco.MjData(fk_model)
hid = fk_model.body("head").id
head = np.empty((len(qpos), 3))
for i in range(len(qpos)):
    fk_data.qpos[:] = qpos[i]
    mujoco.mj_kinematics(fk_model, fk_data)
    head[i] = fk_data.xpos[hid]
state = np.concatenate([qpos[:, :5], qvel[:, :5], head, qpos[:, 5:8]], axis=1)
print(f"episodes: {len(files)}, samples: {len(moving)} downsampled, "
      f"moving {moving.sum()}/{len(moving)}, kept after idle-subsample: {len(state)}")

mean_s, std_s = state.mean(0), np.maximum(state.std(0), 1e-2)
mean_a, std_a = act.mean(0), np.maximum(act.std(0), 1e-2)
sn = (state - mean_s) / std_s
an = (act - mean_a) / std_a

rng = np.random.default_rng(SEED)
idx = rng.permutation(len(sn))
n_val = int(len(sn) * VAL_RATIO)
val_idx, tr_idx = idx[:n_val], idx[n_val:]
Xtr, Ytr, Xva, Yva = sn[tr_idx], an[tr_idx], sn[val_idx], an[val_idx]

ds, da = Xtr.shape[1], Ytr.shape[1]
params = {
    "W1": rng.normal(0, np.sqrt(2 / ds), (ds, HIDDEN)), "b1": np.zeros(HIDDEN),
    "W2": rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, HIDDEN)), "b2": np.zeros(HIDDEN),
    "W3": rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, da)), "b3": np.zeros(da),
}
adam_m = {k: np.zeros_like(v) for k, v in params.items()}
adam_v = {k: np.zeros_like(v) for k, v in params.items()}


def forward(p, x):
    h1 = np.maximum(x @ p["W1"] + p["b1"], 0)
    h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
    return h2 @ p["W3"] + p["b3"], h1, h2


def backward(p, x, h1, h2, y, out):
    n = len(x)
    d = 2 * (out - y) / n
    g = {}
    g["W3"] = h2.T @ d
    g["b3"] = d.sum(0)
    dh2 = d @ p["W3"].T
    dh2 = dh2 * (h2 > 0)
    g["W2"] = h1.T @ dh2
    g["b2"] = dh2.sum(0)
    dh1 = dh2 @ p["W2"].T
    dh1 = dh1 * (h1 > 0)
    g["W1"] = x.T @ dh1
    g["b1"] = dh1.sum(0)
    return g


def mseloss(pred, y):
    return float(np.mean((pred - y) ** 2))


beta1, beta2, eps = 0.9, 0.999, 1e-8
step = 0
for epoch in range(1, EPOCHS + 1):
    order = rng.permutation(len(Xtr))
    for start in range(0, len(order), BATCH):
        b = order[start:start + BATCH]
        x, y = Xtr[b], Ytr[b]
        out, h1, h2 = forward(params, x)
        grads = backward(params, x, h1, h2, y, out)
        step += 1
        for k in params:
            adam_m[k] = beta1 * adam_m[k] + (1 - beta1) * grads[k]
            adam_v[k] = beta2 * adam_v[k] + (1 - beta2) * grads[k] ** 2
            mh = adam_m[k] / (1 - beta1 ** step)
            vh = adam_v[k] / (1 - beta2 ** step)
            params[k] -= LR * mh / (np.sqrt(vh) + eps)

    if epoch % 200 == 0 or epoch == 1:
        tr_mse = mseloss(forward(params, Xtr)[0], Ytr)
        va_mse = mseloss(forward(params, Xva)[0], Yva)
        va_rmse_rad = np.sqrt(va_mse * np.mean(std_a ** 2))
        print(f"epoch {epoch:4d}  train_mse {tr_mse:.5f}  val_mse {va_mse:.5f}  "
              f"val_rmse ≈ {np.degrees(va_rmse_rad):.2f}°")

np.savez("bc_model.npz", **params,
         mean_s=mean_s, std_s=std_s, mean_a=mean_a, std_a=std_a)
print("saved -> bc_model.npz")
