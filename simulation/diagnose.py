import glob
import numpy as np
import mujoco

m = np.load("bc_model.npz")
W1, b1, W2, b2, W3, b3 = (m[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"])
mean_s, std_s, mean_a, std_a = m["mean_s"], m["std_s"], m["mean_a"], m["std_a"]

files = sorted(glob.glob("demos/episode_*.npz"))
print("=== demos ===")
for f in files:
    d = np.load(f)
    cup = d["qpos"][:, 5:8]
    disp = np.linalg.norm(cup[-1, :2] - cup[0, :2])
    dur = d["t"][-1] - d["t"][0]
    print(f"{f.split(chr(92))[-1]}: {len(d['t']):6d} steps, {dur:6.1f}s, "
          f"ctrl_range=[{d['ctrl'].min():5.2f},{d['ctrl'].max():5.2f}], cup_disp={disp:.3f}m")


def predict(s, qpos):
    sn = (s - mean_s) / std_s
    h1 = np.maximum(sn @ W1 + b1, 0)
    h2 = np.maximum(h1 @ W2 + b2, 0)
    delta = (h2 @ W3 + b3) * std_a + mean_a
    return np.clip(qpos[:5] + delta, -1.57, 1.57)


model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
jid = model.joint("cup_joint").id
CQ = model.jnt_qposadr[jid]
HEAD_ID = model.body("head").id

print("\n=== policy rollout (no viewer, 40s sim time) ===")
cases = [
    ((0.0, -0.23), np.zeros(5)),
    ((0.05, -0.25), np.zeros(5)),
    ((-0.05, -0.21), np.zeros(5)),
    ((0.0, -0.23), np.array([0.4, -0.3, 0.2, 0.0, 0.0])),
]
for cup_xy, start_pose in cases:
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.ctrl[:] = 0
    data.qpos[:5] = start_pose
    data.qpos[CQ + 3] = 1.0
    data.qpos[CQ:CQ + 3] = [cup_xy[0], cup_xy[1], 0.04]
    mujoco.mj_forward(model, data)
    start = data.qpos[CQ:CQ + 3].copy()
    ctrl_smooth = predict(np.concatenate([data.qpos[:5], data.qvel[:5], data.xpos[HEAD_ID], data.qpos[CQ:CQ + 3]]), data.qpos)
    snaps = {}
    for i in range(20000):
        s = np.concatenate([data.qpos[:5], data.qvel[:5], data.xpos[HEAD_ID], data.qpos[CQ:CQ + 3]])
        pred = predict(s, data.qpos)
        ctrl_smooth = 0.9 * ctrl_smooth + 0.1 * pred
        data.ctrl[:] = ctrl_smooth
        mujoco.mj_step(model, data)
        t = data.time
        for mark in [2, 5, 10, 20, 40]:
            if mark not in snaps and t >= mark:
                snaps[mark] = (data.qpos[:5].copy(), ctrl_smooth.copy())
    cup_d = np.linalg.norm(data.qpos[CQ:CQ + 2] - start[:2])
    print(f"\ncup at {cup_xy}, start qpos={np.round(start_pose, 2)}: final cup_disp={cup_d:.3f}m")
    for mark in sorted(snaps):
        q, c = snaps[mark]
        print(f"  t={mark:2d}s qpos={np.round(q, 2)} ctrl={np.round(c, 2)}")

print("\n=== demo ctrl at same time marks (first 3 episodes) ===")
for f in files[:3]:
    d = np.load(f)
    print(f)
    for mark in [2, 5, 10, 20]:
        i = np.searchsorted(d["t"], d["t"][0] + mark)
        if i < len(d["t"]):
            print(f"  t={mark:2d}s qpos={np.round(d['qpos'][i, :5], 2)} ctrl={np.round(d['ctrl'][i], 2)}")
