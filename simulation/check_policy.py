import argparse
import glob

import numpy as np
import mujoco
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser(description="离线对比 ONNX 策略输出与演示真实动作")
    p.add_argument("--onnx", default="myarm_policy.onnx")
    p.add_argument("--demos-dir", default="demos")
    p.add_argument("--decimate", type=int, default=10)
    p.add_argument("--episodes", type=int, default=10, help="用前 N 条做对比")
    p.add_argument("--good", type=float, default=0.05, help="判定达标 L1 阈值(rad)")
    args = p.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    chunk = sess.get_outputs()[0].shape[1]
    state_dim = sess.get_inputs()[0].shape[1]
    print(f"模型: state {state_dim}d -> chunk {chunk}")

    model = mujoco.MjModel.from_xml_path("robot.xml")
    data = mujoco.MjData(model)
    hid = model.body("head").id

    def head_z(qpos12):
        data.qpos[:] = qpos12
        mujoco.mj_kinematics(model, data)
        return float(data.xpos[hid, 2])

    files = sorted(glob.glob(f"{args.demos_dir}/episode_*.npz"))[:args.episodes]
    errs, pred_min_z, real_min_z = [], [], []
    for f in files:
        d = np.load(f)
        q, a = d["qpos"], d["ctrl"]
        q, a = q[::args.decimate], a[::args.decimate]
        for i in range(0, len(q) - chunk - 1, chunk):
            state = np.concatenate([q[i, :5], q[i, 5:8]]).astype(np.float32)[None, :]
            pred = sess.run(None, {"state": state})[0][0]
            real = a[i:i + chunk]
            errs.append(np.abs(pred - real).mean(axis=1).mean())

            for j in range(0, chunk, 10):
                pred_min_z.append(head_z(np.concatenate([pred[j], q[i, 5:12]])))
                real_min_z.append(head_z(np.concatenate([real[j], q[i, 5:12]])))

    errs = np.array(errs)
    print(f"对比样本: {len(errs)} 个动作块 (来自 {len(files)} 条演示)")
    print(f"L1 误差(rad):  mean={errs.mean():.4f} ({np.degrees(errs.mean()):.2f} deg)  "
          f"50%={np.percentile(errs, 50):.4f}  90%={np.percentile(errs, 90):.4f}")
    print(f"head 最低高度:  演示动作 {min(real_min_z):.3f} m   策略预测 {min(pred_min_z):.3f} m")
    if errs.mean() < args.good:
        print("结论: 达标, 可以进仿真评估")
    elif errs.mean() < 0.1:
        print("结论: 接近达标, 可进仿真试试, 建议再续训")
    else:
        print(f"结论: 欠拟合 (mean L1 > {args.good}), 继续训练或提高学习率")


if __name__ == "__main__":
    main()
