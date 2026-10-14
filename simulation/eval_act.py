import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
from pynput import keyboard

import calib

CUP_XR = (-0.07, 0.07)
CUP_YR = (-0.26, -0.21)
CUP_Z = 0.04
DECIMATE = 10


def main():
    p = argparse.ArgumentParser(description="在 MuJoCo 仿真中回放 ONNX ACT 策略")
    p.add_argument("--onnx", default="act_policy.onnx")
    p.add_argument("--chunk-exec", type=int, default=100,
                   help="每次推理执行动作块中的前多少步")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="headless 模式下的步数限制")
    p.add_argument("--calib", default="bubble.json", help="校准文件路径")
    p.add_argument("--no-calib", action="store_true",
                   help="不加载校准文件, 退回统一的 ±1.57 限位")
    args = p.parse_args()

    cal = None
    if not args.no_calib:
        cal = calib.load(args.calib)
        problems = cal.problems()
        if problems:
            raise SystemExit("校准文件自检失败:\n  " + "\n  ".join(problems))
        print(f"关节限位({args.calib}, homing 单位: {cal.homing_unit}):")
        print(cal.table())

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    model = mujoco.MjModel.from_xml_path("robot.xml")
    data = mujoco.MjData(model)
    cup_jid = model.joint("cup_joint").id
    CUP_Q = model.jnt_qposadr[cup_jid]

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
    if not args.headless:
        listener.start()

    rng = np.random.default_rng()

    def reset_scene(arm="home"):
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
        data.qpos[CUP_Q + 4:CUP_Q + 7] = 0.0
        mujoco.mj_forward(model, data)

    def get_obs():
        return np.concatenate([data.qpos[:5], data.qpos[CUP_Q:CUP_Q + 3]]).astype(np.float32)[None, :]

    def new_chunk():
        actions = sess.run(None, {"state": get_obs()})[0][0]
        return list(actions[:args.chunk_exec])

    reset_scene()
    queue = new_chunk()
    cup_start = data.qpos[CUP_Q:CUP_Q + 3].copy()
    step_dt = model.opt.timestep
    next_step = time.perf_counter()
    last_print = 0.0
    prev_pressed = set()
    phys_step = 0

    import contextlib
    ctx = mujoco.viewer.launch_passive(model, data) if not args.headless else contextlib.nullcontext()
    try:
        with ctx:
            while args.headless or ctx.is_running():
                if 'r' in pressed and 'r' not in prev_pressed:
                    reset_scene()
                    queue = new_chunk()
                    cup_start = data.qpos[CUP_Q:CUP_Q + 3].copy()
                    print(f"\nreset -> cup {np.round(cup_start, 3)}")
                if 'p' in pressed and 'p' not in prev_pressed:
                    reset_scene(arm="random")
                    queue = new_chunk()
                    cup_start = data.qpos[CUP_Q:CUP_Q + 3].copy()
                    print(f"\nrandom -> qpos {np.round(data.qpos[:5], 2)}")
                prev_pressed = set(pressed)

                if phys_step % DECIMATE == 0 and not queue:
                    queue = new_chunk()

                if queue:
                    action = np.clip(queue[0], -1.57, 1.57)
                    if cal is not None:
                        action = cal.clip_radians(action)
                    data.ctrl[:] = action
                    if phys_step % DECIMATE == DECIMATE - 1:
                        queue.pop(0)

                mujoco.mj_step(model, data)
                phys_step += 1
                if not args.headless:
                    ctx.sync()

                if args.headless and args.max_steps and phys_step >= args.max_steps:
                    break

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
                    print(f"\rt={data.time:6.1f}s  cup_disp={disp:.3f}m  "
                          f"qpos={np.round(data.qpos[:5], 2)}  queue={len(queue)}", end="")
    finally:
        if not args.headless:
            listener.stop()
    print()


if __name__ == "__main__":
    main()
