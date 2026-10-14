"""RK3588 板端 ACT 推理(结合 bubble.json 校准做限位/方向保护)。

部署到板子时, 需要把 calib.py / bubble.json 与本文件放同一目录。

安全策略(按顺序):
    1. 策略输出先截断到 ±1.57(与训练动作域一致)
    2. 再按 bubble.json 换算出的各关节弧度软限位截断
    3. 限速: 每步(1/FPS 秒)目标角变化不超过 --max-deg-per-step, 防止突跳
    4. 换算成舵机原始值后, 硬限幅到 [range_min, range_max] 才发送
    5. 若输出严重越限(超过限位 --abort-margin-deg), 立即停机退出

方向: drive_mode bit0=1 的关节, 弧度增大对应舵机原始值减小, 由 calib.ArmCalib
统一处理; 装配方向与仿真相反的关节, 改 bubble.json 里该关节的 drive_mode=1 即可。
"""

import argparse
import time

import numpy as np
import onnxruntime as ort

import calib

FPS = 50
CTRL_LIMIT = 1.57  # 策略动作域上限(与训练一致)


def read_joint_raws():
    raise NotImplementedError("读取 5 个舵机原始位置(0..4095), 按校准文件 id 顺序返回")


def read_cup_xyz():
    raise NotImplementedError("读取杯子/目标 xyz, 返回 shape (3,)")


def send_raw_targets(raws):
    raise NotImplementedError("把 5 个舵机目标原始值(已按校准限位)写入舵机")


def read_state():
    raise NotImplementedError("(--no-calib 模式) 读取 5 个关节角 + 杯子 xyz, 返回 shape (8,) float32")


def send_action(joint_targets):
    raise NotImplementedError("(--no-calib 模式) 把 5 个关节目标角写入舵机/控制器")


def main():
    p = argparse.ArgumentParser(description="RK3588 板端 ACT 推理")
    p.add_argument("--model", default="act_policy.onnx")
    p.add_argument("--calib", default="bubble.json")
    p.add_argument("--no-calib", action="store_true",
                   help="不加载校准文件(仅联调用, 无限位保护)")
    p.add_argument("--steps-per-infer", type=int, default=100,
                   help="每次推理执行多少个动作(默认=chunk_size, 即每 2s 推理一次)")
    p.add_argument("--max-steps", type=int, default=1000, help="总步数, 0 为无限")
    p.add_argument("--max-deg-per-step", type=float, default=2.0,
                   help="限速: 单步(1/FPS 秒)允许的最大角度变化(度)")
    p.add_argument("--abort-margin-deg", type=float, default=20.0,
                   help="策略输出超出关节限位达到该角度时立即停机(度)")
    args = p.parse_args()

    cal = None
    if not args.no_calib:
        cal = calib.load(args.calib)
        problems = cal.problems()
        if problems:
            raise SystemExit("校准文件自检失败, 拒绝运行真机:\n  " + "\n  ".join(problems))
        print(f"已加载校准 {args.calib} (homing 单位: {cal.homing_unit})")
        print(cal.table())

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    max_step = np.radians(args.max_deg_per_step)
    abort_margin = np.radians(args.abort_margin_deg)
    last_rad = cal.raw_to_rad(read_joint_raws()) if cal is not None else None
    clip_warn_t = 0.0

    queue = []
    step = 0
    next_t = time.perf_counter()
    while args.max_steps == 0 or step < args.max_steps:
        if not queue:
            if cal is not None:
                state = np.concatenate([cal.raw_to_rad(read_joint_raws()), read_cup_xyz()])
            else:
                state = read_state()
            state = state.astype(np.float32)[None, :]
            t0 = time.perf_counter()
            queue = list(sess.run(None, {"state": state})[0][0])
            dt = (time.perf_counter() - t0) * 1000
            print(f"step {step}: 推理 {len(queue)} 个动作, 耗时 {dt:.0f} ms")

        action = np.clip(np.asarray(queue.pop(0), dtype=float), -CTRL_LIMIT, CTRL_LIMIT)
        if cal is not None:
            beyond = cal.beyond_radians(action)
            if np.max(beyond) > abort_margin:
                print(f"[ABORT] step {step}: 策略输出越限 {np.degrees(np.max(beyond)):.1f} 度, "
                      f"action={np.round(action, 3).tolist()}, "
                      f"限位={np.round(cal.rad_limits, 3).tolist()}, 停机")
                break
            if np.max(beyond) > 0 and time.perf_counter() - clip_warn_t > 5.0:
                clip_warn_t = time.perf_counter()
                print(f"[warn] step {step}: 策略输出超出关节限位, 已截断 {np.degrees(np.max(beyond)):.1f} 度")
            action = cal.clip_radians(action)
            if last_rad is not None:
                action = last_rad + np.clip(action - last_rad, -max_step, max_step)
            last_rad = action
            send_raw_targets(cal.rad_to_raw(action))
        else:
            send_action(action)
        step += 1

        next_t += 1.0 / FPS
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.perf_counter()


if __name__ == "__main__":
    main()
