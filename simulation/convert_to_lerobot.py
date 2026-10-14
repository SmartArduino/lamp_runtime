import argparse
import glob
import os

import numpy as np


def load_episode(path, trim, action_mode, decimate):
    d = np.load(path)
    q, a, t = d["qpos"], d["ctrl"], d["t"]
    if trim:
        arm_move = (np.abs(a).max(1) > 0.05) | (np.abs(q[:, :5]).max(1) > 0.05)
        i0 = int(arm_move.argmax()) if arm_move.any() else 0
        cup_d = np.linalg.norm(np.diff(q[:, 5:7], axis=0), axis=1)
        i1 = len(q) - 1
        if (cup_d > 1e-4).any():
            i1 = min(len(q) - 1, int(np.where(cup_d > 1e-4)[0][-1]) + 500)
        q, a, t = q[i0:i1 + 1], a[i0:i1 + 1], t[i0:i1 + 1]
    dt = float(np.median(np.diff(t)))
    q, a = q[::decimate], a[::decimate]
    state = np.concatenate([q[:, :5], q[:, 5:8]], axis=1)
    action = a - q[:, :5] if action_mode == "delta" else a
    return state.astype(np.float32), action.astype(np.float32), 1.0 / (dt * decimate)


def make_dataset(repo_id, fps, root, features):
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            raise SystemExit(
                "lerobot 未安装或缺少依赖, 请执行: pip install 'lerobot[dataset]'"
            )
    return LeRobotDataset, LeRobotDataset.create(
        repo_id=repo_id, fps=fps, root=root, features=features
    )


def add_frame_with_task(dataset, frame, task, mode):
    if mode == "kwarg":
        dataset.add_frame(frame, task=task)
    elif mode == "frame":
        dataset.add_frame({**frame, "task": task})
    else:
        try:
            dataset.add_frame(frame, task=task)
            return "kwarg"
        except TypeError:
            try:
                dataset.add_frame({**frame, "task": task})
                return "frame"
            except (TypeError, ValueError, KeyError):
                dataset.add_frame(frame)
                return "save"
    return mode


def save_episode_with_task(dataset, task, mode):
    if mode == "save":
        try:
            dataset.save_episode(task=task)
        except TypeError:
            dataset.save_episode()
    else:
        dataset.save_episode()


def main():
    p = argparse.ArgumentParser(
        description="把 record_demo.py 录的 demos/episode_*.npz 转成 LeRobotDataset",
        epilog="示例: python convert_to_lerobot.py --repo-id local/myarm_push",
    )
    p.add_argument("--demos-dir", default="demos")
    p.add_argument("--repo-id", required=True, help="如 local/myarm_push 或 <HF用户名>/<数据集名>")
    p.add_argument("--task", default="Push the cup to the target zone")
    p.add_argument("--decimate", type=int, default=10, help="降采样倍数, 10 即 500Hz->50fps")
    p.add_argument("--action", choices=["abs", "delta"], default="abs",
                   help="abs=关节位置目标(推荐), delta=ctrl-qpos(与 train_bc.py 一致, 评估时要加回当前关节角)")
    p.add_argument("--root", default=None, help="数据集保存目录, 默认 ~/.cache/huggingface/lerobot/")
    p.add_argument("--no-trim", action="store_true", help="不裁剪首尾空闲段")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.demos_dir, "episode_*.npz")))
    if not files:
        raise SystemExit(f"{args.demos_dir}/ 下没有 episode_*.npz")

    _, _, fps = load_episode(files[0], not args.no_trim, args.action, args.decimate)
    fps = round(fps)

    features = {
        "observation.state": {"dtype": "float32", "shape": (8,)},
        "action": {"dtype": "float32", "shape": (5,)},
    }
    DatasetCls, dataset = make_dataset(args.repo_id, fps, args.root, features)

    total, mode = 0, None
    for f in files:
        state, action, _ = load_episode(f, not args.no_trim, args.action, args.decimate)
        for s, a in zip(state, action):
            mode = add_frame_with_task(dataset, {"observation.state": s, "action": a},
                                       args.task, mode)
        save_episode_with_task(dataset, args.task, mode)
        total += len(state)
        print(f"{f}: {len(state)} frames")

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()

    check = DatasetCls(args.repo_id, root=args.root)
    sample = check[0]
    print(f"\n完成: {len(files)} episodes / {total} frames @ {fps} fps")
    print(f"observation.state shape={tuple(sample['observation.state'].shape)}, "
          f"action shape={tuple(sample['action'].shape)}")
    print(f"训练: lerobot-train --dataset.repo_id={args.repo_id} --policy.type=act "
          f"--output_dir=outputs/train/act_myarm")


if __name__ == "__main__":
    main()
