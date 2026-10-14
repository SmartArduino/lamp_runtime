import argparse
import json
import os

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import OBS_STATE
from lerobot.policies.factory import make_policy


class ActOnnx(torch.nn.Module):

    def __init__(self, policy, s_mean, s_std, a_mean, a_std):
        super().__init__()
        self.model = policy.model
        self.register_buffer("s_mean", torch.as_tensor(s_mean, dtype=torch.float32))
        self.register_buffer("s_std", torch.as_tensor(s_std, dtype=torch.float32))
        self.register_buffer("a_mean", torch.as_tensor(a_mean, dtype=torch.float32))
        self.register_buffer("a_std", torch.as_tensor(a_std, dtype=torch.float32))

    def forward(self, state):
        sn = (state - self.s_mean) / self.s_std
        out = self.model({OBS_STATE: sn})[0]
        return out * self.a_std + self.a_mean


def load_stats_from_json(path):
    with open(path) as f:
        d = json.load(f)
    return (np.array(d["observation.state"]["mean"], np.float32),
            np.maximum(np.array(d["observation.state"]["std"], np.float32), 1e-6),
            np.array(d["action"]["mean"], np.float32),
            np.maximum(np.array(d["action"]["std"], np.float32), 1e-6))


def compute_stats_from_demos(demos_dir, decimate):
    from add_norm_to_onnx import compute_stats
    return compute_stats(demos_dir, decimate)


def main():
    p = argparse.ArgumentParser(description="导出 ACT 策略为 ONNX(含归一化, 原始值进/原始值出)")
    p.add_argument("--pretrained", default=None,
                   help="训练产物目录, 如 outputs/train/act_myarm_v5/checkpoints/200000/pretrained_model")
    p.add_argument("--repo-id", default="local/myarm_act", help="训练用的数据集(读取其精确归一化stats)")
    p.add_argument("--stats-json", default=None, help="直接指定 meta/stats.json 路径(优先于 repo-id)")
    p.add_argument("--demos-dir", default=None, help="无数据集时从演示npz近似计算stats")
    p.add_argument("--decimate", type=int, default=10)
    p.add_argument("--out", default="myarm_policy_raw.onnx")
    args = p.parse_args()

    ds = LeRobotDataset(args.repo_id)
    state_dim = ds.meta.features["observation.state"]["shape"][0]
    action_dim = ds.meta.features["action"]["shape"][0]

    if args.stats_json:
        stats = load_stats_from_json(args.stats_json)
        print(f"stats 来源: {args.stats_json}")
    else:
        stats_path = os.path.join(str(ds.meta.root), "meta", "stats.json")
        if os.path.exists(stats_path):
            stats = load_stats_from_json(stats_path)
            print(f"stats 来源: {stats_path}")
        elif args.demos_dir:
            stats = compute_stats_from_demos(args.demos_dir, args.decimate)
            print(f"stats 来源: 由 {args.demos_dir} 近似计算")
        else:
            raise SystemExit("找不到 stats.json, 请用 --stats-json 或 --demos-dir 指定来源")
    s_mean, s_std, a_mean, a_std = stats

    if args.pretrained:
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(args.pretrained).to("cpu")
    else:
        cfg = ACTConfig()
        cfg.push_to_hub = False
        policy = make_policy(cfg=cfg, ds_meta=ds.meta)
        print("[警告] 未提供 --pretrained, 使用随机初始化权重(仅用于验证导出流程)")
    policy.eval()

    wrapper = ActOnnx(policy, s_mean, s_std, a_mean, a_std).eval()
    example = torch.zeros(1, state_dim, dtype=torch.float32)
    with torch.no_grad():
        ref = wrapper(example).numpy()

    torch.onnx.export(
        wrapper,
        example,
        args.out,
        input_names=["state"],
        output_names=["action_chunk"],
        opset_version=17,
        dynamo=False,
    )
    print(f"已导出: {args.out}  (原始 state (1,{state_dim}) -> 原始 action_chunk "
          f"(1,{policy.config.chunk_size},{action_dim}), 归一化已内置)")

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"state": example.numpy()})[0]
    diff = float(np.abs(out - ref).max())
    print(f"onnxruntime vs pytorch 最大误差: {diff:.2e}")
    if not np.isfinite(diff) or diff > 1e-4:
        raise SystemExit("误差过大, 请检查导出配置")
    print("验证通过")


if __name__ == "__main__":
    main()
