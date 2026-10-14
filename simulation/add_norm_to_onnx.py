import argparse
import glob

import numpy as np
import onnx
from onnx import TensorProto, helper

from convert_to_lerobot import load_episode


def compute_stats(demos_dir, decimate):
    states, actions = [], []
    for f in sorted(glob.glob(f"{demos_dir}/episode_*.npz")):
        s, a, _ = load_episode(f, True, "abs", decimate)
        states.append(s)
        actions.append(a)
    states = np.concatenate(states).astype(np.float32)
    actions = np.concatenate(actions).astype(np.float32)
    return states.mean(0), np.maximum(states.std(0), 1e-6), actions.mean(0), np.maximum(actions.std(0), 1e-6)


def main():
    p = argparse.ArgumentParser(description="把 state 归一化 / action 反归一化烘焙进 ONNX 图")
    p.add_argument("--infile", default="myarm_policy.onnx")
    p.add_argument("--outfile", default=None)
    p.add_argument("--demos-dir", default="demos")
    p.add_argument("--decimate", type=int, default=10)
    args = p.parse_args()
    outfile = args.outfile or args.infile.replace(".onnx", "_raw.onnx")

    s_mean, s_std, a_mean, a_std = compute_stats(args.demos_dir, args.decimate)
    print("state mean/std:", np.round(s_mean, 3), "/", np.round(s_std, 3))
    print("action mean/std:", np.round(a_mean, 3), "/", np.round(a_std, 3))

    model = onnx.load(args.infile)
    g = model.graph
    IN, OUT = "state", "action_chunk"
    assert g.input[0].name == IN and g.output[0].name == OUT, "模型输入输出名不符合预期"

    in_dims = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    out_dims = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]

    g.input[0].name = "state_norm"
    del g.input[0]
    for node in g.node:
        for i, name in enumerate(node.input):
            if name == IN:
                node.input[i] = "state_norm"
    for node in g.node:
        for i, name in enumerate(node.output):
            if name == OUT:
                node.output[i] = "action_norm"
    for o in list(g.output):
        if o.name == OUT:
            g.output.remove(o)

    g.input.append(helper.make_tensor_value_info(IN, TensorProto.FLOAT, in_dims))
    g.output.append(helper.make_tensor_value_info(OUT, TensorProto.FLOAT, out_dims))

    def const(name, arr):
        g.initializer.append(onnx.numpy_helper.from_array(arr.astype(np.float32), name))

    const("norm.s_mean", s_mean)
    const("norm.s_std", s_std)
    const("norm.a_mean", a_mean)
    const("norm.a_std", a_std)

    pre = [
        helper.make_node("Sub", [IN, "norm.s_mean"], ["norm.t1"]),
        helper.make_node("Div", ["norm.t1", "norm.s_std"], ["state_norm"]),
    ]
    post = [
        helper.make_node("Mul", ["action_norm", "norm.a_std"], ["norm.t2"]),
        helper.make_node("Add", ["norm.t2", "norm.a_mean"], [OUT]),
    ]
    for i, node in enumerate(pre):
        g.node.insert(i, node)
    g.node.extend(post)

    onnx.checker.check_model(model)
    onnx.save(model, outfile)
    print(f"已生成: {outfile}  (原始值进, 原始值出)")


if __name__ == "__main__":
    main()
