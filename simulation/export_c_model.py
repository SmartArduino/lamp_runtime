import numpy as np

m = np.load("bc_model.npz")
W1, b1, W2, b2, W3, b3 = (m[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"])
ms, ss, ma, sa = m["mean_s"], m["std_s"], m["mean_a"], m["std_a"]

W1f = W1 / ss[:, None]
b1f = b1 - (ms / ss) @ W1
W3f = W3 * sa
b3f = b3 * sa + ma


def forward_orig(s):
    sn = (s - ms) / ss
    h1 = np.maximum(sn @ W1 + b1, 0)
    h2 = np.maximum(h1 @ W2 + b2, 0)
    return np.clip((h2 @ W3 + b3) * sa + ma, -1.57, 1.57)


def forward_folded(s):
    h1 = np.maximum(s @ W1f + b1f, 0)
    h2 = np.maximum(h1 @ W2 + b2, 0)
    return np.clip(h2 @ W3f + b3f, -1.57, 1.57)


rng = np.random.default_rng(0)
test = rng.uniform(-1.57, 1.57, (1024, W1.shape[0]))
diff = np.abs(forward_orig(test) - forward_folded(test)).max()
print(f"fold check max diff: {diff:.2e}")
assert diff < 1e-5


def emit(f, name, a):
    a = np.ascontiguousarray(a, dtype=np.float32).ravel()
    rows = []
    for i in range(0, len(a), 8):
        rows.append(", ".join(f"{v:.9g}" for v in a[i:i + 8]))
    f.write(f"static const float {name}[{len(a)}] = {{\n    " + ",\n    ".join(rows) + ",\n};\n\n")


with open("bc_model.h", "w") as f:
    f.write("#ifndef BC_MODEL_H\n#define BC_MODEL_H\n\n")
    f.write(f"#define BC_IN_DIM {W1.shape[0]}\n")
    f.write(f"#define BC_H1 {W1.shape[1]}\n")
    f.write(f"#define BC_H2 {W2.shape[1]}\n")
    f.write(f"#define BC_OUT {W3.shape[1]}\n\n")
    emit(f, "W1", W1f.T)
    emit(f, "b1", b1f)
    emit(f, "W2", W2.T)
    emit(f, "b2", b2)
    emit(f, "W3", W3f.T)
    emit(f, "b3", b3f)
    f.write("#endif\n")

print("saved -> bc_model.h")
