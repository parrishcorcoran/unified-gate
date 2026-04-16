"""Export gate_k20.pt to a flat binary for C++ consumption.

Layout (all float32, little-endian):
  Header:
    [4 bytes]  uint32  K (number of features = 20)
    [4 bytes]  uint32  hidden_dim (= 64)
    [4 bytes]  uint32  n_thresholds (= 4)
  Feature indices:
    [K * 4 bytes]  uint32  feature_indices[K]
  Normalization:
    [K * 4 bytes]  float32  mu[K]
    [K * 4 bytes]  float32  sd[K]
  MLP weights (layer 0: K → hidden):
    [hidden * K * 4 bytes]  float32  W0  (row-major)
    [hidden * 4 bytes]      float32  b0
  MLP weights (layer 1: hidden → hidden):
    [hidden * hidden * 4 bytes]  float32  W1
    [hidden * 4 bytes]           float32  b1
  MLP weights (layer 2: hidden → 1):
    [hidden * 4 bytes]  float32  W2  (1 × hidden, stored as hidden floats)
    [4 bytes]           float32  b2
  Thresholds:
    [n_thresholds * 8 bytes]  float32 lambda, float32 tau  (interleaved pairs)

Total: 4*3 + 20*4 + 20*4 + 64*20*4 + 64*4 + 64*64*4 + 64*4 + 64*4 + 4 + 4*8
     = 12 + 80 + 80 + 5120 + 256 + 16384 + 256 + 256 + 4 + 32
     = 22480 bytes (~22 KB)
"""
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def export(ckpt_path: str, out_path: str):
    g = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    K = int(g["K"])
    hidden = int(g["mlp_hidden"])
    indices = np.asarray(g["feature_indices"], dtype=np.uint32)
    mu = np.asarray(g["mu"], dtype=np.float32)
    sd = np.asarray(g["sd"], dtype=np.float32)
    thresholds = g["thresholds"]

    state = g["mlp_state"]
    W0 = state["net.0.weight"].float().numpy()  # [hidden, K]
    b0 = state["net.0.bias"].float().numpy()     # [hidden]
    W1 = state["net.2.weight"].float().numpy()  # [hidden, hidden]
    b1 = state["net.2.bias"].float().numpy()
    W2 = state["net.4.weight"].float().numpy()  # [1, hidden]
    b2 = state["net.4.bias"].float().numpy()     # [1]

    assert W0.shape == (hidden, K), f"W0 shape {W0.shape} != ({hidden}, {K})"
    assert W1.shape == (hidden, hidden)
    assert W2.shape == (1, hidden)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", K, hidden, len(thresholds)))
        f.write(indices.tobytes())
        f.write(mu.tobytes())
        f.write(sd.tobytes())
        f.write(W0.tobytes())
        f.write(b0.tobytes())
        f.write(W1.tobytes())
        f.write(b1.tobytes())
        f.write(W2.flatten().tobytes())
        f.write(b2.flatten().tobytes())
        for lam in sorted(thresholds.keys()):
            f.write(struct.pack("<ff", float(lam), float(thresholds[lam])))

    size = Path(out_path).stat().st_size
    print(f"exported {out_path}  ({size} bytes)")
    print(f"  K={K}  hidden={hidden}  thresholds={len(thresholds)}")

    # Verify roundtrip
    verify_roundtrip(out_path, g)


def verify_roundtrip(bin_path: str, g: dict):
    """Load the binary back and verify inference matches PyTorch."""
    data = open(bin_path, "rb").read()
    off = 0

    def read_u32():
        nonlocal off
        v = struct.unpack_from("<I", data, off)[0]; off += 4; return v
    def read_f32(n):
        nonlocal off
        v = np.frombuffer(data, dtype=np.float32, count=n, offset=off); off += n * 4
        return v.copy()

    K = read_u32(); hidden = read_u32(); n_thresh = read_u32()
    indices = np.frombuffer(data, dtype=np.uint32, count=K, offset=off).copy(); off += K * 4
    mu = read_f32(K); sd = read_f32(K)
    W0 = read_f32(hidden * K).reshape(hidden, K)
    b0 = read_f32(hidden)
    W1 = read_f32(hidden * hidden).reshape(hidden, hidden)
    b1 = read_f32(hidden)
    W2 = read_f32(hidden).reshape(1, hidden)
    b2 = read_f32(1)

    # Test inference on random input
    rng = np.random.default_rng(42)
    x_raw = rng.standard_normal((10, 70)).astype(np.float32)
    x = (x_raw[:, indices] - mu) / (sd + 1e-9)

    # C-style forward
    h = np.maximum(0, x @ W0.T + b0)
    h = np.maximum(0, h @ W1.T + b1)
    logits = (h @ W2.T + b2).squeeze(-1)
    scores_bin = 1.0 / (1.0 + np.exp(-logits))

    # PyTorch forward
    from unified_gate.gate import _GateMLP
    mlp = _GateMLP(K, hidden)
    mlp.load_state_dict(g["mlp_state"])
    mlp.eval()
    with torch.no_grad():
        scores_pt = torch.sigmoid(mlp(torch.from_numpy(x))).numpy()

    max_diff = np.abs(scores_bin - scores_pt).max()
    print(f"  roundtrip max|Δ| = {max_diff:.2e}  {'PASS' if max_diff < 1e-5 else 'FAIL'}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "gate_k20.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else ckpt.replace(".pt", ".bin")
    export(ckpt, out)
