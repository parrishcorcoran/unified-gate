"""End-to-end: run the gate on cached BitNet 2B hidden states, measure the
held-out skip/fidelity frontier, and compare against the frontier stored in
``gate_k20.pt``.

Requires the parent MedusaBitNet repo for the cached data layout:
  --medusabitnet-root <path-to-MedusaBitNet>

Usage:
  python scripts/reproduce.py --medusabitnet-root /path/to/MedusaBitNet

Expected output: held-out frontier within a few percent of the stored values
(small differences arise from K-means cluster centers and boundary-token
tokenizer sets, which are estimated from array inputs here vs. the full
tokenizer vocabulary in training).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_gate import Gate, extract_all_features, N_FEATURES


SEQ_LEN = 2048
HIDDEN = 2560
VOCAB = 128256
N_SEQS = 48
TRAIN_SPLIT = 36


def load_bf16_memmap(path, n_seqs, seq_len, hidden):
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    per_seq = seq_len * hidden
    seqs = []
    for si in range(n_seqs):
        chunk = mm[si * per_seq : (si + 1) * per_seq]
        raw = (chunk.astype(np.uint32) << 16)
        seqs.append(raw.view(np.float32).reshape(seq_len, hidden).copy())
    return seqs


def frontier(scores, correct, targets=(0.85, 0.90, 0.95, 0.99)):
    order = np.argsort(-scores)
    sorted_c = correct[order]
    cum = np.cumsum(sorted_c)
    counts = np.arange(1, len(sorted_c) + 1)
    fid = cum / counts
    out = []
    for lam in targets:
        ok = fid >= lam
        if not ok.any():
            out.append((lam, 0.0, 0.0))
            continue
        i = int(np.where(ok)[0][-1])
        out.append((lam, (i + 1) / len(sorted_c), float(fid[i])))
    return out


def fit_training_cluster_centers(h29_train, K=32, subsample=20000, seed=0):
    """Reproduce the training-time K-means centers exactly.

    Concatenate hidden states from the first 20 training seqs, subsample, fit
    K-means with 20 Lloyd iterations. This matches MedusaBitNet's
    test_physics_apertures.py::aperture_cluster.
    """
    train_arr = np.concatenate(h29_train, axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(train_arr), size=min(subsample, len(train_arr)), replace=False)
    kmeans_data = torch.from_numpy(train_arr[sub])
    idx0 = rng.choice(len(kmeans_data), size=K, replace=False)
    centers = kmeans_data[idx0].clone()
    for _ in range(20):
        dists = torch.cdist(kmeans_data, centers)
        assigns = dists.argmin(dim=-1)
        for k in range(K):
            mask = assigns == k
            if mask.sum() > 0:
                centers[k] = kmeans_data[mask].mean(dim=0)
    return centers.numpy()


def build_boundary_sets(tokenizer_dir):
    """Scan tokenizer vocab, return (period_ids, newline_ids) as sets."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    V = tok.vocab_size
    enders, newlines = set(), set()
    for tid in range(V):
        s = tok.decode([tid])
        if "\n" in s:
            newlines.add(tid)
        stripped = s.strip()
        if stripped and stripped[-1] in ".!?":
            enders.add(tid)
    return enders, newlines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medusabitnet-root", required=True, type=Path)
    p.add_argument("--gate", default=str(Path(__file__).resolve().parents[1] / "gate_k20.pt"))
    p.add_argument("--n-seqs", type=int, default=N_SEQS)
    p.add_argument("--tokenizer-dir", default=None,
                   help="Path to HF tokenizer; defaults to <root>/models/bitnet-b1.58-2B-4T")
    args = p.parse_args()

    root = args.medusabitnet_root
    assert (root / "data" / "hidden_gguf_v2.bin").exists(), f"missing {root}/data/hidden_gguf_v2.bin"

    sys.path.insert(0, str(root))
    from model import MedusaHeads

    print(f"Loading MedusaBitNet artifacts from {root}...")
    ckpt = torch.load(root / "checkpoints" / "full_gguf_shift" / "medusa_heads_step1000.pt",
                      map_location="cpu", weights_only=True)
    heads = MedusaHeads(HIDDEN, VOCAB, 4, 1, dtype=torch.bfloat16).eval()
    heads.load_state_dict(ckpt["heads"])
    lm_head = torch.load(root / "data" / "lm_head.pt", map_location="cpu",
                         weights_only=True).to(torch.bfloat16)
    lm_head_np = lm_head.float().numpy()

    tokens_mm = np.memmap(root / "data" / "tokens.bin", dtype=np.uint32, mode="r")
    print("  loading hidden states at layers 5, 15, and 29 (result_norm)...")
    h29_all = load_bf16_memmap(root / "data" / "hidden_gguf_v2.bin", args.n_seqs, SEQ_LEN, HIDDEN)
    h15_all = load_bf16_memmap(root / "data" / "hidden_gguf_layer15.bin", args.n_seqs, SEQ_LEN, HIDDEN)
    h5_all  = load_bf16_memmap(root / "data" / "hidden_gguf_layer5.bin", args.n_seqs, SEQ_LEN, HIDDEN)

    print("  fitting K-means cluster centers on 20 training sequences...")
    centers = fit_training_cluster_centers(h29_all[:20])

    print("  scanning tokenizer vocab for sentence-boundary IDs...")
    tokenizer_dir = args.tokenizer_dir or str(root / "models" / "bitnet-b1.58-2B-4T")
    period_ids, newline_ids = build_boundary_sets(tokenizer_dir)
    print(f"    |period_ids|={len(period_ids)}  |newline_ids|={len(newline_ids)}")

    print("Extracting features per sequence...")
    feats_test = []
    labels_test = []
    with torch.no_grad():
        for si in range(args.n_seqs):
            is_test = si >= TRAIN_SPLIT
            if not is_test:
                continue

            h_last = h29_all[si]
            h_mid = h15_all[si]
            h_early = h5_all[si]
            tokens = np.asarray(tokens_mm[si * SEQ_LEN : (si + 1) * SEQ_LEN], dtype=np.uint32)

            h_t = torch.from_numpy(h_last).to(torch.bfloat16)
            logits = heads(h_t.unsqueeze(0), lm_head)[0].float().numpy()  # [T, K, V]

            # label = head-0 argmax matches the verifier's next-token prediction
            # verifier prediction: argmax of lm_head @ h[t+1]  — i.e. one-step lookahead
            h_next = torch.from_numpy(h_last[1:]).to(torch.bfloat16)
            vlogits = h_next @ lm_head.T.to(torch.bfloat16)
            vpred = vlogits.float().argmax(-1).numpy()
            h0pred = logits[:, 0, :].argmax(-1)

            X = extract_all_features(
                hidden_last=h_last, hidden_mid=h_mid, hidden_early=h_early,
                head_logits=logits, lm_head=lm_head_np, tokens=tokens,
                period_ids=period_ids, newline_ids=newline_ids,
                cluster_centers=centers,
            )
            # labels correspond to ts = np.arange(6, SEQ_LEN - 2)
            ts = np.arange(6, SEQ_LEN - 2, dtype=np.int64)
            y = (h0pred[ts] == vpred[ts]).astype(np.float32)
            assert X.shape[0] == y.shape[0], f"shape mismatch {X.shape} vs {y.shape}"

            feats_test.append(X)
            labels_test.append(y)
            print(f"  seq {si+1}/{args.n_seqs}  shape={X.shape}")

    X_test = np.concatenate(feats_test, axis=0)
    y_test = np.concatenate(labels_test, axis=0)
    print(f"\nTest set: {X_test.shape}, positives rate = {y_test.mean():.4f}")

    print("\nLoading gate...")
    gate = Gate(args.gate)
    scores = gate.score(X_test)

    print("\n=== Measured frontier (this run) ===")
    measured = frontier(scores, y_test)
    for lam, skip, fid in measured:
        print(f"  λ={lam:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

    print("\n=== Stored frontier (gate_k20.pt) ===")
    for lam, skip, fid in gate.frontier:
        print(f"  λ={lam:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")


if __name__ == "__main__":
    main()
