"""Load and inspect gate_k20.pt — show the deployable artifact's contents.

Usage: python scripts/inspect_gate.py

No training data or dependencies on the parent MedusaBitNet repo — this
just unpacks the 26 KB checkpoint and prints what's inside.
"""
import torch

g = torch.load("gate_k20.pt", weights_only=False, map_location="cpu")

print(f"=== unified-gate K={g['K']} deployment artifact ===\n")

print(f"MLP: {g['K']} inputs → {g['mlp_hidden']} hidden → 1 output")
print(f"Trained with seed={g['seed']} for {g['epochs']} epochs\n")

print("=== Top-20 features (gradient-importance ranked) ===")
for rank, (idx, name) in enumerate(zip(g["feature_indices"], g["feature_names"]), 1):
    print(f"  {rank:2d}. [src_idx={idx:3d}] {name}")

print("\n=== Measured held-out frontier ===")
for (lam, skip, fid) in g["frontier"]:
    print(f"  λ={lam:.2f}  skip={skip:.4f}  fidelity={fid:.4f}")

print("\n=== Calibrated τ thresholds (score cutoffs for each λ) ===")
for lam, tau in g["thresholds"].items():
    print(f"  λ={lam:.2f}  τ={tau:.4f}")

print("\n=== Normalization stats ===")
print(f"  mu.shape = {g['mu'].shape}, sd.shape = {g['sd'].shape}")
print(f"  mu[0:5] = {g['mu'][:5]}")
print(f"  sd[0:5] = {g['sd'][:5]}")

print("\n=== MLP weights ===")
for k, v in g["mlp_state"].items():
    print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
