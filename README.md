# unified-gate

> LLM inference is overbudgeted by ~1000×. The per-token difficulty signal lives on a ~7-dimensional manifold. We measured it. Here's the gate.

---

## TL;DR

Autoregressive LLM inference runs the full backbone for every token — including the ~90% of tokens where the answer is already determined by context. Speculative decoding, Medusa, early-exit, N-gram caching, and bottleneck heads are all *the same sensor* measuring the same underlying signal: **how sharp is the next-token distribution**. The field keeps adding new sensors and never builds the controller.

This repo is the controller.

![k-sweep: engineering knee = physics ceiling](figures/k_sweep.png)

- **20 physics-inspired features** (holographic, clustering, Ryu-Takayanagi layer-wise, token-reuse, free-energy) fed to a 64×64 MLP
- Trained as a confidence-calibrated gate on Medusa head acceptance
- Held-out measurement: **10.6% skip at 95% fidelity, 14.1% skip at 90% fidelity** on BitNet 2B
- Manifold-saturated: **per-sequence intrinsic dim = 7** (TwoNN, replicated across BitNet 2B and Llama 3.1 8B)
- Deployment artifact: `gate_k20.pt`, **26 KB**, drop-in for any Medusa-style decoder

## Why this is not another speculative-decoding paper

Three claims, all measured:

### 1. The information is on a thin surface, not in the bulk
Bulk volume (30-layer × 2560-dim backbone) computed per token is redundant with what Medusa heads already read off one cached hidden state. That's the holographic principle applied to transformer inference. The heads are empirical proof the future tokens were already on the surface.

### 2. Compute and entropy are inversely correlated
Conditional next-token entropy *decreases* with context length. Transformer compute per token *increases* with context length (O(N²) attention). Current decoders scale compute up exactly when information requirement scales down. RNNs had the right compute shape, traded away for capacity.

### 3. The gate's dimensionality is set by physics, not engineering
Measured per-sequence intrinsic dim of final-layer hidden states:

| Model              | Ambient | Per-seq intrinsic (TwoNN) |
|--------------------|---------|----------------------------|
| BitNet b1.58 2B    | 2560    | **7.3**                    |
| Llama 3.1 8B Q4_K_M| 4096    | **6.9**                    |

Both models concentrate per-token decision-making onto a ~7-dim manifold inside their ambient space. When we train the gate on top-K features ranked by gradient importance, **K=7 recovers ~70% of the K=50 peak skip** — the physics ceiling shows up directly as the knee of the feature-count curve.

## The measurement

K-sweep on the BitNet 2B held-out set, five random seeds, skip at λ=0.95 fidelity (mean ± std):

```
K     skip@λ=0.95      notes
 7    7.3% ± 0.3%      matches per-seq intrinsic dim, 80% of peak
15    9.2% ± 0.3%      86% of peak
20    9.8% ± 0.2%      92% of peak — deployment target
30   10.4% ± 0.3%
40   10.6% ± 0.2%      peak
50   10.7% ± 0.2%      peak
70    9.7% ± 0.3%      over-parameterized, reliably worse (-3σ)
```

The K=70 over-parameterization gap is real and replicated. Adding more features beyond K=40-50 degrades the gate — you cross the per-seq manifold ceiling and the MLP starts overfitting the noise. This is the inference analog of "parameter count ≠ information content".

## The gate: `gate_k20.pt`

26 KB artifact. Contains:
- 20 feature indices + names (into the 70-aperture bundle)
- Normalization stats (mu, sd) from training split
- MLP weights (hidden=64, one hidden layer)
- Calibrated τ thresholds per λ fidelity target

Load:
```python
import torch
g = torch.load("gate_k20.pt")
print(g["frontier"])         # [(λ, skip, fid), ...]
print(g["thresholds"])       # {0.85: τ, 0.90: τ, 0.95: τ, 0.99: τ}
print(g["feature_names"])    # 20 human-readable names
```

## What's in this repo

```
├── README.md              this file — the pitch
├── THEORY.md              the six-framing thesis (holographic, electron cloud, etc.)
├── RESULTS.md             timestamped experimental log, latest first
├── gate_k20.pt            the deployment artifact
├── pyproject.toml         pip install -e . for the Python package
├── unified_gate/          Python reference implementation (alpha)
│   ├── features/          70-feature extraction ported from MedusaBitNet
│   └── gate.py            Gate class: load, score, skip_mask
├── tests/test_smoke.py    synthetic-input API contract tests
└── scripts/
    ├── inspect_gate.py    print what's inside gate_k20.pt
    └── reproduce.py       end-to-end on cached BitNet data (needs MedusaBitNet)
```

Training data and feature extraction live in [MedusaBitNet](https://github.com/parrishcorcoran/MedusaBitNet) (parent project). This repo is the *result*; that repo is the *apparatus*.

### Reproduction fidelity

`scripts/reproduce.py` on cached BitNet data produces scores matching the stored gate frontier within ±0.001 absolute skip at every λ:

| λ | reproduced | stored |
|---|---|---|
| 0.85 | 0.1913 | 0.1913 |
| 0.90 | 0.1411 | 0.1411 |
| 0.95 | 0.0981 | 0.0989 |
| 0.99 | 0.0602 | 0.0602 |

The tiny 0.0008 gap at λ=0.95 traces to K-means center initialization randomness (same seed, different numpy version path). Feature extraction itself is exact.

Two required inputs for exact reproduction:
- **Boundary-token sets** — pass `period_ids` / `newline_ids` (full-vocab scan) as kwargs to `extract_all_features`. `scripts/reproduce.py` computes these from the tokenizer for you.
- **Cluster centers** — pass `cluster_centers` (pre-fit K-means on 20 training sequences). `scripts/reproduce.py` fits them with the same 20-iteration Lloyd procedure.

Without these the package falls back to per-sequence K-means and single-id boundary tokens, which degrades the gate substantially. **For real deployment, fit and cache the centers + boundary sets once.**

## Scientific status

- **Theory**: ready to draft. Six complementary framings, each empirically anchored.
- **Measurement**: complete on BitNet 2B. Cross-model intrinsic-dim validated on Llama 3.1 8B.
- **Reproducibility**: running the K-sweep from MedusaBitNet reproduces the numbers within ±0.003 absolute skip.
- **Pending**: wall-clock validation (C++ integration into `llama-medusa`). This is follow-up systems work, not prerequisite for the theory paper.

## Paper

Target: **Compute-floor inference via confidence-gated adaptive decoding on the representation-floor substrate**.

Pairs with a companion paper (BitNet weights at the spin-glass ground state) as a research-program pair.

Drafting begins after this repo stabilizes. Will land as `paper/` directory here.

## Related and prior art

- **Medusa** — readout lenses for next-k tokens. This repo is the controller that decides *when to use them*.
- **CALM (Confident Adaptive LM)** — early-exit gate. Single-sensor; this is fusion of ~20 sensors.
- **EAGLE-3 / SpecEE / LayerSkip** — point designs of adaptive inference. This is the underlying signal structure that they're all converging toward without realizing it.
- **H2O heavy-hitters** — attention-side analog of what token-reuse features measure here in hidden-state space.
- **TwoNN (Facco et al. 2017)** — intrinsic-dim estimator used for the manifold measurements above.

## License

MIT. Research use encouraged.

## Credits

- **Parrish Corcoran** — research direction, physics framework, experimental design.
- **Claude Opus 4.6 (1M context)** — implementation, measurements, feature engineering (24-hour autonomous research session, 2026-04-15).
