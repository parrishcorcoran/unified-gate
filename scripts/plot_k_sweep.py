"""Plot the headline K-sweep finding: skip@λ=0.95 vs K, with 5-seed error bars.

Shows the over-parameterization effect visually — the K=40-50 peak and the
drop at K=70. This is the plot to put on social media.

Numbers baked from MedusaBitNet's test_k_robustness.py 5-seed sweep
(commit 5398c0b). Regenerate with that script if you retrain the gate.

Output: figures/k_sweep.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 5-seed K-robustness results (mean, std) at λ=0.95 from test_k_robustness.py
K_VALUES   = [15,    20,    25,    30,    40,    50,    70]
MEAN_SKIPS = [0.0917, 0.0977, 0.1009, 0.1045, 0.1061, 0.1066, 0.0974]
STD_SKIPS  = [0.0027, 0.0016, 0.0018, 0.0027, 0.0021, 0.0016, 0.0034]


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)

    ax.errorbar(
        K_VALUES, [100 * m for m in MEAN_SKIPS],
        yerr=[100 * s for s in STD_SKIPS],
        fmt="o-", color="#2b6cb0", linewidth=2.2, markersize=8,
        capsize=4, capthick=1.5, ecolor="#718096",
        label="measured (5-seed mean ± std)",
    )

    # Mark peak and the over-parameterization drop
    peak_i = int(np.argmax(MEAN_SKIPS))
    ax.axhline(100 * MEAN_SKIPS[peak_i], linestyle="--", color="#48bb78",
               alpha=0.5, linewidth=1.0)
    ax.annotate(
        f"peak (K={K_VALUES[peak_i]}): {100 * MEAN_SKIPS[peak_i]:.1f}%",
        xy=(K_VALUES[peak_i], 100 * MEAN_SKIPS[peak_i]),
        xytext=(K_VALUES[peak_i] + 2, 100 * MEAN_SKIPS[peak_i] + 0.3),
        fontsize=10, color="#2f855a",
    )
    ax.annotate(
        "over-parameterized\n(~3σ drop)",
        xy=(70, 100 * MEAN_SKIPS[-1]),
        xytext=(55, 9.2),
        fontsize=10, color="#c53030",
        arrowprops=dict(arrowstyle="->", color="#c53030", alpha=0.75),
    )

    # Physics-ceiling line at intrinsic dim 7
    ax.axvline(7, linestyle=":", color="#805ad5", alpha=0.7)
    ax.text(7.3, 8.7, "intrinsic dim ≈ 7\n(TwoNN, per-seq)",
            fontsize=9, color="#553c9a")

    ax.set_xlabel("K  (number of top-importance features in gate MLP)", fontsize=11)
    ax.set_ylabel("skip @ λ=0.95 fidelity (%)", fontsize=11)
    ax.set_title(
        "unified-gate:  engineering knee = physics ceiling\n"
        "BitNet b1.58 2B, held-out, 5-seed replication",
        fontsize=12, pad=12,
    )
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_xlim(0, 75)
    ax.set_ylim(7.5, 11.2)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    out_path = Path(__file__).resolve().parents[1] / "figures" / "k_sweep.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}  ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
