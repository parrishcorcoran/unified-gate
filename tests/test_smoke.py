"""Smoke test: verify the package imports and runs on synthetic inputs.

This test does NOT verify numerical fidelity against the trained gate's
training distribution — for that, see scripts/reproduce.py which runs on
cached BitNet data. This test just ensures the API contract holds.
"""
import numpy as np

from unified_gate import Gate, extract_all_features, FEATURE_NAMES, N_FEATURES


def _synthetic_inputs(T=64, H=32, V=200, K=4, seed=0):
    rng = np.random.default_rng(seed)
    hidden_last = rng.standard_normal((T, H)).astype(np.float32)
    hidden_mid = rng.standard_normal((T, H)).astype(np.float32)
    hidden_early = rng.standard_normal((T, H)).astype(np.float32)
    head_logits = rng.standard_normal((T, K, V)).astype(np.float32)
    lm_head = rng.standard_normal((V, H)).astype(np.float32)
    tokens = rng.integers(0, V, size=T, dtype=np.uint32)
    return hidden_last, hidden_mid, hidden_early, head_logits, lm_head, tokens


def test_feature_count_consistent():
    assert N_FEATURES == 70
    assert len(FEATURE_NAMES) == N_FEATURES


def test_extract_all_features_shape():
    T = 64
    inputs = _synthetic_inputs(T=T)
    X = extract_all_features(*inputs)
    assert X.shape == (T - 8, N_FEATURES), f"expected ({T-8}, {N_FEATURES}), got {X.shape}"
    assert X.dtype == np.float32
    assert np.isfinite(X).all(), "non-finite values in extracted features"


def test_gate_score_range():
    inputs = _synthetic_inputs(T=64)
    X = extract_all_features(*inputs)
    gate = Gate("gate_k20.pt")
    scores = gate.score(X)
    assert scores.shape == (X.shape[0],)
    assert ((scores >= 0.0) & (scores <= 1.0)).all(), "scores outside [0, 1]"


def test_gate_skip_mask_frontier():
    inputs = _synthetic_inputs(T=128)
    X = extract_all_features(*inputs)
    gate = Gate("gate_k20.pt")
    mask_95 = gate.skip_mask(X, fidelity=0.95)
    mask_85 = gate.skip_mask(X, fidelity=0.85)
    assert mask_95.dtype == bool
    # Lower λ → looser threshold → at least as many skips
    assert mask_85.sum() >= mask_95.sum()


def test_gate_checkpoint_contents():
    g = Gate("gate_k20.pt")
    assert g.K == 20
    assert len(g.feature_names) == 20
    assert len(g.feature_indices) == 20
    assert all(0 <= i < N_FEATURES for i in g.feature_indices)


if __name__ == "__main__":
    test_feature_count_consistent()
    test_extract_all_features_shape()
    test_gate_score_range()
    test_gate_skip_mask_frontier()
    test_gate_checkpoint_contents()
    print("all smoke tests passed")
