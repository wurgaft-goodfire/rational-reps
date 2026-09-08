"""Unit tests for the array-side analysis: the readout projection, the comparison metrics, and
the alignment / decoding helpers."""


def test_readout_projection_keeps_exactly_what_the_readout_reads():
    # H = a signal in the centered readout's row space + a component the readout cannot read: the
    # projection has rank(D_c) coordinates, lifts back to the signal alone, and zeroes the rest
    import numpy as np
    from src.code.analysis_utils import readout_basis, readout_projection
    rng = np.random.default_rng(0)
    D = np.c_[rng.normal(size=(6, 2)), np.ones((6, 1)), np.zeros((6, 2))]   # reads dims 0-1; dim 2 is a
    signal = np.c_[rng.normal(size=(50, 2)), np.zeros((50, 3))]             # common offset, centered away
    junk = np.c_[np.zeros((50, 2)), rng.normal(size=(50, 3))]
    B = readout_basis(D)
    Hp = readout_projection(signal + junk, D)
    assert B.shape == (2, 5) and Hp.shape == (50, 2)
    assert np.allclose(Hp @ B, signal, atol=1e-10)
    assert np.allclose(readout_projection(junk, D), 0.0, atol=1e-10)


def test_cca_ignores_float32_scale_junk_directions():
    # features computed in float32 with an exact affine constraint (e.g. probabilities summing
    # to 1) carry a junk singular direction at ~1e-6 of the top; the basis tolerance must reject
    # it, or the mean canonical correlation is silently dragged down by a noise dimension
    import numpy as np
    from src.code.analysis_utils import cca
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((500, 7))
    W = np.exp(np.concatenate([Z, np.zeros((500, 1))], 1))
    W = W / W.sum(1, keepdims=True)                                # exact rank 7
    H = np.tanh(Z @ rng.standard_normal((7, 32))) + 0.05 * rng.standard_normal((500, 32))
    s0 = np.linalg.svd(W - W.mean(0), compute_uv=False)[0]
    u = rng.standard_normal(500); u /= np.linalg.norm(u)
    v = rng.standard_normal(8); v /= np.linalg.norm(v)
    W_noisy = W + 1.5e-6 * s0 * np.outer(u, v)                     # junk direction at float32 scale
    assert abs(cca(H, W) - cca(H, W_noisy)) < 0.02


def test_align_representations_rotation_separates_what_linear_cannot():
    import numpy as np
    from src.code.analysis_utils import align_representations
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(400, 4))
    Q = np.linalg.qr(rng.normal(size=(4, 4)))[0]

    rotated = ref @ Q                                       # pure rotation: both methods perfect
    for method in ("linear", "rotation"):
        aligned, r2 = align_representations(rotated, ref, method=method)
        assert r2 > 0.999
        assert np.allclose(aligned, ref, atol=1e-6)         # aligned output lives in ref's space

    stretched = (ref * np.array([5.0, 1.0, 1.0, 1.0])) @ Q  # per-axis stretch: rotation can't fix it
    _, r2_lin = align_representations(stretched, ref, method="linear")
    _, r2_rot = align_representations(stretched, ref, method="rotation")
    assert r2_lin > 0.999 and r2_rot < 0.9

    wide = np.c_[ref @ Q, 0.01 * rng.normal(size=(400, 60))]   # wide hidden: PCA-truncate then rotate
    _, r2 = align_representations(wide, ref, method="rotation")
    assert r2 > 0.99


def test_heldout_decode_r2_recovers_linear_map_and_respects_groups():
    import numpy as np
    from src.code.analysis_utils import heldout_decode_r2
    rng = np.random.default_rng(0)
    F = rng.normal(size=(80, 3))
    groups = np.repeat(np.arange(40), 2)                    # duplicated prefixes share a group
    H = np.repeat(F[::2] @ rng.normal(size=(3, 32)), 2, axis=0)  # hidden is a linear image of F
    assert heldout_decode_r2(H, np.repeat(F[::2], 2, 0), groups, seed=0) > 0.95
    assert heldout_decode_r2(H, rng.normal(size=(80, 3)), groups, seed=0) < 0.3


def test_heldout_decode_r2_ignores_units_flat_on_the_train_split():
    """A saturated unit (tanh-flat over the training prefixes, moving only on a held-out one) carries
    no fitted information: standardizing it by its ~0 train std turns that flat noise into an O(1)
    regressor, then multiplies the held-out excursion by the same huge factor and wrecks R². Such
    units must be dropped, not rescaled."""
    import numpy as np
    from src.code.analysis_utils import heldout_decode_r2
    rng = np.random.default_rng(0)
    F = rng.normal(size=(80, 3))
    groups = np.arange(80)
    train_mask = np.arange(80) < 60
    flat = 1e-9 * rng.normal(size=80); flat[-1] = 50.0       # flat on train, excursion on one test row
    H = F @ rng.normal(size=(3, 16))
    clean = heldout_decode_r2(H, F, groups, train_mask=train_mask)
    assert clean > 0.95
    assert heldout_decode_r2(np.c_[H, flat], F, groups, train_mask=train_mask) > clean - 0.01
