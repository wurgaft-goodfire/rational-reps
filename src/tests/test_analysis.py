"""Unit tests for the array-side analysis: the readout projection, the comparison metrics, and
the alignment / decoding helpers."""


def test_readout_basis_spans_exactly_what_the_readout_reads():
    # H = a signal in the centered readout's row space + a component the readout cannot read: in
    # the basis' coordinates the signal has rank(D_c) components and lifts back exactly, the rest
    # vanishes
    import numpy as np
    from src.code.analysis_utils import readout_basis
    rng = np.random.default_rng(0)
    D = np.c_[rng.normal(size=(6, 2)), np.ones((6, 1)), np.zeros((6, 2))]   # reads dims 0-1; dim 2 is a
    signal = np.c_[rng.normal(size=(50, 2)), np.zeros((50, 3))]             # common offset, centered away
    junk = np.c_[np.zeros((50, 2)), rng.normal(size=(50, 3))]
    B = readout_basis(D)
    assert B.shape == (2, 5) and np.allclose(B @ B.T, np.eye(2))
    assert np.allclose(((signal + junk) @ B.T) @ B, signal, atol=1e-10)
    assert np.allclose(junk @ B.T, 0.0, atol=1e-10)


def test_representation_is_the_last_layer_at_the_positions_the_readout_reads():
    # static task: the last position, whatever the model's input length (a readout reads one row,
    # a sequence model reads [start, item] or [start, a, b]); sequence task: every position
    import numpy as np
    from src.code.analysis_utils import representation
    H = np.arange(2 * 4 * 3 * 5.0).reshape(2, 4, 3, 5)                  # (layers, B, T, width)
    assert np.array_equal(representation(H, static=True), H[-1][:, -1])
    assert np.array_equal(representation(H, static=False), H[-1])
    assert representation(None, static=True) is None


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


def test_rigid_align_matches_a_rotated_copy_where_truncate_then_rotate_cannot():
    """Aligning a hidden code to a predicted one must rotate in the full space and only then take the
    displayed modes. Truncating each side to its own top modes first fails whenever the spectrum is
    degenerate: each side's PCA picks an arbitrary basis of the tied block, so the two slices differ
    and no in-plane rotation can match them."""
    import numpy as np
    from src.code.analysis_utils import align_representations, rigid_align
    rng = np.random.default_rng(0)
    U = np.linalg.qr(rng.normal(size=(600, 4)))[0]
    E = U * np.array([1.0, 0.6, 0.6, 0.3])                 # modes 2 and 3 tied, as for a grid or a tree
    Q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    H = E @ Q + 1e-6 * rng.normal(size=E.shape)             # the same code in another basis

    X = rigid_align(H, E)                                   # H expressed in E's coordinates, by an isometry
    assert np.allclose(X, E, atol=1e-4)
    assert np.allclose(X[:, :2], E[:, :2], atol=1e-4)      # so the displayed modes agree point by point

    plane, r2 = align_representations(H, E[:, :2], method="rotation")   # truncate H to 2 dims, then rotate
    assert r2 < 0.9                                          # the slice is the wrong one

    Hn = H[:, :3]                                            # narrower hidden code: a partial isometry
    Xn = rigid_align(Hn, E)
    assert Xn.shape == E.shape
