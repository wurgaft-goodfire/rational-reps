"""Sparse structured attributes: N words, K factors (structure specs) each active w.p. p at a
uniform position, outcomes the factors' node tokens. Lc(x,(k,j)) = z_k alpha_k (1 + kappa_k(pos_k, j))."""
import numpy as np
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import KernelDataset, factors, feature_cube
from src.code.geometry import factor_logcond, factor_planes

RING = {"structure": "ring", "n": 8, "cooccurrence": "gaussian", "gamma": 2.0}


def test_one_node_factors_reproduce_feature_cube():
    # a factor with a single node has kappa = 0, so Lc = z_k alpha_k: the feature cube exactly
    K, N, p = 5, 64, 0.3
    L, info = factors([{"structure": "ring", "n": 1}] * K, N=N, p=p, decay=0.8, scale=1.5, seed=3)
    L_cube, info_cube = feature_cube(N, K, p, decay=0.8, scale=1.5, seed=3)
    assert np.allclose(L, L_cube)
    assert np.array_equal(info["Z"], info_cube["Z"])


def test_rings_decompose_into_orthogonal_circle_planes_and_presence():
    K = 3
    L, info = factors([RING] * K, N=2048, p=0.3, decay=0.8, scale=8.0, seed=0)
    _, E, D, S = factor_logcond(L)
    # rank 3K - 1: a circle plane per factor plus K - 1 presence dimensions (softmax gauge)
    r = 3 * K - 1
    assert S[r - 1] > 100 * S[r]
    planes = factor_planes(D[:r], info["blocks"])
    for (m, U), b in zip(planes, info["blocks"]):
        assert U.shape[1] == 2                                            # each block is a plane ...
        rad = np.linalg.norm(U.T @ (D[:r, b] - m[:, None]), axis=0)
        assert rad.std() / rad.mean() < 0.05                              # ... holding a circle (finite-N ellipse)
    # planes mutually orthogonal and orthogonal to the presence directions; exact in expectation,
    # O(1/sqrt(Np)) from the sampled positions
    for i in range(K):
        for j in range(i + 1, K):
            assert np.abs(planes[i][1].T @ planes[j][1]).max() < 0.05
        for m, _ in planes:
            assert np.abs(planes[i][1].T @ m).max() / np.linalg.norm(m) < 0.05
    M = np.stack([m for m, _ in planes], 1)
    assert np.linalg.matrix_rank(M, tol=1e-6 * np.abs(M).max()) == K - 1  # presence spans K - 1 dims


def test_single_ring_factor_is_a_circle_with_analytic_scale():
    # one factor: no presence dimension survives the softmax, the code is exactly the circle,
    # each Fourier mode carrying alpha^2 gamma^2 n/4 per active word
    N, n, gamma, scale = 4096, 8, 2.0, 8.0
    L, info = factors([dict(RING, n=n, gamma=gamma)], N=N, p=0.3, decay=0.8, scale=scale, seed=1)
    _, _, _, S = factor_logcond(L)
    assert S[1] > 1e6 * S[2]
    n_active = info["Z"].sum()
    s_pred = np.sqrt(n_active * n / 4) * scale * gamma
    assert np.allclose(S[:2], s_pred, rtol=0.05)


def test_plane_angles_recover_the_latent_positions():
    K = 3
    L, info = factors([RING] * K, N=2048, p=0.3, decay=0.8, scale=8.0, seed=2)
    _, E, D, S = factor_logcond(L)
    r = 3 * K - 1
    E, D = E[:, :r], D[:r]
    for k, ((m, U), b) in enumerate(zip(factor_planes(D, info["blocks"]), info["blocks"])):
        X = E @ U                                                          # words in factor k's plane
        active = info["Z"][:, k] == 1
        assert np.abs(X[~active]).max() < 1e-6                             # inactive words at the origin
        tok = np.arctan2(*(U.T @ (D[:, b] - m[:, None]))[::-1])            # token angles in the plane
        word = np.arctan2(X[active, 1], X[active, 0])
        # a word sits at the angle of its own position's token, up to the finite-N ellipticity
        err = np.angle(np.exp(1j * (word - tok[info["positions"][active, k]])))
        assert np.abs(err).max() < 0.05


def test_dense_activity_removes_the_presence_dimensions():
    # p = 1: every word carries every factor, so presence is a shared offset and the
    # item-centred code is the 2K circle dimensions alone
    K = 3
    for p, rank in ((0.3, 3 * K - 1), (1.0, 2 * K)):
        L, _ = factors([RING] * K, N=1024, p=p, decay=0.8, scale=8.0, seed=0)
        _, E, _, S = factor_logcond(L)
        Ec = E - E.mean(0)
        assert np.linalg.matrix_rank(Ec, tol=1e-6 * np.sqrt(S[0])) == rank


def test_factors_dataset_is_bipartite():
    cfg = Config(OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "factors",
        "structure_kwargs": {"factors": [RING, RING], "N": 32, "p": 0.3, "decay": 0.8, "scale": 8.0, "seed": 2},
        "random_seed": 0, "model_kwargs": {"model_type": "linear"}, "training_kwargs": {"batch_size": 16},
    }))
    ds = KernelDataset(cfg)
    assert ds.bipartite and ds.vocab_size == 32 + 16 and ds.n_outcomes == 16
    assert np.allclose(ds.conditional, np.log(np.asarray(ds.P.cpu())), atol=1e-6)
    assert np.allclose(ds.features, ds.info["Z"])


def test_factor_planes_figure_renders():
    from src.code import plot_utils
    L, info = factors([RING, RING], N=64, p=0.3, decay=0.8, scale=8.0, seed=0)
    _, E, D, _ = factor_logcond(L)
    fig = plot_utils.plot_factor_planes(E, D, {"model": (E, D.T)}, info, scores={"model": 1.0})   # readout is (C, d)
    assert len(fig.axes) == 2 * 2                                         # (predicted + 1 model) x 2 factors
