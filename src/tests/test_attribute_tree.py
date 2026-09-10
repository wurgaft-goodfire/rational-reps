"""Category tree whose nodes own attribute factors; items are every (leaf, positions) combination.
Lc(x,(a,j)) = alpha_a [owner on x's path] (1 + kappa_a(z_a, j))."""
import numpy as np
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import KernelDataset, attribute_tree
from src.code.geometry import factor_logcond, factor_planes

RING = {"structure": "ring", "n": 8, "cooccurrence": "gaussian", "gamma": 1.0, "alpha": 2.0}
# binary root split (nodes 1, 2 are the categories): one shared ring at the root, one private ring each
MINIMAL = dict(attributes=[dict(RING, node=0), dict(RING, node=1), dict(RING, node=2)], n=1, k=2)


def _code(**kw):
    L, info = attribute_tree(**kw)
    _, E, D, S = factor_logcond(L)
    r = int((S > 1e-8 * S[0]).sum())
    return L, info, E[:, :r], D[:r], S[:r]


def test_minimal_instance_has_the_analytic_spectrum():
    # rank 1 (root split) + 2 x 3 rings = 7; items enumerated, so the moments are exact:
    # shared ring pair sqrt(N n/4) alpha gamma, root split sqrt(N) alpha sqrt(2n)/2, private pairs sqrt(N/2 n/4) alpha gamma
    L, info, E, D, S = _code(**MINIMAL)
    N, n, alpha, gamma = L.shape[0], 8, 2.0, 1.0
    assert N == 2 * 8 * 8
    s = np.linalg.svd(E - E.mean(0), compute_uv=False)
    assert s[6] > 1e-3 * s[0] and (s[7:] < 1e-8 * s[0]).all()
    shared = np.sqrt(N * n / 4) * alpha * gamma
    split = np.sqrt(N) * alpha * np.sqrt(2 * n) / 2
    private = np.sqrt(N / 2 * n / 4) * alpha * gamma
    # E carries sqrt(S), so its singular values squared are the log-conditional's
    assert np.allclose(s[:7] ** 2, [split, shared, shared] + [private] * 4, rtol=1e-6)


def test_planes_are_orthogonal_and_the_circles_exact():
    L, info, E, D, S = _code(**MINIMAL)
    planes = factor_planes(D, info["blocks"])
    for (m, U), b in zip(planes, info["blocks"]):
        assert U.shape[1] == 2
        rad = np.linalg.norm(U.T @ (D[:, b] - m[:, None]), axis=0)
        assert rad.std() / rad.mean() < 1e-6
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.abs(planes[i][1].T @ planes[j][1]).max() < 1e-8
    axis = planes[1][0] - planes[2][0]                    # the category axis: the private blocks' presence contrast
    for _, U in planes:
        assert np.abs(U.T @ axis).max() / np.linalg.norm(axis) < 1e-8


def test_plane_occupancy_follows_the_tree():
    L, info, E, D, S = _code(**MINIMAL)
    leaf = info["leaf"]
    assert (info["Z"][:, 0] == 1).all()                                       # shared: everyone
    assert np.array_equal(info["Z"][:, 1] == 1, leaf == 1) and np.array_equal(info["Z"][:, 2] == 1, leaf == 2)
    for a, ((m, U), b) in enumerate(zip(factor_planes(D, info["blocks"]), info["blocks"])):
        X = E @ U
        tok = np.arctan2(*(U.T @ (D[:, b] - m[:, None]))[::-1])
        on = info["Z"][:, a] == 1
        if (~on).any():
            assert np.abs(X[~on]).max() < 1e-9                                # non-owners at the origin
        word = np.arctan2(X[on, 1], X[on, 0])
        assert np.abs(np.angle(np.exp(1j * (word - tok[info["positions"][on, a]])))).max() < 1e-6


def test_top_three_components_are_the_category_and_the_shared_ring():
    # PC1 is the split (every item contrasts), PC2-3 the shared ring: the 3-D view is two stacked circles
    L, info, E, D, S = _code(**MINIMAL)
    U, s, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    cat = (info["leaf"] == 1).astype(float); cat -= cat.mean()
    corr = [abs(np.corrcoef(U[:, k], cat)[0, 1]) for k in range(3)]
    assert corr[0] > 1 - 1e-8 and max(corr[1:]) < 1e-6
    r = np.linalg.norm(U[:, 1:3] * s[1:3], axis=1)
    assert r.std() / r.mean() < 1e-6                                          # every item on the shared circle


def test_a_split_whose_children_own_nothing_is_invisible():
    # every attribute at the root: the categories differ in nothing the outcomes can see
    L, info, E, D, S = _code(attributes=[dict(RING, node=0)] * 3, n=1, k=2)
    assert len(S) == 6                                                        # 3 rings, no split dimension
    leaf = info["leaf"]
    assert np.allclose(E[leaf == 1], E[leaf == 2])                            # same items, same code


SEASONS = {"structure": "ring", "n": 12, "m": 4, "cooccurrence": "gaussian", "gamma": 2.0, "alpha": 2.0}
TAG = {"structure": "ring", "n": 1, "cooccurrence": "gaussian", "gamma": 1.0, "alpha": 2.0}
# months x hemispheres -> seasons: a shared 12-position ring read out on 4 coarse tokens, and a hemisphere token each
HEMISPHERES = dict(attributes=[dict(SEASONS, node=0), dict(TAG, node=1), dict(TAG, node=2)], n=1, k=2)


def test_coarse_outcome_attribute_separates_positions_from_tokens():
    L, info = attribute_tree(**HEMISPHERES)
    assert L.shape == (2 * 12, 4 + 1 + 1)                                    # 24 items, 6 tokens
    assert info["blocks"][0] == slice(0, 4) and info["blocks"][1] == slice(4, 5)
    assert set(np.unique(info["positions"][:, 0])) == set(range(12))


def test_hemispheres_are_two_stacked_season_circles():
    L, info, E, D, S = _code(**HEMISPHERES)
    Ec = E - E.mean(0)
    s = np.linalg.svd(Ec, compute_uv=False)
    assert s[2] > 1e-3 * s[0] and (s[3:] < 1e-8 * s[0]).all()               # rank 3: season pair + hemisphere axis
    U, s, _ = np.linalg.svd(Ec, full_matrices=False)
    ring = [k for k in range(3) if abs(np.corrcoef(U[:, k], info["leaf"])[0, 1]) < 1e-6]
    assert len(ring) == 2                                                     # two components carry no hemisphere
    X = Ec @ np.linalg.svd(Ec, full_matrices=False)[2][ring].T
    r = np.linalg.norm(X, axis=1)
    assert r.std() / r.mean() < 1e-6                                          # everyone on one circle
    # the same season phase sits at the same angle in both hemispheres
    th = np.arctan2(X[:, 1], X[:, 0])
    north, south = info["leaf"] == 1, info["leaf"] == 2
    assert np.allclose(np.sort(th[north]), np.sort(th[south]), atol=1e-8)


def test_attribute_tree_dataset_is_bipartite():
    cfg = Config(OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "attribute_tree", "structure_kwargs": MINIMAL,
        "random_seed": 0, "model_kwargs": {"model_type": "linear"}, "training_kwargs": {"batch_size": 16},
    }))
    ds = KernelDataset(cfg)
    assert ds.bipartite and ds.n_items == 128 and ds.n_outcomes == 24 and ds.vocab_size == 128 + 24


def test_attribute_tree_figure_renders():
    from src.code import plot_utils
    L, info, E, D, S = _code(**MINIMAL)
    fig = plot_utils.plot_attribute_tree(E, D, {"model": (E, D.T)}, info, scores={"model": 1.0})
    assert len(fig.axes) == 2 * (1 + 3 + 1)                                   # per row: 3-D view, 3 planes, Gram
    L, info, E, D, S = _code(**HEMISPHERES)
    fig = plot_utils.plot_attribute_tree(E, D, {}, info, colors=info["positions"][:, 0] / 12)
    assert len(fig.axes) == 1 + 1 + 1                                         # one-token attributes have no plane panel
