"""A tree whose levels have shapes: the children of every node at level l form the structure ``levels[l]``
(ring, chain, complete) instead of a complete graph. Two leaves are similar at level l only if they share the
parent above it, so the code is a direct sum ordered by level: the root's shape on every leaf, then one private
copy of the next shape per node."""
import numpy as np
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import KernelDataset, graph_conditional, tree_conditional, tree_parents
from src.code.geometry import factor_logcond

RING = {"structure": "ring", "n": 12, "cooccurrence": "gaussian", "gamma": 1.0, "alpha": 2.0}
CHAIN = {"structure": "chain", "n": 3, "cooccurrence": "gaussian", "gamma": 1.0, "alpha": 1.0}


def _code(levels):
    L = tree_conditional(levels=levels)
    _, E, D, S = factor_logcond(L)
    r = int((S > 1e-8 * S[0]).sum())
    return L, E[:, :r], S[:r]


def test_levels_set_the_branching_per_level():
    parents = tree_parents(levels=[RING, CHAIN])
    assert len(parents) == 1 + 12 + 36                                        # root, 12 months, 36 leaves
    assert parents[1] == 0 and parents[2] == 1 and parents[5] == 0             # DFS pre-order: month, its 3 leaves, next month


def test_ring_over_chain_is_a_shared_circle_with_private_lines():
    L, E, S = _code([RING, CHAIN])
    assert L.shape == (36, 36)                                                # leaves predict leaves
    Ec = E - E.mean(0)
    U, s, _ = np.linalg.svd(Ec, full_matrices=False)
    # the root ring (2 modes, every leaf), then 12 private chain lines (the chain's linear mode, one per
    # month), then the chain's quadratic mode, a column effect constant over a month's rows: 12 such
    # block-constant pieces share their 12-dim row space with the ring's block-constant rows, so 10
    # survive. Rank 2 + 12 + 10 = 24, and item-centring removes the mean row, which lies in that
    # block-constant space: 23.
    assert s[22] > 1e-3 * s[0] and (s[23:] < 1e-8 * s[0]).all()
    assert np.allclose(s[:2], s[0]) and s[2] < 0.3 * s[0]                     # the ring pair leads
    X = U[:, :2] * s[:2]
    r = np.linalg.norm(X, axis=1)
    assert r.std() / r.mean() < 1e-3                                          # every leaf on the root circle
    month = np.repeat(np.arange(12), 3)
    th = np.arctan2(X[:, 1], X[:, 0])
    gap = np.abs(np.angle(np.exp(1j * (th[:, None] - th[None]))))
    assert (gap[month[:, None] == month[None]] < 1e-3).all()                  # a month's leaves share an angle
    assert (gap[month[:, None] != month[None]] > 0.4).all()                   # months are spread around it
    # the deeper part of the code is private per month: across months the table's Gram is the ring's alone
    kr = graph_conditional({key: v for key, v in RING.items() if key != "alpha"})
    kr = RING["alpha"] * (kr - kr.mean(1, keepdims=True))
    M1 = kr[month][:, month]
    Lc = L - L.mean(1, keepdims=True)
    cross = month[:, None] != month[None]
    assert np.allclose((Lc @ Lc.T)[cross], (M1 @ M1.T)[cross], atol=1e-10)


def test_complete_levels_give_an_ultrametric():
    complete = {"structure": "complete", "n": 3, "cooccurrence": "gaussian", "gamma": 1.0, "alpha": 1.0}
    L, E, S = _code([complete, complete])
    G = (E - E.mean(0)) @ (E - E.mean(0)).T
    parent = np.repeat(np.arange(3), 3)
    off = ~np.eye(9, dtype=bool)
    same = off & (parent[:, None] == parent[None]); diff = parent[:, None] != parent[None]
    assert G[same].std() < 1e-8 and G[diff].std() < 1e-8 and G[same].mean() > G[diff].mean()


def test_default_tree_is_unchanged_by_the_levels_argument():
    assert np.allclose(tree_conditional(n=2, k=3), tree_conditional(n=2, k=3, levels=None))
    assert np.allclose(graph_conditional({"structure": "tree", "n": 2, "k": 3, "gamma": 0.85}), tree_conditional(n=2, k=3))


def test_levels_tree_dataset_is_a_connected_graph():
    cfg = Config(OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "tree", "structure_kwargs": {"levels": [RING, CHAIN]},
        "random_seed": 0, "model_kwargs": {"model_type": "linear"}, "training_kwargs": {"batch_size": 16},
    }))
    ds = KernelDataset(cfg)
    assert not ds.bipartite and ds.n_items == 36 and ds.vocab_size == 36
