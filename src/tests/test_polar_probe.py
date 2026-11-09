import numpy as np
import torch

from src.code.relations import KING, grid_graph, king_description, graph_distances, incidence


def test_grid_graph_links_exactly_the_unit_step_neighbours():
    """Algorithm 1: n distinct lattice points grown by signed unit steps, an edge (i, j, k) whenever j sits
    one positive step along axis k from i."""
    rng = np.random.default_rng(0)
    pos, edges = grid_graph(6, 2, rng)
    assert len({tuple(p) for p in pos}) == 6
    want = {(i, j, k) for i in range(6) for j in range(6) for k in range(2)
            if np.array_equal(pos[j] - pos[i], np.eye(2, dtype=int)[k])}
    assert set(edges) == want and len(edges) >= 5                       # connected: grown from one point


def test_distances_and_incidence_follow_the_edges():
    pos = np.array([[0], [1], [2]]); edges = [(0, 1, 0), (1, 2, 0)]
    assert graph_distances(3, edges).tolist() == [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
    M = incidence(3, edges, 1)
    assert M[0, 1, 0] == 1 and M[1, 0, 0] == -1 and M[0, 2, 0] == 0


def test_king_description_ordinality():
    """Header, then each edge once in a random direction, then 'Who are a, b, and c?' in vocabulary order."""
    rng = np.random.default_rng(3)
    names = ["d", "f", "c"]
    text, listing = king_description("ordinality", names, [(0, 1, 0), (1, 2, 0)], rng)
    assert text.startswith(KING["ordinality"]["header"].replace("[]", "d, f, c"))
    facts = text.split("**Scene description:**")[1].strip().split("\n")[0]
    assert len(facts.rstrip(".").split(". ")) == 2                     # two facts, one per edge
    assert any(f in facts for f in ("f is greater than d", "d is less than f"))
    assert any(f in facts for f in ("c is greater than f", "f is less than c"))
    assert text.endswith("Who are d, f, and c?") and listing == names


def test_polar_probe_recovers_a_planted_polar_code():
    """Codes that are a polar code (position on a lattice) in a random 8-dim subspace of 64 dims plus noise:
    the trained probe's distances and directions recover the graphs."""
    from src.code.polar_probe import evaluate, train_probe
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2, 64))
    def sample(k):
        out = []
        for _ in range(k):
            pos, edges = grid_graph(5, 2, rng)
            out.append((pos @ A + 0.05 * rng.normal(size=(5, 64)), graph_distances(5, edges), incidence(5, edges, 2)))
        return out
    probe = train_probe(sample(200), n_types=2, rank=8, epochs=200, lr=1e-2, seed=0)
    test = sample(50)
    ex, ty = evaluate(probe, test)
    # the planted code itself is the ceiling: type Spearman stays below 1 because unlinked pairs (diagonal
    # neighbours) have non-zero cosines with the prototypes where the gold incidence is 0
    oracle = {"B": np.eye(2), "P": np.eye(2), "mu": np.zeros(2), "sd": np.ones(2)}
    ex0, ty0 = evaluate(oracle, [(H @ np.linalg.pinv(A), D, M) for H, D, M in test])
    assert ex > ex0 - 0.05 and ty > ty0 - 0.05
