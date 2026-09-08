"""Hypothesis-space structures: intervals, rectangles, subtrees (Bayesian generalization) and
keys (key finding). Each is a bipartite kernel log p(x | h) with its own prior."""
import numpy as np
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import (KernelDataset, interval_hypotheses, rectangle_hypotheses,
                           subtree_hypotheses, key_profiles, KK_MAJOR, KK_MINOR)


def _rows(L):
    P = np.exp(L)
    return P, P.sum(1)


def test_intervals_are_uniform_on_their_span_with_an_erlang_prior():
    L, info = interval_hypotheses(n=4, prior="erlang", sigma=2.0)
    assert L.shape == (10, 4) and info["bipartite"]
    P, sums = _rows(L)
    assert np.allclose(sums, 1.0)
    for row, (a, b) in zip(P, info["hypotheses"]):
        inside = np.arange(1, 5)
        assert np.allclose(row[(inside >= a) & (inside <= b)], 1.0 / (b - a + 1))
        assert np.all(row[(inside < a) | (inside > b)] == 0.0)
    sizes = info["hypotheses"][:, 1] - info["hypotheses"][:, 0] + 1
    g = sizes * np.exp(-sizes / 2.0)
    assert np.allclose(info["prior"], g / g.sum())


def test_rectangles_are_uniform_on_their_area():
    L, info = rectangle_hypotheses(n=3, prior="uniform")
    assert L.shape == (36, 9)
    P, sums = _rows(L)
    assert np.allclose(sums, 1.0)
    areas = np.prod(info["hypotheses"][:, [1, 3]] - info["hypotheses"][:, [0, 2]] + 1, axis=1)
    assert np.allclose(P.max(1), 1.0 / areas)
    assert np.allclose((P > 0).sum(1), areas)
    assert np.allclose(info["prior"], 1.0 / 36)


def test_subtrees_are_uniform_on_their_leaves():
    L, info = subtree_hypotheses(n=2, k=2)
    assert L.shape == (7, 4)
    P, sums = _rows(L)
    assert np.allclose(sums, 1.0)
    assert np.allclose(P[0], 0.25)                                 # root: all leaves
    leaf_rows = P[[i for i, ls in enumerate(info["leaves"]) if len(ls) == 1]]
    assert leaf_rows.shape == (4, 4) and np.allclose(np.sort(leaf_rows.max(1)), 1.0)
    internal = [i for i, ls in enumerate(info["leaves"]) if len(ls) == 2]
    assert len(internal) == 2 and np.allclose(P[internal].max(1), 0.5)


def test_keys_are_rotated_profiles_and_beta_sharpens():
    L, info = key_profiles(beta=1.0)
    assert L.shape == (24, 12)
    P, sums = _rows(L)
    assert np.allclose(sums, 1.0)
    keys = info["keys"]
    c_major = P[[i for i, (t, m) in enumerate(keys) if t == 0 and m == "major"][0]]
    g_major = P[[i for i, (t, m) in enumerate(keys) if t == 7 and m == "major"][0]]
    assert np.allclose(c_major, np.asarray(KK_MAJOR) / np.sum(KK_MAJOR))
    assert np.allclose(g_major, np.roll(c_major, 7))
    a_minor = P[[i for i, (t, m) in enumerate(keys) if t == 9 and m == "minor"][0]]
    assert np.allclose(a_minor, np.roll(np.asarray(KK_MINOR) / np.sum(KK_MINOR), 9))
    P3 = np.exp(key_profiles(beta=3.0)[0])
    assert np.array_equal(P3.argmax(1), P.argmax(1))
    ent = lambda Q: -(Q * np.log(Q)).sum(1)
    assert np.all(ent(P3) < ent(P))


def _dataset(structure, infer, **kw):
    return KernelDataset(Config(OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": structure, "structure_kwargs": kw,
        "sample_mode": "sequence", "infer_latent": infer, "random_seed": 0,
        "training_kwargs": {"batch_size": 4, "context_length": 6}})))


def test_dataset_takes_the_structure_prior_and_bayes_matches_brute_force():
    ds = _dataset("intervals", infer=True, n=4, prior="erlang", sigma=2.0)
    L, info = interval_hypotheses(n=4, prior="erlang", sigma=2.0)
    assert ds.bipartite and ds.n_items == 10 and ds.n_outcomes == 4
    assert np.allclose(ds.prior.cpu().numpy(), info["prior"])
    # observe items 2 then 3 (tokens 1, 2): posterior ∝ prior * prod p(x | h), with the lapsed kernel
    ids = torch.tensor([[ds.start_token, 1, 2, 0]], device=ds.device)
    post = ds.optimal_predictions(ids)[0, 2].double().cpu().numpy()  # after two observations
    P = ds.P.double().cpu().numpy()
    brute = info["prior"] * P[:, 1] * P[:, 2]; brute /= brute.sum()
    assert np.allclose(post, brute, atol=1e-6)
    # the next-token target is the posterior predictive of the same belief
    nxt = _dataset("intervals", infer=False, n=4, prior="erlang", sigma=2.0)
    pred = nxt.optimal_predictions(ids)[0, 2].double().cpu().numpy()
    assert np.allclose(pred, brute @ P, atol=1e-6)
